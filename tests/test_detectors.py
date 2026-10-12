from datetime import timedelta

import pytest

from llm_token_saver.detect import FindingKind
from llm_token_saver.detect.cache import cache_rebuild_finding
from llm_token_saver.detect.context import analyze_context
from llm_token_saver.detect.prefix import fixed_prefix_finding
from llm_token_saver.detect.traffic import api_error_finding
from llm_token_saver.schema import BlockKind, Dataset, Marker, MarkerKind
from tests.builders import ONE_CHAR_PER_TOKEN, PRICES, T0, ThreadBuilder

OPUS_55 = {"read": 0.20, "write_1h": 8.0}


def dataset(*builders: ThreadBuilder) -> Dataset:
    return Dataset(threads=[builder.thread for builder in builders])


def test_fixed_prefix_reports_session_and_subagent_starts() -> None:
    main = ThreadBuilder("main").block(1_000, kind=BlockKind.USER_TEXT).call(read=25_000)
    agent = ThreadBuilder("agent-1", sidechain=True).block(500).call(read=10_500)
    data = dataset(main, agent)
    finding = fixed_prefix_finding(analyze_context(data, PRICES, ONE_CHAR_PER_TOKEN), data)
    assert finding is not None
    assert finding.kind is FindingKind.INFO
    assert "~24.0K tokens" in finding.summary and "~10.0K" in finding.summary
    assert finding.evidence[0].cost_usd == pytest.approx(2 * 1_000 * OPUS_55["read"] / 1e6)


def test_idle_session_rebuild_costs_the_write_premium() -> None:
    builder = (
        ThreadBuilder()
        .call(write_1h=500_000, at=T0)
        .call(write_1h=500_000, at=T0 + timedelta(hours=2))
    )
    finding = cache_rebuild_finding(dataset(builder), PRICES)
    assert finding is not None
    assert finding.measured and finding.ceiling
    assert finding.savings_usd == pytest.approx(
        500_000 * (OPUS_55["write_1h"] - OPUS_55["read"]) / 1e6
    )
    assert finding.evidence[0].label == "idle for more than an hour"


def test_model_switch_and_changed_prefix_are_told_apart() -> None:
    switch = ThreadBuilder("a").call(write=100_000).call(write=100_000, model="claude-sonnet-5-5")
    changed = ThreadBuilder("b").call(write=100_000).call(write=100_000)
    finding = cache_rebuild_finding(dataset(switch, changed), PRICES)
    assert finding is not None
    assert {e.label for e in finding.evidence} == {
        "model switched mid-session",
        "prompt prefix changed",
    }


def test_a_healthy_loop_has_no_rebuilds() -> None:
    healthy = ThreadBuilder().call(write=100_000).call(read=100_000, write=2_000)
    assert cache_rebuild_finding(dataset(healthy), PRICES) is None


def test_rebuilds_after_a_compaction_are_expected() -> None:
    builder = ThreadBuilder().call(write=100_000).compact().call(write=100_000)
    assert cache_rebuild_finding(dataset(builder), PRICES) is None


def test_api_errors_in_window_are_counted() -> None:
    builder = ThreadBuilder().call(read=10)
    builder.thread.markers.append(
        Marker(session_id="s", thread_id="t", seq=99, kind=MarkerKind.API_ERROR, timestamp=T0)
    )
    finding = api_error_finding(dataset(builder))
    assert finding is not None and finding.summary.startswith("1 requests failed")
