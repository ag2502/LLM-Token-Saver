from datetime import UTC, datetime, timedelta

import pytest

from llm_token_saver.pricing import PriceTable
from llm_token_saver.report.format import middle_truncate, percent, tokens
from llm_token_saver.schema import Call
from llm_token_saver.stats import compute_stats
from llm_token_saver.store import Store

START = datetime(2026, 10, 1, 9, 0, tzinfo=UTC)


def call(n: int, *, session: str = "s1", model: str = "claude-opus-5-5", **tokens: int) -> Call:
    return Call(
        call_id=f"c{n}",
        source="test",
        session_id=session,
        thread_id=session,
        seq=n,
        timestamp=START + timedelta(hours=n),
        model=model,
        project=f"~/{session}",
        effort="high",
        **tokens,  # type: ignore[arg-type]
    )


def test_totals_split_and_breakdowns() -> None:
    calls = [
        call(0, cache_read=1_000_000, output=1_000),
        call(1, cache_read=1_000_000, cache_write_1h=100_000, output=1_000),
        call(2, session="s2", model="claude-sonnet-5-5", input_uncached=1_000_000),
        call(3, session="s2", model="mystery-model", output=5_000),
    ]
    stats = compute_stats(Store(calls, PriceTable.bundled()))
    assert (stats.calls, stats.sessions, stats.threads) == (4, 2, 2)
    assert stats.cost.cache_read == pytest.approx(0.40)
    assert stats.cost.cache_write_1h == pytest.approx(0.80)
    assert stats.cost.output == pytest.approx(0.04)
    assert stats.cost.input_uncached == pytest.approx(2.00)
    assert stats.cost.total == pytest.approx(3.24)
    assert [group.key for group in stats.by_model] == [
        "claude-sonnet-5-5",
        "claude-opus-5-5",
        "mystery-model",
    ]
    assert [(group.key, group.calls) for group in stats.unpriced_models] == [("mystery-model", 1)]
    assert stats.top_sessions[0].session_id == "s2"
    assert stats.top_sessions[1].peak_prompt_tokens == 1_100_000
    assert stats.tokens.prompt == 3_100_000
    assert stats.cache_hit_rate == pytest.approx(2_000_000 / 3_100_000)
    assert stats.prompt_p50 == 1_000_000
    assert [row.calls for row in stats.by_day] == [4]


def test_days_follow_local_offset() -> None:
    late_evening_utc = [call(14), call(15)]  # 23:00 and 00:00 UTC
    stats = compute_stats(Store(late_evening_utc, PriceTable.bundled()), utc_offset_seconds=0)
    assert len(stats.by_day) == 2
    ist = compute_stats(Store(late_evening_utc, PriceTable.bundled()), utc_offset_seconds=19800)
    assert len(ist.by_day) == 1


def test_formatting_helpers() -> None:
    assert tokens(950) == "950"
    assert tokens(12_400) == "12.4K"
    assert tokens(3_420_000_000) == "3.4B"
    assert tokens(716_055) == "716K"
    assert percent(0.004) == "<1%"
    assert percent(0.58) == "58%"
    assert middle_truncate("short") == "short"
    label = middle_truncate("~/Library/Application Support/Claude/scratch-workspaces/x/y-2026")
    assert len(label) == 44 and label.startswith("~/Library") and label.endswith("y-2026")
