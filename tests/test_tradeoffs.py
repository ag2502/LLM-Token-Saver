from datetime import timedelta

import pytest

from llm_token_saver.detect import FindingKind
from llm_token_saver.detect.context import analyze_context
from llm_token_saver.detect.effort import effort_finding
from llm_token_saver.detect.routing import repriced, subagent_model_finding, upgrade_finding
from llm_token_saver.schema import Dataset
from tests.builders import ONE_CHAR_PER_TOKEN, PRICES, T0, ThreadBuilder

OPUS_55_OUTPUT = 20.0


def dataset(*builders: ThreadBuilder) -> Dataset:
    return Dataset(threads=[builder.thread for builder in builders])


def test_effort_saving_compares_with_the_high_effort_baseline() -> None:
    builder = ThreadBuilder()
    for _ in range(20):
        builder.call(read=1_000, output=200, thinking=100, effort="high")
    for _ in range(5):
        builder.call(read=1_000, output=500, thinking=400, effort="max")
    data = dataset(builder)
    finding = effort_finding(data, PRICES, analyze_context(data, PRICES, ONE_CHAR_PER_TOKEN))
    assert finding is not None
    assert finding.kind is FindingKind.TRADEOFF
    assert finding.savings_usd == pytest.approx(5 * 300 * OPUS_55_OUTPUT / 1e6)
    assert "~400 thinking tokens each vs ~100 at high" in finding.evidence[0].note


def test_upgrade_counts_only_calls_after_the_successor_was_in_use() -> None:
    old = ThreadBuilder("old").call(read=1_000_000, model="claude-opus-5", at=T0)
    new = ThreadBuilder("new").call(read=10, at=T0 + timedelta(days=10))
    late = ThreadBuilder("late").call(
        read=1_000_000, model="claude-opus-5", at=T0 + timedelta(days=20)
    )
    finding = upgrade_finding(dataset(old, new, late), PRICES)
    assert finding is not None
    assert finding.savings_usd == pytest.approx(1_000_000 * (0.50 - 0.20) / 1e6)
    assert finding.evidence[0].label == "claude-opus-5 → claude-opus-5-5"
    assert finding.evidence[0].note == "1 calls"


def test_subagent_step_down_is_a_price_only_ceiling() -> None:
    agent = ThreadBuilder("agent-1", sidechain=True).call(read=1_000_000, output=10_000)
    finding = subagent_model_finding(dataset(agent), PRICES)
    assert finding is not None and finding.ceiling
    opus = 1_000_000 * 0.20 / 1e6 + 10_000 * 20 / 1e6
    sonnet = 1_000_000 * 0.10 / 1e6 + 10_000 * 10 / 1e6
    assert finding.savings_usd == pytest.approx(opus - sonnet)


def test_repricing_across_tokenizer_generations_scales_tokens() -> None:
    call = ThreadBuilder().call(read=1_000_000, model="claude-sonnet-4-5").thread.calls[0]
    cost = repriced(call, "claude-sonnet-5-5", PRICES)
    assert cost == pytest.approx(1_000_000 * (3.3 / 2.5) * 0.10 / 1e6)
