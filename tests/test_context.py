from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from llm_token_saver.detect import CostPool, Finding, FindingKind, rank
from llm_token_saver.detect.context import (
    COMPACTION_KEEPS,
    analyze_context,
    duplicate_context_finding,
    stale_context_finding,
    tool_output_findings,
)
from llm_token_saver.estimate import TokenEstimator
from llm_token_saver.pricing import PriceTable
from llm_token_saver.schema import Block, BlockKind, Call, Dataset, Marker, MarkerKind, Thread

PRICES = PriceTable.bundled()
ONE_CHAR_PER_TOKEN = TokenEstimator({"claude-opus-5-5": 1.0})
T0 = datetime(2026, 10, 1, tzinfo=UTC)
READ, WRITE_5M = 0.20, 5.0  # Claude Opus 5.5, USD per million tokens


class ThreadBuilder:
    def __init__(self) -> None:
        self.seq = 0
        self.thread = Thread(thread_id="t", session_id="s", source="test", project="~/p")

    def block(
        self,
        chars: int,
        kind: BlockKind = BlockKind.TOOL_RESULT,
        tool: str = "Bash",
        detail: str = "",
        content_hash: str = "",
    ) -> ThreadBuilder:
        self.thread.blocks.append(
            Block(
                session_id="s",
                thread_id="t",
                seq=self._next(),
                kind=kind,
                chars=chars,
                tool=tool,
                detail=detail,
                content_hash=content_hash,
            )
        )
        return self

    def call(
        self, read: int = 0, write: int = 0, uncached: int = 0, output: int = 0
    ) -> ThreadBuilder:
        seq = self._next()
        self.thread.calls.append(
            Call(
                call_id=f"c{seq}",
                source="test",
                session_id="s",
                thread_id="t",
                seq=seq,
                timestamp=T0 + timedelta(minutes=seq),
                model="claude-opus-5-5",
                input_uncached=uncached,
                cache_read=read,
                cache_write_5m=write,
                output=output,
            )
        )
        return self

    def compact(self) -> ThreadBuilder:
        self.thread.markers.append(
            Marker(session_id="s", thread_id="t", seq=self._next(), kind=MarkerKind.COMPACTION)
        )
        return self

    def _next(self) -> int:
        self.seq += 1
        return self.seq


def analyze(builder: ThreadBuilder, window_start: datetime | None = None, **kwargs: int):
    dataset = Dataset(threads=[builder.thread], window_start=window_start)
    return analyze_context(dataset, PRICES, ONE_CHAR_PER_TOKEN, **kwargs)


def test_a_block_is_billed_at_write_then_read_rates() -> None:
    builder = ThreadBuilder().block(1_000).call(write=11_000).call(read=11_000)
    context = analyze(builder)
    [carried] = context.blocks
    assert carried.calls == 2
    assert carried.cost == pytest.approx(1_000 * (WRITE_5M + READ) / 1e6)
    assert context.unseen_cost == pytest.approx(10_000 * (WRITE_5M + READ) / 1e6)
    assert context.prompt_cost == pytest.approx(11_000 * (WRITE_5M + READ) / 1e6)
    assert context.first_prefix_tokens == {"t": 10_000}


def test_compaction_stops_charging_older_blocks() -> None:
    builder = ThreadBuilder().block(1_000).call(read=5_000).compact().block(500).call(read=5_000)
    first, second = analyze(builder).blocks
    assert (first.calls, second.calls) == (1, 1)
    assert (first.epoch, second.epoch) == (0, 1)


def test_calls_before_the_window_are_not_charged() -> None:
    builder = ThreadBuilder().block(1_000).call(read=5_000).call(read=5_000)
    context = analyze(builder, window_start=T0 + timedelta(minutes=3))
    [carried] = context.blocks
    assert carried.calls == 1
    assert carried.cost == pytest.approx(1_000 * READ / 1e6)


def test_block_estimates_larger_than_the_prompt_are_scaled_to_fit() -> None:
    builder = ThreadBuilder().block(10_000).call(read=5_000)
    context = analyze(builder)
    assert context.unseen_cost == 0
    assert context.blocks[0].cost == pytest.approx(context.prompt_cost)


def test_stale_cost_counts_content_older_than_the_threshold() -> None:
    builder = ThreadBuilder().block(1_000)
    for _ in range(4):
        builder.call(read=1_000)
    context = analyze(builder, stale_after_calls=2)
    assert context.stale_cost == pytest.approx(2 * 1_000 * READ / 1e6)
    finding = stale_context_finding(context, Dataset(threads=[builder.thread]))
    assert finding is not None
    assert finding.savings_usd == pytest.approx(context.stale_cost * (1 - COMPACTION_KEEPS))


@st.composite
def random_threads(draw: st.DrawFn) -> ThreadBuilder:
    builder = ThreadBuilder()
    steps = draw(st.lists(st.integers(0, 9), min_size=1, max_size=40))
    for step in steps:
        if step < 4:
            builder.block(draw(st.integers(0, 60_000)))
        elif step < 9:
            builder.call(
                read=draw(st.integers(0, 300_000)),
                write=draw(st.integers(0, 80_000)),
                uncached=draw(st.integers(0, 20_000)),
            )
        else:
            builder.compact()
    return builder.call(read=draw(st.integers(0, 300_000)))


@settings(max_examples=200, deadline=None)
@given(random_threads())
def test_attributed_costs_always_add_up_to_the_bill(builder: ThreadBuilder) -> None:
    context = analyze(builder)
    billed = sum(cost.prompt for call in builder.thread.calls if (cost := PRICES.cost(call)))
    attributed = sum(item.cost for item in context.blocks) + context.unseen_cost
    assert context.prompt_cost == pytest.approx(billed, rel=1e-9, abs=1e-12)
    assert attributed == pytest.approx(billed, rel=1e-9, abs=1e-12)
    assert all(item.cost >= -1e-12 for item in context.blocks)


def test_tool_output_saving_assumes_results_capped_at_2000_tokens() -> None:
    builder = ThreadBuilder().block(5_000, detail="npm test").call(read=20_000)
    [finding] = tool_output_findings(analyze(builder))
    assert finding.title == "Long command output stays in context"
    assert finding.savings_usd == pytest.approx(3_000 * READ / 1e6)
    assert finding.evidence[0].label == "npm test"


def test_duplicates_count_only_within_the_same_context() -> None:
    builder = (
        ThreadBuilder()
        .block(800, tool="Read", detail="a.py", content_hash="h1")
        .call(read=10_000)
        .block(800, tool="Read", detail="a.py", content_hash="h1")
        .call(read=10_000)
        .compact()
        .block(800, tool="Read", detail="a.py", content_hash="h1")
        .call(read=10_000)
    )
    context = analyze(builder)
    finding = duplicate_context_finding(context)
    assert finding is not None
    assert finding.savings_usd == pytest.approx(context.blocks[1].cost)
    assert "1 tool results" in finding.summary


def finding(savings: float, kind: FindingKind = FindingKind.FREE_WIN) -> Finding:
    return Finding(
        detector="d", title=f"f{savings}", kind=kind, summary="", fix="", savings_usd=savings
    )


def test_rank_shrinks_overlapping_savings_and_puts_info_last() -> None:
    ranked = rank(
        [finding(50), finding(0, FindingKind.INFO), finding(60), finding(30, FindingKind.TRADEOFF)],
        {CostPool.PROMPT: 100.0},
    )
    assert [f.title for f in ranked] == ["f60", "f50", "f30", "f0"]
    assert [round(f.savings_usd, 6) for f in ranked] == [60.0, 20.0, 6.0, 0.0]
