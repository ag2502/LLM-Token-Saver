"""Detector 7: thinking spent at the highest effort levels.

Thinking is billed as output, and most of it stays in the context (see `estimate.py`), so a
high-effort turn costs more when it is generated and again on every later call. The saving
compares each max/xhigh turn with the average thinking of the same model's high-effort turns.
"""

from __future__ import annotations

from collections import defaultdict

from llm_token_saver.detect.base import CostPool, Evidence, Finding, FindingKind
from llm_token_saver.detect.context import CarriedContext
from llm_token_saver.pricing import PriceTable
from llm_token_saver.pricing.engine import PER_MILLION
from llm_token_saver.schema import BlockKind, Dataset

ELEVATED_EFFORTS = ("max", "xhigh")
BASELINE_EFFORT = "high"
MIN_BASELINE_CALLS = 20


def effort_finding(dataset: Dataset, prices: PriceTable, context: CarriedContext) -> Finding | None:
    totals: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
    for call in dataset.calls:
        tally = totals[(call.model, call.effort)]
        tally[0] += 1
        tally[1] += call.thinking
    baseline = {
        model: thinking / calls
        for (model, effort), (calls, thinking) in totals.items()
        if effort == BASELINE_EFFORT and calls >= MIN_BASELINE_CALLS
    }
    generated: dict[str, float] = defaultdict(float)
    for call in dataset.calls:
        rates = prices.rates_for(call)
        if call.effort in ELEVATED_EFFORTS and call.model in baseline and rates is not None:
            excess = max(0.0, call.thinking - baseline[call.model])
            generated[call.effort] += excess * rates.output / PER_MILLION
    carried: dict[str, float] = defaultdict(float)
    for effort in ELEVATED_EFFORTS:
        shrink = _pooled_shrink(totals, baseline, effort)
        carried[effort] = shrink * sum(
            item.cost
            for item in context.blocks
            if item.block.kind is BlockKind.THINKING and item.block.detail == effort
        )
    savings = sum(generated.values()) + sum(carried.values())
    if savings <= 0:
        return None
    evidence = []
    for effort in ELEVATED_EFFORTS:
        calls, thinking = _pooled(totals, effort, baseline)
        base_calls, base_thinking = _pooled(totals, BASELINE_EFFORT, baseline)
        if calls and base_calls:
            evidence.append(
                Evidence(
                    f"{effort} effort",
                    generated[effort] + carried[effort],
                    f"{calls:,} turns, ~{thinking / calls:,.0f} thinking tokens each vs "
                    f"~{base_thinking / base_calls:,.0f} at high",
                )
            )
    return Finding(
        detector="effort",
        title="Highest-effort turns think much more",
        kind=FindingKind.TRADEOFF,
        summary=(
            "Turns at max and xhigh effort produced more thinking than the same model's "
            "high-effort turns. Thinking is billed as output and most of it stays in the "
            "context, so it is paid for again on later calls. The saving assumes those turns "
            "had thought as much as a typical high-effort turn; hard problems may need more."
        ),
        fix=(
            "Make high the default (/effort high, or `effortLevel`, per model under "
            "`modelSettings` in ~/.claude/settings.json) and raise it for the problems that "
            "need it."
        ),
        savings_usd=savings,
        pool=CostPool.OUTPUT,
        evidence=evidence,
    )


def _pooled(
    totals: dict[tuple[str, str], list[int]], effort: str, baseline: dict[str, float]
) -> tuple[int, int]:
    """Calls and thinking at `effort`, across models that have a high-effort baseline."""
    calls = thinking = 0
    for (model, level), (n, t) in totals.items():
        if level == effort and model in baseline:
            calls += n
            thinking += t
    return calls, thinking


def _pooled_shrink(
    totals: dict[tuple[str, str], list[int]], baseline: dict[str, float], effort: str
) -> float:
    calls, thinking = _pooled(totals, effort, baseline)
    base_calls, base_thinking = _pooled(totals, BASELINE_EFFORT, baseline)
    if not calls or not base_calls or not thinking:
        return 0.0
    return max(0.0, 1 - (base_thinking / base_calls) / (thinking / calls))
