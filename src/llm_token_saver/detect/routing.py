"""Detector 8 (price-only): what the same calls would cost on another model.

These are price comparisons, not quality claims: the router in a later release decides which
calls can move with a quality guarantee.
"""

from __future__ import annotations

import dataclasses
from collections import defaultdict
from datetime import datetime, timedelta

from llm_token_saver.detect.base import Evidence, Finding, FindingKind
from llm_token_saver.estimate import default_chars_per_token
from llm_token_saver.pricing import PriceTable, normalize_model_id
from llm_token_saver.schema import Call, Dataset

RECENT_DAYS = 7
SUBAGENT_STEP_DOWN = {"claude-opus-5-5": "claude-sonnet-5-5", "claude-opus-5": "claude-sonnet-5-5"}


def repriced(call: Call, model: str, prices: PriceTable) -> float | None:
    """Cost of `call` on `model`, adjusting token counts across tokenizer generations."""
    factor = default_chars_per_token(call.model) / default_chars_per_token(model)
    moved = dataclasses.replace(
        call,
        model=model,
        input_uncached=round(call.input_uncached * factor),
        cache_read=round(call.cache_read * factor),
        cache_write_5m=round(call.cache_write_5m * factor),
        cache_write_1h=round(call.cache_write_1h * factor),
        output=round(call.output * factor),
        thinking=round(call.thinking * factor),
    )
    cost = prices.cost(moved)
    return cost.total if cost else None


def upgrade_finding(dataset: Dataset, prices: PriceTable) -> Finding | None:
    """Calls on an older model after its cheaper successor was already in use (or recently)."""
    first_seen: dict[str, datetime] = {}
    last_seen: datetime | None = None
    for thread in dataset.threads:
        for call in thread.calls:
            model = normalize_model_id(call.model)
            if model not in first_seen or call.timestamp < first_seen[model]:
                first_seen[model] = call.timestamp
            last_seen = max(last_seen or call.timestamp, call.timestamp)
    if last_seen is None:
        return None
    recent = last_seen - timedelta(days=RECENT_DAYS)
    rows: dict[tuple[str, str], list[float]] = defaultdict(lambda: [0, 0.0, 0.0])
    for call in dataset.calls:
        price = prices.lookup(call.model)
        successor = price.successor if price else None
        current = prices.cost(call)
        if successor is None or current is None:
            continue
        available_from = first_seen.get(successor, recent)
        if call.timestamp < available_from:
            continue
        upgraded = repriced(call, successor, prices)
        if upgraded is None:
            continue
        row = rows[(normalize_model_id(call.model), successor)]
        row[0] += 1
        row[1] += current.total
        row[2] += current.total - upgraded
    savings = sum(row[2] for row in rows.values())
    if savings <= 0:
        return None
    return Finding(
        detector="model_upgrade",
        title="Older models were used where a cheaper successor exists",
        kind=FindingKind.TRADEOFF,
        summary=(
            "These calls ran on a model whose newer successor in the same tier costs less "
            "per token, after you had started using the successor (or in the last "
            f"{RECENT_DAYS} days). Newer models usually match or beat their predecessors, "
            "but check the switch on your own work."
        ),
        fix="Switch with /model, or set `model` in ~/.claude/settings.json.",
        savings_usd=savings,
        evidence=[
            Evidence(f"{old} → {new}", saving, f"{int(n):,} calls")
            for (old, new), (n, _, saving) in sorted(rows.items(), key=lambda kv: -kv[1][2])
        ],
    )


def subagent_model_finding(dataset: Dataset, prices: PriceTable) -> Finding | None:
    """Price-only ceiling: subagent runs on Opus repriced on Sonnet 5.5."""
    calls = savings = 0.0
    for thread in dataset.threads:
        if not thread.is_sidechain:
            continue
        for call in thread.calls:
            target = SUBAGENT_STEP_DOWN.get(normalize_model_id(call.model))
            current = prices.cost(call)
            if target is None or current is None or not dataset.in_window(call):
                continue
            cheaper = repriced(call, target, prices)
            if cheaper is not None:
                calls += 1
                savings += current.total - cheaper
    if savings <= 0:
        return None
    return Finding(
        detector="subagent_model",
        title="Subagents ran on Opus",
        kind=FindingKind.TRADEOFF,
        summary=(
            f"{int(calls):,} subagent calls ran on Opus. Priced on Sonnet 5.5 they would cost "
            "less; this is an upper bound on price alone, not a quality judgement."
        ),
        fix=(
            "Give routine subagents (search, exploration, summaries) a cheaper model with the "
            "`model` field of the agent definition, for example `model: sonnet`."
        ),
        savings_usd=savings,
        ceiling=True,
    )
