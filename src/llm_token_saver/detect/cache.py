"""Detector: prompt-cache rebuilds in the middle of a conversation.

In a healthy loop each call reads everything the previous call sent from the cache and writes
only what was appended since. A call that instead writes most of the conversation again paid
the write price on tokens that should have been cheap reads.
"""

from __future__ import annotations

from collections import defaultdict

from llm_token_saver.detect.base import Evidence, Finding, FindingKind
from llm_token_saver.pricing import PriceTable
from llm_token_saver.pricing.engine import PER_MILLION
from llm_token_saver.schema import Call, Dataset, Marker, MarkerKind
from llm_token_saver.timeline import timeline

MIN_REBUILD_TOKENS = 10_000
MISSED_SHARE = 0.5
"""A call counts as a rebuild when it read less than this share of the previous prompt."""


def rebuild_cause(previous: Call, call: Call) -> str:
    if previous.model != call.model:
        return "model switched mid-session"
    ttl = 3_600 if previous.cache_write_1h or call.cache_write_1h else 300
    idle = (call.timestamp - previous.timestamp).total_seconds()
    if idle > ttl:
        return f"idle past the {'1-hour' if ttl == 3_600 else '5-minute'} cache lifetime"
    return "prompt prefix changed"


def cache_rebuild_finding(dataset: Dataset, prices: PriceTable) -> Finding | None:
    extra_by_cause: dict[str, list[float]] = defaultdict(list)
    for thread in dataset.threads:
        previous: Call | None = None
        compacted = False
        for event in timeline(thread):
            if isinstance(event, Marker):
                compacted = compacted or event.kind is MarkerKind.COMPACTION
                continue
            if not isinstance(event, Call):
                continue
            call = event
            if previous is not None and not compacted and dataset.in_window(call):
                extra = _rebuild_extra_cost(previous, call, prices)
                if extra > 0:
                    extra_by_cause[rebuild_cause(previous, call)].append(extra)
            previous, compacted = call, False
    total = sum(sum(costs) for costs in extra_by_cause.values())
    if total <= 0:
        return None
    count = sum(len(costs) for costs in extra_by_cause.values())
    return Finding(
        detector="cache_rebuilds",
        title="The prompt cache was rebuilt mid-session",
        kind=FindingKind.FREE_WIN,
        summary=(
            f"{count:,} calls wrote most of their conversation to the cache again instead of "
            "reading it. The saving is what those tokens cost above the cache-read price."
        ),
        fix=(
            "A pause longer than the cache lifetime makes the next call re-write the whole "
            "conversation. Before stepping away from a long session, /compact it (or /clear "
            "if the task is done) so the rebuild is small. Switching models mid-session also "
            "rebuilds the cache, because caches are per model."
        ),
        savings_usd=total,
        measured=True,
        ceiling=True,
        evidence=[
            Evidence(cause, sum(costs), f"{len(costs):,} rebuilds")
            for cause, costs in sorted(extra_by_cause.items(), key=lambda kv: -sum(kv[1]))
        ],
    )


def _rebuild_extra_cost(previous: Call, call: Call, prices: PriceTable) -> float:
    expected = previous.prompt_tokens
    missed = min(expected - call.cache_read, call.cache_write)
    if call.cache_write < MIN_REBUILD_TOKENS or missed <= MISSED_SHARE * expected:
        return 0.0
    rates = prices.rates_for(call)
    if rates is None:
        return 0.0
    write_rate = (
        call.cache_write_5m * rates.cache_write_5m + call.cache_write_1h * rates.cache_write_1h
    ) / call.cache_write
    return missed * (write_rate - rates.cache_read) / PER_MILLION
