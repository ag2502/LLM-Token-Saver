"""Detector 6: request-level waste: failed and retried calls, repeated requests, and scheduled
jobs that could use the Batch API."""

from __future__ import annotations

from collections import defaultdict
from itertools import pairwise

from llm_token_saver.detect.base import Evidence, Finding, FindingKind
from llm_token_saver.pricing import PriceTable
from llm_token_saver.report.format import plural
from llm_token_saver.schema import Call, Dataset, MarkerKind

BURST_GAP_SECONDS = 10
MIN_BURST_CALLS = 50
BURST_SHARE = 0.9
"""A service is batch-shaped when this share of its independent requests arrive in bursts."""


def api_error_finding(dataset: Dataset) -> Finding | None:
    errors = [
        marker
        for thread in dataset.threads
        for marker in thread.markers
        if marker.kind is MarkerKind.API_ERROR
        and (
            dataset.window_start is None
            or (marker.timestamp is not None and marker.timestamp >= dataset.window_start)
        )
    ]
    if not errors:
        return None
    sessions = len({marker.session_id for marker in errors})
    return Finding(
        detector="api_errors",
        title="Requests failed and were retried",
        kind=FindingKind.INFO,
        summary=(
            f"{len(errors):,} requests failed and were retried across {sessions:,} sessions. "
            "Failed requests report no usage, so they add waiting time rather than cost here."
        ),
        fix="Nothing to change unless errors cluster at certain hours or in certain projects.",
    )


def duplicate_request_finding(dataset: Dataset, prices: PriceTable) -> Finding | None:
    """Byte-identical requests (model, prompt and parameters) sent more than once."""
    groups: dict[str, list[Call]] = defaultdict(list)
    for call in dataset.calls:
        if call.request_hash:
            groups[call.request_hash].append(call)
    repeats: dict[str, list[float]] = defaultdict(list)
    for calls in groups.values():
        calls.sort(key=lambda call: call.timestamp)
        for repeat in calls[1:]:
            cost = prices.cost(repeat)
            if cost is not None:
                repeats[repeat.project].append(cost.total)
    total = sum(sum(costs) for costs in repeats.values())
    if total <= 0:
        return None
    count = sum(len(costs) for costs in repeats.values())
    return Finding(
        detector="duplicate_requests",
        title="Identical requests are sent again and again",
        kind=FindingKind.FREE_WIN,
        summary=(
            f"{plural(count, 'request')} repeated an earlier request byte for byte (same model, "
            "prompt and parameters). Unless a fresh answer is wanted each time, a response "
            "cache could serve them."
        ),
        fix=(
            "Cache responses keyed on the model, prompt and sampling parameters, with an expiry "
            "that matches how often the answer may change."
        ),
        savings_usd=total,
        measured=True,
        evidence=[
            Evidence(service, sum(costs), plural(len(costs), "repeat"))
            for service, costs in sorted(repeats.items(), key=lambda kv: -sum(kv[1]))
        ],
    )


def batchable_finding(dataset: Dataset, prices: PriceTable) -> Finding | None:
    """Independent requests that arrive in dense bursts, the shape of a scheduled job."""
    sizes = {thread.thread_id: len(thread.calls) for thread in dataset.threads}
    independent: dict[str, list[Call]] = defaultdict(list)
    for call in dataset.calls:
        if sizes.get(call.thread_id) == 1 and not call.is_batch:
            independent[call.project].append(call)
    rows: list[tuple[str, int, int, float]] = []
    for service, calls in independent.items():
        calls.sort(key=lambda call: call.timestamp)
        bursts: list[list[Call]] = [[calls[0]]]
        for previous, call in pairwise(calls):
            if (call.timestamp - previous.timestamp).total_seconds() <= BURST_GAP_SECONDS:
                bursts[-1].append(call)
            else:
                bursts.append([call])
        large = [burst for burst in bursts if len(burst) >= MIN_BURST_CALLS]
        bursty = [call for burst in large for call in burst]
        if not bursty or len(bursty) < BURST_SHARE * len(calls):
            continue
        cost = sum(c.total for call in bursty if (c := prices.cost(call)) is not None)
        rows.append((service, len(bursty), len(large), cost * (1 - prices.batch_multiplier)))
    if not rows:
        return None
    rows.sort(key=lambda row: -row[3])
    return Finding(
        detector="batchable",
        title="Scheduled jobs pay full price for answers nobody waits for",
        kind=FindingKind.FREE_WIN,
        summary=(
            "These services send independent requests in dense bursts (each request within "
            f"{BURST_GAP_SECONDS} seconds of the last), the shape of a scheduled job. If nobody "
            "is waiting on the results, the Batch API answers within 24 hours at half price."
        ),
        fix=(
            "Send these requests through the provider's Batch API (Claude Message Batches, "
            "OpenAI Batch, Gemini Batch): 50% off every token, and the discount stacks with "
            "prompt caching."
        ),
        savings_usd=sum(row[3] for row in rows),
        evidence=[
            Evidence(service, saving, f"{calls:,} requests in {plural(bursts, 'burst')}")
            for service, calls, bursts, saving in rows
        ],
    )
