"""Detector 3: prompt-cache misses that a change to how the prompt is built would fix.

Requests are grouped into families (service and model) and compared with the request before
them. Two causes are named:

- **A volatile value early in the prompt.** Consecutive system prompts or tool lists differ
  only in a short window (a timestamp, a request id), so the cache entry never matches and
  the stable text around it is written again on every call.
- **No caching at all.** A family shares a prefix long enough to cache, but no request ever
  read or wrote the cache.

The saving prices the stable tokens at the rate they would have cost with the value moved
after the cache breakpoint (or caching turned on): a cache read when the previous request was
within the cache lifetime, a cache write otherwise.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import pairwise

from llm_token_saver.detect.base import Evidence, Finding, FindingKind, fresh_rate
from llm_token_saver.estimate import TokenEstimator
from llm_token_saver.pricing import PriceTable, normalize_model_id
from llm_token_saver.pricing.engine import PER_MILLION
from llm_token_saver.report.format import plural
from llm_token_saver.schema import Call, Dataset, PromptPart

VOLATILE_WINDOW_CHARS = 200
"""A change no longer than this inside a system prompt or tool list reads as a dynamic value."""

CONTEXT_CHARS = 40
"""Stable characters shown before a volatile value, to say where it is."""

FIVE_MINUTES, ONE_HOUR = 300, 3_600


@dataclass(frozen=True, slots=True)
class PromptChange:
    """Where a prompt first differs from the one before it."""

    part_index: int
    kind: str
    offset: int
    """Characters into the part where the difference starts."""
    window: int
    """Characters that differ."""
    stable_chars: int
    """Characters of the prompt up to the end of the changed part, outside the window."""
    context: str
    """The stable text just before the change."""


def common_prefix_length(a: str, b: str) -> int:
    """Length of the shared start of two strings (binary search over C-speed slice compares)."""
    low, high = 0, min(len(a), len(b))
    while low < high:
        middle = (low + high + 1) // 2
        if a[:middle] == b[:middle]:
            low = middle
        else:
            high = middle - 1
    return low


def common_suffix_length(a: str, b: str) -> int:
    low, high = 0, min(len(a), len(b))
    while low < high:
        middle = (low + high + 1) // 2
        if a[len(a) - middle :] == b[len(b) - middle :]:
            low = middle
        else:
            high = middle - 1
    return low


def prompt_change(previous: list[PromptPart], current: list[PromptPart]) -> PromptChange | None:
    """The first difference, if it is a small edit inside a system prompt or tool list."""
    stable = 0
    for index, (before, after) in enumerate(zip(previous, current, strict=False)):
        if before.content_hash == after.content_hash:
            stable += after.chars
            continue
        if after.kind not in {"system", "tools"} or before.kind != after.kind:
            return None
        head = common_prefix_length(before.text, after.text)
        tail = common_suffix_length(before.text[head:], after.text[head:])
        window = after.chars - head - tail
        if window > VOLATILE_WINDOW_CHARS:
            return None
        return PromptChange(
            part_index=index,
            kind=after.kind,
            offset=head,
            window=window,
            stable_chars=stable + after.chars - window,
            context=after.text[max(0, head - CONTEXT_CHARS) : head],
        )
    return None


def shared_prefix_chars(previous: list[PromptPart], current: list[PromptPart]) -> int:
    chars = 0
    for before, after in zip(previous, current, strict=False):
        if before.content_hash == after.content_hash:
            chars += after.chars
            continue
        return chars + common_prefix_length(before.text, after.text)
    return chars


def prompt_cache_findings(
    dataset: Dataset, prices: PriceTable, estimator: TokenEstimator
) -> list[Finding]:
    families: dict[tuple[str, str], list[Call]] = defaultdict(list)
    for call in dataset.calls:
        if call.prompt:
            families[(call.project, normalize_model_id(call.model))].append(call)
    volatile: list[tuple[str, int, float, PromptChange]] = []
    uncached: list[tuple[str, int, float]] = []
    for (service, model), calls in families.items():
        calls.sort(key=lambda call: call.timestamp)
        ratio = estimator.ratio(model)
        caches = any(call.cache_read or call.cache_write for call in calls)
        changed: list[tuple[float, PromptChange]] = []
        never_cached = 0.0
        for previous, call in pairwise(calls):
            rates = prices.rates_for(call)
            if rates is None or previous.prompt is None or call.prompt is None:
                continue
            gap = (call.timestamp - previous.timestamp).total_seconds()
            ttl = ONE_HOUR if call.cache_write_1h else FIVE_MINUTES
            change = prompt_change(previous.prompt, call.prompt)
            if caches and change is not None and change.window:
                tokens = max(0.0, change.stable_chars / ratio - call.cache_read)
                paid = (
                    rates.cache_write_5m if call.cache_write >= tokens else fresh_rate(call, rates)
                )
                ideal = rates.cache_read if gap <= ttl else rates.cache_write_5m
                changed.append((tokens * (paid - ideal) / PER_MILLION, change))
            elif not caches:
                tokens = shared_prefix_chars(previous.prompt, call.prompt) / ratio
                if tokens >= prices.min_cacheable_tokens(model):
                    ideal = rates.cache_read if gap <= FIVE_MINUTES else rates.cache_write_5m
                    never_cached += tokens * (rates.input - ideal) / PER_MILLION
        saving = sum(amount for amount, _ in changed)
        if saving > 0:
            most_common = max(changed, key=lambda item: item[0])[1]
            volatile.append((service, len(changed), saving, most_common))
        if never_cached > 0:
            uncached.append((service, len(calls), never_cached))
    findings = []
    if volatile:
        findings.append(_volatile_finding(volatile))
    if uncached:
        findings.append(_uncached_finding(uncached))
    return findings


def _volatile_finding(rows: list[tuple[str, int, float, PromptChange]]) -> Finding:
    rows.sort(key=lambda row: -row[2])
    service, _, _, example = rows[0]
    where = f"the {example.kind} prompt" if example.kind == "system" else "the tool list"
    return Finding(
        detector="prompt_cache",
        title="A changing value at the top of the prompt defeats the cache",
        kind=FindingKind.FREE_WIN,
        summary=(
            f"In {service}, consecutive requests differ in just "
            f"{plural(example.window, 'character')} of {where}, {example.offset:,} characters "
            "in, right after "
            f'"{example.context.strip()}". Everything around that value is written to the '
            "cache again on every request instead of being read."
        ),
        fix=(
            "Move the changing value (a timestamp, request id or user name) out of the system "
            "prompt into the latest user message, after the last cache breakpoint, so the "
            "prefix stays byte-identical between requests."
        ),
        savings_usd=sum(row[2] for row in rows),
        evidence=[
            Evidence(service, saving, f"{calls:,} requests rewrote the cached prefix")
            for service, calls, saving, _ in rows
        ],
    )


def _uncached_finding(rows: list[tuple[str, int, float]]) -> Finding:
    rows.sort(key=lambda row: -row[2])
    return Finding(
        detector="prompt_cache",
        title="A shared prompt prefix is never cached",
        kind=FindingKind.FREE_WIN,
        summary=(
            "These services send the same long prompt prefix on consecutive requests, but no "
            "request read from or wrote to the prompt cache."
        ),
        fix=(
            "On Claude, add a `cache_control` breakpoint at the end of the shared prefix (or "
            "turn on automatic caching). OpenAI and Gemini cache long prefixes automatically: "
            "check that the prefix is byte-identical and comes first."
        ),
        savings_usd=sum(row[2] for row in rows),
        evidence=[
            Evidence(service, saving, f"{calls:,} requests") for service, calls, saving in rows
        ],
    )
