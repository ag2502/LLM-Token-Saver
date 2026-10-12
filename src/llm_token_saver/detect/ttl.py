"""Detector 4: whether the 5-minute or the 1-hour prompt cache is cheaper for this traffic.

A 5-minute cache write costs 1.25x the input price and a 1-hour one 2x; a cache read
refreshes either. With requests seconds apart the 5-minute cache never expires and its cheaper
writes win. With pauses of 5 to 60 minutes it expires and the whole prompt is written again,
which the 1-hour cache avoids. Each conversation is replayed under both lifetimes from its
real request times and prompt sizes, then compared per configuration group (a service, or a
Claude Code project's main sessions and subagents, which have separate settings).
"""

from __future__ import annotations

from collections import defaultdict

from llm_token_saver.detect.base import Evidence, Finding, FindingKind
from llm_token_saver.pricing import PriceTable
from llm_token_saver.pricing.engine import PER_MILLION
from llm_token_saver.report.format import plural
from llm_token_saver.schema import Call, Dataset

FIVE_MINUTES, ONE_HOUR = 300, 3_600


def simulate(calls: list[Call], prices: PriceTable, ttl: int) -> float:
    """Prompt cost of a conversation if every request were cached with lifetime `ttl`.

    Each request reads what the previous one cached, unless the gap exceeded the lifetime,
    and writes the rest of its prompt.
    """
    total = 0.0
    previous: Call | None = None
    for call in calls:
        rates = prices.rates_for(call)
        if rates is None:
            continue
        write = rates.cache_write_1h if ttl == ONE_HOUR else rates.cache_write_5m
        prompt = call.prompt_tokens
        expired = previous is None or (call.timestamp - previous.timestamp).total_seconds() > ttl
        if expired or previous is None:
            total += prompt * write
        else:
            reused = min(previous.prompt_tokens, prompt)
            total += reused * rates.cache_read + (prompt - reused) * write
        previous = call
    return total / PER_MILLION


def ttl_findings(dataset: Dataset, prices: PriceTable) -> list[Finding]:
    groups: dict[tuple[str, str, bool], list[list[Call]]] = defaultdict(list)
    for thread in dataset.threads:
        calls = [call for call in thread.calls if dataset.in_window(call)]
        if len(calls) >= 2 and any(call.cache_read or call.cache_write for call in calls):
            groups[(thread.source, thread.project, thread.is_sidechain)].append(calls)
    shorter: list[tuple[str, str, bool, int, float]] = []
    longer: list[tuple[str, str, bool, int, float]] = []
    for (source, project, sidechain), conversations in groups.items():
        written_1h = sum(c.cache_write_1h for calls in conversations for c in calls)
        written_5m = sum(c.cache_write_5m for calls in conversations for c in calls)
        current = ONE_HOUR if written_1h > written_5m else FIVE_MINUTES
        other = FIVE_MINUTES if current == ONE_HOUR else ONE_HOUR
        saving = sum(
            simulate(c, prices, current) - simulate(c, prices, other) for c in conversations
        )
        if saving > 0:
            row = (source, project, sidechain, len(conversations), saving)
            (shorter if other == FIVE_MINUTES else longer).append(row)
    findings = []
    if shorter:
        findings.append(_finding(shorter, FIVE_MINUTES))
    if longer:
        findings.append(_finding(longer, ONE_HOUR))
    return findings


def _finding(rows: list[tuple[str, str, bool, int, float]], target: int) -> Finding:
    rows.sort(key=lambda row: -row[4])
    claude_code = any(source == "claude_code" for source, *_ in rows)
    apps = any(source != "claude_code" for source, *_ in rows)
    fixes = []
    ttl = "5m" if target == FIVE_MINUTES else "1h"
    if claude_code:
        fixes.append(
            f'In Claude Code, set `"promptCacheTtl": "{ttl}"` (main sessions) or '
            f'`"subagentPromptCacheTtl": "{ttl}"` (subagents) in that project\'s '
            ".claude/settings.json."
        )
    if apps:
        fixes.append(
            'In your app, use `cache_control: {"type": "ephemeral"}` (5 minutes, the default).'
            if target == FIVE_MINUTES
            else 'In your app, use `cache_control: {"type": "ephemeral", "ttl": "1h"}`.'
        )
    if target == FIVE_MINUTES:
        title = "The 1-hour cache costs more than it saves here"
        summary = (
            "These conversations rarely pause for more than 5 minutes, so the 1-hour cache's "
            "doubled write price buys nothing: replayed with the 5-minute cache, every request "
            "still reads the cache and writes cost 1.25x instead of 2x the input price."
        )
    else:
        title = "Pauses of 5 to 60 minutes keep expiring the cache"
        summary = (
            "These conversations often pause for more than 5 minutes, so the 5-minute cache "
            "expires and the whole prompt is written again. Replayed with the 1-hour cache, "
            "those requests read the cache instead, which outweighs its 2x write price."
        )
    return Finding(
        detector=f"cache_ttl_{ttl}",
        title=title,
        kind=FindingKind.FREE_WIN,
        summary=summary,
        fix=" ".join(fixes),
        savings_usd=sum(row[4] for row in rows),
        evidence=[
            Evidence(
                f"{'subagents in ' if sidechain else ''}{project or source}",
                saving,
                plural(conversations, "conversation"),
            )
            for source, project, sidechain, conversations, saving in rows
        ],
    )
