"""Carried-context cost: what each piece of context cost while it stayed in the prompt.

Every call re-sends the whole conversation, so a tool result added at turn 10 is paid for
again on turns 11, 12, ... until the session ends or compacts. To attribute a call's prompt
cost to the content inside it, the replay lays the prompt out in order

    [unseen prefix: system prompt, tool definitions, ...][block 1][block 2] ... [newest block]

and bills it the way prompt caching does: the first `cache_read` tokens at the cache-read
rate, the next `cache_write` tokens at the cache-write rate, and the uncached tail at the
input rate. Block sizes are estimates (see `estimate.py`); when they add up to more than the
billed prompt they are scaled down to fit, so attributed costs always sum to the bill.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import defaultdict
from dataclasses import dataclass, field

from llm_token_saver.detect.base import Evidence, Finding, FindingKind
from llm_token_saver.estimate import TokenEstimator
from llm_token_saver.pricing import PriceTable, Rates
from llm_token_saver.pricing.engine import PER_MILLION
from llm_token_saver.report.format import percent, plural, tokens
from llm_token_saver.schema import Block, BlockKind, Call, Dataset, Marker, MarkerKind, Thread
from llm_token_saver.timeline import timeline

STALE_AFTER_CALLS = 100
"""Content older than this many calls counts as stale."""

TOOL_OUTPUT_CAP_TOKENS = 2_000
"""The output size the tool-output saving assumes (about 5,000 characters)."""

COMPACTION_KEEPS = 0.1
"""Share of stale content assumed to survive as a compaction summary."""

MIN_DUPLICATE_TOKENS = 50


@dataclass(slots=True)
class CarriedBlock:
    block: Block
    tokens: int
    epoch: int
    """Which compaction-delimited stretch of its thread the block belongs to."""
    cost: float = 0.0
    """USD paid to keep this block in the prompt across in-window calls."""
    calls: int = 0
    """In-window calls whose prompt included this block."""


@dataclass(slots=True)
class CarriedContext:
    blocks: list[CarriedBlock] = field(default_factory=list)
    prompt_cost: float = 0.0
    """USD of every in-window prompt; block costs plus `unseen_cost` add up to this."""
    unseen_cost: float = 0.0
    """Prompt cost not attributable to a logged block: the fixed prefix and the harness."""
    stale_cost: float = 0.0
    stale_by_thread: dict[str, float] = field(default_factory=dict)
    first_prefix_tokens: dict[str, int] = field(default_factory=dict)
    """Unseen prefix size at each thread's first call: system prompt, tools, and so on."""
    prefix_token_cost: float = 0.0
    """USD one more token of fixed prefix would have cost across all in-window calls."""
    stale_after_calls: int = STALE_AFTER_CALLS


def analyze_context(
    dataset: Dataset,
    prices: PriceTable,
    estimator: TokenEstimator,
    *,
    stale_after_calls: int = STALE_AFTER_CALLS,
) -> CarriedContext:
    result = CarriedContext(stale_after_calls=stale_after_calls)
    for thread in dataset.threads:
        _replay(thread, dataset, prices, estimator, result)
    return result


class _Epoch:
    """Blocks in context since the last compaction, with lazily summed per-token rates."""

    def __init__(self, index: int) -> None:
        self.index = index
        self.items: list[CarriedBlock] = []
        self.cum: list[int] = [0]
        self.entry_clock: list[int] = []
        self.entry_window_calls: list[int] = []
        self._rate_steps: list[float] = [0.0]  # difference array of per-token rates
        self._direct: list[float] = []  # token x rate for blocks a boundary cuts through

    @property
    def tokens(self) -> int:
        return self.cum[-1]

    def add(self, item: CarriedBlock, clock: int, window_calls: int) -> None:
        self.items.append(item)
        self.cum.append(self.cum[-1] + item.tokens)
        self.entry_clock.append(clock)
        self.entry_window_calls.append(window_calls)
        self._rate_steps.append(0.0)
        self._direct.append(0.0)

    def tokens_entered_by(self, clock: int) -> int:
        """Tokens of the blocks that entered before call number `clock` was made."""
        return self.cum[bisect_right(self.entry_clock, clock)]

    def charge(self, start: float, end: float, rate: float) -> None:
        """Bill `rate` USD per million tokens on block positions [start, end)."""
        start, end = max(start, 0.0), min(end, float(self.cum[-1]))
        if end <= start or rate == 0.0:
            return
        first = bisect_right(self.cum, start) - 1
        last = bisect_left(self.cum, end) - 1
        if first == last:
            self._direct[first] += (end - start) * rate
            return
        self._direct[first] += (self.cum[first + 1] - start) * rate
        if last > first + 1:
            self._rate_steps[first + 1] += rate
            self._rate_steps[last] -= rate
        self._direct[last] += (end - self.cum[last]) * rate

    def close(self, window_calls: int) -> list[CarriedBlock]:
        rate = 0.0
        for index, item in enumerate(self.items):
            rate += self._rate_steps[index]
            item.cost += (rate * item.tokens + self._direct[index]) / PER_MILLION
            item.calls = window_calls - self.entry_window_calls[index]
        return self.items


def _replay(
    thread: Thread,
    dataset: Dataset,
    prices: PriceTable,
    estimator: TokenEstimator,
    result: CarriedContext,
) -> None:
    epoch = _Epoch(0)
    clock = 0  # calls made so far in this thread, in or out of the window
    window_calls = 0
    model = thread.calls[0].model if thread.calls else ""
    stale_cost = 0.0
    for event in timeline(thread):
        if isinstance(event, Marker):
            if event.kind is MarkerKind.COMPACTION:
                result.blocks.extend(epoch.close(window_calls))
                epoch = _Epoch(epoch.index + 1)
            continue
        if isinstance(event, Block):
            item = CarriedBlock(event, estimator.tokens(event, model), epoch.index)
            epoch.add(item, clock, window_calls)
            continue
        call: Call = event
        model = call.model
        prompt = call.prompt_tokens
        live = epoch.tokens
        scale = prompt / live if live > prompt else 1.0
        unseen = prompt - live * scale
        if clock == 0:
            result.first_prefix_tokens[thread.thread_id] = round(unseen)
        age_clock, clock = clock, clock + 1
        rates = prices.rates_for(call)
        if rates is None or not dataset.in_window(call):
            continue
        stale_tokens = (
            epoch.tokens_entered_by(age_clock - result.stale_after_calls)
            if age_clock >= result.stale_after_calls
            else 0
        )
        stale_end = unseen + scale * stale_tokens
        segments = _segments(call, rates)
        for start, end, rate in segments:
            if end <= start:
                continue
            result.unseen_cost += _overlap(start, end, 0.0, unseen) * rate / PER_MILLION
            stale_cost += _overlap(start, end, unseen, stale_end) * rate / PER_MILLION
            result.prompt_cost += (end - start) * rate / PER_MILLION
            if end > unseen and scale > 0:
                epoch.charge(
                    (max(start, unseen) - unseen) / scale, (end - unseen) / scale, rate * scale
                )
        first_rate = next((rate for start, end, rate in segments if end > start), 0.0)
        result.prefix_token_cost += first_rate / PER_MILLION
        window_calls += 1
    result.blocks.extend(epoch.close(window_calls))
    result.stale_cost += stale_cost
    if stale_cost:
        result.stale_by_thread[thread.thread_id] = stale_cost


def _segments(call: Call, rates: Rates) -> list[tuple[float, float, float]]:
    """Prompt spans in billing order: cache reads, then cache writes, then uncached input."""
    read_end = call.cache_read
    write_end = read_end + call.cache_write
    write_rate = (
        (call.cache_write_5m * rates.cache_write_5m + call.cache_write_1h * rates.cache_write_1h)
        / call.cache_write
        if call.cache_write
        else 0.0
    )
    return [
        (0.0, read_end, rates.cache_read),
        (read_end, write_end, write_rate),
        (write_end, write_end + call.input_uncached, rates.input),
    ]


def _overlap(start: float, end: float, low: float, high: float) -> float:
    return max(0.0, min(end, high) - max(start, low))


# Findings ------------------------------------------------------------------------------------

_TOOL_GROUPS = {
    "Bash": (
        "Long command output stays in context",
        "Make commands print less: pipe through `tail -n 40` or `grep`, use quiet flags "
        "(`pytest -q`, `npm ci --silent`, `git log --oneline -n 20`), and print only the "
        "lines you need from a file instead of `cat`-ing it. To cap every result, lower "
        "`BASH_MAX_OUTPUT_LENGTH` (default 30000 characters) in the `env` block of "
        "~/.claude/settings.json.",
    ),
    "Read": (
        "Whole-file reads stay in context",
        "Read the part you need: ask for a line range, or Grep for the section first. Every "
        "full read is re-sent on each later call until the session ends or compacts.",
    ),
    "MCP tools": (
        "Large MCP tool results stay in context",
        "Ask MCP tools for narrower results (filters, limits, text instead of full dumps). "
        "`MAX_MCP_OUTPUT_TOKENS` (default 25000) caps each response.",
    ),
    "Web": (
        "Fetched pages stay in context",
        "Fetch with a specific question so only the relevant part comes back.",
    ),
    "Subagents": (
        "Long subagent reports stay in context",
        "Ask subagents to return a short summary instead of everything they found.",
    ),
    "Other tools": (
        "Large tool results stay in context",
        "Keep tool results small: narrower searches and patterns return less.",
    ),
}


_GROUP_NOUNS = {
    "Bash": "command result",
    "Read": "file read",
    "MCP tools": "MCP tool result",
    "Web": "web result",
    "Subagents": "subagent report",
    "Other tools": "other tool result",
}


def tool_group(tool: str) -> str:
    if tool in {"Bash", "Read"}:
        return tool
    if tool.startswith("mcp__"):
        return "MCP tools"
    if tool in {"WebFetch", "WebSearch"}:
        return "Web"
    if tool in {"Agent", "Task"}:
        return "Subagents"
    return "Other tools"


def tool_output_findings(context: CarriedContext) -> list[Finding]:
    """Detector 1: tool results that kept costing money on every later call."""
    by_group: dict[str, list[CarriedBlock]] = defaultdict(list)
    for item in context.blocks:
        if item.block.kind is BlockKind.TOOL_RESULT and item.cost > 0:
            by_group[tool_group(item.block.tool)].append(item)
    findings: list[Finding] = []
    for group, items in by_group.items():
        oversized = [item for item in items if item.tokens > TOOL_OUTPUT_CAP_TOKENS]
        savings = sum(
            item.cost * (item.tokens - TOOL_OUTPUT_CAP_TOKENS) / item.tokens for item in oversized
        )
        if savings <= 0:
            continue
        total = sum(item.cost for item in items)
        title, fix = _TOOL_GROUPS[group]
        findings.append(
            Finding(
                detector="carried_context",
                title=title,
                kind=FindingKind.FREE_WIN,
                summary=(
                    f"{plural(len(items), _GROUP_NOUNS[group])} were re-sent on later calls, "
                    f"{percent(total / context.prompt_cost)} of prompt spend. "
                    f"{len(oversized):,} were longer than ~{TOOL_OUTPUT_CAP_TOKENS:,} tokens; "
                    "the saving assumes each is cut to that size."
                ),
                fix=fix,
                savings_usd=savings,
                evidence=_top_details(items),
            )
        )
    return findings


def screenshot_finding(context: CarriedContext) -> Finding | None:
    """Images stay in context like text, at roughly 1,750 tokens each."""
    images = [item for item in context.blocks if item.block.kind is BlockKind.IMAGE]
    cost = sum(item.cost for item in images)
    if not images or cost <= 0:
        return None
    return Finding(
        detector="carried_context",
        title="Screenshots stay in context",
        kind=FindingKind.INFO,
        summary=(
            f"{len(images):,} images (~{tokens(sum(i.tokens for i in images))} tokens) were "
            f"re-sent on later calls, {percent(cost / context.prompt_cost)} of prompt spend."
        ),
        fix="Prefer text or DOM snapshots over screenshots when the layout doesn't matter.",
        evidence=[Evidence("All screenshots", cost, f"{len(images):,} images")],
    )


def stale_context_finding(context: CarriedContext, dataset: Dataset) -> Finding | None:
    """Content far behind the current turn that every call still paid to re-read."""
    if context.stale_cost <= 0 or context.prompt_cost <= 0:
        return None
    compactions = sum(
        1
        for thread in dataset.threads
        for marker in thread.markers
        if marker.kind is MarkerKind.COMPACTION
    )
    threads = {thread.thread_id: thread for thread in dataset.threads}
    worst = sorted(context.stale_by_thread.items(), key=lambda item: -item[1])[:5]
    return Finding(
        detector="stale_context",
        title="Long sessions keep re-reading old context",
        kind=FindingKind.FREE_WIN,
        summary=(
            f"{percent(context.stale_cost / context.prompt_cost)} of prompt spend paid to "
            f"re-read content more than {context.stale_after_calls} calls old. Sessions "
            f"compacted {compactions} times in this period. The saving assumes a compaction "
            f"or fresh session keeps about {percent(COMPACTION_KEEPS)} of that content."
        ),
        fix=(
            "Start a fresh session (/clear) when you switch to an unrelated task, and compact "
            "earlier: set `CLAUDE_CODE_AUTO_COMPACT_WINDOW` (for example 300000) in the `env` "
            "block of ~/.claude/settings.json so sessions compact at that size instead of "
            "near the full context window."
        ),
        savings_usd=context.stale_cost * (1 - COMPACTION_KEEPS),
        evidence=[
            Evidence(_thread_label(threads.get(thread_id), thread_id), cost)
            for thread_id, cost in worst
        ],
    )


def duplicate_context_finding(context: CarriedContext) -> Finding | None:
    """Detector 2: the same tool output added to a context that already held it."""
    seen: set[tuple[str, int, str, str]] = set()
    duplicates: list[CarriedBlock] = []
    for item in context.blocks:
        block = item.block
        if block.kind is not BlockKind.TOOL_RESULT or not block.content_hash:
            continue
        key = (block.thread_id, item.epoch, block.tool, block.content_hash)
        if key in seen:
            if item.tokens >= MIN_DUPLICATE_TOKENS:
                duplicates.append(item)
        else:
            seen.add(key)
    savings = sum(item.cost for item in duplicates)
    if savings <= 0:
        return None
    reads = sum(1 for item in duplicates if item.block.tool == "Read")
    rereads = f" ({plural(reads, 're-read')} of unchanged files)" if reads else ""
    return Finding(
        detector="duplicate_context",
        title="The same tool output entered the context twice",
        kind=FindingKind.FREE_WIN,
        summary=(
            f"{plural(len(duplicates), 'tool result')} repeated output already in the "
            f"context{rereads}. Each copy is re-sent on every later call."
        ),
        fix=(
            "When a file hasn't changed, ask Claude to use what it already read instead of "
            "reading it again; re-run commands only after something changed."
        ),
        savings_usd=savings,
        evidence=_top_details(duplicates),
    )


def _top_details(items: list[CarriedBlock], limit: int = 5) -> list[Evidence]:
    by_detail: dict[str, list[CarriedBlock]] = defaultdict(list)
    for item in items:
        label = item.block.detail or item.block.tool or "(unlabeled)"
        by_detail[label].append(item)
    ranked = sorted(by_detail.items(), key=lambda pair: -sum(i.cost for i in pair[1]))
    return [
        Evidence(
            label,
            sum(i.cost for i in group),
            f"{plural(len(group), 'result')}, "
            f"~{tokens(sum(i.tokens for i in group) / len(group))} tokens each",
        )
        for label, group in ranked[:limit]
    ]


def _thread_label(thread: Thread | None, thread_id: str) -> str:
    if thread is None or not thread.calls:
        return thread_id
    started = thread.calls[0].timestamp.astimezone()
    kind = "subagent in " if thread.is_sidechain else ""
    return f"{kind}{thread.project or '?'} ({started:%Y-%m-%d %H:%M})"
