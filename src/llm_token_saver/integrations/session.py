"""One-session summaries for Claude Code's SessionEnd hook.

The hook receives `{"session_id", "transcript_path", "cwd", ...}` on stdin. Its stdout only
reaches Claude Code's debug log, so the summary goes to a history file and, on macOS, a
notification.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from llm_token_saver.detect.context import analyze_context, tool_group
from llm_token_saver.estimate import TokenEstimator
from llm_token_saver.ingest.claude_code import read_thread
from llm_token_saver.pricing import PriceTable
from llm_token_saver.report.format import percent, tokens
from llm_token_saver.schema import BlockKind, Dataset


@dataclass(frozen=True, slots=True)
class SessionSummary:
    session_id: str
    project: str
    started: str
    ended: str
    calls: int
    subagent_runs: int
    cost_usd: float
    peak_prompt_tokens: int
    cache_hit_rate: float
    stale_share: float
    """Share of prompt spend paid to re-read content more than 100 calls old."""
    top_tool: str
    top_tool_cost_usd: float

    def headline(self, money: str) -> str:
        parts = [
            f"{money} API-equivalent",
            f"{self.calls:,} calls",
            f"peak context {tokens(self.peak_prompt_tokens)}",
        ]
        if self.stale_share >= 0.1:
            parts.append(f"{percent(self.stale_share)} re-reading old context")
        elif self.top_tool and self.top_tool_cost_usd > 0:
            parts.append(f"{self.top_tool} output was the biggest carried cost")
        return " · ".join(parts)


def session_files(transcript: Path, session_id: str | None) -> list[Path]:
    """The main transcript plus its subagent transcripts."""
    session_id = session_id or transcript.stem
    subagents = sorted((transcript.parent / session_id / "subagents").glob("*.jsonl"))
    return [transcript, *subagents]


def summarize_session(
    transcript: Path, session_id: str | None = None, prices: PriceTable | None = None
) -> SessionSummary | None:
    prices = prices or PriceTable.bundled()
    threads = [t for path in session_files(transcript, session_id) if (t := read_thread(path))]
    if not threads:
        return None
    dataset = Dataset(threads=threads)
    calls = dataset.calls
    costs = [cost for call in calls if (cost := prices.cost(call)) is not None]
    prompt_tokens = sum(call.prompt_tokens for call in calls)
    context = analyze_context(dataset, prices, TokenEstimator.fit(threads))
    by_group: dict[str, float] = {}
    for item in context.blocks:
        if item.block.kind is BlockKind.TOOL_RESULT:
            group = tool_group(item.block.tool)
            by_group[group] = by_group.get(group, 0.0) + item.cost
    top_tool, top_cost = max(by_group.items(), key=lambda kv: kv[1], default=("", 0.0))
    main = next((t for t in threads if not t.is_sidechain), threads[0])
    timestamps = [call.timestamp for call in calls]
    return SessionSummary(
        session_id=main.session_id,
        project=main.project,
        started=min(timestamps).isoformat(),
        ended=max(timestamps).isoformat(),
        calls=len(calls),
        subagent_runs=sum(1 for t in threads if t.is_sidechain),
        cost_usd=sum(cost.total for cost in costs),
        peak_prompt_tokens=max(call.prompt_tokens for call in calls),
        cache_hit_rate=sum(c.cache_read for c in calls) / prompt_tokens if prompt_tokens else 0.0,
        stale_share=context.stale_cost / context.prompt_cost if context.prompt_cost else 0.0,
        top_tool=top_tool,
        top_tool_cost_usd=top_cost,
    )


def append_history(summary: SessionSummary, history: Path) -> None:
    history.parent.mkdir(parents=True, exist_ok=True)
    record = {"recorded_at": datetime.now(UTC).isoformat(), **asdict(summary)}
    with history.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")
