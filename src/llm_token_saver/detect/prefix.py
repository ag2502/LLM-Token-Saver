"""Detector 5: the fixed prefix every call re-sends before the conversation starts."""

from __future__ import annotations

import statistics

from llm_token_saver.detect.base import Evidence, Finding, FindingKind
from llm_token_saver.detect.context import CarriedContext
from llm_token_saver.report.format import percent, tokens
from llm_token_saver.schema import Dataset


def fixed_prefix_finding(context: CarriedContext, dataset: Dataset) -> Finding | None:
    sidechain = {
        thread.thread_id: thread.is_sidechain
        for thread in dataset.threads
        if thread.source == "claude_code"  # the wording below describes Claude Code sessions
    }
    main = [n for tid, n in context.first_prefix_tokens.items() if sidechain.get(tid) is False]
    subagents = [n for tid, n in context.first_prefix_tokens.items() if sidechain.get(tid)]
    if not (main or subagents) or context.prompt_cost <= 0:
        return None
    sizes = []
    if main:
        sizes.append(f"sessions start with ~{tokens(statistics.median(main))} tokens")
    if subagents:
        sizes.append(f"subagents with ~{tokens(statistics.median(subagents))}")
    return Finding(
        detector="fixed_prefix",
        title="A fixed prefix is re-sent on every call",
        kind=FindingKind.INFO,
        summary=(
            f"Before your first message, {' and '.join(sizes)} of system prompt, tool "
            "definitions, CLAUDE.md and MCP instructions. Every call re-reads it; together "
            "with other content the logs don't show it took "
            f"{percent(context.unseen_cost / context.prompt_cost)} of prompt spend."
        ),
        fix=(
            "Keep CLAUDE.md to what every session needs, disconnect MCP servers a project "
            "doesn't use, and run /context to see what the prefix holds."
        ),
        evidence=[
            Evidence("Each 1,000 tokens of prefix", context.prefix_token_cost * 1_000),
            Evidence("Prefix and unlogged content", context.unseen_cost),
        ],
    )
