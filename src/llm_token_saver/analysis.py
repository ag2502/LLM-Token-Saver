"""Run every detector over a dataset and assemble the report model."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime

from llm_token_saver.detect import CostPool, Finding, rank
from llm_token_saver.detect.cache import cache_rebuild_finding
from llm_token_saver.detect.context import (
    analyze_context,
    duplicate_context_finding,
    screenshot_finding,
    stale_context_finding,
    tool_output_findings,
)
from llm_token_saver.detect.effort import effort_finding
from llm_token_saver.detect.prefix import fixed_prefix_finding
from llm_token_saver.detect.prompt import duplicate_chunk_finding
from llm_token_saver.detect.prompt_cache import prompt_cache_findings
from llm_token_saver.detect.routing import (
    routable_calls_finding,
    subagent_model_finding,
    upgrade_finding,
)
from llm_token_saver.detect.traffic import (
    api_error_finding,
    batchable_finding,
    duplicate_request_finding,
)
from llm_token_saver.detect.ttl import ttl_findings
from llm_token_saver.estimate import TokenEstimator
from llm_token_saver.pricing import PriceTable
from llm_token_saver.schema import Dataset
from llm_token_saver.stats import Stats, compute_stats
from llm_token_saver.store import Store

DAYS_PER_MONTH = 30


@dataclass(frozen=True, slots=True)
class Report:
    title: str
    stats: Stats
    findings: list[Finding]
    window_days: float
    prices_source: str
    prices_retrieved: date
    generated_at: datetime
    chars_per_token: dict[str, float]
    group_noun: str = "project"
    """What `stats.by_project` groups by: Claude Code projects, or services in app traces."""

    def monthly(self, usd: float) -> float:
        """Scale an amount over the analysis window to 30 days."""
        return usd * DAYS_PER_MONTH / self.window_days


def build_report(
    dataset: Dataset,
    prices: PriceTable,
    *,
    title: str,
    top: int = 10,
    utc_offset_seconds: int = 0,
    group_noun: str = "project",
) -> Report:
    stats = compute_stats(
        Store(dataset.calls, prices), top=top, utc_offset_seconds=utc_offset_seconds
    )
    estimator = TokenEstimator.fit(dataset.threads)
    context = analyze_context(dataset, prices, estimator)
    candidates = [
        *tool_output_findings(context),
        stale_context_finding(context, dataset),
        duplicate_context_finding(context),
        screenshot_finding(context),
        fixed_prefix_finding(context, dataset),
        cache_rebuild_finding(dataset, prices),
        api_error_finding(dataset),
        effort_finding(dataset, prices, context),
        upgrade_finding(dataset, prices),
        subagent_model_finding(dataset, prices),
        *prompt_cache_findings(dataset, prices, estimator),
        *ttl_findings(dataset, prices),
        duplicate_request_finding(dataset, prices),
        batchable_finding(dataset, prices),
        duplicate_chunk_finding(dataset, prices, estimator),
        routable_calls_finding(dataset, prices),
    ]
    findings = rank(
        [finding for finding in candidates if finding is not None],
        {
            CostPool.PROMPT: stats.cost.total - stats.cost.output,
            CostPool.OUTPUT: stats.cost.output,
        },
    )
    span_days = (stats.last - stats.first).total_seconds() / 86_400
    return Report(
        title=title,
        stats=stats,
        findings=findings,
        window_days=max(span_days, 1.0),
        prices_source=prices.source,
        prices_retrieved=prices.retrieved,
        generated_at=datetime.now(UTC),
        chars_per_token=dict(estimator.chars_per_token),
        group_noun=group_noun,
    )
