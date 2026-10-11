"""Machine-readable report: every amount in USD, with the display currency alongside."""

from __future__ import annotations

import json
from typing import Any

from llm_token_saver.analysis import Report
from llm_token_saver.config import Currency


def report_to_dict(report: Report, currency: Currency) -> dict[str, Any]:
    stats = report.stats
    return {
        "title": report.title,
        "generated_at": report.generated_at.isoformat(),
        "window": {
            "first_call": stats.first.isoformat(),
            "last_call": stats.last.isoformat(),
            "days": round(report.window_days, 2),
        },
        "prices": {
            "source": report.prices_source,
            "retrieved": report.prices_retrieved.isoformat(),
            "currency": "USD",
        },
        "display_currency": {"code": currency.code, "per_usd": currency.per_usd},
        "spend_usd": {
            "total": stats.cost.total,
            "cache_read": stats.cost.cache_read,
            "cache_write_5m": stats.cost.cache_write_5m,
            "cache_write_1h": stats.cost.cache_write_1h,
            "output": stats.cost.output,
            "input_uncached": stats.cost.input_uncached,
        },
        "usage": {
            "calls": stats.calls,
            "sessions": stats.sessions,
            "threads": stats.threads,
            "cache_hit_rate": stats.cache_hit_rate,
            "prompt_tokens_p50": stats.prompt_p50,
            "prompt_tokens_p90": stats.prompt_p90,
            "thinking_tokens": stats.tokens.thinking,
            "output_tokens": stats.tokens.output,
        },
        "chars_per_token": report.chars_per_token,
        "findings": [
            {
                "detector": finding.detector,
                "title": finding.title,
                "kind": finding.kind.value,
                "savings_usd": finding.savings_usd,
                "savings_usd_per_month": report.monthly(finding.savings_usd),
                "measured": finding.measured,
                "ceiling": finding.ceiling,
                "summary": finding.summary,
                "fix": finding.fix,
                "evidence": [
                    {"label": e.label, "cost_usd": e.cost_usd, "note": e.note}
                    for e in finding.evidence
                ],
            }
            for finding in report.findings
        ],
    }


def report_to_json(report: Report, currency: Currency) -> str:
    return json.dumps(report_to_dict(report, currency), indent=2)
