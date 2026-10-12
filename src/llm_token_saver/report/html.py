"""Self-contained HTML report: one file, inline CSS and SVG, no network requests."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

from jinja2 import Environment, PackageLoader, select_autoescape
from markupsafe import Markup, escape

from llm_token_saver import __version__
from llm_token_saver.analysis import Report
from llm_token_saver.config import Currency
from llm_token_saver.detect import Finding, FindingKind
from llm_token_saver.report.format import middle_truncate, percent, tokens

KIND_LABELS = {
    FindingKind.FREE_WIN: "Free win",
    FindingKind.TRADEOFF: "Tradeoff",
    FindingKind.INFO: "Good to know",
}
MIN_SHOWN_MONTHLY_USD = 0.5

# Token types in a fixed order, each tied to one categorical slot (color follows the entity).
_SPLIT = (
    ("Cache reads", "cache_read", 1),
    ("Cache writes, 1-hour", "cache_write_1h", 2),
    ("Cache writes, 5-minute", "cache_write_5m", 3),
    ("Output (incl. thinking)", "output", 4),
    ("Uncached input", "input_uncached", 5),
)

_CHART_WIDTH, _CHART_HEIGHT = 720, 220
_PLOT_LEFT, _PLOT_RIGHT, _PLOT_TOP, _PLOT_BOTTOM = 52, 8, 12, 28
_MAX_BAR = 24
_GAP = 2


def render_html(report: Report, currency: Currency) -> str:
    env = Environment(
        loader=PackageLoader("llm_token_saver", "report/templates"),
        autoescape=select_autoescape(["html", "j2"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    env.filters["code"] = inline_code
    return env.get_template("report.html.j2").render(view=_view(report, currency))


def inline_code(text: str) -> Markup:
    """Escape text, then render `backticked` spans as <code>."""
    escaped = str(escape(text))
    return Markup(re.sub(r"`([^`]+)`", r"<code>\1</code>", escaped))


@dataclass(frozen=True, slots=True)
class Bar:
    x: float
    path: str
    hit_x: float
    hit_width: float
    label: str
    value: str


def nice_ticks(maximum: float, target: int = 4) -> list[float]:
    """Round axis ticks from zero that cover `maximum` in about `target` steps."""
    if maximum <= 0:
        return [0.0, 1.0]
    raw = maximum / target
    magnitude = 10 ** math.floor(math.log10(raw))
    step = next(m * magnitude for m in (1, 2, 2.5, 5, 10) if m * magnitude >= raw)
    steps = math.ceil(maximum / step - 1e-9)
    return [round(i * step, 10) for i in range(steps + 1)]


def column_path(x: float, y: float, width: float, height: float, radius: float = 4.0) -> str:
    """A column with rounded top corners and a square base."""
    r = min(radius, width / 2, height)
    bottom = y + height
    return (
        f"M{x:.2f},{bottom:.2f}V{y + r:.2f}Q{x:.2f},{y:.2f} {x + r:.2f},{y:.2f}"
        f"H{x + width - r:.2f}Q{x + width:.2f},{y:.2f} {x + width:.2f},{y + r:.2f}"
        f"V{bottom:.2f}Z"
    )


def _view(report: Report, currency: Currency) -> dict[str, Any]:
    stats = report.stats
    total = stats.cost.total
    actionable = [
        f
        for f in report.findings
        if f.kind is not FindingKind.INFO and report.monthly(f.savings_usd) >= MIN_SHOWN_MONTHLY_USD
    ]
    free = sum(f.savings_usd for f in actionable if f.kind is FindingKind.FREE_WIN)
    split = []
    for label, field, slot in _SPLIT:
        cost = getattr(stats.cost, field)
        split.append(
            {
                "label": label,
                "slot": slot,
                "value": currency.format(cost),
                "share": cost / total if total else 0.0,
                "share_text": percent(cost / total if total else 0.0),
                "tokens": tokens(getattr(stats.tokens, field)),
            }
        )
    return {
        "title": report.title,
        "group_noun": report.group_noun,
        "session_noun": "sessions" if report.group_noun == "project" else "conversations",
        "version": __version__,
        "generated": f"{report.generated_at.astimezone():%Y-%m-%d %H:%M}",
        "period": f"{stats.first.astimezone():%Y-%m-%d} → {stats.last.astimezone():%Y-%m-%d}",
        "calls": f"{stats.calls:,}",
        "sessions": f"{stats.sessions:,}",
        "subagent_runs": f"{stats.subagent_threads:,}",
        "prices_source": report.prices_source,
        "prices_retrieved": f"{report.prices_retrieved:%Y-%m-%d}",
        "monthly_spend": currency.format(report.monthly(total)),
        "min_shown": currency.format(MIN_SHOWN_MONTHLY_USD),
        "window_spend": currency.format(total),
        "tiles": [
            {
                "label": "Free wins per month",
                "value": currency.format(report.monthly(free)),
                "note": f"{percent(free / total)} of spend" if total else "",
            },
            {
                "label": "Median context per call",
                "value": f"{tokens(stats.prompt_p50)} tokens",
                "note": f"p90 {tokens(stats.prompt_p90)}",
            },
            {
                "label": "Cache hit rate",
                "value": percent(stats.cache_hit_rate),
                "note": "of prompt tokens read from cache",
            },
            {
                "label": "Thinking share of output",
                "value": percent(stats.tokens.thinking / max(stats.tokens.output, 1)),
                "note": f"{tokens(stats.tokens.thinking)} thinking tokens",
            },
        ],
        "split": split,
        "findings": [_finding(f, report, currency) for f in actionable],
        "info": [
            _finding(f, report, currency) for f in report.findings if f.kind is FindingKind.INFO
        ],
        "daily": _daily_chart(report, currency),
        "by_model": _groups(stats.by_model, total, currency),
        "by_project": _groups(stats.by_project, total, currency),
        "by_effort": _groups(stats.by_effort, total, currency),
        "sessions_table": [
            {
                "started": f"{row.started.astimezone():%Y-%m-%d %H:%M}",
                "project": middle_truncate(row.project or "-", 48),
                "calls": f"{row.calls:,}",
                "peak": tokens(row.peak_prompt_tokens),
                "cost": currency.format(row.cost),
            }
            for row in stats.top_sessions
        ],
        "chars_per_token": ", ".join(
            f"{model} {ratio:.2f}" for model, ratio in sorted(report.chars_per_token.items())
        ),
    }


def _finding(finding: Finding, report: Report, currency: Currency) -> dict[str, Any]:
    monthly = currency.format(report.monthly(finding.savings_usd))
    return {
        "title": finding.title,
        "kind": KIND_LABELS[finding.kind],
        "kind_class": finding.kind.value,
        "saving": f"up to {monthly}" if finding.ceiling else monthly,
        "has_saving": finding.savings_usd > 0,
        "basis": "measured" if finding.measured else "estimated",
        "summary": finding.summary,
        "fix": finding.fix,
        "evidence": [
            {
                "label": middle_truncate(e.label, 64),
                "cost": currency.format(e.cost_usd) if e.cost_usd is not None else "",
                "note": e.note,
            }
            for e in finding.evidence[:5]
        ],
    }


def _groups(groups: list[Any], total: float, currency: Currency) -> list[dict[str, str]]:
    return [
        {
            "key": middle_truncate(group.key or "-", 48),
            "calls": f"{group.calls:,}",
            "cost": currency.format(group.cost),
            "share": percent(group.cost / total if total else 0.0),
        }
        for group in groups
    ]


def _daily_chart(report: Report, currency: Currency) -> dict[str, Any]:
    days = report.stats.by_day
    plot_width = _CHART_WIDTH - _PLOT_LEFT - _PLOT_RIGHT
    plot_height = _CHART_HEIGHT - _PLOT_TOP - _PLOT_BOTTOM
    ticks = nice_ticks(max((day.cost for day in days), default=0.0))
    top = ticks[-1] or 1.0
    band = plot_width / max(len(days), 1)
    width = max(min(_MAX_BAR, band - _GAP), 1.0)
    baseline = _PLOT_TOP + plot_height
    bars = []
    for index, day in enumerate(days):
        height = plot_height * day.cost / top
        x = _PLOT_LEFT + index * band + (band - width) / 2
        bars.append(
            Bar(
                x=x,
                path=column_path(x, baseline - height, width, height) if height > 0 else "",
                hit_x=_PLOT_LEFT + index * band,
                hit_width=band,
                label=f"{day.day:%a %Y-%m-%d} · {day.calls:,} calls",
                value=currency.format(day.cost),
            )
        )
    label_every = max(1, math.ceil(len(days) / 8))
    return {
        "width": _CHART_WIDTH,
        "height": _CHART_HEIGHT,
        "plot_left": _PLOT_LEFT,
        "plot_right": _CHART_WIDTH - _PLOT_RIGHT,
        "plot_top": _PLOT_TOP,
        "baseline": baseline,
        "ticks": [
            {"y": baseline - plot_height * tick / top, "label": currency.format_axis(tick)}
            for tick in ticks
        ],
        "bars": bars,
        "x_labels": [
            {"x": bar.x + width / 2, "label": f"{day.day:%b %d}"}
            for index, (bar, day) in enumerate(zip(bars, days, strict=True))
            if index % label_every == 0
        ],
        "rows": [
            {"day": f"{day.day:%Y-%m-%d}", "calls": f"{day.calls:,}", "cost": bar.value}
            for bar, day in zip(bars, days, strict=True)
        ],
        "summary": (
            f"Daily spend over {len(days)} days; highest "
            f"{currency.format(max((d.cost for d in days), default=0.0))}."
        ),
    }
