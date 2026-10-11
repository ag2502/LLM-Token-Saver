"""Findings: what a detector reports, and how findings are ranked together."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from enum import StrEnum


class FindingKind(StrEnum):
    FREE_WIN = "free_win"
    """Saves money without lowering answer quality."""

    TRADEOFF = "tradeoff"
    """Saves money but can lower quality; measure before adopting."""

    INFO = "info"
    """Worth knowing; no saving is attached."""


class CostPool(StrEnum):
    """The part of the bill a saving comes out of."""

    PROMPT = "prompt"
    OUTPUT = "output"


@dataclass(frozen=True, slots=True)
class Evidence:
    """One supporting row: a label, an optional cost (rendered in the display currency), a note."""

    label: str
    cost_usd: float | None = None
    note: str = ""


@dataclass(frozen=True, slots=True)
class Finding:
    detector: str
    title: str
    kind: FindingKind
    summary: str
    fix: str
    savings_usd: float = 0.0
    """Estimated saving over the analysis window, in USD."""
    pool: CostPool = CostPool.PROMPT
    measured: bool = False
    """True when the saving comes straight from billed usage, False when it is estimated."""
    ceiling: bool = False
    """True when the saving is an upper bound rather than an expected value."""
    evidence: list[Evidence] = field(default_factory=list)


def rank(findings: list[Finding], pool_totals: dict[CostPool, float]) -> list[Finding]:
    """Order findings by saving, shrinking overlaps so savings never sum past the bill.

    Free wins are credited first, then tradeoffs, matching the order fixes should be applied.
    Within a pool, savings compose as independent fractions: if one fix saves 30% of prompt
    spend and the next saves 20%, the second is credited 20% of the 70% that remains.
    """
    order = {FindingKind.FREE_WIN: 0, FindingKind.TRADEOFF: 1, FindingKind.INFO: 2}
    remaining = dict(pool_totals)
    credited: list[Finding] = []
    for finding in sorted(findings, key=lambda f: (order[f.kind], -f.savings_usd)):
        total = pool_totals.get(finding.pool, 0.0)
        if finding.kind is FindingKind.INFO or finding.savings_usd <= 0 or total <= 0:
            credited.append(finding)
            continue
        share = min(finding.savings_usd / total, 1.0) * remaining[finding.pool]
        remaining[finding.pool] -= share
        credited.append(dataclasses.replace(finding, savings_usd=share))
    return sorted(credited, key=lambda f: (f.kind is FindingKind.INFO, -f.savings_usd))
