"""Number formatting shared by the terminal and HTML renderers."""

from __future__ import annotations


def tokens(count: float) -> str:
    """Compact token counts: 950, 12.4K, 3.4M, 1.2B."""
    for limit, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(count) >= limit:
            value = count / limit
            return f"{value:.1f}{suffix}" if value < 100 else f"{value:.0f}{suffix}"
    return f"{count:.0f}"


def percent(share: float) -> str:
    if 0 < share < 0.005:
        return "<1%"
    return f"{share:.0%}"


def middle_truncate(text: str, width: int = 44) -> str:
    """Keep both ends of long labels such as project paths: `~/Library/Ap…scratch-2026-09-14`."""
    if len(text) <= width:
        return text
    keep = width - 1
    return text[: keep // 2] + "…" + text[len(text) - (keep - keep // 2) :]


def plural(count: int, noun: str, plural_form: str | None = None) -> str:
    """`1 result`, `3 results`, `2 file reads`."""
    word = noun if count == 1 else plural_form or f"{noun}s"
    return f"{count:,} {word}"
