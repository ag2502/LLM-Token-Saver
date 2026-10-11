"""Rich terminal output."""

from __future__ import annotations

from rich.console import Console
from rich.table import Table

from llm_token_saver.config import Currency
from llm_token_saver.pricing import PriceTable
from llm_token_saver.report.format import middle_truncate, percent, tokens
from llm_token_saver.stats import Group, Stats


def render_stats(
    stats: Stats, *, currency: Currency, prices: PriceTable, console: Console, title: str
) -> None:
    total = stats.cost.total
    days = max(1, (stats.last.date() - stats.first.date()).days + 1)
    console.print(f"[bold]{title}[/bold] · {stats.first:%Y-%m-%d} → {stats.last:%Y-%m-%d}")
    console.print(
        f"{stats.calls:,} calls · {stats.sessions:,} sessions · "
        f"{stats.threads:,} threads ({stats.subagent_threads:,} subagent runs)"
    )
    console.print(
        f"\nAPI-equivalent cost [bold]{currency.format(total)}[/bold]"
        f" (≈{currency.format(total / days)}/day) at list prices retrieved "
        f"{prices.retrieved:%Y-%m-%d}. On a subscription this is usage, not a bill."
    )

    split = Table(title="Where the money went", title_justify="left")
    split.add_column("Token type")
    split.add_column("Cost", justify="right")
    split.add_column("Share", justify="right")
    split.add_column("Tokens", justify="right")
    rows = [
        ("Cache reads (re-read context)", stats.cost.cache_read, stats.tokens.cache_read),
        ("Cache writes, 1-hour TTL", stats.cost.cache_write_1h, stats.tokens.cache_write_1h),
        ("Cache writes, 5-minute TTL", stats.cost.cache_write_5m, stats.tokens.cache_write_5m),
        ("Output (incl. thinking)", stats.cost.output, stats.tokens.output),
        ("Uncached input", stats.cost.input_uncached, stats.tokens.input_uncached),
    ]
    for label, cost, count in sorted(rows, key=lambda row: -row[1]):
        split.add_row(label, currency.format(cost), percent(cost / total), tokens(count))
    console.print()
    console.print(split)

    console.print(
        f"\nContext per call: median {tokens(stats.prompt_p50)}, p90 {tokens(stats.prompt_p90)},"
        f" p99 {tokens(stats.prompt_p99)} tokens · cache hit rate {stats.cache_hit_rate:.0%}"
        f" · thinking is {percent(stats.tokens.thinking / max(stats.tokens.output, 1))}"
        " of output tokens"
    )

    for heading, groups in (
        ("By model", stats.by_model),
        ("By project", stats.by_project),
        ("By effort", stats.by_effort),
    ):
        console.print()
        console.print(_group_table(heading, groups, total, currency))

    sessions = Table(title="Most expensive sessions", title_justify="left")
    sessions.add_column("Started")
    sessions.add_column("Project")
    sessions.add_column("Calls", justify="right")
    sessions.add_column("Peak context", justify="right")
    sessions.add_column("Cost", justify="right")
    for row in stats.top_sessions:
        sessions.add_row(
            f"{row.started:%Y-%m-%d %H:%M}",
            middle_truncate(row.project) if row.project else "-",
            f"{row.calls:,}",
            tokens(row.peak_prompt_tokens),
            currency.format(row.cost),
        )
    console.print()
    console.print(sessions)

    if stats.unpriced_models:
        names = ", ".join(f"{group.key} ({group.calls:,} calls)" for group in stats.unpriced_models)
        console.print(f"\n[yellow]Not priced (missing from the price table):[/yellow] {names}")


def _group_table(heading: str, groups: list[Group], total: float, currency: Currency) -> Table:
    table = Table(title=heading, title_justify="left")
    table.add_column(heading.removeprefix("By ").capitalize())
    table.add_column("Calls", justify="right")
    table.add_column("Cost", justify="right")
    table.add_column("Share", justify="right")
    for group in groups:
        table.add_row(
            middle_truncate(group.key) if group.key else "-",
            f"{group.calls:,}",
            currency.format(group.cost),
            percent(group.cost / total if total else 0.0),
        )
    return table
