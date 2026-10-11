"""Command-line interface: `tokensaver` (also installed as `llm-token-saver`)."""

from __future__ import annotations

import re
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from llm_token_saver import __version__
from llm_token_saver.config import ConfigError, Settings, load_settings
from llm_token_saver.ingest import claude_code
from llm_token_saver.pricing import PriceTable
from llm_token_saver.report.terminal import render_stats
from llm_token_saver.stats import compute_stats
from llm_token_saver.store import Store

app = typer.Typer(
    name="tokensaver",
    help="Find out why your LLM bill is high, and what to fix first.",
    no_args_is_help=True,
    add_completion=False,
)
claude_code_app = typer.Typer(
    help="Analyze your Claude Code usage from local session logs (read-only).",
    no_args_is_help=True,
)
app.add_typer(claude_code_app, name="claude-code")

SinceOption = Annotated[
    str,
    typer.Option(help="How far back to look: a duration (12h, 7d, 2w) or a date (2026-10-01)."),
]
ProjectsDirOption = Annotated[
    Path | None,
    typer.Option(help="Claude Code projects directory. Default: ~/.claude/projects."),
]


def _show_version(value: bool) -> None:
    if value:
        typer.echo(f"llm-token-saver {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            callback=_show_version,
            is_eager=True,
            help="Show the version and exit.",
        ),
    ] = False,
) -> None:
    """LLM Token Saver."""


@claude_code_app.command("stats")
def claude_code_stats(
    since: SinceOption = "30d",
    projects_dir: ProjectsDirOption = None,
    top: Annotated[int, typer.Option(help="Rows to show in each breakdown.")] = 10,
) -> None:
    """Where your Claude Code tokens went: by token type, model, project and session."""
    settings = _settings()
    start = parse_since(since)
    root = (projects_dir or settings.claude_code.projects_dir).expanduser()
    dataset = claude_code.load(root, since=start)
    console = Console()
    if not dataset.calls:
        console.print(f"No Claude Code calls found in {root} since {start:%Y-%m-%d %H:%M} UTC.")
        raise typer.Exit()
    prices = PriceTable.bundled()
    stats = compute_stats(Store(dataset.calls, prices), top=top, utc_offset_seconds=_utc_offset())
    render_stats(
        stats, currency=settings.currency, prices=prices, console=console, title="Claude Code"
    )


def parse_since(value: str, *, now: datetime | None = None) -> datetime:
    """`7d`, `12h`, `2w` before `now`, or local midnight of an ISO date, as UTC."""
    now = now or datetime.now(UTC)
    match = re.fullmatch(r"(\d+)\s*([hdw])", value.strip().lower())
    if match:
        amount, unit = int(match[1]), match[2]
        return now - {"h": timedelta(hours=amount), "d": timedelta(days=amount)}.get(
            unit, timedelta(weeks=amount)
        )
    try:
        day = date.fromisoformat(value.strip())
    except ValueError:
        raise typer.BadParameter(
            "use a duration like 12h, 7d or 2w, or a date like 2026-10-01", param_hint="--since"
        ) from None
    return datetime(day.year, day.month, day.day).astimezone().astimezone(UTC)


def _settings() -> Settings:
    try:
        return load_settings()
    except ConfigError as exc:
        typer.echo(f"Config error: {exc}", err=True)
        raise typer.Exit(code=2) from exc


def _utc_offset() -> int:
    offset = datetime.now().astimezone().utcoffset()
    return int(offset.total_seconds()) if offset else 0
