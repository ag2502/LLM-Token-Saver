"""Command-line interface: `tokensaver` (also installed as `llm-token-saver`)."""

from __future__ import annotations

import re
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from llm_token_saver import __version__
from llm_token_saver.analysis import build_report
from llm_token_saver.config import ConfigError, Settings, load_settings
from llm_token_saver.ingest import claude_code
from llm_token_saver.pricing import PriceTable
from llm_token_saver.report.html import render_html
from llm_token_saver.report.json_export import report_to_json
from llm_token_saver.report.terminal import render_report, render_stats
from llm_token_saver.schema import Dataset
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
    settings, dataset = _load_claude_code(since, projects_dir)
    prices = PriceTable.bundled()
    stats = compute_stats(Store(dataset.calls, prices), top=top, utc_offset_seconds=_utc_offset())
    render_stats(
        stats, currency=settings.currency, prices=prices, console=Console(), title="Claude Code"
    )


@claude_code_app.command("report")
def claude_code_report(
    since: SinceOption = "30d",
    projects_dir: ProjectsDirOption = None,
    json_path: Annotated[
        Path | None, typer.Option("--json", help="Also write the report as JSON to this file.")
    ] = None,
    html_path: Annotated[
        Path | None,
        typer.Option("--html", help="Also write a self-contained HTML report to this file."),
    ] = None,
) -> None:
    """What is wasting tokens in your Claude Code sessions, ranked by monthly saving."""
    settings, dataset = _load_claude_code(since, projects_dir)
    report = build_report(
        dataset, PriceTable.bundled(), title="Claude Code", utc_offset_seconds=_utc_offset()
    )
    console = Console()
    render_report(report, currency=settings.currency, console=console)
    if json_path is not None:
        json_path.write_text(report_to_json(report, settings.currency), encoding="utf-8")
        console.print(f"\nWrote {json_path}")
    if html_path is not None:
        html_path.write_text(render_html(report, settings.currency), encoding="utf-8")
        console.print(f"\nWrote {html_path}")


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


def _load_claude_code(since: str, projects_dir: Path | None) -> tuple[Settings, Dataset]:
    settings = _settings()
    start = parse_since(since)
    root = (projects_dir or settings.claude_code.projects_dir).expanduser()
    dataset = claude_code.load(root, since=start)
    if not dataset.calls:
        Console().print(f"No Claude Code calls found in {root} since {start:%Y-%m-%d %H:%M} UTC.")
        raise typer.Exit()
    return settings, dataset


def _settings() -> Settings:
    try:
        return load_settings()
    except ConfigError as exc:
        typer.echo(f"Config error: {exc}", err=True)
        raise typer.Exit(code=2) from exc


def _utc_offset() -> int:
    offset = datetime.now().astimezone().utcoffset()
    return int(offset.total_seconds()) if offset else 0
