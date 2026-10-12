"""Command-line interface: `tokensaver` (also installed as `llm-token-saver`)."""

from __future__ import annotations

import json
import re
import sys
import traceback
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from llm_token_saver import __version__, demo
from llm_token_saver.analysis import Report, build_report
from llm_token_saver.config import ConfigError, Settings, home_dir, load_settings
from llm_token_saver.detect import FindingKind
from llm_token_saver.ingest import claude_code, otel
from llm_token_saver.integrations import macos
from llm_token_saver.integrations.session import append_history, summarize_session
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
schedule_app = typer.Typer(
    help="Write the weekly report automatically (macOS launchd).", no_args_is_help=True
)
claude_code_app.add_typer(schedule_app, name="schedule")

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
    notify: Annotated[
        bool, typer.Option(help="Show a macOS notification with the headline numbers.")
    ] = False,
) -> None:
    """What is wasting tokens in your Claude Code sessions, ranked by monthly saving."""
    settings, dataset = _load_claude_code(since, projects_dir)
    report = build_report(
        dataset, PriceTable.bundled(), title="Claude Code", utc_offset_seconds=_utc_offset()
    )
    _emit(report, settings, json_path=json_path, html_path=html_path, notify=notify)


@claude_code_app.command("session-end")
def claude_code_session_end(
    notify: Annotated[bool, typer.Option(help="Show a macOS notification.")] = True,
) -> None:
    """Summarize one session from a Claude Code SessionEnd hook (hook JSON on stdin).

    Appends the summary to ~/.tokensaver/sessions.jsonl. Never fails the hook: errors go to
    ~/.tokensaver/hook-errors.log.
    """
    try:
        payload = json.loads(sys.stdin.read() or "{}")
        transcript = Path(str(payload.get("transcript_path", ""))).expanduser()
        if not transcript.is_file():
            return
        summary = summarize_session(transcript, session_id=payload.get("session_id"))
        if summary is None:
            return
        settings = load_settings()
        append_history(summary, home_dir() / "sessions.jsonl")
        if notify:
            project = Path(summary.project).name or "Claude Code"
            macos.notify(summary.headline(settings.currency.format(summary.cost_usd)), project)
    except Exception:  # a hook must never get in the way of ending a session
        log = home_dir() / "hook-errors.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a", encoding="utf-8") as fh:
            fh.write(f"{datetime.now(UTC).isoformat()}\n{traceback.format_exc()}\n")


JsonOption = Annotated[
    Path | None, typer.Option("--json", help="Also write the report as JSON to this file.")
]
HtmlOption = Annotated[
    Path | None,
    typer.Option("--html", help="Also write a self-contained HTML report to this file."),
]


@app.command("analyze")
def analyze(
    paths: Annotated[
        list[Path],
        typer.Argument(
            exists=True, help="OTLP/JSON trace files or directories (.json, .jsonl, .ndjson)."
        ),
    ],
    since: Annotated[
        str | None,
        typer.Option(help="Only count calls after this: a duration (7d) or a date."),
    ] = None,
    json_path: JsonOption = None,
    html_path: HtmlOption = None,
) -> None:
    """Find waste in any LLM app from OpenTelemetry GenAI traces."""
    settings = _settings()
    dataset = otel.load(paths, since=parse_since(since) if since else None)
    if not dataset.calls:
        typer.echo("No GenAI chat calls found in those traces.")
        raise typer.Exit()
    report = build_report(
        dataset,
        PriceTable.bundled(),
        title="LLM app traces",
        utc_offset_seconds=_utc_offset(),
        group_noun="service",
    )
    _emit(report, settings, json_path=json_path, html_path=html_path, notify=False)


@app.command("demo")
def demo_command(
    out: Annotated[Path, typer.Option(help="Where to write the demo traces.")] = Path(
        "tokensaver-demo/traces.jsonl"
    ),
    days: Annotated[int, typer.Option(min=1, max=31)] = 7,
    seed: Annotated[int, typer.Option(help="Change for a different but repeatable week.")] = 7,
    html_path: HtmlOption = None,
) -> None:
    """Write a week of synthetic app traces with planted waste, then analyze them."""
    traces = demo.generate(seed=seed, days=days)
    path = traces.write(out)
    console = Console()
    console.print(f"Wrote {path}. Planted problems the analysis should find:")
    for planted in traces.planted:
        console.print(f"  - {planted.service}: {planted.description}")
    console.print()
    analyze([path], since=None, json_path=None, html_path=html_path)


def _emit(
    report: Report,
    settings: Settings,
    *,
    json_path: Path | None,
    html_path: Path | None,
    notify: bool,
) -> None:
    console = Console()
    render_report(report, currency=settings.currency, console=console)
    for path, text in (
        (json_path, lambda: report_to_json(report, settings.currency)),
        (html_path, lambda: render_html(report, settings.currency)),
    ):
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text(), encoding="utf-8")
            console.print(f"\nWrote {path}")
    if notify:
        money = settings.currency.format
        free = sum(f.savings_usd for f in report.findings if f.kind is FindingKind.FREE_WIN)
        macos.notify(
            f"{money(report.monthly(report.stats.cost.total))}/month API-equivalent; free wins "
            f"about {money(report.monthly(free))}/month."
            + (f" Report: {html_path}" if html_path else ""),
            title="Token Saver weekly report",
        )


@schedule_app.command("install")
def schedule_install(
    weekday: Annotated[int, typer.Option(min=0, max=6, help="0 = Sunday, 1 = Monday.")] = 1,
    hour: Annotated[int, typer.Option(min=0, max=23)] = 9,
    minute: Annotated[int, typer.Option(min=0, max=59)] = 0,
) -> None:
    """Write a 7-day report to ~/.tokensaver/reports/ every week and notify when it is ready."""
    if not macos.is_macos():
        typer.echo("Scheduling uses launchd and is only available on macOS.", err=True)
        raise typer.Exit(code=1)
    reports = home_dir() / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    program = [
        sys.executable,
        "-m",
        "llm_token_saver",
        "claude-code",
        "report",
        "--since",
        "7d",
        "--html",
        str(reports / "latest.html"),
        "--json",
        str(reports / "latest.json"),
        "--notify",
    ]
    plist = macos.weekly_report_plist(
        program, weekday=weekday, hour=hour, minute=minute, log_dir=reports
    )
    path = macos.install_agent(plist)
    day = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"][weekday]
    typer.echo(f"Installed {path}: every {day} at {hour:02d}:{minute:02d}.")


@schedule_app.command("remove")
def schedule_remove() -> None:
    """Stop the weekly report."""
    path = macos.remove_agent()
    typer.echo(f"Removed {path}." if path else "No weekly report was scheduled.")


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
