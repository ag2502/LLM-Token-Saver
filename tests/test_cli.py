from datetime import UTC, datetime
from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner

from llm_token_saver import __version__
from llm_token_saver.cli import app, parse_since
from tests.transcripts import assistant, write_transcript

runner = CliRunner()


def test_version_flag_prints_package_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.output.strip() == f"llm-token-saver {__version__}"


def test_no_arguments_shows_help() -> None:
    result = runner.invoke(app, [])
    assert "Find out why your LLM bill is high" in result.output


def test_claude_code_stats_on_a_transcript(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TOKENSAVER_HOME", str(tmp_path / "home"))
    write_transcript(
        tmp_path / "projects" / "p" / "s.jsonl",
        [assistant("2026-10-01T10:00:00Z", "m1", read=1_000_000, output=2_000)],
    )
    result = runner.invoke(
        app,
        ["claude-code", "stats", "--projects-dir", str(tmp_path / "projects"), "--since", "3650d"],
    )
    assert result.exit_code == 0, result.output
    assert "API-equivalent cost" in result.output
    assert "claude-opus-5-5" in result.output


def test_claude_code_stats_with_no_logs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TOKENSAVER_HOME", str(tmp_path / "home"))
    result = runner.invoke(app, ["claude-code", "stats", "--projects-dir", str(tmp_path)])
    assert result.exit_code == 0
    assert "No Claude Code calls found" in result.output


def test_bad_config_exits_with_an_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TOKENSAVER_HOME", str(tmp_path))
    (tmp_path / "config.toml").write_text('[currency]\ncode = "EUR"\n')
    result = runner.invoke(app, ["claude-code", "stats", "--projects-dir", str(tmp_path)])
    assert result.exit_code == 2


NOW = datetime(2026, 10, 11, 12, 0, tzinfo=UTC)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("12h", datetime(2026, 10, 11, 0, 0, tzinfo=UTC)),
        ("7d", datetime(2026, 10, 4, 12, 0, tzinfo=UTC)),
        ("2w", datetime(2026, 9, 27, 12, 0, tzinfo=UTC)),
    ],
)
def test_parse_since_durations(value: str, expected: datetime) -> None:
    assert parse_since(value, now=NOW) == expected


def test_parse_since_date_is_local_midnight() -> None:
    parsed = parse_since("2026-10-01")
    assert parsed.tzinfo is UTC
    assert parsed.astimezone().strftime("%Y-%m-%d %H:%M") == "2026-10-01 00:00"


def test_parse_since_rejects_garbage() -> None:
    with pytest.raises(typer.BadParameter):
        parse_since("last tuesday")
