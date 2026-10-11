import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from llm_token_saver.analysis import build_report
from llm_token_saver.cli import app
from llm_token_saver.detect import FindingKind
from llm_token_saver.ingest import claude_code
from llm_token_saver.pricing import PriceTable
from tests.transcripts import assistant, user, write_transcript

runner = CliRunner()


def noisy_session(path: Path) -> Path:
    """One 200,000-character `npm test` log that later calls keep re-reading."""
    tool_use = {"type": "tool_use", "id": "tu-1", "name": "Bash", "input": {"command": "npm test"}}
    entries = [
        assistant("2026-10-01T10:00:00Z", "m1", content=[tool_use], write=30_000, output=100),
        user(
            "2026-10-01T10:00:05Z",
            [{"type": "tool_result", "tool_use_id": "tu-1", "content": "x" * 200_000}],
        ),
    ]
    entries += [
        assistant(f"2026-10-01T10:{minute:02d}:00Z", f"m{minute}", read=110_200, write=100)
        for minute in range(2, 12)
    ]
    return write_transcript(path, entries)


def test_report_ranks_the_noisy_command(tmp_path: Path) -> None:
    noisy_session(tmp_path / "p" / "s.jsonl")
    report = build_report(claude_code.load(tmp_path), PriceTable.bundled(), title="Test")
    actionable = [f for f in report.findings if f.kind is not FindingKind.INFO]
    assert actionable[0].title == "Long command output stays in context"
    assert actionable[0].evidence[0].label == "npm test"
    assert report.window_days == 1.0
    assert report.monthly(1.0) == 30.0


def test_report_command_prints_findings_and_writes_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TOKENSAVER_HOME", str(tmp_path / "home"))
    noisy_session(tmp_path / "projects" / "p" / "s.jsonl")
    out = tmp_path / "report.json"
    result = runner.invoke(
        app,
        [
            "claude-code",
            "report",
            "--projects-dir",
            str(tmp_path / "projects"),
            "--since",
            "3650d",
            "--json",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Where to save" in result.output
    assert "Long command output stays in context" in result.output
    data = json.loads(out.read_text())
    assert data["prices"]["currency"] == "USD"
    assert data["usage"]["calls"] == 11
    top = data["findings"][0]
    assert top["detector"] == "carried_context"
    assert top["savings_usd_per_month"] == pytest.approx(top["savings_usd"] * 30)
