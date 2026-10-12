import json
import os
import plistlib
import subprocess
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from llm_token_saver.cli import app
from llm_token_saver.integrations import macos
from llm_token_saver.integrations.session import SessionSummary, summarize_session
from tests.transcripts import assistant, write_transcript

runner = CliRunner()


def session(tmp_path: Path) -> Path:
    project = tmp_path / "projects" / "-work-app"
    main = write_transcript(
        project / "sess-1.jsonl",
        [
            assistant("2026-10-01T10:00:00Z", "m1", write=20_000, output=500),
            assistant("2026-10-01T10:01:00Z", "m2", read=20_000, write=600, output=400),
        ],
    )
    write_transcript(
        project / "sess-1" / "subagents" / "agent-a1.jsonl",
        [assistant("2026-10-01T10:00:30Z", "s1", write=5_000, output=100, sidechain=True)],
    )
    return main


def test_summary_covers_the_session_and_its_subagents(tmp_path: Path) -> None:
    summary = summarize_session(session(tmp_path), "sess-1")
    assert summary is not None
    assert (summary.calls, summary.subagent_runs, summary.session_id) == (3, 1, "sess-1")
    assert summary.peak_prompt_tokens == 20_602
    assert summary.cost_usd > 0
    assert summary.headline("$0.15").startswith("$0.15 API-equivalent · 3 calls · peak context")


class Recorder:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def __call__(self, args: list[str], **_: Any) -> subprocess.CompletedProcess[bytes]:
        self.calls.append(list(args))
        return subprocess.CompletedProcess(args, 0, b"", b"")


def test_session_end_hook_appends_history_and_notifies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TOKENSAVER_HOME", str(tmp_path / "home"))
    recorder = Recorder()
    monkeypatch.setattr(macos.subprocess, "run", recorder)
    monkeypatch.setattr(macos, "is_macos", lambda: True)
    payload = {"session_id": "sess-1", "transcript_path": str(session(tmp_path)), "reason": "other"}
    result = runner.invoke(app, ["claude-code", "session-end"], input=json.dumps(payload))
    assert result.exit_code == 0
    [line] = (tmp_path / "home" / "sessions.jsonl").read_text().splitlines()
    record = json.loads(line)
    assert record["calls"] == 3 and record["project"] == "/work/app"
    [osascript] = recorder.calls
    assert osascript[0] == "osascript" and osascript[-1] == "app"
    assert osascript[-2].endswith("peak context 20.6K")


@pytest.mark.parametrize("stdin", ["not json", json.dumps({"transcript_path": "/nope.jsonl"})])
def test_session_end_never_fails_the_hook(
    stdin: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TOKENSAVER_HOME", str(tmp_path))
    result = runner.invoke(app, ["claude-code", "session-end", "--no-notify"], input=stdin)
    assert result.exit_code == 0
    assert not (tmp_path / "sessions.jsonl").exists()
    assert (tmp_path / "hook-errors.log").exists() == (stdin == "not json")


def test_notification_text_is_passed_as_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = Recorder()
    monkeypatch.setattr(macos.subprocess, "run", recorder)
    monkeypatch.setattr(macos, "is_macos", lambda: True)
    assert macos.notify('say "hi"; do shell script "rm"', "t")
    assert recorder.calls[0][-2:] == ['say "hi"; do shell script "rm"', "t"]


def test_weekly_plist_runs_on_the_chosen_day(tmp_path: Path) -> None:
    plist = plistlib.loads(
        macos.weekly_report_plist(["tokensaver"], weekday=1, hour=9, minute=30, log_dir=tmp_path)
    )
    assert plist["Label"] == macos.LABEL
    assert plist["StartCalendarInterval"] == {"Weekday": 1, "Hour": 9, "Minute": 30}
    assert plist["StandardOutPath"] == str(tmp_path / "weekly.log")


def test_install_and_remove_agent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = Recorder()
    monkeypatch.setattr(macos.subprocess, "run", recorder)
    path = macos.install_agent(b"<plist/>", tmp_path)
    assert path.read_bytes() == b"<plist/>"
    domain = f"gui/{os.getuid()}"
    assert recorder.calls == [
        ["launchctl", "bootout", domain, str(path)],
        ["launchctl", "bootstrap", domain, str(path)],
    ]
    assert macos.remove_agent(tmp_path) == path
    assert not path.exists()
    assert macos.remove_agent(tmp_path) is None


def test_schedule_is_macos_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(macos, "is_macos", lambda: False)
    result = runner.invoke(app, ["claude-code", "schedule", "install"])
    assert result.exit_code == 1


def test_headline_prefers_the_stale_context_hint() -> None:
    summary = SessionSummary(
        session_id="s",
        project="~/p",
        started="",
        ended="",
        calls=10,
        subagent_runs=0,
        cost_usd=1.0,
        peak_prompt_tokens=500_000,
        cache_hit_rate=0.98,
        stale_share=0.4,
        top_tool="Bash",
        top_tool_cost_usd=0.2,
    )
    assert summary.headline("$1.00").endswith("40% re-reading old context")
