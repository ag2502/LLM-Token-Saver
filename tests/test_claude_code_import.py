import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from llm_token_saver.ingest.claude_code import (
    command_family,
    find_transcripts,
    load,
    read_thread,
    tool_detail,
)
from llm_token_saver.schema import BlockKind, MarkerKind


def assistant(
    ts: str,
    message_id: str,
    *,
    request_id: str = "req",
    model: str = "claude-opus-5-5",
    content: list[dict[str, Any]] | None = None,
    output: int = 10,
    uncached: int = 2,
    read: int = 0,
    write: int = 0,
    write_1h: int = 0,
    thinking: int = 0,
    sidechain: bool = False,
) -> dict[str, Any]:
    return {
        "type": "assistant",
        "sessionId": "sess-1",
        "cwd": "/work/app",
        "isSidechain": sidechain,
        "timestamp": ts,
        "requestId": request_id,
        "perTurnEffort": "max",
        "message": {
            "id": message_id,
            "model": model,
            "stop_reason": "tool_use",
            "content": content or [],
            "usage": {
                "input_tokens": uncached,
                "cache_read_input_tokens": read,
                "cache_creation_input_tokens": write,
                "cache_creation": {
                    "ephemeral_5m_input_tokens": write - write_1h,
                    "ephemeral_1h_input_tokens": write_1h,
                },
                "output_tokens": output,
                "output_tokens_details": {"thinking_tokens": thinking},
                "speed": "standard",
                "service_tier": "standard",
            },
        },
    }


def user(ts: str, content: Any) -> dict[str, Any]:
    return {"type": "user", "sessionId": "sess-1", "timestamp": ts, "message": {"content": content}}


def attachment(payload: dict[str, Any]) -> dict[str, Any]:
    return {"type": "attachment", "attachment": payload}


def write_transcript(path: Path, entries: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(entry) for entry in entries) + "\n")
    return path


def test_split_response_lines_become_one_call_with_final_usage(tmp_path: Path) -> None:
    path = write_transcript(
        tmp_path / "sess-1.jsonl",
        [
            assistant("2026-10-01T10:00:00Z", "msg-1", output=10, read=500, write=100),
            assistant("2026-10-01T10:00:05Z", "msg-1", output=120, read=500, write=100),
        ],
    )
    thread = read_thread(path)
    assert thread is not None
    assert len(thread.calls) == 1
    call = thread.calls[0]
    assert call.output == 120
    assert call.seq == 0
    assert call.timestamp == datetime(2026, 10, 1, 10, 0, tzinfo=UTC)
    assert call.effort == "max"
    output_block = next(b for b in thread.blocks if b.kind is BlockKind.ASSISTANT_OUTPUT)
    assert (output_block.seq, output_block.tokens) == (1, 120)


def test_cache_writes_split_by_ttl(tmp_path: Path) -> None:
    path = write_transcript(
        tmp_path / "s.jsonl",
        [assistant("2026-10-01T10:00:00Z", "m", read=1000, write=300, write_1h=200, thinking=4)],
    )
    thread = read_thread(path)
    assert thread is not None
    call = thread.calls[0]
    assert (call.cache_write_5m, call.cache_write_1h, call.cache_read) == (100, 200, 1000)
    assert call.prompt_tokens == 2 + 1000 + 300
    assert call.thinking == 4


def test_tool_results_carry_tool_name_detail_and_hash(tmp_path: Path) -> None:
    tool_use = {
        "type": "tool_use",
        "id": "tu-1",
        "name": "Bash",
        "input": {"command": "cd app && uv run pytest -q"},
    }
    screenshot = {"type": "tool_use", "id": "tu-2", "name": "mcp__browser__screenshot", "input": {}}
    path = write_transcript(
        tmp_path / "s.jsonl",
        [
            assistant("2026-10-01T10:00:00Z", "m1", content=[tool_use, screenshot]),
            user(
                "2026-10-01T10:00:01Z",
                [
                    {"type": "tool_result", "tool_use_id": "tu-1", "content": "5 passed"},
                    {
                        "type": "tool_result",
                        "tool_use_id": "tu-2",
                        "content": [{"type": "image"}, {"type": "text", "text": "ok"}],
                    },
                ],
            ),
            assistant("2026-10-01T10:00:02Z", "m2"),
        ],
    )
    thread = read_thread(path)
    assert thread is not None
    results = [b for b in thread.blocks if b.kind is BlockKind.TOOL_RESULT]
    assert [(b.tool, b.detail, b.chars) for b in results] == [
        ("Bash", "uv run pytest", 8),
        ("mcp__browser__screenshot", "browser", 2),
    ]
    assert results[0].content_hash and results[0].content_hash != results[1].content_hash
    images = [b for b in thread.blocks if b.kind is BlockKind.IMAGE]
    assert [b.tool for b in images] == ["mcp__browser__screenshot"]


def test_bookkeeping_attachments_are_skipped(tmp_path: Path) -> None:
    path = write_transcript(
        tmp_path / "s.jsonl",
        [
            attachment({"type": "prompt_snapshot", "systemPrompt": "x" * 900}),
            attachment({"type": "total_tokens_reminder", "text": "y" * 50}),
            attachment({"type": "skill_listing", "text": "z" * 300}),
            assistant("2026-10-01T10:00:00Z", "m1"),
        ],
    )
    thread = read_thread(path)
    assert thread is not None
    attachments = [b for b in thread.blocks if b.kind is BlockKind.ATTACHMENT]
    assert [(b.tool, b.chars) for b in attachments] == [("skill_listing", 300)]


def test_compactions_and_api_errors_become_markers(tmp_path: Path) -> None:
    path = write_transcript(
        tmp_path / "s.jsonl",
        [
            assistant("2026-10-01T10:00:00Z", "m1"),
            {"type": "system", "subtype": "api_error", "timestamp": "2026-10-01T10:01:00Z"},
            {
                "type": "system",
                "subtype": "compact_boundary",
                "timestamp": "2026-10-01T10:02:00Z",
                "compactMetadata": {"preTokens": 900_000, "postTokens": 40_000},
            },
            assistant("2026-10-01T10:03:00Z", "m2"),
        ],
    )
    thread = read_thread(path)
    assert thread is not None
    assert [(m.kind, m.seq) for m in thread.markers] == [
        (MarkerKind.API_ERROR, 1),
        (MarkerKind.COMPACTION, 2),
    ]
    compaction = thread.markers[1]
    assert (compaction.pre_tokens, compaction.post_tokens) == (900_000, 40_000)


def test_subagent_files_are_their_own_sidechain_thread(tmp_path: Path) -> None:
    path = write_transcript(
        tmp_path / "proj" / "sess-1" / "subagents" / "agent-abc.jsonl",
        [assistant("2026-10-01T10:00:00Z", "m1", sidechain=True)],
    )
    thread = read_thread(path)
    assert thread is not None
    assert (thread.thread_id, thread.session_id) == ("agent-abc", "sess-1")
    assert thread.is_sidechain
    assert thread.project == "/work/app"
    assert thread.calls[0].is_sidechain


def test_files_without_billed_calls_are_ignored(tmp_path: Path) -> None:
    synthetic = assistant("2026-10-01T10:00:00Z", "m1", model="<synthetic>")
    path = write_transcript(tmp_path / "s.jsonl", [synthetic, {"type": "ai-title"}])
    path.write_text(path.read_text() + "not json\n")
    assert read_thread(path) is None


def test_find_transcripts_skips_files_untouched_since_cutoff(tmp_path: Path) -> None:
    old = write_transcript(tmp_path / "a" / "old.jsonl", [assistant("2026-01-01T00:00:00Z", "m")])
    new = write_transcript(tmp_path / "b" / "new.jsonl", [assistant("2026-10-01T00:00:00Z", "m")])
    old_time = datetime(2026, 1, 2, tzinfo=UTC).timestamp()
    os.utime(old, (old_time, old_time))
    assert find_transcripts(tmp_path, since=datetime(2026, 9, 1, tzinfo=UTC)) == [new]
    dataset = load(tmp_path, since=datetime(2026, 9, 1, tzinfo=UTC))
    assert [t.thread_id for t in dataset.threads] == ["new"]


def test_window_keeps_earlier_calls_out_of_spend(tmp_path: Path) -> None:
    write_transcript(
        tmp_path / "s.jsonl",
        [assistant("2026-08-01T00:00:00Z", "early"), assistant("2026-10-01T00:00:00Z", "late")],
    )
    dataset = load(tmp_path, since=datetime(2026, 9, 1, tzinfo=UTC))
    assert len(dataset.threads[0].calls) == 2
    assert [call.call_id for call in dataset.calls] == ["late"]


@pytest.mark.parametrize(
    ("command", "family"),
    [
        ("cd app && uv run pytest -q tests/", "uv run pytest"),
        ("git log --oneline -20 | head", "git log"),
        ("FOO=1 BAR=2 npm run build", "npm run build"),
        ("python3 -m pytest -x", "python3 -m pytest"),
        ("npx tsc --noEmit", "npx tsc"),
        ("/usr/bin/curl -s https://x", "curl"),
        ("git -C repo status", "git"),
        ("cd only", ""),
    ],
)
def test_command_family(command: str, family: str) -> None:
    assert command_family(command) == family


def test_tool_detail_for_files_and_urls() -> None:
    home = str(Path.home())
    assert tool_detail("Read", {"file_path": f"{home}/proj/a.py"}) == "~/proj/a.py"
    assert tool_detail("WebFetch", {"url": "https://docs.example.com/x?y=1"}) == "docs.example.com"
    assert tool_detail("Agent", {"subagent_type": "Explore"}) == "Explore"
