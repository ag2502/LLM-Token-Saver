"""Builders for synthetic Claude Code transcript lines."""

import json
from pathlib import Path
from typing import Any


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
