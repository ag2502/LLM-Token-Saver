"""Import Claude Code session transcripts (`~/.claude/projects/**/*.jsonl`).

Format facts this importer relies on, checked against real transcripts (October 2026):

- One JSON object per line. `type` is "assistant", "user", "attachment", "system", or
  bookkeeping that is skipped.
- One model response is split over several "assistant" lines (one per content block) that
  share `message.id` and `requestId`. Every line repeats `message.usage`, and the output
  count grows across those lines, so the last line holds the final usage.
- Subagent runs live in `<session>/subagents/agent-<id>.jsonl`. They carry the parent's
  `sessionId` but have a context of their own, so each file is its own thread.
- `system` lines with subtype "compact_boundary" mark a compaction (with token counts in
  `compactMetadata`); subtype "api_error" marks a failed request that was retried.
- Attachment types in `_BOOKKEEPING_ATTACHMENTS` never reach the model: fitting call-to-call
  usage deltas against attachment sizes gives them a weight of about zero.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from llm_token_saver.schema import (
    Block,
    BlockKind,
    Call,
    Dataset,
    Marker,
    MarkerKind,
    Thread,
)

SOURCE = "claude_code"
SYNTHETIC_MODEL = "<synthetic>"
_BOOKKEEPING_ATTACHMENTS = frozenset({"prompt_snapshot", "queued_command", "total_tokens_reminder"})
_FILE_TOOLS = frozenset({"Read", "Edit", "MultiEdit", "Write", "NotebookEdit"})


def default_projects_dir() -> Path:
    return Path.home() / ".claude" / "projects"


def find_transcripts(root: Path, *, since: datetime | None = None) -> list[Path]:
    """Transcript files under `root`, skipping files last written before `since`."""
    paths = sorted(root.glob("**/*.jsonl"))
    if since is None:
        return paths
    cutoff = since.timestamp()
    return [path for path in paths if path.stat().st_mtime >= cutoff]


def load(root: Path | None = None, *, since: datetime | None = None) -> Dataset:
    """Import every transcript under `root` that may contain calls at or after `since`."""
    root = (root or default_projects_dir()).expanduser()
    threads = [
        thread for path in find_transcripts(root, since=since) if (thread := read_thread(path))
    ]
    return Dataset(threads=threads, window_start=since)


def read_thread(path: Path) -> Thread | None:
    """Parse one transcript file; None if it contains no billed calls."""
    builder = _ThreadBuilder(thread_id=path.stem)
    with path.open(encoding="utf-8", errors="replace") as fh:
        for seq, line in enumerate(fh):
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(entry, dict):
                builder.add(seq, entry)
    return builder.build()


def command_family(command: str) -> str:
    """The program a shell command runs: `cd app && uv run pytest -q` -> `uv run pytest`."""
    for segment in re.split(r"&&|\|\||;|\|", command):
        words = segment.strip().split()
        while words and _is_env_assignment(words[0]):
            words = words[1:]
        while words and words[0] in {"sudo", "time", "env", "nohup", "exec"}:
            words = words[1:]
        if not words or words[0] in {"cd", "pushd", "popd", "export", "source", ".", "set"}:
            continue
        head = os.path.basename(words[0])
        rest = words[1:]
        family = [head]
        if head in {"python", "python3"} and rest[:1] == ["-m"] and len(rest) > 1:
            family += ["-m", rest[1]]
        elif head in {"npx", "uvx", "bunx", "pipx"} and rest and not rest[0].startswith("-"):
            family.append(rest[0])
        elif head in _SUBCOMMAND_TOOLS and rest and not rest[0].startswith("-"):
            family.append(rest[0])
            if rest[0] in {"run", "exec", "x"} and len(rest) > 1 and not rest[1].startswith("-"):
                family.append(rest[1])
        return " ".join(family)
    return ""


_SUBCOMMAND_TOOLS = frozenset(
    {
        "git", "npm", "pnpm", "yarn", "bun", "uv", "cargo", "go", "docker", "kubectl", "pip",
        "pip3", "brew", "poetry", "gh", "make", "deno", "dotnet", "terraform", "helm",
    }
)  # fmt: skip


def tool_detail(name: str, tool_input: dict[str, Any]) -> str:
    """A short label for what a tool call touched, used to group its results."""
    if name == "Bash":
        return command_family(str(tool_input.get("command", "")))
    if name in _FILE_TOOLS:
        path = tool_input.get("file_path") or tool_input.get("notebook_path") or ""
        return shorten_path(str(path))
    if name in {"Grep", "Glob"}:
        return str(tool_input.get("pattern", ""))[:60]
    if name == "WebFetch":
        return urlparse(str(tool_input.get("url", ""))).netloc
    if name in {"Agent", "Task"}:
        return str(tool_input.get("subagent_type", ""))
    if name.startswith("mcp__"):
        parts = name.split("__")
        return parts[1] if len(parts) > 2 else ""
    return ""


def shorten_path(path: str) -> str:
    home = str(Path.home())
    return "~" + path[len(home) :] if path.startswith(home + os.sep) else path


def _is_env_assignment(word: str) -> bool:
    name, sep, _ = word.partition("=")
    return bool(sep) and name.isidentifier()


def _parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _content_hash(text: str) -> str:
    return hashlib.blake2b(text.encode("utf-8", "replace"), digest_size=8).hexdigest()


def _text_length(value: Any) -> int:
    """Characters of text in an attachment payload, ignoring its `type` tag."""
    if isinstance(value, str):
        return len(value)
    if isinstance(value, dict):
        return sum(_text_length(item) for key, item in value.items() if key != "type")
    if isinstance(value, list):
        return sum(_text_length(item) for item in value)
    return 0


@dataclass(slots=True)
class _Response:
    """The pieces of one model response gathered across its transcript lines."""

    message_id: str
    first_seq: int
    last_seq: int
    timestamp: datetime | None
    model: str
    usage: dict[str, Any]
    effort: str
    stop_reason: str


class _ThreadBuilder:
    def __init__(self, thread_id: str) -> None:
        self.thread_id = thread_id
        self.session_id = ""
        self.project = ""
        self.is_sidechain = False
        self.responses: dict[tuple[str, str], _Response] = {}
        self.blocks: list[Block] = []
        self.markers: list[Marker] = []
        self.tool_uses: dict[str, tuple[str, str]] = {}
        self.last_timestamp: datetime | None = None

    def add(self, seq: int, entry: dict[str, Any]) -> None:
        if not self.session_id and isinstance(entry.get("sessionId"), str):
            self.session_id = entry["sessionId"]
        if not self.project and isinstance(entry.get("cwd"), str):
            self.project = shorten_path(entry["cwd"])
        if entry.get("isSidechain") is True:
            self.is_sidechain = True
        timestamp = _parse_timestamp(entry.get("timestamp")) or self.last_timestamp
        self.last_timestamp = timestamp
        kind = entry.get("type")
        if kind == "assistant":
            self._add_assistant(seq, entry, timestamp)
        elif kind == "user":
            self._add_user(seq, entry)
        elif kind == "attachment":
            self._add_attachment(seq, entry)
        elif kind == "system":
            self._add_system(seq, entry, timestamp)

    def _add_assistant(self, seq: int, entry: dict[str, Any], timestamp: datetime | None) -> None:
        message = entry.get("message")
        if not isinstance(message, dict):
            return
        for block in _content_blocks(message):
            if block.get("type") == "tool_use" and isinstance(block.get("id"), str):
                name = str(block.get("name", ""))
                tool_input = block.get("input")
                detail = tool_detail(name, tool_input) if isinstance(tool_input, dict) else ""
                self.tool_uses[block["id"]] = (name, detail)
        usage = message.get("usage")
        model = str(message.get("model", ""))
        if not isinstance(usage, dict) or not model or model == SYNTHETIC_MODEL:
            return
        message_id = str(message.get("id", ""))
        key = (message_id, str(entry.get("requestId", "")))
        effort = entry.get("perTurnEffort") or entry.get("effort") or ""
        previous = self.responses.get(key)
        stop_reason = message.get("stop_reason") or (previous.stop_reason if previous else "")
        self.responses[key] = _Response(
            message_id=message_id,
            first_seq=previous.first_seq if previous else seq,
            last_seq=seq,
            timestamp=previous.timestamp if previous else timestamp,
            model=model,
            usage=usage,  # later lines carry the final counts
            effort=str(effort),
            stop_reason=str(stop_reason),
        )

    def _add_user(self, seq: int, entry: dict[str, Any]) -> None:
        message = entry.get("message")
        if not isinstance(message, dict):
            return
        content = message.get("content")
        if isinstance(content, str):
            self._text_block(seq, BlockKind.USER_TEXT, content)
            return
        for block in _content_blocks(message):
            block_type = block.get("type")
            if block_type == "text":
                self._text_block(seq, BlockKind.USER_TEXT, str(block.get("text", "")))
            elif block_type == "image":
                self._block(seq, BlockKind.IMAGE)
            elif block_type == "tool_result":
                tool, detail = self.tool_uses.get(str(block.get("tool_use_id", "")), ("", ""))
                self._add_tool_result(seq, block.get("content"), tool, detail)

    def _add_tool_result(self, seq: int, content: Any, tool: str, detail: str) -> None:
        if isinstance(content, str):
            self._text_block(seq, BlockKind.TOOL_RESULT, content, tool, detail)
            return
        if not isinstance(content, list):
            return
        texts: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "image":
                self._block(seq, BlockKind.IMAGE, tool=tool, detail=detail)
            elif isinstance(item.get("text"), str):
                texts.append(item["text"])
        if texts:
            self._text_block(seq, BlockKind.TOOL_RESULT, "\n".join(texts), tool, detail)

    def _add_attachment(self, seq: int, entry: dict[str, Any]) -> None:
        attachment = entry.get("attachment")
        if not isinstance(attachment, dict):
            return
        kind = str(attachment.get("type", ""))
        if kind in _BOOKKEEPING_ATTACHMENTS:
            return
        chars = _text_length(attachment)
        if chars:
            detail = shorten_path(str(attachment.get("filename") or ""))
            self._block(seq, BlockKind.ATTACHMENT, chars=chars, tool=kind, detail=detail)

    def _add_system(self, seq: int, entry: dict[str, Any], timestamp: datetime | None) -> None:
        subtype = entry.get("subtype")
        if subtype == "compact_boundary":
            meta = entry.get("compactMetadata")
            meta = meta if isinstance(meta, dict) else {}
            self.markers.append(
                Marker(
                    session_id=self.session_id,
                    thread_id=self.thread_id,
                    seq=seq,
                    kind=MarkerKind.COMPACTION,
                    timestamp=timestamp,
                    pre_tokens=int(meta.get("preTokens") or 0),
                    post_tokens=int(meta.get("postTokens") or 0),
                )
            )
        elif subtype == "api_error":
            self.markers.append(
                Marker(
                    session_id=self.session_id,
                    thread_id=self.thread_id,
                    seq=seq,
                    kind=MarkerKind.API_ERROR,
                    timestamp=timestamp,
                )
            )

    def _text_block(
        self, seq: int, kind: BlockKind, text: str, tool: str = "", detail: str = ""
    ) -> None:
        if text:
            self._block(seq, kind, chars=len(text), tool=tool, detail=detail, text=text)

    def _block(
        self,
        seq: int,
        kind: BlockKind,
        *,
        chars: int = 0,
        tool: str = "",
        detail: str = "",
        text: str = "",
    ) -> None:
        self.blocks.append(
            Block(
                session_id=self.session_id,
                thread_id=self.thread_id,
                seq=seq,
                kind=kind,
                chars=chars,
                tool=tool,
                detail=detail,
                content_hash=_content_hash(text) if text else "",
            )
        )

    def build(self) -> Thread | None:
        calls: list[Call] = []
        outputs: list[Block] = []
        for response in self.responses.values():
            call = self._call(response)
            if call is None:
                continue
            calls.append(call)
            # The response enters the context right after it was generated: its visible
            # part and, separately, its thinking (which later prompts mostly keep).
            outputs.append(
                Block(
                    session_id=self.session_id,
                    thread_id=self.thread_id,
                    seq=response.last_seq,
                    kind=BlockKind.ASSISTANT_OUTPUT,
                    tokens=call.output - call.thinking,
                )
            )
            if call.thinking:
                outputs.append(
                    Block(
                        session_id=self.session_id,
                        thread_id=self.thread_id,
                        seq=response.last_seq,
                        kind=BlockKind.THINKING,
                        tokens=call.thinking,
                    )
                )
        if not calls:
            return None
        blocks = sorted([*self.blocks, *outputs], key=lambda block: block.seq)
        for block in blocks:
            block.session_id = self.session_id
        for marker in self.markers:
            marker.session_id = self.session_id
        return Thread(
            thread_id=self.thread_id,
            session_id=self.session_id,
            source=SOURCE,
            project=self.project,
            is_sidechain=self.is_sidechain,
            calls=sorted(calls, key=lambda call: call.seq),
            blocks=blocks,
            markers=self.markers,
        )

    def _call(self, response: _Response) -> Call | None:
        if response.timestamp is None:
            return None
        usage = response.usage
        written = int(usage.get("cache_creation_input_tokens") or 0)
        by_ttl = usage.get("cache_creation")
        by_ttl = by_ttl if isinstance(by_ttl, dict) else {}
        write_1h = min(int(by_ttl.get("ephemeral_1h_input_tokens") or 0), written)
        details = usage.get("output_tokens_details")
        thinking = int(details.get("thinking_tokens") or 0) if isinstance(details, dict) else 0
        output = int(usage.get("output_tokens") or 0)
        return Call(
            call_id=response.message_id,
            source=SOURCE,
            session_id=self.session_id,
            thread_id=self.thread_id,
            seq=response.first_seq,
            timestamp=response.timestamp,
            model=response.model,
            input_uncached=int(usage.get("input_tokens") or 0),
            cache_read=int(usage.get("cache_read_input_tokens") or 0),
            cache_write_5m=written - write_1h,
            cache_write_1h=write_1h,
            output=output,
            thinking=min(thinking, output),
            speed=str(usage.get("speed") or "standard"),
            effort=response.effort,
            inference_geo=str(usage.get("inference_geo") or ""),
            is_batch=usage.get("service_tier") == "batch",
            is_sidechain=self.is_sidechain,
            project=self.project,
            stop_reason=response.stop_reason,
        )


def _content_blocks(message: dict[str, Any]) -> Iterable[dict[str, Any]]:
    content = message.get("content")
    if isinstance(content, list):
        return (block for block in content if isinstance(block, dict))
    return ()
