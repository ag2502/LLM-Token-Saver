"""Import OpenTelemetry traces that follow the GenAI semantic conventions.

Input is OTLP/JSON: one `ExportTraceServiceRequest` object (`{"resourceSpans": [...]}`) per
file, per line (the Collector's file exporter), or a list of them. Directories are searched
for `.json`, `.jsonl` and `.ndjson` files.

Attribute names follow the current conventions, with the older spellings still accepted:

- usage: `gen_ai.usage.input_tokens` (which, per the conventions, *includes* cache reads and
  writes), `gen_ai.usage.output_tokens`, `gen_ai.usage.cache_read.input_tokens`,
  `gen_ai.usage.cache_write.input_tokens` (formerly `cache_creation`), and
  `gen_ai.usage.reasoning.output_tokens`. When an instrumentation reports Anthropic's native
  `input_tokens` (which exclude cache) instead, the total is smaller than the cache counts and
  the value is read as the uncached remainder.
- content (opt-in in instrumentations): `gen_ai.system_instructions`,
  `gen_ai.tool.definitions` and `gen_ai.input.messages` as JSON, or OpenLLMetry's indexed
  `gen_ai.prompt.<n>.role` / `gen_ai.prompt.<n>.content`.
- `anthropic.cache_creation.ephemeral_1h_input_tokens`, a TokenSaver extension, marks cache
  writes made with the 1-hour TTL; without it writes are priced as 5-minute writes.

Each conversation (`gen_ai.conversation.id`, else the trace) becomes a thread. When requests
in a conversation carry their messages, the content added between two requests becomes
context blocks, so carried-context analysis works for chat apps as well as Claude Code.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from llm_token_saver.schema import (
    Block,
    BlockKind,
    Call,
    Dataset,
    Marker,
    MarkerKind,
    PromptPart,
    Thread,
)

SOURCE = "otel"
TRACE_SUFFIXES = frozenset({".json", ".jsonl", ".ndjson"})
CHAT_OPERATIONS = frozenset({"chat", "text_completion", "generate_content"})

_INPUT = ("gen_ai.usage.input_tokens", "gen_ai.usage.prompt_tokens", "llm.usage.prompt_tokens")
_OUTPUT = (
    "gen_ai.usage.output_tokens",
    "gen_ai.usage.completion_tokens",
    "llm.usage.completion_tokens",
)
_CACHE_READ = ("gen_ai.usage.cache_read.input_tokens", "gen_ai.usage.cache_read_input_tokens")
_CACHE_WRITE = (
    "gen_ai.usage.cache_write.input_tokens",
    "gen_ai.usage.cache_creation.input_tokens",
    "gen_ai.usage.cache_creation_input_tokens",
)
_REASONING = ("gen_ai.usage.reasoning.output_tokens",)
_MODEL = ("gen_ai.response.model", "gen_ai.request.model")
_PROVIDER = ("gen_ai.provider.name", "gen_ai.system")
CACHE_WRITE_1H = "anthropic.cache_creation.ephemeral_1h_input_tokens"


def find_trace_files(paths: Iterable[Path]) -> list[Path]:
    found: list[Path] = []
    for path in paths:
        if path.is_dir():
            found += sorted(
                p for p in path.rglob("*") if p.is_file() and p.suffix in TRACE_SUFFIXES
            )
        elif path.is_file():
            found.append(path)
    return found


def load(paths: Iterable[Path], *, since: datetime | None = None) -> Dataset:
    """Import every GenAI call in the trace files under `paths`."""
    by_thread: dict[tuple[str, str], list[_Request]] = defaultdict(list)
    for path in find_trace_files(paths):
        for resource, span in iter_spans(path):
            request = _request(resource, span)
            if request is not None:
                by_thread[(request.call.project, request.thread_key)].append(request)
    threads = [_thread(requests) for requests in by_thread.values()]
    threads.sort(key=lambda thread: thread.calls[0].timestamp)
    return Dataset(threads=threads, window_start=since)


def iter_spans(path: Path) -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
    """(resource attributes, span) for every span in an OTLP/JSON file."""
    for request in _export_requests(path):
        for resource_spans in request.get("resourceSpans") or []:
            resource = attributes((resource_spans.get("resource") or {}).get("attributes"))
            scopes = resource_spans.get("scopeSpans") or resource_spans.get(
                "instrumentationLibrarySpans", []
            )
            for scope in scopes:
                for span in scope.get("spans") or []:
                    if isinstance(span, dict):
                        yield resource, span


def attributes(items: Any) -> dict[str, Any]:
    """OTLP `KeyValue` list (or an already-flat mapping) to a plain dict."""
    if isinstance(items, dict):
        return dict(items)
    result: dict[str, Any] = {}
    for item in items or []:
        if isinstance(item, dict) and "key" in item:
            result[item["key"]] = any_value(item.get("value") or {})
    return result


def any_value(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    if "stringValue" in value:
        return value["stringValue"]
    if "intValue" in value:
        return int(value["intValue"])
    if "doubleValue" in value:
        return float(value["doubleValue"])
    if "boolValue" in value:
        return bool(value["boolValue"])
    if "arrayValue" in value:
        return [any_value(item) for item in (value["arrayValue"] or {}).get("values", [])]
    if "kvlistValue" in value:
        return attributes((value["kvlistValue"] or {}).get("values", []))
    return value.get("bytesValue")


def _export_requests(path: Path) -> Iterator[dict[str, Any]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        for line in text.splitlines():
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                yield item
        return
    if isinstance(data, list):
        yield from (item for item in data if isinstance(item, dict))
    elif isinstance(data, dict):
        yield data


@dataclass(slots=True)
class _Part:
    """A message part with what the thread builder needs to know about it."""

    role: str
    kind: str
    text: str
    tool: str = ""
    tool_call_id: str = ""

    @property
    def key(self) -> tuple[str, str]:
        return (self.role, _hash(self.text))


@dataclass(slots=True)
class _Request:
    call: Call
    thread_key: str
    messages: list[_Part] | None


def _request(resource: dict[str, Any], span: dict[str, Any]) -> _Request | None:
    attrs = attributes(span.get("attributes"))
    total_input = _first_int(attrs, _INPUT)
    operation = attrs.get("gen_ai.operation.name")
    if operation not in CHAT_OPERATIONS and total_input is None:
        return None
    try:
        start_ns = int(span.get("startTimeUnixNano") or 0)
        end_ns = int(span.get("endTimeUnixNano") or start_ns)
    except (TypeError, ValueError):
        return None
    if not start_ns:
        return None
    read = _first_int(attrs, _CACHE_READ) or 0
    written = _first_int(attrs, _CACHE_WRITE) or 0
    total_input = total_input or 0
    # Conventions: input_tokens includes cache. Native Anthropic counts exclude it.
    uncached = total_input - read - written if total_input >= read + written else total_input
    written_1h = min(int(attrs.get(CACHE_WRITE_1H) or 0), written)
    output = _first_int(attrs, _OUTPUT) or 0
    reasoning = min(_first_int(attrs, _REASONING) or 0, output)
    model = str(_first(attrs, _MODEL) or "")
    service = str(resource.get("service.name") or attrs.get("service.name") or "")
    trace_id = str(span.get("traceId") or "")
    conversation = str(attrs.get("gen_ai.conversation.id") or "")
    system, tools, messages = _content(attrs)
    prompt = _prompt(system, tools, messages)
    call = Call(
        call_id=str(attrs.get("gen_ai.response.id") or span.get("spanId") or ""),
        source=SOURCE,
        session_id=conversation or trace_id,
        thread_id=conversation or trace_id,
        seq=0,
        timestamp=datetime.fromtimestamp(start_ns / 1e9, UTC),
        model=model,
        input_uncached=uncached,
        cache_read=read,
        cache_write_5m=written - written_1h,
        cache_write_1h=written_1h,
        output=output,
        thinking=reasoning,
        project=service,
        stop_reason=_finish_reason(attrs.get("gen_ai.response.finish_reasons")),
        provider=str(_first(attrs, _PROVIDER) or ""),
        trace_id=trace_id,
        latency_ms=(end_ns - start_ns) / 1e6,
        request_hash=_request_hash(model, prompt, attrs),
        prompt=prompt,
    )
    return _Request(call=call, thread_key=conversation or trace_id, messages=messages)


def _content(attrs: dict[str, Any]) -> tuple[list[str], Any, list[_Part] | None]:
    system: list[str] = []
    for item in _json(attrs.get("gen_ai.system_instructions")) or []:
        text = item.get("content") if isinstance(item, dict) else item
        if isinstance(text, str):
            system.append(text)
    tools = _json(attrs.get("gen_ai.tool.definitions"))
    raw = _json(attrs.get("gen_ai.input.messages"))
    if raw is None:
        raw = _indexed_messages(attrs)
    if raw is None:
        return system, tools, None
    parts: list[_Part] = []
    for message in raw:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "")
        message_parts = message.get("parts")
        if message_parts is None and "content" in message:
            message_parts = [{"type": "text", "content": message["content"]}]
        for part in message_parts or []:
            if isinstance(part, dict):
                parts.append(_message_part(role, part))
    # A system-role message is part of the system prompt for prefix purposes.
    while parts and parts[0].role == "system":
        system.append(parts.pop(0).text)
    return system, tools, parts


def _message_part(role: str, part: dict[str, Any]) -> _Part:
    kind = str(part.get("type") or "text")
    call_id = str(part.get("id") or "")
    if kind in {"text", "reasoning"}:
        return _Part(role, kind, _as_text(part.get("content")))
    if kind == "tool_call":
        name = str(part.get("name") or "")
        arguments = part.get("arguments")
        return _Part(role, kind, f"{name}({_as_text(arguments)})", tool=name, tool_call_id=call_id)
    if kind == "tool_call_response":
        return _Part(role, kind, _as_text(part.get("response")), tool_call_id=call_id)
    if kind == "blob":
        return _Part(role, "blob", "")
    return _Part(role, kind, _as_text(part.get("content")))


def _indexed_messages(attrs: dict[str, Any]) -> list[dict[str, Any]] | None:
    """OpenLLMetry's `gen_ai.prompt.<n>.role` / `gen_ai.prompt.<n>.content` attributes."""
    indexed: dict[int, dict[str, Any]] = defaultdict(dict)
    for key, value in attrs.items():
        pieces = key.split(".")
        if len(pieces) == 4 and pieces[:2] == ["gen_ai", "prompt"] and pieces[2].isdigit():
            indexed[int(pieces[2])][pieces[3]] = value
    if not indexed:
        return None
    return [
        {"role": indexed[i].get("role", ""), "content": indexed[i].get("content", "")}
        for i in sorted(indexed)
    ]


def _prompt(system: list[str], tools: Any, messages: list[_Part] | None) -> list[PromptPart] | None:
    if not system and not tools and messages is None:
        return None
    parts: list[PromptPart] = []
    if tools:
        parts.append(PromptPart("tools", text=json.dumps(tools, sort_keys=True)))
    parts += [PromptPart("system", role="system", text=text) for text in system]
    parts += [PromptPart("message", role=p.role, text=p.text) for p in messages or []]
    for part in parts:
        part.content_hash = _hash(f"{part.kind}\0{part.role}\0{part.text}")
    return parts


def _thread(requests: list[_Request]) -> Thread:
    requests.sort(key=lambda request: request.call.timestamp)
    first = requests[0].call
    thread = Thread(
        thread_id=first.thread_id or first.call_id,
        session_id=first.session_id,
        source=SOURCE,
        project=first.project,
    )
    seq = 0
    previous: list[tuple[str, str]] | None = None
    tool_names: dict[str, str] = {}
    for request in requests:
        call = request.call
        if request.messages is not None:
            for part in request.messages:
                if part.kind == "tool_call":
                    tool_names[part.tool_call_id] = part.tool
            keys = [part.key for part in request.messages]
            if previous is not None and keys[: len(previous)] == previous:
                added = request.messages[len(previous) :]
                while added and added[0].role == "assistant":
                    added = added[1:]  # the previous reply, already counted as its output
            else:
                if previous is not None:
                    thread.markers.append(_marker(thread, seq, MarkerKind.COMPACTION))
                    seq += 1
                added = request.messages
            for part in added:
                thread.blocks.append(_block(thread, seq, part, tool_names))
                seq += 1
            previous = keys
        call.seq = seq
        call.session_id, call.thread_id = thread.session_id, thread.thread_id
        thread.calls.append(call)
        seq += 1
        thread.blocks.append(
            Block(
                session_id=thread.session_id,
                thread_id=thread.thread_id,
                seq=seq,
                kind=BlockKind.ASSISTANT_OUTPUT,
                tokens=call.output - call.thinking,
            )
        )
        if call.thinking:
            thread.blocks.append(
                Block(
                    session_id=thread.session_id,
                    thread_id=thread.thread_id,
                    seq=seq,
                    kind=BlockKind.THINKING,
                    tokens=call.thinking,
                )
            )
        seq += 1
    return thread


def _block(thread: Thread, seq: int, part: _Part, tool_names: dict[str, str]) -> Block:
    if part.kind == "blob":
        kind = BlockKind.IMAGE
    elif part.kind == "tool_call_response" or part.role == "tool":
        kind = BlockKind.TOOL_RESULT
    elif part.role == "assistant":
        kind = BlockKind.ASSISTANT_OUTPUT
    else:
        kind = BlockKind.USER_TEXT
    return Block(
        session_id=thread.session_id,
        thread_id=thread.thread_id,
        seq=seq,
        kind=kind,
        chars=len(part.text),
        tool=tool_names.get(part.tool_call_id, "") if kind is BlockKind.TOOL_RESULT else "",
        content_hash=_hash(part.text) if part.text else "",
    )


def _marker(thread: Thread, seq: int, kind: MarkerKind) -> Marker:
    return Marker(session_id=thread.session_id, thread_id=thread.thread_id, seq=seq, kind=kind)


def _request_hash(model: str, prompt: list[PromptPart] | None, attrs: dict[str, Any]) -> str:
    if prompt is None:
        return ""
    params = {
        key: attrs.get(key)
        for key in ("gen_ai.request.max_tokens", "gen_ai.request.temperature")
        if key in attrs
    }
    body = json.dumps(
        [model, [p.content_hash for p in prompt], params], sort_keys=True, default=str
    )
    return _hash(body)


def _first(attrs: dict[str, Any], names: tuple[str, ...]) -> Any:
    for name in names:
        if attrs.get(name) not in (None, ""):
            return attrs[name]
    return None


def _first_int(attrs: dict[str, Any], names: tuple[str, ...]) -> int | None:
    value = _first(attrs, names)
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _finish_reason(value: Any) -> str:
    if isinstance(value, list):
        return str(value[0]) if value else ""
    return str(value or "")


def _json(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return None
    return value


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True, default=str)


def _hash(text: str) -> str:
    return hashlib.blake2b(text.encode("utf-8", "replace"), digest_size=8).hexdigest()
