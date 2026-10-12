import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from llm_token_saver import otlp
from llm_token_saver.ingest.otel import CACHE_WRITE_1H, load
from llm_token_saver.schema import BlockKind, MarkerKind

T0 = datetime(2026, 10, 1, 12, 0, tzinfo=UTC)


def chat(
    seconds: float, attrs: dict[str, Any], *, trace: str = "t1", span_id: str = "s1"
) -> dict[str, Any]:
    base = {"gen_ai.operation.name": "chat", "gen_ai.provider.name": "anthropic"}
    return otlp.span(
        trace_id=trace,
        span_id=span_id,
        start=T0 + timedelta(seconds=seconds),
        duration=timedelta(seconds=2),
        attrs=base | attrs,
    )


def messages(*items: tuple[str, list[dict[str, Any]]]) -> str:
    return json.dumps([{"role": role, "parts": parts} for role, parts in items])


def text(content: str) -> dict[str, Any]:
    return {"type": "text", "content": content}


def write(path: Path, *requests: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(list(requests)))
    return path


def test_current_conventions_count_cache_inside_input_tokens(tmp_path: Path) -> None:
    span = chat(
        0,
        {
            "gen_ai.request.model": "claude-opus-5-5",
            "gen_ai.response.model": "claude-opus-5-5",
            "gen_ai.response.id": "msg_1",
            "gen_ai.conversation.id": "conv-1",
            "gen_ai.usage.input_tokens": 10_300,
            "gen_ai.usage.cache_read.input_tokens": 10_000,
            "gen_ai.usage.cache_write.input_tokens": 200,
            "gen_ai.usage.output_tokens": 50,
            "gen_ai.usage.reasoning.output_tokens": 20,
            "gen_ai.system_instructions": json.dumps([text("You are helpful.")]),
            "gen_ai.tool.definitions": json.dumps([{"name": "search"}]),
            "gen_ai.input.messages": messages(("user", [text("hi")])),
        },
    )
    dataset = load([write(tmp_path / "a.json", otlp.export_request([span], "support-bot"))])
    [call] = dataset.calls
    assert (call.input_uncached, call.cache_read, call.cache_write_5m) == (100, 10_000, 200)
    assert (call.output, call.thinking) == (50, 20)
    assert (call.model, call.project, call.thread_id) == (
        "claude-opus-5-5",
        "support-bot",
        "conv-1",
    )
    assert call.latency_ms == 2_000
    assert call.prompt is not None
    assert [(p.kind, p.text) for p in call.prompt] == [
        ("tools", '[{"name": "search"}]'),
        ("system", "You are helpful."),
        ("message", "hi"),
    ]
    assert call.request_hash


def test_native_anthropic_counts_are_the_uncached_remainder(tmp_path: Path) -> None:
    span = chat(
        0,
        {
            "gen_ai.request.model": "claude-opus-5-5",
            "gen_ai.usage.input_tokens": 100,
            "gen_ai.usage.cache_read.input_tokens": 10_000,
            "gen_ai.usage.cache_creation.input_tokens": 200,
            CACHE_WRITE_1H: 150,
        },
    )
    [call] = load([write(tmp_path / "a.json", otlp.export_request([span], "s"))]).calls
    assert (call.input_uncached, call.cache_write_5m, call.cache_write_1h) == (100, 50, 150)
    assert call.prompt is None and call.request_hash == ""


def test_openllmetry_indexed_attributes(tmp_path: Path) -> None:
    span = chat(
        0,
        {
            "gen_ai.system": "anthropic",
            "gen_ai.request.model": "claude-haiku-5-5",
            "gen_ai.usage.prompt_tokens": 900,
            "gen_ai.usage.completion_tokens": 9,
            "gen_ai.usage.cache_read_input_tokens": 0,
            "gen_ai.prompt.0.role": "system",
            "gen_ai.prompt.0.content": "Classify tickets.",
            "gen_ai.prompt.1.role": "user",
            "gen_ai.prompt.1.content": "My card was charged twice",
        },
    )
    [call] = load([write(tmp_path / "a.json", otlp.export_request([span], "router"))]).calls
    assert (call.input_uncached, call.output, call.provider) == (900, 9, "anthropic")
    assert call.prompt is not None
    assert [(p.kind, p.role) for p in call.prompt] == [("system", "system"), ("message", "user")]


def test_jsonl_files_in_directories_and_non_chat_spans(tmp_path: Path) -> None:
    tool_span = otlp.span(
        trace_id="t",
        span_id="x",
        start=T0,
        duration=timedelta(seconds=1),
        attrs={"gen_ai.operation.name": "execute_tool", "gen_ai.tool.name": "search"},
    )
    lines = [
        otlp.export_request([chat(0, {"gen_ai.usage.input_tokens": 10})], "a"),
        otlp.export_request([tool_span], "a"),
        otlp.export_request([chat(5, {"gen_ai.usage.input_tokens": 20}, trace="t2")], "a"),
    ]
    nested = tmp_path / "traces" / "day1.jsonl"
    nested.parent.mkdir(parents=True)
    nested.write_text("\n".join(json.dumps(line) for line in lines) + "\nnot json\n")
    dataset = load([tmp_path / "traces"])
    assert sorted(call.input_uncached for call in dataset.calls) == [10, 20]


def test_conversation_turns_become_context_blocks(tmp_path: Path) -> None:
    conversation = {"gen_ai.conversation.id": "c1", "gen_ai.request.model": "claude-opus-5-5"}
    first = chat(0, conversation | {"gen_ai.input.messages": messages(("user", [text("q1")]))})
    second = chat(
        60,
        conversation
        | {
            "gen_ai.input.messages": messages(
                ("user", [text("q1")]),
                ("assistant", [{"type": "tool_call", "id": "tc1", "name": "grep"}]),
                ("tool", [{"type": "tool_call_response", "id": "tc1", "response": "x" * 500}]),
                ("user", [text("q2")]),
            )
        },
        span_id="s2",
    )
    rewritten = chat(
        120,
        conversation | {"gen_ai.input.messages": messages(("user", [text("summary")]))},
        span_id="s3",
    )
    dataset = load(
        [write(tmp_path / "a.json", otlp.export_request([rewritten, first, second], "agent"))]
    )
    [thread] = dataset.threads
    inputs = [b for b in thread.blocks if b.kind is not BlockKind.ASSISTANT_OUTPUT]
    kinds = [(b.kind, b.tool, b.chars) for b in inputs]
    assert kinds == [
        (BlockKind.USER_TEXT, "", 2),
        (BlockKind.TOOL_RESULT, "grep", 500),
        (BlockKind.USER_TEXT, "", 2),
        (BlockKind.USER_TEXT, "", 7),
    ]
    assert [m.kind for m in thread.markers] == [MarkerKind.COMPACTION]
    first_call, second_call, _ = thread.calls
    assert first_call.seq < inputs[1].seq < second_call.seq


def test_window_and_flat_attribute_maps(tmp_path: Path) -> None:
    flat = {
        "traceId": "t",
        "spanId": "s",
        "startTimeUnixNano": str(int(T0.timestamp() * 1e9)),
        "attributes": {"gen_ai.operation.name": "chat", "gen_ai.usage.input_tokens": 7},
    }
    later = chat(3_600, {"gen_ai.usage.input_tokens": 8}, trace="t9")
    dataset = load(
        [write(tmp_path / "a.json", otlp.export_request([flat, later], "s"))],
        since=T0 + timedelta(minutes=30),
    )
    assert [call.input_uncached for call in dataset.calls] == [8]
    assert len(dataset.threads) == 2
