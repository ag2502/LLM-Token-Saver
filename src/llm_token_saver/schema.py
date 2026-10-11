"""Normalized records shared by importers, detectors and reports.

Token counts follow the Anthropic API convention: `input_uncached` is only the part of a
prompt billed at the full input rate. Cache reads and cache writes are counted separately, so
a call's whole prompt is `input_uncached + cache_read + cache_write_5m + cache_write_1h`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum


class BlockKind(StrEnum):
    """What a piece of context is."""

    USER_TEXT = "user_text"
    TOOL_RESULT = "tool_result"
    ATTACHMENT = "attachment"
    IMAGE = "image"
    ASSISTANT_OUTPUT = "assistant_output"


class MarkerKind(StrEnum):
    """Timeline events that are not billed calls."""

    COMPACTION = "compaction"
    API_ERROR = "api_error"


@dataclass(slots=True)
class Call:
    """One billed model request."""

    call_id: str
    source: str
    session_id: str
    thread_id: str
    seq: int
    timestamp: datetime
    model: str
    input_uncached: int = 0
    cache_read: int = 0
    cache_write_5m: int = 0
    cache_write_1h: int = 0
    output: int = 0
    thinking: int = 0
    speed: str = "standard"
    effort: str = ""
    inference_geo: str = ""
    is_batch: bool = False
    is_sidechain: bool = False
    project: str = ""
    stop_reason: str = ""

    @property
    def cache_write(self) -> int:
        return self.cache_write_5m + self.cache_write_1h

    @property
    def prompt_tokens(self) -> int:
        """Everything the model read for this call, cached or not."""
        return self.input_uncached + self.cache_read + self.cache_write


@dataclass(slots=True)
class Block:
    """A piece of content that entered a thread's context.

    `tokens` is set when the size is measured (an assistant turn's output tokens); otherwise
    the size is estimated from `chars` later, because transcripts do not record it.
    """

    session_id: str
    thread_id: str
    seq: int
    kind: BlockKind
    chars: int = 0
    tokens: int | None = None
    tool: str = ""
    detail: str = ""
    content_hash: str = ""


@dataclass(slots=True)
class Marker:
    """A compaction or an API error on a thread's timeline."""

    session_id: str
    thread_id: str
    seq: int
    kind: MarkerKind
    timestamp: datetime | None = None
    pre_tokens: int = 0
    post_tokens: int = 0


@dataclass(slots=True)
class Thread:
    """One context timeline: a main session, or a single subagent run inside it.

    Calls, blocks and markers share one `seq` ordering, so walking them merged by `seq`
    replays the order in which content entered the context and calls were made.
    """

    thread_id: str
    session_id: str
    source: str
    project: str = ""
    is_sidechain: bool = False
    calls: list[Call] = field(default_factory=list)
    blocks: list[Block] = field(default_factory=list)
    markers: list[Marker] = field(default_factory=list)


@dataclass(slots=True)
class Dataset:
    """Everything one import produced.

    Threads are kept whole even when they started before `window_start`, because earlier
    content is still part of the context that later calls paid to read. Only calls at or
    after `window_start` count toward spend.
    """

    threads: list[Thread] = field(default_factory=list)
    window_start: datetime | None = None

    def in_window(self, call: Call) -> bool:
        return self.window_start is None or call.timestamp >= self.window_start

    @property
    def calls(self) -> list[Call]:
        """Calls inside the window, across all threads."""
        return [call for thread in self.threads for call in thread.calls if self.in_window(call)]
