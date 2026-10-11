"""Replay a thread's calls, context blocks and markers in the order they happened."""

from __future__ import annotations

from typing import TypeAlias

from llm_token_saver.schema import Block, Call, Marker, Thread

Event: TypeAlias = Call | Block | Marker


def timeline(thread: Thread) -> list[Event]:
    """Calls, blocks and markers merged by `seq`.

    A response's output block can share its call's `seq` (single-line responses); the call
    sorts first because its output only enters the context after it was generated.
    """
    keyed: list[tuple[int, int, Event]] = [
        *((call.seq, 0, call) for call in thread.calls),
        *((block.seq, 1, block) for block in thread.blocks),
        *((marker.seq, 2, marker) for marker in thread.markers),
    ]
    keyed.sort(key=lambda item: (item[0], item[1]))
    return [event for _, _, event in keyed]
