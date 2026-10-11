"""A fluent builder for synthetic threads used across detector tests."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from llm_token_saver.estimate import TokenEstimator
from llm_token_saver.pricing import PriceTable
from llm_token_saver.schema import Block, BlockKind, Call, Marker, MarkerKind, Thread

PRICES = PriceTable.bundled()
ONE_CHAR_PER_TOKEN = TokenEstimator({"claude-opus-5-5": 1.0})
T0 = datetime(2026, 10, 1, tzinfo=UTC)
READ, WRITE_5M = 0.20, 5.0  # Claude Opus 5.5, USD per million tokens


class ThreadBuilder:
    def __init__(self, thread_id: str = "t", *, sidechain: bool = False) -> None:
        self.seq = 0
        self.thread = Thread(
            thread_id=thread_id,
            session_id="s",
            source="test",
            project="~/p",
            is_sidechain=sidechain,
        )

    def block(
        self,
        chars: int,
        kind: BlockKind = BlockKind.TOOL_RESULT,
        tool: str = "Bash",
        detail: str = "",
        content_hash: str = "",
    ) -> ThreadBuilder:
        self.thread.blocks.append(
            Block(
                session_id="s",
                thread_id=self.thread.thread_id,
                seq=self._next(),
                kind=kind,
                chars=chars,
                tool=tool,
                detail=detail,
                content_hash=content_hash,
            )
        )
        return self

    def call(
        self,
        read: int = 0,
        write: int = 0,
        uncached: int = 0,
        output: int = 0,
        *,
        write_1h: int = 0,
        thinking: int = 0,
        effort: str = "",
        model: str = "claude-opus-5-5",
        at: datetime | None = None,
    ) -> ThreadBuilder:
        seq = self._next()
        self.thread.calls.append(
            Call(
                call_id=f"{self.thread.thread_id}-c{seq}",
                source="test",
                session_id=self.thread.session_id,
                thread_id=self.thread.thread_id,
                seq=seq,
                timestamp=at or T0 + timedelta(minutes=seq),
                model=model,
                input_uncached=uncached,
                cache_read=read,
                cache_write_5m=write,
                cache_write_1h=write_1h,
                output=output,
                thinking=thinking,
                effort=effort,
                is_sidechain=self.thread.is_sidechain,
            )
        )
        return self

    def compact(self) -> ThreadBuilder:
        self.thread.markers.append(
            Marker(
                session_id="s",
                thread_id=self.thread.thread_id,
                seq=self._next(),
                kind=MarkerKind.COMPACTION,
            )
        )
        return self

    def _next(self) -> int:
        self.seq += 1
        return self.seq
