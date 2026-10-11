"""In-process DuckDB tables over priced calls, for aggregate queries."""

from __future__ import annotations

import json
import tempfile
from collections.abc import Sequence
from datetime import UTC
from pathlib import Path
from typing import Any

import duckdb

from llm_token_saver.pricing import PriceTable
from llm_token_saver.schema import Call

CALL_COLUMNS: tuple[tuple[str, str], ...] = (
    ("call_id", "VARCHAR"),
    ("source", "VARCHAR"),
    ("session_id", "VARCHAR"),
    ("thread_id", "VARCHAR"),
    ("project", "VARCHAR"),
    ("ts", "TIMESTAMP"),  # UTC
    ("model", "VARCHAR"),
    ("is_sidechain", "BOOLEAN"),
    ("effort", "VARCHAR"),
    ("speed", "VARCHAR"),
    ("input_uncached", "BIGINT"),
    ("cache_read", "BIGINT"),
    ("cache_write_5m", "BIGINT"),
    ("cache_write_1h", "BIGINT"),
    ("output", "BIGINT"),
    ("thinking", "BIGINT"),
    ("priced", "BOOLEAN"),
    ("cost_input", "DOUBLE"),
    ("cost_cache_read", "DOUBLE"),
    ("cost_cache_write_5m", "DOUBLE"),
    ("cost_cache_write_1h", "DOUBLE"),
    ("cost_output", "DOUBLE"),
)


class Store:
    """A `calls` table (tokens plus USD cost per token type) in an in-memory DuckDB.

    Unpriced calls (models missing from the price table) keep their tokens with zero cost
    and `priced = false`, so totals never silently include guesses.
    """

    def __init__(self, calls: Sequence[Call], prices: PriceTable) -> None:
        self.con = duckdb.connect(":memory:")
        self._load_calls(calls, prices)

    def rows(self, sql: str, params: Sequence[Any] | None = None) -> list[tuple[Any, ...]]:
        return self.con.execute(sql, params or []).fetchall()

    def one(self, sql: str, params: Sequence[Any] | None = None) -> tuple[Any, ...]:
        row = self.con.execute(sql, params or []).fetchone()
        if row is None:
            raise LookupError("query returned no rows")
        return row

    def _load_calls(self, calls: Sequence[Call], prices: PriceTable) -> None:
        # Bulk-load through a temporary NDJSON file: DuckDB parses it natively, which is far
        # faster than binding Python values one by one.
        with tempfile.TemporaryDirectory(prefix="tokensaver-") as tmp:
            path = Path(tmp) / "calls.ndjson"
            with path.open("w", encoding="utf-8") as fh:
                for call in calls:
                    fh.write(json.dumps(_call_row(call, prices)))
                    fh.write("\n")
            columns = (
                "{" + ", ".join(f"'{name}': '{sql_type}'" for name, sql_type in CALL_COLUMNS) + "}"
            )
            self.con.execute(
                "CREATE TABLE calls AS SELECT * FROM read_json(?, format = 'newline_delimited', "
                f"columns = {columns})",
                [str(path)],
            )
        self.con.execute(
            "CREATE VIEW priced_calls AS SELECT *, "
            "cost_input + cost_cache_read + cost_cache_write_5m + cost_cache_write_1h "
            "AS cost_prompt, "
            "cost_input + cost_cache_read + cost_cache_write_5m + cost_cache_write_1h "
            "+ cost_output AS cost_total, "
            "input_uncached + cache_read + cache_write_5m + cache_write_1h AS prompt_tokens "
            "FROM calls"
        )


def _call_row(call: Call, prices: PriceTable) -> dict[str, Any]:
    cost = prices.cost(call)
    return {
        "call_id": call.call_id,
        "source": call.source,
        "session_id": call.session_id,
        "thread_id": call.thread_id,
        "project": call.project,
        "ts": call.timestamp.astimezone(UTC).replace(tzinfo=None).isoformat(),
        "model": call.model,
        "is_sidechain": call.is_sidechain,
        "effort": call.effort,
        "speed": call.speed,
        "input_uncached": call.input_uncached,
        "cache_read": call.cache_read,
        "cache_write_5m": call.cache_write_5m,
        "cache_write_1h": call.cache_write_1h,
        "output": call.output,
        "thinking": call.thinking,
        "priced": cost is not None,
        "cost_input": cost.input_uncached if cost else 0.0,
        "cost_cache_read": cost.cache_read if cost else 0.0,
        "cost_cache_write_5m": cost.cache_write_5m if cost else 0.0,
        "cost_cache_write_1h": cost.cache_write_1h if cost else 0.0,
        "cost_output": cost.output if cost else 0.0,
    }
