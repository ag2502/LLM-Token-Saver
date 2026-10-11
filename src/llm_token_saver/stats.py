"""Spend and usage summaries over the `calls` table of a Store."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

from llm_token_saver.store import Store


@dataclass(frozen=True, slots=True)
class CostSplit:
    """USD by token type."""

    input_uncached: float
    cache_read: float
    cache_write_5m: float
    cache_write_1h: float
    output: float

    @property
    def total(self) -> float:
        return (
            self.input_uncached
            + self.cache_read
            + self.cache_write_5m
            + self.cache_write_1h
            + self.output
        )


@dataclass(frozen=True, slots=True)
class TokenSplit:
    input_uncached: int
    cache_read: int
    cache_write_5m: int
    cache_write_1h: int
    output: int
    thinking: int

    @property
    def prompt(self) -> int:
        return self.input_uncached + self.cache_read + self.cache_write_5m + self.cache_write_1h


@dataclass(frozen=True, slots=True)
class Group:
    """One row of a breakdown."""

    key: str
    calls: int
    cost: float


@dataclass(frozen=True, slots=True)
class DayRow:
    day: date
    calls: int
    cost: float


@dataclass(frozen=True, slots=True)
class SessionRow:
    session_id: str
    project: str
    started: datetime
    calls: int
    cost: float
    peak_prompt_tokens: int


@dataclass(frozen=True, slots=True)
class Stats:
    calls: int
    sessions: int
    threads: int
    subagent_threads: int
    first: datetime
    last: datetime
    cost: CostSplit
    tokens: TokenSplit
    prompt_p50: int
    prompt_p90: int
    prompt_p99: int
    by_model: list[Group]
    by_project: list[Group]
    by_effort: list[Group]
    by_day: list[DayRow]
    top_sessions: list[SessionRow]
    unpriced_models: list[Group]

    @property
    def cache_hit_rate(self) -> float:
        """Share of all prompt tokens that were served from the cache."""
        return self.tokens.cache_read / self.tokens.prompt if self.tokens.prompt else 0.0


def compute_stats(store: Store, *, top: int = 10, utc_offset_seconds: int = 0) -> Stats:
    """Summarize every call in `store`; days are local to `utc_offset_seconds`."""
    (
        calls,
        sessions,
        threads,
        subagent_threads,
        first,
        last,
        *split,
    ) = store.one(
        """
        SELECT count(*), count(DISTINCT session_id), count(DISTINCT thread_id),
               count(DISTINCT thread_id) FILTER (is_sidechain), min(ts), max(ts),
               sum(cost_input), sum(cost_cache_read), sum(cost_cache_write_5m),
               sum(cost_cache_write_1h), sum(cost_output),
               sum(input_uncached), sum(cache_read), sum(cache_write_5m),
               sum(cache_write_1h), sum(output), sum(thinking)
        FROM calls
        """
    )
    if not calls:
        raise ValueError("no calls to summarize")
    p50, p90, p99 = store.one(
        "SELECT quantile_disc(prompt_tokens, [0.5, 0.9, 0.99]) FROM priced_calls"
    )[0]
    return Stats(
        calls=calls,
        sessions=sessions,
        threads=threads,
        subagent_threads=subagent_threads,
        first=first,
        last=last,
        cost=CostSplit(*(float(value or 0.0) for value in split[:5])),
        tokens=TokenSplit(*(int(value or 0) for value in split[5:])),
        prompt_p50=int(p50),
        prompt_p90=int(p90),
        prompt_p99=int(p99),
        by_model=_groups(store, "model", top),
        by_project=_groups(store, "project", top),
        by_effort=_groups(store, "coalesce(nullif(effort, ''), 'unset')", top),
        by_day=[
            DayRow(day=day, calls=n, cost=cost)
            for day, n, cost in store.rows(
                """
                SELECT CAST(ts + to_seconds(?) AS DATE) AS day, count(*), sum(cost_total)
                FROM priced_calls GROUP BY day ORDER BY day
                """,
                [utc_offset_seconds],
            )
        ],
        top_sessions=[
            SessionRow(*row)
            for row in store.rows(
                """
                SELECT session_id, any_value(project), min(ts), count(*), sum(cost_total),
                       max(prompt_tokens)
                FROM priced_calls GROUP BY session_id ORDER BY 5 DESC LIMIT ?
                """,
                [top],
            )
        ],
        unpriced_models=[
            Group(key=model, calls=n, cost=0.0)
            for model, n in store.rows(
                "SELECT model, count(*) FROM calls WHERE NOT priced GROUP BY model ORDER BY 2 DESC"
            )
        ],
    )


def _groups(store: Store, expression: str, top: int) -> list[Group]:
    return [
        Group(key=str(key), calls=n, cost=cost)
        for key, n, cost in store.rows(
            f"SELECT {expression} AS key, count(*), sum(cost_total) FROM priced_calls "
            "GROUP BY key ORDER BY 3 DESC, 2 DESC LIMIT ?",
            [top],
        )
    ]
