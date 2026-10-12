"""Synthetic LLM-app traces with planted waste, for trying the tool and testing the detectors.

Every scenario is one service. Each planted problem comes with its ground-truth saving,
computed here from the scenario's own parameters (not by running the detectors), so tests can
check that each detector recovers it. `agent-loop` is a clean control with nothing to find.

Synthetic text is built from 4-character pseudo-words, so a span's character counts and its
reported token counts agree exactly (4 characters per token).
"""

from __future__ import annotations

import json
import random
import string
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from llm_token_saver import otlp
from llm_token_saver.ingest.otel import CACHE_WRITE_1H
from llm_token_saver.pricing import PriceTable
from llm_token_saver.schema import Call

CHARS_PER_TOKEN = 4
START = datetime(2026, 10, 5, tzinfo=UTC)  # a Monday
OPUS, SONNET, HAIKU = "claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5"


@dataclass(frozen=True, slots=True)
class Planted:
    """One deliberately wasteful pattern and what fixing it would save over the demo period."""

    detector: str
    service: str
    savings_usd: float
    description: str


@dataclass(slots=True)
class DemoTraces:
    requests: list[dict[str, Any]] = field(default_factory=list)
    planted: list[Planted] = field(default_factory=list)

    def write(self, path: Path) -> Path:
        """Write one OTLP/JSON export request per line."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            for request in self.requests:
                fh.write(json.dumps(request, separators=(",", ":")) + "\n")
        return path


class _Scenario:
    """Shared helpers: deterministic text, span ids and per-token prices."""

    def __init__(self, service: str, seed: int, days: int) -> None:
        self.service = service
        self.rng = random.Random(f"{seed}:{service}")
        self.days = days
        self.prices = PriceTable.bundled()
        self.spans: list[dict[str, Any]] = []
        self._ids = 0

    def text(self, tokens: int) -> str:
        letters = string.ascii_lowercase
        choice = self.rng.choice
        return "".join(
            choice(letters) + choice(letters) + choice(letters) + " " for _ in range(tokens)
        )

    def arrivals(self, day: int, start_hour: float, end_hour: float, gap: tuple[float, float]):
        """Request start times on `day` between the hours, with random gaps in seconds."""
        at = START + timedelta(days=day, hours=start_hour)
        end = START + timedelta(days=day, hours=end_hour)
        while at < end:
            yield at
            at += timedelta(seconds=self.rng.uniform(*gap))

    def emit(
        self,
        at: datetime,
        model: str,
        *,
        uncached: int,
        read: int = 0,
        write: int = 0,
        write_1h: int = 0,
        output: int,
        conversation: str = "",
        system: str | None = None,
        messages: list[dict[str, Any]] | None = None,
    ) -> Call:
        """Record one chat span and return the equivalent call for ground-truth pricing."""
        self._ids += 1
        attrs: dict[str, Any] = {
            "gen_ai.operation.name": "chat",
            "gen_ai.provider.name": "anthropic",
            "gen_ai.request.model": model,
            "gen_ai.response.model": model,
            "gen_ai.response.id": f"{self.service}-{self._ids}",
            "gen_ai.conversation.id": conversation or None,
            "gen_ai.usage.input_tokens": uncached + read + write + write_1h,
            "gen_ai.usage.cache_read.input_tokens": read,
            "gen_ai.usage.cache_write.input_tokens": write + write_1h,
            CACHE_WRITE_1H: write_1h or None,
            "gen_ai.usage.output_tokens": output,
            "gen_ai.request.max_tokens": 1024,
        }
        if system is not None:
            attrs["gen_ai.system_instructions"] = json.dumps([{"type": "text", "content": system}])
        if messages is not None:
            attrs["gen_ai.input.messages"] = json.dumps(messages)
        self.spans.append(
            otlp.span(
                trace_id=f"{self.service}-trace-{conversation or self._ids}",
                span_id=f"{self._ids:016x}",
                start=at,
                duration=timedelta(seconds=2),
                attrs=attrs,
                name=f"chat {model}",
            )
        )
        return Call(
            call_id=str(self._ids),
            source="demo",
            session_id=conversation,
            thread_id=conversation,
            seq=self._ids,
            timestamp=at,
            model=model,
            input_uncached=uncached,
            cache_read=read,
            cache_write_5m=write,
            cache_write_1h=write_1h,
            output=output,
        )

    def cost(self, call: Call) -> float:
        cost = self.prices.cost(call)
        assert cost is not None
        return cost.total

    def rate(self, model: str, kind: str) -> float:
        rates = self.prices.rates(model)
        assert rates is not None
        return getattr(rates, kind) / 1e6


def _user(text: str) -> dict[str, Any]:
    return {"role": "user", "parts": [{"type": "text", "content": text}]}


def support_bot(seed: int, days: int) -> tuple[_Scenario, list[Planted]]:
    """A timestamp at the top of a cached system prompt: the cache never hits."""
    s = _Scenario("support-bot", seed, days)
    template, saving = s.text(2_500), 0.0
    for day in range(days):
        previous: datetime | None = None
        for at in s.arrivals(day, 9, 12, (120, 240)):
            system = f"Current time: {at:%Y-%m-%dT%H:%M:%SZ} \n" + template  # 9 + 2,500 tokens
            s.emit(at, OPUS, uncached=150, write=2_509, output=200, system=system,
                   messages=[_user(s.text(150))])  # fmt: skip
            if previous is not None and (at - previous).total_seconds() <= 300:
                saving += 2_500 * (s.rate(OPUS, "cache_write_5m") - s.rate(OPUS, "cache_read"))
            previous = at
    return s, [
        Planted("prompt_cache", s.service, saving, "timestamp at the top of the system prompt")
    ]


def docs_qa(seed: int, days: int) -> tuple[_Scenario, list[Planted]]:
    """Retrieval returns overlapping chunks, so two of six chunks are sent twice."""
    s = _Scenario("docs-qa", seed, days)
    system, saving = s.text(600), 0.0
    for day in range(days):
        first = True
        for at in s.arrivals(day, 10, 13, (120, 240)):
            chunks = [s.text(200)[:-2] + "\n\n" for _ in range(4)]  # 200 tokens each
            question = s.text(50)
            body = "".join(chunks + chunks[:2]) + question  # 6 x 200 + 50 tokens
            s.emit(at, SONNET, uncached=1_250, read=0 if first else 600,
                   write=600 if first else 0, output=250, system=system,
                   messages=[_user(body)])  # fmt: skip
            saving += 2 * 200 * s.rate(SONNET, "input")
            first = False
    return s, [Planted("duplicate_chunks", s.service, saving, "duplicated retrieval chunks")]


def ticket_router(seed: int, days: int) -> tuple[_Scenario, list[Planted]]:
    """One-word ticket labels from the most capable model."""
    s = _Scenario("ticket-router", seed, days)
    saving = 0.0
    for day in range(days):
        first = True
        for at in s.arrivals(day, 8, 20, (180, 360)):
            call = s.emit(at, OPUS, uncached=200, read=0 if first else 1_500,
                          write=1_500 if first else 0, output=3)  # fmt: skip
            cheap = Call(**{**_fields(call), "model": HAIKU})
            saving += s.cost(call) - s.cost(cheap)
            first = False
    return s, [Planted("routable_calls", s.service, saving, "Opus labeling tickets")]


def nightly_digest(seed: int, days: int) -> tuple[_Scenario, list[Planted]]:
    """A cron job that fires 120 independent summaries at 02:00 through the synchronous API."""
    s = _Scenario("nightly-digest", seed, days)
    total = 0.0
    for day in range(days):
        at = START + timedelta(days=day, hours=2)
        for index in range(120):
            call = s.emit(at, SONNET, uncached=3_000, read=0 if index == 0 else 800,
                          write=800 if index == 0 else 0, output=400)  # fmt: skip
            total += s.cost(call)
            at += timedelta(seconds=s.rng.uniform(2, 3))
    return s, [Planted("batchable", s.service, total / 2, "nightly job not using the Batch API")]


def faq_bot(seed: int, days: int) -> tuple[_Scenario, list[Planted]]:
    """1-hour cache writes on conversations whose turns are never more than 3 minutes apart."""
    s = _Scenario("faq-bot", seed, days)
    saving = 0.0
    premium = s.rate(OPUS, "cache_write_1h") - s.rate(OPUS, "cache_write_5m")
    for day in range(days):
        at = START + timedelta(days=day, hours=9)
        conversation = 0
        while at < START + timedelta(days=day, hours=14):
            conversation += 1
            prompt = 4_000 + 1_200
            for turn in range(8):
                written = prompt if turn == 0 else 1_500
                s.emit(at, OPUS, uncached=0, read=prompt - written, write_1h=written,
                       output=300, conversation=f"faq-{day}-{conversation}")  # fmt: skip
                saving += written * premium
                prompt += 1_500
                at += timedelta(seconds=s.rng.uniform(60, 180))
            at += timedelta(seconds=s.rng.uniform(60, 180))
    return s, [Planted("cache_ttl_5m", s.service, saving, "1-hour cache on steady traffic")]


def faq_widget(seed: int, days: int) -> tuple[_Scenario, list[Planted]]:
    """Users click the same five suggested questions; every click is a fresh request."""
    s = _Scenario("faq-widget", seed, days)
    system = s.text(300)
    questions = [s.text(40) for _ in range(5)]
    seen: set[int] = set()
    saving = 0.0
    for day in range(days):
        for at in s.arrivals(day, 8, 20, (300, 900)):
            pick = s.rng.randrange(len(questions))
            call = s.emit(at, SONNET, uncached=340, output=120, system=system,
                          messages=[_user(questions[pick])])  # fmt: skip
            if pick in seen:
                saving += s.cost(call)
            seen.add(pick)
    return s, [Planted("duplicate_requests", s.service, saving, "identical requests repeated")]


def weekly_reports(seed: int, days: int) -> tuple[_Scenario, list[Planted]]:
    """An analyst's day-long conversation over a large data dictionary, with 10-30 minute
    pauses: every call misses the 5-minute cache."""
    s = _Scenario("weekly-reports", seed, days)
    saving = 0.0
    write_5m, write_1h = s.rate(OPUS, "cache_write_5m"), s.rate(OPUS, "cache_write_1h")
    read = s.rate(OPUS, "cache_read")
    for day in range(days):
        prompt = 20_000
        for turn, at in enumerate(s.arrivals(day, 9, 17, (600, 1_800))):
            s.emit(at, OPUS, uncached=0, write=prompt, output=800,
                   conversation=f"reports-{day}")  # fmt: skip
            if turn == 0:
                saving += prompt * (write_5m - write_1h)
            else:
                previous = prompt - 1_000
                saving += prompt * write_5m - (previous * read + 1_000 * write_1h)
            prompt += 1_000
    return s, [Planted("cache_ttl_1h", s.service, saving, "5-minute cache on 10-30 minute gaps")]


def agent_loop(seed: int, days: int) -> tuple[_Scenario, list[Planted]]:
    """Clean control: a well-cached agent loop with nothing to fix."""
    s = _Scenario("agent-loop", seed, days)
    for day in range(days):
        at = START + timedelta(days=day, hours=9)
        conversation = 0
        while at < START + timedelta(days=day, hours=14):
            conversation += 1
            prompt = 3_100
            for turn in range(15):
                written = prompt if turn == 0 else 900
                s.emit(at, SONNET, uncached=0, read=prompt - written, write=written,
                       output=300, conversation=f"agent-{day}-{conversation}")  # fmt: skip
                prompt += 900
                at += timedelta(seconds=s.rng.uniform(10, 40))
            at += timedelta(minutes=s.rng.uniform(15, 25))
    return s, []


SCENARIOS: dict[str, Callable[[int, int], tuple[_Scenario, list[Planted]]]] = {
    "support-bot": support_bot,
    "docs-qa": docs_qa,
    "ticket-router": ticket_router,
    "nightly-digest": nightly_digest,
    "faq-bot": faq_bot,
    "faq-widget": faq_widget,
    "weekly-reports": weekly_reports,
    "agent-loop": agent_loop,
}


def generate(*, seed: int = 7, days: int = 7, services: list[str] | None = None) -> DemoTraces:
    traces = DemoTraces()
    for name in services or list(SCENARIOS):
        scenario, planted = SCENARIOS[name](seed, days)
        traces.requests.append(otlp.export_request(scenario.spans, scenario.service))
        traces.planted += planted
    return traces


def _fields(call: Call) -> dict[str, Any]:
    return {name: getattr(call, name) for name in call.__dataclass_fields__}
