from datetime import UTC, datetime, timedelta

import pytest

from llm_token_saver.detect.prompt_cache import (
    common_prefix_length,
    common_suffix_length,
    prompt_cache_findings,
    prompt_change,
)
from llm_token_saver.estimate import TokenEstimator
from llm_token_saver.pricing import PriceTable
from llm_token_saver.schema import Call, Dataset, PromptPart, Thread

T0 = datetime(2026, 10, 5, 9, tzinfo=UTC)
PRICES = PriceTable.bundled()
FOUR_CHARS = TokenEstimator({"claude-opus-5-5": 4.0})


def parts(system: str, user: str = "question") -> list[PromptPart]:
    built = [PromptPart("system", "system", system), PromptPart("message", "user", user)]
    for part in built:
        part.content_hash = f"{part.kind}:{part.text}"
    return built


def call(n: int, prompt: list[PromptPart], seconds: float, **tokens: int) -> Call:
    return Call(
        call_id=f"c{n}",
        source="otel",
        session_id=f"c{n}",
        thread_id=f"c{n}",
        seq=0,
        timestamp=T0 + timedelta(seconds=seconds),
        model="claude-opus-5-5",
        project="svc",
        prompt=prompt,
        **tokens,  # type: ignore[arg-type]
    )


def dataset(*calls: Call) -> Dataset:
    return Dataset(threads=[Thread(c.thread_id, c.session_id, "otel", calls=[c]) for c in calls])


@pytest.mark.parametrize(
    ("a", "b", "prefix", "suffix"),
    [("abcdef", "abcxef", 3, 2), ("same", "same", 4, 4), ("", "x", 0, 0), ("ab", "abc", 2, 0)],
)
def test_common_prefix_and_suffix(a: str, b: str, prefix: int, suffix: int) -> None:
    assert common_prefix_length(a, b) == prefix
    assert common_suffix_length(a, b) == suffix


def test_a_timestamp_in_the_system_prompt_is_a_small_window() -> None:
    template = "x" * 4_000
    change = prompt_change(
        parts(f"Now: 10:04:13\n{template}"), parts(f"Now: 10:07:41\n{template}", "other")
    )
    assert change is not None
    assert (change.kind, change.offset, change.window) == ("system", 9, 4)
    assert change.stable_chars == len(f"Now: 10:04:13\n{template}") - 4
    assert change.context == "Now: 10:0"


def test_different_user_messages_are_not_a_volatile_prefix() -> None:
    assert prompt_change(parts("same"), parts("same", "a different question")) is None
    assert prompt_change(parts("a" * 500), parts("b" * 500)) is None


def test_volatile_prefix_saving_uses_read_price_within_the_cache_lifetime() -> None:
    template = "x" * 8_000  # 2,000 tokens at 4 characters each
    calls = [
        call(
            i,
            parts(f"t={i:04d}\n{template}"),
            seconds=i * 60,
            input_uncached=10,
            cache_write_5m=2_002,
        )
        for i in range(3)
    ]
    [finding] = prompt_cache_findings(dataset(*calls), PRICES, FOUR_CHARS)
    stable_tokens = (len(f"t=0000\n{template}") - 1) / 4
    assert finding.savings_usd == pytest.approx(2 * stable_tokens * (5.0 - 0.20) / 1e6)


def test_a_shared_prefix_that_is_never_cached() -> None:
    shared = "y" * 8_000
    calls = [
        call(i, parts(shared, f"q{i}"), seconds=i * 60, input_uncached=2_010) for i in range(3)
    ]
    [finding] = prompt_cache_findings(dataset(*calls), PRICES, FOUR_CHARS)
    assert finding.title == "A shared prompt prefix is never cached"
    tokens = (8_000 + 1) / 4  # the system prompt plus the "q" both requests start with
    assert finding.savings_usd == pytest.approx(2 * tokens * (4.0 - 0.20) / 1e6)


def test_prefixes_below_the_cacheable_minimum_are_left_alone() -> None:
    calls = [
        call(i, parts("z" * 800, f"q{i}"), seconds=i * 60, input_uncached=210) for i in range(3)
    ]
    assert prompt_cache_findings(dataset(*calls), PRICES, FOUR_CHARS) == []
