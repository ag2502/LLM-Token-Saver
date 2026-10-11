from datetime import UTC, date, datetime

import pytest

from llm_token_saver.pricing import PriceTable, normalize_model_id
from llm_token_saver.schema import Call

TABLE = PriceTable.bundled()


def make_call(model: str, **tokens: int | str | bool) -> Call:
    return Call(
        call_id="c1",
        source="test",
        session_id="s1",
        thread_id="t1",
        seq=0,
        timestamp=datetime(2026, 10, 1, tzinfo=UTC),
        model=model,
        **tokens,  # type: ignore[arg-type]
    )


def test_bundled_table_metadata() -> None:
    assert TABLE.source.startswith("https://platform.claude.com/")
    assert TABLE.retrieved == date(2026, 10, 11)
    assert {"claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5"} <= set(TABLE.models)


@pytest.mark.parametrize("model_id", sorted(PriceTable.bundled().models))
def test_table_matches_published_multipliers(model_id: str) -> None:
    """Guards against typos: writes are 1.25x / 2x input; reads 0.1x, 0.05x or 0.025x."""
    rates = TABLE.models[model_id].standard
    assert rates.cache_write_5m == pytest.approx(1.25 * rates.input)
    assert rates.cache_write_1h == pytest.approx(2 * rates.input)
    read_multiplier = round(rates.cache_read / rates.input, 4)
    assert read_multiplier in {0.1, 0.05, 0.025}


def test_opus_5_5_cost_by_token_type() -> None:
    call = make_call(
        "claude-opus-5-5",
        input_uncached=1_000,
        cache_read=1_000_000,
        cache_write_5m=100_000,
        cache_write_1h=200_000,
        output=10_000,
    )
    cost = TABLE.cost(call)
    assert cost is not None
    assert cost.input_uncached == pytest.approx(0.004)
    assert cost.cache_read == pytest.approx(0.20)
    assert cost.cache_write_5m == pytest.approx(0.50)
    assert cost.cache_write_1h == pytest.approx(1.60)
    assert cost.output == pytest.approx(0.20)
    assert cost.total == pytest.approx(2.504)


def test_sonnet_5_5_cache_reads_are_half_of_opus_5_5() -> None:
    sonnet = TABLE.rates("claude-sonnet-5-5")
    opus = TABLE.rates("claude-opus-5-5")
    assert sonnet is not None and opus is not None
    assert sonnet.cache_read == pytest.approx(0.10)
    assert opus.cache_read == pytest.approx(0.20)


def test_haiku_5_5_long_prompts_pay_higher_prices_for_the_whole_request() -> None:
    at_threshold = TABLE.rates("claude-haiku-5-5", prompt_tokens=100_000)
    above = TABLE.rates("claude-haiku-5-5", prompt_tokens=100_001)
    assert at_threshold is not None and above is not None
    assert at_threshold.input == pytest.approx(0.10)
    assert above.input == pytest.approx(0.50)
    assert above.cache_read == pytest.approx(0.05)
    # The threshold counts cache reads too: a mostly cached long prompt still pays the tier.
    call = make_call("claude-haiku-5-5", input_uncached=10, cache_read=150_000, output=100)
    cost = TABLE.cost(call)
    assert cost is not None
    assert cost.cache_read == pytest.approx(150_000 * 0.05 / 1e6)


def test_fast_mode_doubles_supported_opus_models_only() -> None:
    fast = TABLE.rates("claude-opus-5-5", speed="fast")
    assert fast is not None
    assert (fast.input, fast.output) == pytest.approx((8.0, 40.0))
    assert fast.cache_read == pytest.approx(0.40)
    standard_billed = TABLE.rates("claude-opus-4-6", speed="fast")
    assert standard_billed is not None
    assert standard_billed.input == pytest.approx(5.0)


def test_batch_and_us_geography_stack() -> None:
    rates = TABLE.rates("claude-opus-5-5", batch=True, inference_geo="us")
    assert rates is not None
    assert rates.input == pytest.approx(4.0 * 0.5 * 1.1)
    assert rates.cache_read == pytest.approx(0.20 * 0.5 * 1.1)


def test_us_geography_only_applies_from_claude_4_6() -> None:
    older = TABLE.rates("claude-opus-4-5", inference_geo="us")
    assert older is not None
    assert older.input == pytest.approx(5.0)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("claude-opus-5-5", "claude-opus-5-5"),
        ("us.anthropic.claude-sonnet-4-5-20250929-v1:0", "claude-sonnet-4-5"),
        ("claude-opus-4-5@20251101", "claude-opus-4-5"),
        ("anthropic/claude-opus-5-5", "claude-opus-5-5"),
        ("claude-3-5-haiku-20241022", "claude-3-5-haiku"),
        ("claude-opus-4-6[1m]", "claude-opus-4-6"),
        ("Claude-Haiku-4-5-20251001", "claude-haiku-4-5"),
    ],
)
def test_normalize_model_id(raw: str, expected: str) -> None:
    assert normalize_model_id(raw) == expected


def test_unknown_models_are_unpriced_not_guessed() -> None:
    assert TABLE.lookup("claude-opus-4-9") is None
    assert TABLE.cost(make_call("<synthetic>", output=10)) is None
