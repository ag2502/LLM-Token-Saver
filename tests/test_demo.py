from pathlib import Path

import pytest

from llm_token_saver import demo
from llm_token_saver.ingest import otel
from llm_token_saver.pricing import PriceTable


@pytest.fixture(scope="module")
def traces() -> demo.DemoTraces:
    return demo.generate(days=2)


def test_generation_is_deterministic(traces: demo.DemoTraces) -> None:
    again = demo.generate(days=2)
    assert again.requests == traces.requests
    assert again.planted == traces.planted


def test_every_scenario_but_the_control_plants_a_saving(traces: demo.DemoTraces) -> None:
    planted = {p.service: p for p in traces.planted}
    assert set(planted) == set(demo.SCENARIOS) - {"agent-loop"}
    assert all(p.savings_usd > 0 for p in planted.values())


def test_traces_round_trip_through_the_importer(traces: demo.DemoTraces, tmp_path: Path) -> None:
    dataset = otel.load([traces.write(tmp_path / "demo.jsonl")])
    services = {call.project for call in dataset.calls}
    assert services == set(demo.SCENARIOS)
    prices = PriceTable.bundled()
    assert all(prices.cost(call) is not None for call in dataset.calls)
    support = [c for c in dataset.calls if c.project == "support-bot"]
    assert all(c.cache_write_5m == 2_509 and c.cache_read == 0 for c in support)
    assert all(c.prompt is not None and c.prompt[0].kind == "system" for c in support)
    faq = [c for c in dataset.calls if c.project == "faq-bot"]
    assert all(c.cache_write_1h > 0 and c.cache_write_5m == 0 for c in faq)


def test_synthetic_text_is_four_characters_per_token(
    traces: demo.DemoTraces, tmp_path: Path
) -> None:
    dataset = otel.load([traces.write(tmp_path / "demo.jsonl")])
    for call in dataset.calls:
        if call.prompt and call.project == "docs-qa":
            chars = sum(part.chars for part in call.prompt)
            assert chars == call.prompt_tokens * demo.CHARS_PER_TOKEN
