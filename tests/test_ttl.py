from datetime import UTC, datetime, timedelta

import pytest

from llm_token_saver.detect.ttl import FIVE_MINUTES, ONE_HOUR, simulate, ttl_findings
from llm_token_saver.pricing import PriceTable
from llm_token_saver.schema import Call, Dataset, Thread

T0 = datetime(2026, 10, 5, 9, tzinfo=UTC)
PRICES = PriceTable.bundled()
W5, W1H, READ = 5.0, 8.0, 0.20  # Claude Opus 5.5, USD per million tokens


def conversation(prompts: list[int], gaps: list[float], *, one_hour: bool) -> list[Call]:
    calls, at, previous = [], T0, 0
    for index, prompt in enumerate(prompts):
        written = prompt - previous if index else prompt
        calls.append(
            Call(
                call_id=f"c{index}",
                source="otel",
                session_id="s",
                thread_id="s",
                seq=index,
                timestamp=at,
                model="claude-opus-5-5",
                cache_read=prompt - written,
                cache_write_1h=written if one_hour else 0,
                cache_write_5m=0 if one_hour else written,
                project="svc",
            )
        )
        previous = prompt
        at += timedelta(seconds=gaps[index] if index < len(gaps) else 0)
    return calls


def test_simulate_reads_within_the_lifetime_and_rewrites_after_it() -> None:
    calls = conversation([10_000, 11_000, 12_000], gaps=[60, 600], one_hour=False)
    five = simulate(calls, PRICES, FIVE_MINUTES)
    assert five == pytest.approx((10_000 * W5 + 10_000 * READ + 1_000 * W5 + 12_000 * W5) / 1e6)
    hour = simulate(calls, PRICES, ONE_HOUR)
    assert hour == pytest.approx(
        (10_000 * W1H + 10_000 * READ + 1_000 * W1H + 11_000 * READ + 1_000 * W1H) / 1e6
    )


def test_steady_traffic_on_the_one_hour_cache_should_use_five_minutes() -> None:
    calls = conversation([5_000, 6_000, 7_000], gaps=[60, 60], one_hour=True)
    [finding] = ttl_findings(
        Dataset([Thread("s", "s", "otel", project="svc", calls=calls)]), PRICES
    )
    assert finding.detector == "cache_ttl_5m"
    assert finding.savings_usd == pytest.approx(7_000 * (W1H - W5) / 1e6)
    assert "cache_control" in finding.fix and "promptCacheTtl" not in finding.fix


def test_no_finding_when_the_current_lifetime_is_already_cheapest() -> None:
    calls = conversation([5_000, 6_000, 7_000], gaps=[60, 60], one_hour=False)
    assert (
        ttl_findings(Dataset([Thread("s", "s", "otel", project="svc", calls=calls)]), PRICES) == []
    )


def test_claude_code_fix_names_the_settings() -> None:
    calls = conversation([50_000, 51_000, 52_000], gaps=[1_200, 1_200], one_hour=False)
    thread = Thread("s", "s", "claude_code", project="~/p", calls=calls)
    [finding] = ttl_findings(Dataset([thread]), PRICES)
    assert finding.detector == "cache_ttl_1h"
    assert '"promptCacheTtl": "1h"' in finding.fix
