"""Every detector must recover the saving planted in the demo traces, within 10%."""

import pytest

from llm_token_saver import demo
from llm_token_saver.detect import Finding
from llm_token_saver.detect.prompt import duplicate_chunk_finding
from llm_token_saver.detect.prompt_cache import prompt_cache_findings
from llm_token_saver.detect.routing import routable_calls_finding
from llm_token_saver.detect.traffic import batchable_finding, duplicate_request_finding
from llm_token_saver.detect.ttl import ttl_findings
from llm_token_saver.estimate import TokenEstimator
from llm_token_saver.ingest import otel
from llm_token_saver.pricing import PriceTable
from llm_token_saver.schema import Dataset

TOLERANCE = 0.10


@pytest.fixture(scope="module")
def traces() -> demo.DemoTraces:
    return demo.generate(days=2)


@pytest.fixture(scope="module")
def dataset(traces: demo.DemoTraces, tmp_path_factory: pytest.TempPathFactory) -> Dataset:
    return otel.load([traces.write(tmp_path_factory.mktemp("demo") / "traces.jsonl")])


def planted(traces: demo.DemoTraces, detector: str) -> float:
    [match] = [p for p in traces.planted if p.detector == detector]
    return match.savings_usd


def only_service(finding: Finding, service: str) -> float:
    return sum(e.cost_usd or 0.0 for e in finding.evidence if e.label == service)


def test_volatile_system_prompt(traces: demo.DemoTraces, dataset: Dataset) -> None:
    estimator = TokenEstimator.fit(dataset.threads)
    [finding] = prompt_cache_findings(dataset, PriceTable.bundled(), estimator)
    assert finding.title.startswith("A changing value")
    assert only_service(finding, "support-bot") == pytest.approx(
        planted(traces, "prompt_cache"), rel=TOLERANCE
    )


def test_cache_lifetime_choices(traces: demo.DemoTraces, dataset: Dataset) -> None:
    findings = {f.detector: f for f in ttl_findings(dataset, PriceTable.bundled())}
    assert set(findings) == {"cache_ttl_5m", "cache_ttl_1h"}
    assert only_service(findings["cache_ttl_5m"], "faq-bot") == pytest.approx(
        planted(traces, "cache_ttl_5m"), rel=TOLERANCE
    )
    assert only_service(findings["cache_ttl_1h"], "weekly-reports") == pytest.approx(
        planted(traces, "cache_ttl_1h"), rel=TOLERANCE
    )


def test_repeated_requests(traces: demo.DemoTraces, dataset: Dataset) -> None:
    finding = duplicate_request_finding(dataset, PriceTable.bundled())
    assert finding is not None
    assert [e.label for e in finding.evidence] == ["faq-widget"]
    assert finding.savings_usd == pytest.approx(
        planted(traces, "duplicate_requests"), rel=TOLERANCE
    )


def test_batchable_scheduled_job(traces: demo.DemoTraces, dataset: Dataset) -> None:
    finding = batchable_finding(dataset, PriceTable.bundled())
    assert finding is not None
    assert [e.label for e in finding.evidence] == ["nightly-digest"]
    assert finding.savings_usd == pytest.approx(planted(traces, "batchable"), rel=TOLERANCE)


def test_duplicated_retrieval_chunks(traces: demo.DemoTraces, dataset: Dataset) -> None:
    estimator = TokenEstimator.fit(dataset.threads)
    finding = duplicate_chunk_finding(dataset, PriceTable.bundled(), estimator)
    assert finding is not None
    assert [e.label for e in finding.evidence] == ["docs-qa"]
    assert finding.savings_usd == pytest.approx(planted(traces, "duplicate_chunks"), rel=TOLERANCE)


def test_short_answers_from_a_frontier_model(traces: demo.DemoTraces, dataset: Dataset) -> None:
    finding = routable_calls_finding(dataset, PriceTable.bundled())
    assert finding is not None and finding.ceiling
    assert [e.label for e in finding.evidence] == [
        "ticket-router: claude-opus-5-5 → claude-haiku-5-5"
    ]
    assert finding.savings_usd == pytest.approx(planted(traces, "routable_calls"), rel=TOLERANCE)
