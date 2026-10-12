"""Every detector must recover the saving planted in the demo traces, within 10%."""

import pytest

from llm_token_saver import demo
from llm_token_saver.detect import Finding
from llm_token_saver.detect.prompt_cache import prompt_cache_findings
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
