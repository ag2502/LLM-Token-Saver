import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from llm_token_saver import demo
from llm_token_saver.analysis import build_report
from llm_token_saver.cli import app
from llm_token_saver.detect import FindingKind
from llm_token_saver.ingest import otel
from llm_token_saver.pricing import PriceTable

runner = CliRunner()


def report_for(traces: demo.DemoTraces, path: Path):
    dataset = otel.load([traces.write(path)])
    return build_report(dataset, PriceTable.bundled(), title="t", group_noun="service")


def test_the_clean_control_has_nothing_to_fix(tmp_path: Path) -> None:
    report = report_for(demo.generate(days=2, services=["agent-loop"]), tmp_path / "c.jsonl")
    savings = [f.title for f in report.findings if f.kind is not FindingKind.INFO and f.savings_usd]
    assert savings == []


def test_the_full_demo_surfaces_every_planted_problem(tmp_path: Path) -> None:
    traces = demo.generate(days=2)
    report = report_for(traces, tmp_path / "d.jsonl")
    found = {f.detector for f in report.findings if f.savings_usd > 0}
    assert {p.detector for p in traces.planted} <= found


def test_analyze_command_writes_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TOKENSAVER_HOME", str(tmp_path / "home"))
    path = demo.generate(days=1).write(tmp_path / "t.jsonl")
    out = tmp_path / "report.json"
    result = runner.invoke(app, ["analyze", str(path), "--json", str(out)])
    assert result.exit_code == 0, result.output
    assert "conversations" in result.output
    assert "On a subscription" not in result.output
    detectors = {finding["detector"] for finding in json.loads(out.read_text())["findings"]}
    assert {"prompt_cache", "batchable", "cache_ttl_1h"} <= detectors


def test_demo_command_writes_and_analyzes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TOKENSAVER_HOME", str(tmp_path / "home"))
    html = tmp_path / "report.html"
    result = runner.invoke(
        app,
        ["demo", "--out", str(tmp_path / "d.jsonl"), "--days", "1", "--html", str(html)],
    )
    assert result.exit_code == 0, result.output
    assert "Planted problems the analysis should find" in " ".join(result.output.split())
    assert "By service" in html.read_text()


def test_analyze_with_no_chat_spans(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TOKENSAVER_HOME", str(tmp_path / "home"))
    empty = tmp_path / "empty.json"
    empty.write_text('{"resourceSpans": []}')
    result = runner.invoke(app, ["analyze", str(empty)])
    assert result.exit_code == 0
    assert "No GenAI chat calls" in result.output
