import re
from html.parser import HTMLParser
from pathlib import Path

import pytest

from llm_token_saver.analysis import build_report
from llm_token_saver.config import Currency
from llm_token_saver.ingest import claude_code
from llm_token_saver.pricing import PriceTable
from llm_token_saver.report.html import column_path, inline_code, nice_ticks, render_html
from tests.test_report import noisy_session

VOID = {"meta", "br", "hr", "img", "input", "link", "path", "line", "rect", "circle"}


class TagBalance(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.stack: list[str] = []
        self.errors: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag not in VOID:
            self.stack.append(tag)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        pass

    def handle_endtag(self, tag: str) -> None:
        if tag in VOID:
            return
        if not self.stack or self.stack[-1] != tag:
            self.errors.append(f"unexpected </{tag}> with open {self.stack[-3:]}")
            return
        self.stack.pop()


@pytest.fixture
def html(tmp_path: Path) -> str:
    noisy_session(tmp_path / "p" / "s.jsonl")
    report = build_report(claude_code.load(tmp_path), PriceTable.bundled(), title="Claude Code")
    return render_html(report, Currency())


def test_report_is_well_formed_and_self_contained(html: str) -> None:
    parser = TagBalance()
    parser.feed(html)
    assert parser.errors == []
    assert parser.stack == []
    assert html.startswith("<!doctype html>")
    assert "<script src" not in html and "<link" not in html
    external = set(re.findall(r'(?:src|href)="(https?://[^"]+)"', html))
    assert external == {"https://platform.claude.com/docs/en/about-claude/pricing"}


def test_report_shows_findings_and_charts(html: str) -> None:
    assert "Long command output stays in context" in html
    assert "Where the money went" in html
    assert html.count('class="col"') == 1  # one day of data
    assert "<code>&#34;bashOutputMaxChars&#34;: 5000</code>" in html


def test_untrusted_labels_are_escaped(tmp_path: Path) -> None:
    path = noisy_session(tmp_path / "p" / "s.jsonl")
    path.write_text(path.read_text().replace("/work/app", "/work/<script>alert(1)</script>"))
    report = build_report(claude_code.load(tmp_path), PriceTable.bundled(), title="T")
    html = render_html(report, Currency())
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html


def test_inline_code_escapes_before_marking_up() -> None:
    assert str(inline_code("use `a<b` & `c`")) == "use <code>a&lt;b</code> &amp; <code>c</code>"


@pytest.mark.parametrize(
    ("maximum", "ticks"),
    [
        (0, [0, 1]),
        (51.3, [0, 20, 40, 60]),
        (7, [0, 2, 4, 6, 8]),
        (250, [0, 100, 200, 300]),
        (0.9, [0, 0.25, 0.5, 0.75, 1.0]),
    ],
)
def test_nice_ticks(maximum: float, ticks: list[float]) -> None:
    assert nice_ticks(maximum) == pytest.approx(ticks)


def test_column_path_rounds_only_the_top() -> None:
    assert column_path(10, 20, 8, 50) == (
        "M10.00,70.00V24.00Q10.00,20.00 14.00,20.00H14.00Q18.00,20.00 18.00,24.00V70.00Z"
    )
    short = column_path(0, 99, 20, 1)
    assert "V100.00Z" in short and "V100.00Q" in short


def test_axis_money_labels_are_compact() -> None:
    usd = Currency()
    assert [usd.format_axis(v) for v in (0, 20, 2.5, 1_500, 10_000)] == [
        "$0",
        "$20",
        "$2.5",
        "$1.5K",
        "$10K",
    ]
