"""Detector: content repeated inside a single prompt, such as overlapping retrieval chunks."""

from __future__ import annotations

from collections import Counter, defaultdict

from llm_token_saver.detect.base import Evidence, Finding, FindingKind, fresh_rate
from llm_token_saver.estimate import TokenEstimator
from llm_token_saver.pricing import PriceTable
from llm_token_saver.pricing.engine import PER_MILLION
from llm_token_saver.report.format import plural, tokens
from llm_token_saver.schema import Dataset

MIN_CHUNK_CHARS = 200
"""Paragraphs shorter than this (greetings, short instructions) may repeat for good reason."""


def duplicate_chunk_finding(
    dataset: Dataset, prices: PriceTable, estimator: TokenEstimator
) -> Finding | None:
    savings: dict[str, float] = defaultdict(float)
    repeated_tokens: dict[str, float] = defaultdict(float)
    requests: Counter[str] = Counter()
    for call in dataset.calls:
        rates = prices.rates_for(call)
        if not call.prompt or rates is None:
            continue
        counts: Counter[str] = Counter()
        for part in call.prompt:
            if part.kind == "message":
                counts.update(c for c in part.text.split("\n\n") if len(c) >= MIN_CHUNK_CHARS)
        repeated_chars = sum((n - 1) * len(chunk) for chunk, n in counts.items() if n > 1)
        if not repeated_chars:
            continue
        repeated = repeated_chars / estimator.ratio(call.model)
        savings[call.project] += repeated * fresh_rate(call, rates) / PER_MILLION
        repeated_tokens[call.project] += repeated
        requests[call.project] += 1
    total = sum(savings.values())
    if total <= 0:
        return None
    return Finding(
        detector="duplicate_chunks",
        title="The same passage is sent twice in one prompt",
        kind=FindingKind.FREE_WIN,
        summary=(
            f"{plural(sum(requests.values()), 'request')} contained a passage of "
            f"{MIN_CHUNK_CHARS}+ characters more than once in the same prompt, usually "
            "overlapping retrieval results. The model gains nothing from the second copy."
        ),
        fix=(
            "Deduplicate retrieved chunks before building the prompt: hash each chunk, drop "
            "repeats, and merge overlapping windows."
        ),
        savings_usd=total,
        evidence=[
            Evidence(
                service,
                saving,
                f"{plural(requests[service], 'request')}, "
                f"~{tokens(repeated_tokens[service] / requests[service])} repeated tokens each",
            )
            for service, saving in sorted(savings.items(), key=lambda kv: -kv[1])
        ],
    )
