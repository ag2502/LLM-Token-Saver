"""Token estimates for context blocks whose size transcripts do not record.

Constants were fitted on real Claude Code transcripts by regressing each call-to-call growth
in prompt tokens on the characters of content added in between (R² = 0.94): tool output and
user text ran at about 2.5 characters per token on Claude 4.7+ models, a screenshot added
about 1,740 tokens, and the harness added about 100 unlogged tokens per turn.
`TokenEstimator.fit` re-fits the characters-per-token ratio per model on the data at hand.
"""

from __future__ import annotations

import re
import statistics
from collections import defaultdict
from collections.abc import Iterable, Iterator

from llm_token_saver.pricing import normalize_model_id
from llm_token_saver.schema import Block, BlockKind, Call, Marker, MarkerKind, Thread
from llm_token_saver.timeline import timeline

IMAGE_TOKENS = 1_750
TURN_OVERHEAD_TOKENS = 100
NEW_TOKENIZER_CHARS_PER_TOKEN = 2.5
OLD_TOKENIZER_CHARS_PER_TOKEN = 3.3  # pre-4.7 models make ~30% fewer tokens from the same text
MIN_GAP_CHARS = 2_000
MIN_SAMPLES = 20
_RATIO_BOUNDS = (1.5, 6.0)
_VERSION = re.compile(r"claude-(?:(?:opus|sonnet|haiku|fable|mythos)-)?(\d+)(?:-(\d{1,2}))?")


def default_chars_per_token(model: str) -> float:
    """Claude 4.7 and later use a newer tokenizer that makes about 30% more tokens."""
    match = _VERSION.match(normalize_model_id(model))
    if not match:
        return OLD_TOKENIZER_CHARS_PER_TOKEN
    version = (int(match[1]), int(match[2] or 0))
    return NEW_TOKENIZER_CHARS_PER_TOKEN if version >= (4, 7) else OLD_TOKENIZER_CHARS_PER_TOKEN


class TokenEstimator:
    """Characters-per-token ratios by model, fitted from usage where there is enough data."""

    def __init__(self, chars_per_token: dict[str, float] | None = None) -> None:
        self.chars_per_token = chars_per_token or {}

    @classmethod
    def fit(cls, threads: Iterable[Thread]) -> TokenEstimator:
        samples: dict[str, list[float]] = defaultdict(list)
        for thread in threads:
            for model, chars, added in _text_gaps(thread):
                if chars >= MIN_GAP_CHARS and added > 0:
                    samples[model].append(chars / added)
        low, high = _RATIO_BOUNDS
        fitted = {
            model: min(max(statistics.median(ratios), low), high)
            for model, ratios in samples.items()
            if len(ratios) >= MIN_SAMPLES
        }
        return cls(fitted)

    def ratio(self, model: str) -> float:
        return self.chars_per_token.get(normalize_model_id(model)) or default_chars_per_token(model)

    def tokens(self, block: Block, model: str) -> int:
        if block.tokens is not None:
            return block.tokens
        if block.kind is BlockKind.IMAGE:
            return IMAGE_TOKENS
        return round(block.chars / self.ratio(model))


def _text_gaps(thread: Thread) -> Iterator[tuple[str, int, int]]:
    """(model, characters added, tokens added) between consecutive calls on one model.

    Gaps with images or a compaction are skipped: their token growth is not about text.
    """
    previous: Call | None = None
    chars = 0
    clean = True
    for event in timeline(thread):
        if isinstance(event, Call):
            if previous is not None and clean and previous.model == event.model:
                added = (
                    event.prompt_tokens
                    - previous.prompt_tokens
                    - previous.output
                    - TURN_OVERHEAD_TOKENS
                )
                yield normalize_model_id(event.model), chars, added
            previous, chars, clean = event, 0, True
        elif isinstance(event, Marker):
            if event.kind is MarkerKind.COMPACTION:
                clean = False
        elif event.kind is BlockKind.IMAGE:
            clean = False
        elif event.kind is not BlockKind.ASSISTANT_OUTPUT:
            chars += event.chars
