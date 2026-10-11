from datetime import UTC, datetime, timedelta

import pytest

from llm_token_saver.estimate import (
    IMAGE_TOKENS,
    MIN_SAMPLES,
    TURN_OVERHEAD_TOKENS,
    TokenEstimator,
    default_chars_per_token,
)
from llm_token_saver.schema import Block, BlockKind, Call, Marker, MarkerKind, Thread
from llm_token_saver.timeline import timeline

T0 = datetime(2026, 10, 1, tzinfo=UTC)


def make_call(seq: int, prompt: int, output: int = 0, model: str = "claude-opus-5-5") -> Call:
    return Call(
        call_id=f"c{seq}",
        source="test",
        session_id="s",
        thread_id="t",
        seq=seq,
        timestamp=T0 + timedelta(seconds=seq),
        model=model,
        cache_read=prompt,
        output=output,
    )


def make_block(seq: int, kind: BlockKind = BlockKind.TOOL_RESULT, chars: int = 0) -> Block:
    return Block(session_id="s", thread_id="t", seq=seq, kind=kind, chars=chars)


def thread_with_ratio(chars_per_token: float, gaps: int) -> Thread:
    """Each gap adds 3,000 characters of tool output that cost 3000 / ratio tokens."""
    calls, blocks = [], []
    prompt = 10_000
    for i in range(gaps + 1):
        calls.append(make_call(seq=2 * i, prompt=prompt, output=50))
        blocks.append(make_block(seq=2 * i + 1, chars=3_000))
        prompt += 50 + round(3_000 / chars_per_token) + TURN_OVERHEAD_TOKENS
    return Thread(thread_id="t", session_id="s", source="test", calls=calls, blocks=blocks)


def test_default_ratio_by_tokenizer_generation() -> None:
    assert default_chars_per_token("claude-opus-5-5") == 2.5
    assert default_chars_per_token("claude-opus-4-7") == 2.5
    assert default_chars_per_token("claude-sonnet-4-6") == 3.3
    assert default_chars_per_token("claude-3-5-haiku-20241022") == 3.3
    assert default_chars_per_token("gpt-unknown") == 3.3


def test_fit_recovers_the_ratio_from_usage_growth() -> None:
    estimator = TokenEstimator.fit([thread_with_ratio(2.8, gaps=MIN_SAMPLES + 5)])
    assert estimator.ratio("claude-opus-5-5") == pytest.approx(2.8, rel=0.01)


def test_fit_needs_enough_samples() -> None:
    estimator = TokenEstimator.fit([thread_with_ratio(2.8, gaps=MIN_SAMPLES - 5)])
    assert estimator.ratio("claude-opus-5-5") == 2.5


def test_gaps_with_images_or_compactions_are_ignored() -> None:
    thread = thread_with_ratio(2.8, gaps=MIN_SAMPLES + 5)
    thread.blocks = [
        make_block(block.seq, BlockKind.IMAGE) if i % 2 else block
        for i, block in enumerate(thread.blocks)
    ]
    thread.markers = [
        Marker(session_id="s", thread_id="t", seq=block.seq, kind=MarkerKind.COMPACTION)
        for block in thread.blocks[::2]
    ]
    assert TokenEstimator.fit([thread]).ratio("claude-opus-5-5") == 2.5


def test_block_tokens() -> None:
    estimator = TokenEstimator({"claude-opus-5-5": 2.0})
    assert estimator.tokens(make_block(0, chars=1_000), "claude-opus-5-5") == 500
    assert estimator.tokens(make_block(0, BlockKind.IMAGE), "claude-opus-5-5") == IMAGE_TOKENS
    measured = Block(
        session_id="s", thread_id="t", seq=0, kind=BlockKind.ASSISTANT_OUTPUT, tokens=7
    )
    assert estimator.tokens(measured, "claude-opus-5-5") == 7


def test_timeline_puts_a_call_before_its_own_output() -> None:
    call = make_call(seq=3, prompt=10)
    output = Block(session_id="s", thread_id="t", seq=3, kind=BlockKind.ASSISTANT_OUTPUT, tokens=5)
    earlier = make_block(seq=1)
    thread = Thread(
        thread_id="t", session_id="s", source="test", calls=[call], blocks=[output, earlier]
    )
    assert timeline(thread) == [earlier, call, output]
