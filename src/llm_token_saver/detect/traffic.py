"""Detector 6: request-level waste such as failed and retried calls."""

from __future__ import annotations

from llm_token_saver.detect.base import Finding, FindingKind
from llm_token_saver.schema import Dataset, MarkerKind


def api_error_finding(dataset: Dataset) -> Finding | None:
    errors = [
        marker
        for thread in dataset.threads
        for marker in thread.markers
        if marker.kind is MarkerKind.API_ERROR
        and (
            dataset.window_start is None
            or (marker.timestamp is not None and marker.timestamp >= dataset.window_start)
        )
    ]
    if not errors:
        return None
    sessions = len({marker.session_id for marker in errors})
    return Finding(
        detector="api_errors",
        title="Requests failed and were retried",
        kind=FindingKind.INFO,
        summary=(
            f"{len(errors):,} requests failed and were retried across {sessions:,} sessions. "
            "Failed requests report no usage, so they add waiting time rather than cost here."
        ),
        fix="Nothing to change unless errors cluster at certain hours or in certain projects.",
    )
