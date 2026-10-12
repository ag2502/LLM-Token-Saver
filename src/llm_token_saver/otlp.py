"""Build OTLP/JSON trace payloads: the demo traces use this, and so do the tests."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any


def value(item: Any) -> dict[str, Any]:
    """A Python value as an OTLP `AnyValue`."""
    if isinstance(item, bool):
        return {"boolValue": item}
    if isinstance(item, int):
        return {"intValue": str(item)}
    if isinstance(item, float):
        return {"doubleValue": item}
    if isinstance(item, list | tuple):
        return {"arrayValue": {"values": [value(entry) for entry in item]}}
    return {"stringValue": str(item)}


def attributes(mapping: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"key": key, "value": value(item)} for key, item in mapping.items() if item is not None]


def span(
    *,
    trace_id: str,
    span_id: str,
    start: datetime,
    duration: timedelta,
    attrs: dict[str, Any],
    name: str = "chat",
) -> dict[str, Any]:
    start_ns = int(start.timestamp() * 1e9)
    return {
        "traceId": trace_id,
        "spanId": span_id,
        "name": name,
        "kind": 3,
        "startTimeUnixNano": str(start_ns),
        "endTimeUnixNano": str(start_ns + int(duration.total_seconds() * 1e9)),
        "attributes": attributes(attrs),
    }


def export_request(spans: list[dict[str, Any]], service: str) -> dict[str, Any]:
    """One `ExportTraceServiceRequest` holding `spans` for `service`."""
    return {
        "resourceSpans": [
            {
                "resource": {"attributes": attributes({"service.name": service})},
                "scopeSpans": [{"scope": {"name": "llm-token-saver"}, "spans": spans}],
            }
        ]
    }
