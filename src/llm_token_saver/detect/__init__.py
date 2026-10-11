"""Waste detectors. Each turns normalized records into ranked findings."""

from llm_token_saver.detect.base import CostPool, Evidence, Finding, FindingKind, rank

__all__ = ["CostPool", "Evidence", "Finding", "FindingKind", "rank"]
