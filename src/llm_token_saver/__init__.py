"""LLM Token Saver: find out why your LLM bill is high and what to fix first."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("llm-token-saver")
except PackageNotFoundError:  # source checkout that was never installed
    __version__ = "0.0.0"
