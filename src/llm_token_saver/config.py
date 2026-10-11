"""User settings, read from `$TOKENSAVER_HOME/config.toml` (default `~/.tokensaver`).

Example:

    [currency]
    code = "INR"
    per_usd = 88.0          # units of `code` per 1 USD; required for non-USD currencies

    [claude_code]
    projects_dir = "~/.claude/projects"
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

CURRENCY_SYMBOLS = {"USD": "$", "EUR": "€", "INR": "₹", "GBP": "£"}


class ConfigError(Exception):
    """The config file exists but is not valid."""


class Currency(BaseModel):
    """Display currency. Prices are stored in USD and converted only for display."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    code: str = "USD"
    per_usd: float = Field(default=1.0, gt=0)

    @model_validator(mode="after")
    def _rate_required_for_other_currencies(self) -> Self:
        if self.code != "USD" and "per_usd" not in self.model_fields_set:
            raise ValueError(f"set per_usd (how many {self.code} one USD buys)")
        return self

    @property
    def symbol(self) -> str:
        return CURRENCY_SYMBOLS.get(self.code, f"{self.code} ")

    def format(self, usd: float) -> str:
        value = usd * self.per_usd
        if abs(value) >= 100:
            return f"{self.symbol}{value:,.0f}"
        return f"{self.symbol}{value:,.2f}"


class ClaudeCodeSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    projects_dir: Path = Path("~/.claude/projects")


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    currency: Currency = Currency()
    claude_code: ClaudeCodeSettings = ClaudeCodeSettings()
    content: Literal["hashed", "full", "none"] = "hashed"


def home_dir() -> Path:
    return Path(os.environ.get("TOKENSAVER_HOME", "~/.tokensaver")).expanduser()


def load_settings(path: Path | None = None) -> Settings:
    """Read settings from `path` (default: the config file in `home_dir()`), if it exists."""
    path = path or home_dir() / "config.toml"
    if not path.exists():
        return Settings()
    try:
        with path.open("rb") as fh:
            return Settings.model_validate(tomllib.load(fh))
    except (tomllib.TOMLDecodeError, ValidationError) as exc:
        raise ConfigError(f"{path}: {exc}") from exc
