"""Token prices and the cost of a call."""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from datetime import date
from importlib import resources
from typing import Any

from llm_token_saver.schema import Call

PER_MILLION = 1_000_000
_TOKEN_PRICES = ("input", "cache_write_5m", "cache_write_1h", "cache_read", "output")


@dataclass(frozen=True, slots=True)
class Rates:
    """USD per million tokens for each billed token type."""

    input: float
    cache_write_5m: float
    cache_write_1h: float
    cache_read: float
    output: float

    def scaled(self, factor: float) -> Rates:
        if factor == 1.0:
            return self
        return Rates(
            input=self.input * factor,
            cache_write_5m=self.cache_write_5m * factor,
            cache_write_1h=self.cache_write_1h * factor,
            cache_read=self.cache_read * factor,
            output=self.output * factor,
        )


@dataclass(frozen=True, slots=True)
class ModelPrice:
    model_id: str
    name: str
    standard: Rates
    provider: str = "anthropic"
    long_context: Rates | None = None
    long_context_above: int | None = None
    fast_multiplier: float = 1.0
    geo_multiplier: bool = False
    successor: str | None = None
    """A newer model in the same tier, when one exists."""


@dataclass(frozen=True, slots=True)
class Cost:
    """USD cost of one call, split by token type."""

    input_uncached: float = 0.0
    cache_write_5m: float = 0.0
    cache_write_1h: float = 0.0
    cache_read: float = 0.0
    output: float = 0.0

    @property
    def cache_write(self) -> float:
        return self.cache_write_5m + self.cache_write_1h

    @property
    def prompt(self) -> float:
        """Everything paid for reading the prompt."""
        return self.input_uncached + self.cache_write + self.cache_read

    @property
    def total(self) -> float:
        return self.prompt + self.output


_REGION_PREFIX = re.compile(r"^(?:(?:us|eu|apac|jp|au|global)\.)?anthropic\.")
_CONTEXT_SUFFIX = re.compile(r"\[[^\]]*\]$")
_BEDROCK_VERSION = re.compile(r"-v\d+(?::\d+)?$")
_DATE_SUFFIX = re.compile(r"[-@]\d{8}$")
_ISO_DATE_SUFFIX = re.compile(r"-\d{4}-\d{2}-\d{2}$")


def normalize_model_id(model: str) -> str:
    """Map provider spellings of a model to its first-party ID.

    `us.anthropic.claude-sonnet-4-5-20250929-v1:0`, `claude-opus-4-5@20251101`,
    `anthropic/claude-opus-5-5` and `models/gemini-2.5-pro` become `claude-sonnet-4-5`,
    `claude-opus-4-5`, `claude-opus-5-5` and `gemini-2.5-pro`. Unknown names pass through
    unchanged and are reported as unpriced rather than guessed.
    """
    name = model.strip().lower().rsplit("/", 1)[-1]
    name = _CONTEXT_SUFFIX.sub("", name)
    name = _REGION_PREFIX.sub("", name)
    name = _BEDROCK_VERSION.sub("", name)
    return _DATE_SUFFIX.sub("", name)


class PriceTable:
    """Per-model prices plus the modifiers that stack on them."""

    def __init__(
        self,
        models: dict[str, ModelPrice],
        *,
        source: str,
        retrieved: date,
        batch_multiplier: float,
        us_geo_multiplier: float,
    ) -> None:
        self.models = models
        self.source = source
        self.retrieved = retrieved
        self.batch_multiplier = batch_multiplier
        self.us_geo_multiplier = us_geo_multiplier

    @classmethod
    def bundled(cls) -> PriceTable:
        """The price table shipped with the package."""
        text = resources.files("llm_token_saver.pricing").joinpath("prices.toml").read_text("utf-8")
        return cls.from_toml(text)

    @classmethod
    def from_toml(cls, text: str) -> PriceTable:
        data = tomllib.loads(text)
        meta = data["meta"]
        models = {
            model_id: _parse_model(model_id, entry) for model_id, entry in data["models"].items()
        }
        return cls(
            models,
            source=meta["source"],
            retrieved=meta["retrieved"],
            batch_multiplier=meta["batch_multiplier"],
            us_geo_multiplier=meta["us_geo_multiplier"],
        )

    def lookup(self, model: str) -> ModelPrice | None:
        name = normalize_model_id(model)
        price = self.models.get(name)
        if price is None and _ISO_DATE_SUFFIX.search(name):
            # OpenAI snapshots (gpt-4o-2024-08-06) share their base model's price unless the
            # table lists the snapshot itself.
            price = self.models.get(_ISO_DATE_SUFFIX.sub("", name))
        return price

    def rates(
        self,
        model: str,
        *,
        prompt_tokens: int = 0,
        speed: str = "standard",
        batch: bool = False,
        inference_geo: str = "",
    ) -> Rates | None:
        """Effective rates for one request, or None if the model is not in the table."""
        price = self.lookup(model)
        if price is None:
            return None
        rates = price.standard
        if (
            price.long_context is not None
            and price.long_context_above is not None
            and prompt_tokens > price.long_context_above
        ):
            rates = price.long_context
        factor = 1.0
        if speed == "fast":
            factor *= price.fast_multiplier
        if batch:
            factor *= self.batch_multiplier
        if inference_geo == "us" and price.geo_multiplier:
            factor *= self.us_geo_multiplier
        return rates.scaled(factor)

    def rates_for(self, call: Call) -> Rates | None:
        return self.rates(
            call.model,
            prompt_tokens=call.prompt_tokens,
            speed=call.speed,
            batch=call.is_batch,
            inference_geo=call.inference_geo,
        )

    def cost(self, call: Call) -> Cost | None:
        """What `call` cost in USD, or None if its model is not priced."""
        rates = self.rates_for(call)
        if rates is None:
            return None
        return Cost(
            input_uncached=call.input_uncached * rates.input / PER_MILLION,
            cache_write_5m=call.cache_write_5m * rates.cache_write_5m / PER_MILLION,
            cache_write_1h=call.cache_write_1h * rates.cache_write_1h / PER_MILLION,
            cache_read=call.cache_read * rates.cache_read / PER_MILLION,
            output=call.output * rates.output / PER_MILLION,
        )


def _parse_rates(entry: dict[str, Any]) -> Rates:
    return Rates(**{field: float(entry[field]) for field in _TOKEN_PRICES})


def _parse_model(model_id: str, entry: dict[str, Any]) -> ModelPrice:
    long_context = entry.get("long_context")
    return ModelPrice(
        model_id=model_id,
        name=entry["name"],
        standard=_parse_rates(entry),
        provider=str(entry.get("provider", "anthropic")),
        long_context=_parse_rates(long_context) if long_context else None,
        long_context_above=int(long_context["above_prompt_tokens"]) if long_context else None,
        fast_multiplier=float(entry.get("fast_multiplier", 1.0)),
        geo_multiplier=bool(entry.get("geo_multiplier", False)),
        successor=entry.get("successor"),
    )
