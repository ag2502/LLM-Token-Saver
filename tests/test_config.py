from pathlib import Path

import pytest

from llm_token_saver.config import ConfigError, Currency, Settings, home_dir, load_settings


def test_defaults_when_no_config_file(tmp_path: Path) -> None:
    settings = load_settings(tmp_path / "missing.toml")
    assert settings == Settings()
    assert settings.currency.code == "USD"


def test_reads_currency_and_projects_dir(tmp_path: Path) -> None:
    config = tmp_path / "config.toml"
    config.write_text(
        '[currency]\ncode = "INR"\nper_usd = 88.0\n\n[claude_code]\nprojects_dir = "/data/cc"\n'
    )
    settings = load_settings(config)
    assert settings.currency == Currency(code="INR", per_usd=88.0)
    assert settings.claude_code.projects_dir == Path("/data/cc")


def test_non_usd_currency_requires_a_rate(tmp_path: Path) -> None:
    config = tmp_path / "config.toml"
    config.write_text('[currency]\ncode = "EUR"\n')
    with pytest.raises(ConfigError, match="per_usd"):
        load_settings(config)


def test_unknown_keys_are_rejected(tmp_path: Path) -> None:
    config = tmp_path / "config.toml"
    config.write_text("[currency]\ncod = 'USD'\n")
    with pytest.raises(ConfigError):
        load_settings(config)


def test_currency_formatting() -> None:
    assert Currency().format(1578.4) == "$1,578"
    assert Currency().format(2.104) == "$2.10"
    assert Currency(code="INR", per_usd=88.0).format(10) == "₹880"
    assert Currency(code="CHF", per_usd=0.8).format(10) == "CHF 8.00"


def test_home_dir_honours_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("TOKENSAVER_HOME", str(tmp_path))
    assert home_dir() == tmp_path
