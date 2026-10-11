from typer.testing import CliRunner

from llm_token_saver import __version__
from llm_token_saver.cli import app

runner = CliRunner()


def test_version_flag_prints_package_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.output.strip() == f"llm-token-saver {__version__}"


def test_no_arguments_shows_help() -> None:
    result = runner.invoke(app, [])
    assert "Find out why your LLM bill is high" in result.output
