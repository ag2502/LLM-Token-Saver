"""Command-line interface: `tokensaver` (also installed as `llm-token-saver`)."""

from __future__ import annotations

from typing import Annotated

import typer

from llm_token_saver import __version__

app = typer.Typer(
    name="tokensaver",
    help="Find out why your LLM bill is high, and what to fix first.",
    no_args_is_help=True,
    add_completion=False,
)


def _show_version(value: bool) -> None:
    if value:
        typer.echo(f"llm-token-saver {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            callback=_show_version,
            is_eager=True,
            help="Show the version and exit.",
        ),
    ] = False,
) -> None:
    """LLM Token Saver."""
