# LLM Token Saver

Find out **why** your LLM bill is high, fix the waste that costs nothing to fix, and see which
savings would trade away quality.

> **Status: early development.** The Claude Code profiler works today from a source checkout.
> A PyPI release, OpenTelemetry traces for any LLM app, and a router with a quality guarantee
> are next (see [Roadmap](#roadmap)).

## What it tells you

Run it on your Claude Code logs and it answers three questions:

- **Where did the money go?** API-equivalent spend split by token type (cache reads, cache
  writes, output, thinking, uncached input), model, project, effort level, day and session.
- **What is waste?** Each finding comes with an estimated monthly saving, the evidence behind
  it, and a concrete fix:
  - long sessions that keep re-reading old context,
  - prompt-cache rebuilds after idle breaks or mid-session model switches (measured),
  - long command output, whole-file reads and large tool results that ride along in context,
  - the same tool output added to the context twice,
  - max/xhigh effort turns that think far more than high-effort ones (a tradeoff),
  - older models used where a cheaper successor exists, and Opus subagents (price-only).
- **What is a free win and what is a tradeoff?** Free wins save money without lowering answer
  quality. Tradeoffs may cost quality, so they are labeled and never mixed in silently.

## Quickstart

Requires [uv](https://docs.astral.sh/uv/). Nothing leaves your machine, and no API key is
needed: it only reads the session logs Claude Code already writes to `~/.claude/projects`.

```sh
git clone https://github.com/ag2502/LLM-Token-Saver.git
cd LLM-Token-Saver
uv run tokensaver claude-code report                      # ranked findings in the terminal
uv run tokensaver claude-code report --html report.html   # one self-contained HTML file
uv run tokensaver claude-code stats --since 7d            # spend breakdowns only
```

`--since` takes a duration (`12h`, `7d`, `2w`) or a date (`2026-10-01`). `--json PATH` writes the
report in a machine-readable form, with every amount in USD.

## How it works

1. **Import.** Each transcript becomes a thread of billed calls and the content that entered the
   context between them. A response split across several log lines is counted once, with its
   final usage. Subagent runs are tracked as their own threads.
2. **Price.** Calls are priced per token type from the
   [Claude pricing page](https://platform.claude.com/docs/en/about-claude/pricing), including
   per-model cache-read prices, 5-minute and 1-hour cache writes, long-context tiers, fast mode,
   batch and US data-residency modifiers. Unknown models are reported as unpriced, never guessed.
3. **Attribute.** Every call re-sends the whole conversation. Each call's prompt cost is split
   across the content inside it in the order caching bills it (cache reads, then writes, then
   uncached input), so a 30,000-token file read shows what it cost over the hundreds of calls it
   stayed in context. Attributed costs add up to the bill exactly (property-tested).
4. **Estimate sizes.** Transcripts record characters, not tokens, for tool output. Ratios are
   fitted per model from how much each prompt grew between calls, and every finding says whether
   its saving is *measured* (straight from billed usage) or *estimated*.
5. **Rank.** Savings are projected to 30 days and shrunk where fixes overlap, so the list never
   adds up to more than the bill.

## Configuration

Optional. `~/.tokensaver/config.toml` (or `$TOKENSAVER_HOME/config.toml`):

```toml
[currency]
code = "INR"        # display currency; prices stay in USD internally
per_usd = 88.0      # required for anything but USD

[claude_code]
projects_dir = "~/.claude/projects"
```

## Roadmap

1. **Claude Code profiler** — done: import, pricing, carried-context attribution, nine
   detectors, terminal/JSON/HTML reports.
2. **Any LLM app** — OpenTelemetry GenAI traces, synthetic demo traces with planted waste,
   cache-miss causes and TTL choice, first PyPI release.
3. **Router with a quality guarantee** — routes easy requests to a cheaper (model, effort) pair
   only where a Learn-Then-Test certificate bounds how often answers get worse; benchmarked on
   public data.
4. **Gateway** — an Anthropic- and OpenAI-compatible local proxy that logs calls and routes them.

## Development

```sh
uv sync
uv run ruff check . && uv run ruff format --check . && uv run pyright && uv run pytest
```

## License

[Apache-2.0](LICENSE)
