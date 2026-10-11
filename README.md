# LLM Token Saver

Find out **why** your LLM bill is high, fix the waste that costs you nothing to fix, and route
easy calls to cheaper models with a **statistical quality guarantee**.

> **Status: early development.** Nothing is ready to install yet. The roadmap below is the plan,
> not a list of shipped features.

## Why

Observability tools show how many tokens you spent. They don't tell you which of those tokens
were waste, or what fixing it is worth. LLM Token Saver reads your traces and answers:

- **Where does the money go?** Spend by model, session, prompt family and token type
  (uncached input, cache writes, cache reads, output, thinking).
- **What is waste?** Context that rides along for dozens of turns, duplicated context,
  prompt-cache misses and their causes, the wrong cache TTL, bloated prefixes, retries,
  traffic that could use the Batch API.
- **What is each fix worth?** Every finding comes with an estimated monthly saving and a
  concrete fix.
- **Can a cheaper model do it?** A router that picks a cheaper (model, effort) pair only where a
  calibrated certificate bounds how often answers get worse.

## Roadmap

1. Profiler for Claude Code logs (`~/.claude/projects`) with an HTML report.
2. Profiler for any LLM app via OpenTelemetry GenAI traces, first PyPI release.
3. Router with a Learn-Then-Test quality certificate, benchmarked on public data.
4. Local gateway (Anthropic- and OpenAI-compatible) that logs calls and routes them.

## License

[Apache-2.0](LICENSE)
