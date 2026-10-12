# Changelog

## 0.1.0

First release.

### Claude Code

- `tokensaver claude-code report` and `stats`: API-equivalent spend by token type, model,
  project, effort level, day and session, with ranked findings and fixes, as terminal output,
  JSON or one self-contained HTML file.
- Carried-context attribution: each call's prompt cost is split across the content inside it
  in the order prompt caching bills it, so every tool result, file read, screenshot and
  thinking block shows what it cost across the calls that re-read it.
- Findings: stale context in long sessions, cache rebuilds after idle breaks or model
  switches, the 5-minute vs 1-hour cache lifetime, long command output, whole-file reads,
  large tool results, duplicate tool output, highest-effort thinking, older models with a
  cheaper successor, and Opus subagents; plus notes on screenshots, the fixed prefix and API
  errors.
- `tokensaver claude-code session-end` for a `SessionEnd` hook, and `schedule install` for a
  weekly report on macOS.

### Any LLM app

- `tokensaver analyze` reads OpenTelemetry GenAI traces (OTLP/JSON, current and older
  attribute names, OpenLLMetry's indexed prompts).
- Findings: a changing value early in the prompt that defeats the cache, a shared prefix that
  is never cached, the cache lifetime replayed on real request times, repeated identical
  requests, scheduled jobs that could use the Batch API, retrieval chunks sent twice, and
  frontier models answering in a few words.
- `tokensaver demo` writes a week of synthetic traces with planted waste and analyzes them.

### Pricing

- Claude, OpenAI and Gemini list prices from the providers' pricing pages, including cache
  read and write prices, long-context tiers, fast mode, batch and data-residency modifiers.
