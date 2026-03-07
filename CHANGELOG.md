# Changelog

## [0.1.0] - 2026-03-07

### Added

- Initial release
- Type-safe structured output extraction from LLMs via JSON Schema
- OpenAI provider: `response_format` with strict JSON Schema mode
- Anthropic provider: `tool_use` with forced tool choice
- OpenAI-compatible provider: works with DeepSeek, Together, etc.
- `ExtractBuilder` with `IntoFuture` for ergonomic `.await` on builder
- Per-request configuration: model, system prompt, temperature, max_tokens, max_retries, context
- Client-level defaults with per-request overrides
- Single-turn retry with error feedback on parse failure
- Schema transformation: inlines `$ref`, adds `additionalProperties: false` for OpenAI strict mode
- Cost tracking via `tiktoken` (behind `cost-tracking` feature flag, enabled by default)
- Trilingual documentation (English, simplified Chinese, Japanese)
