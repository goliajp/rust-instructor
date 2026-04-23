# Changelog

## [1.3.3] - 2026-04-24

### Changed
- Smoke-test release via the new repo's GitHub Actions publish workflow.
  No code changes.

## [1.3.2] - 2026-04-24

### Changed
- Migrated from `goliajp/airs` mono-repo to standalone `goliajp/rust-instructor`.
  No code changes; `repository` URL updated. `tiktoken` dep switched from workspace
  path to crates.io (`tiktoken = "3.1"`, optional, behind `cost-tracking` feature).

## [1.1.2] - 2026-03-08

### Added

- Examples for streaming (`streaming.rs`), image input (`image.rs`), and provider fallback (`fallback.rs`)
- Streaming, image input, and provider fallback sections to trilingual READMEs

### Fixed

- `BatchBuilder` doc link path resolution
- README model table accuracy

## [1.1.1] - 2026-03-08

### Changed

- Streaming SSE parser: use `buffer.drain()` instead of reallocation per line
- Collapse nested `if let` in Anthropic streaming (clippy)

## [1.1.0] - 2026-03-08

### Added

- SSE streaming via `.on_stream()` callback (OpenAI and Anthropic)
- Image input via `.image()` / `.images()` for vision-capable models
- `Message::user_with_images()` constructor
- Provider fallback via `Client::with_fallback()` — chain multiple providers for auto-failover

## [1.0.0] - 2026-03-07

### Added

- `Validate` trait + closure `.validate()` with error feedback to LLM on retry
- `BatchBuilder` with `tokio::Semaphore` for concurrent multi-prompt extraction
- `extract_many::<T>()` for list extraction via `Vec<T>` wrapper
- Multi-turn conversations via `.messages()` history
- `on_request` / `on_response` lifecycle hooks
- Schema caching via `thread_local` (zero lock contention)
- Anthropic-compatible provider constructor
- 96%+ test coverage (112 tests)

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
