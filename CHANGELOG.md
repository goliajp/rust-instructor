# Changelog

## [1.4.0] - 2026-08-13

Ecosystem catch-up. The provider defaults had fallen one to two model
generations behind, and the Anthropic default was past its published retirement
date.

### Changed

- **Default models moved to each provider's current balanced tier.** OpenAI
  `gpt-4o` → `gpt-5.6-terra`, Anthropic `claude-sonnet-4-20250514` →
  `claude-sonnet-5`, Gemini `gemini-2.5-flash` → `gemini-3.6-flash`. The old
  Anthropic default was deprecated with a published retirement date of
  2026-06-15, so `Client::anthropic(key)` with no `.with_model()` was calling a
  retired id. Pin `.with_model("...")` if you need a specific model to survive
  upgrades.
- **`tiktoken` 3 → 4, with `default-features = false`.** Only
  `tiktoken::pricing` is used, never the tokenizer, so skipping the default
  `vocabs-all` feature keeps roughly 5 MB of vocabulary data out of the binary.
  The price table also gains the current OpenAI, Anthropic, and Gemini
  generations.
- **Default `max_tokens` 4096 (8192 for Gemini) → 16384.** On the current
  Claude and Gemini generations thinking tokens count against the same budget
  as the response, so the old ceiling truncated mid-answer.

### Fixed

- **Cost tracking no longer reports `None` for most real model ids.** Model ids
  were passed to the price table verbatim, but that lookup is an exact match
  and the table spells a generation with a dot (`claude-haiku-4.5`) while the
  APIs use a dash (`claude-haiku-4-5`) and historically appended a release date
  (`claude-sonnet-4-20250514`, `gpt-4o-2024-08-06`). Those ids all priced at
  `None` silently.

  The fix lives upstream in `tiktoken` 4.1, whose `estimate_cost` resolves
  those spellings (and Bedrock / Vertex decoration) itself — normalization is
  the price table's own contract, not something each consumer should reimplement.
  This crate carries none of its own; `tests/cost_tracking.rs` is the contract
  that keeps upstream honest about the spellings this crate depends on.
  Requires `tiktoken >= 4.1.1`.

### Documentation

- Document that `temperature` is not sent to Anthropic. The current Claude
  generations reject a non-default `temperature` with a 400, so the Anthropic
  request never carries it and `with_temperature` / `.temperature()` have no
  effect there. Noted on all three setters and in the READMEs.

### Added

- `Client::with_anthropic_structured_output()` — opt into Anthropic's native
  structured outputs (`output_config.format`) instead of the default forced
  `tool_use`. Opt-in because not every Claude generation, nor every
  Anthropic-compatible proxy, accepts it. Covers both the buffered and
  streaming paths.

## [1.3.4] - 2026-06-07

### Changed
- 防腐 maintenance release. No behavior change.
- Drop `rust-version = "1.94"` pin from `Cargo.toml` — the crate tracks
  stable; the pin was a stale snapshot left over from the migration.
- Switch `benches/schema.rs` from deprecated `criterion::black_box` to
  `std::hint::black_box` (criterion 0.8 deprecation).
- Clean up ~89 `needless_borrows_for_generic_args` clippy hits across
  integration tests (mostly `&server.uri()` → `server.uri()`).

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
