# Changelog

## [2.0.0] - 2026-11-26

2.0 is a breaking release. The default build no longer depends on
`schemars`, `futures`, `thiserror`, `rand` or `jsonrepair`, and `tokio` is no
longer a direct dependency; the crate builds requests and parses responses
without I/O, and the HTTP layer is a trait with reqwest as the default.

### Upgrading from 1.x

Each item lists what to change. Most code needs the first three.

1. **Derive `Schema` instead of `JsonSchema`.** `use instructors::prelude::*`
   brings `Deserialize` and `Schema`.

   ```rust
   // 1.x
   #[derive(Deserialize, JsonSchema)]
   struct Contact { name: String }
   // 2.0
   #[derive(Deserialize, Schema)]
   struct Contact { name: String }
   ```

   The derive reads serde attributes (`rename`, `rename_all`, `default`,
   `skip`, `flatten`, `tag`, `content`, `untagged`, `transparent`,
   `deny_unknown_fields`) and doc comments, and rejects serde attributes that
   change the JSON in a way it cannot see (`deserialize_with`, `try_from`)
   unless you add `#[schema(like = SomeType)]`. A type that only has
   `schemars::JsonSchema` can still be used with feature `schemars` and
   `client.extract_schemars::<T>(..)`. `instructors::JsonSchema` is no longer
   re-exported.

   **Schema properties now follow declaration order** for every provider (in
   1.x they were alphabetical unless `preserve-order` was on). OpenAI
   generates fields in schema order, so answers can change. The
   `preserve-order` feature still exists and does nothing.

2. **Errors.** `Error` is `#[non_exhaustive]`; add a `_ =>` arm.

   | 1.x | 2.0 |
   |---|---|
   | `Error::Http(reqwest::Error)` | `Error::Transport { source: TransportError, .. }` |
   | `Error::Json(serde_json::Error)` | `Error::Api` (malformed response body) or `Error::Parse` |
   | `Error::ExtractionFailed { retries, .. }` | `Error::Parse { attempts, raw, .. }` |
   | `Error::ValidationFailed { retries, .. }` | `Error::Validation { attempts, .. }` |
   | `Error::Other` for refusals | `Error::Refusal { text, .. }` |
   | `Error::Other` for truncation | `Error::Truncated { reason, raw, .. }` |
   | `Error::Other` for unsupported settings | `Error::Unsupported { what, .. }` |
   | `Error::Timeout(Duration)` | `Error::Timeout { after, .. }` |
   | fallback errors joined in one message | `Error::Fallback`; `chain()` and `source()` walk every provider's error |

   Every variant names its `provider`, and `err.usage()` returns what the
   failed extraction cost; `Error::Api` gained `provider`, `retry_after` and
   `usage`, so a pattern on it needs `..`. `refusal()` and `is_truncated()`
   remain. `ValidationError::message` is now a method, not a field.

3. **Usage.** Counters are `u64`. `total_tokens` and `retries` are methods.
   New: `cached_input_tokens`, `cache_write_tokens`, `reasoning_tokens`,
   `requests`. `Usage` is `#[non_exhaustive]`. Anthropic `input_tokens` now
   includes cache reads and writes. The usage of a fallback that succeeded
   includes the requests of the providers that failed before it.

4. `ExtractResult<T>` is now `Extracted<T>` with a `truncated` field and
   `#[non_exhaustive]`; the old name is a deprecated alias at the crate root
   and is no longer in the prelude. `ExtractBuilder` is now `Extract` and
   `BatchBuilder` is `Batch`.

5. **Messages and images.** `Message { role, content, images }` has private
   fields: use `Message::user(..)` / `Message::assistant(..)` and `.with(part)`.
   `ImageInput::Url(u)` → `Part::image_url(u)`, `ImageInput::Base64 { .. }` →
   `Part::image(media_type, data)`; `Message::user_with_images(text, imgs)` →
   `Message::user(text).with(part)`; on the builder `.image(..)` /
   `.images(..)` → `.part(..)`. New: `Part::document` for PDFs.

6. **Retries.** HTTP retries are **on by default**: 3 retries on 408, 409,
   429, 500, 502, 503, 504, 529 and connection errors, honouring
   `retry-after-ms` and `Retry-After` (seconds, or an HTTP date counted from
   the response's `Date` header). An `x-should-retry: true` / `false`
   response header overrides the status. `BackoffConfig` → `RetryPolicy`, `with_retry_backoff` →
   `with_retry`; `with_retry(RetryPolicy::none())` turns them off.
   `with_max_retries(n)` / `.max_retries(n)` → `with_max_attempts(n + 1)` /
   `.max_attempts(n + 1)`.

7. **Re-asks show the model its answer.** After a parse or validation failure
   the next request appends the model's answer as an assistant message and
   the error as a user message, instead of rewriting the prompt with
   "IMPORTANT: A previous extraction attempt produced invalid output".

8. **Hooks.** `on_stream` → `on_delta`. `on_request` receives the
   `&HttpRequest` (URL, headers, body bytes) instead of `(model, prompt)`.
   `PartialValue::to_value()` → `to_json()` (or `to_value()` with feature
   `serde_json`). New: `on_item` for `extract_many`.

9. **Output modes.** `with_anthropic_tool_use()` →
   `with_output_mode(OutputMode::Tool)`; `with_anthropic_structured_output()` →
   `with_output_mode(OutputMode::Schema)`. `openai_compatible` clients for
   DeepSeek, GLM and Qwen hosts now send `response_format: json_object` with
   the schema in the system prompt, since those APIs document only JSON mode;
   `with_output_mode(OutputMode::Schema)` restores the 1.x request. Gemini
   gets `generationConfig.responseFormat.text` with a JSON Schema (1.x sent
   `responseMimeType` with `responseSchema`, or `responseJsonSchema` for a
   recursive type; Google marks those fields deprecated). Integer and number
   `format`s and `const` are not sent to Gemini, which documents neither.

10. **Features and runtime.** `default-features = false` now means no HTTP
    client: add `features = ["reqwest"]` or call `with_transport`. The
    implicit `tiktoken` feature is gone; enable `cost-tracking`. The core
    needs no async runtime; the default reqwest transport still needs tokio.
    TLS is rustls with the ring provider (1.x used aws-lc), so no CMake or C
    toolchain beyond what ring needs.

11. `extract_batch` takes any `IntoIterator<Item = String>`, and the batch can
    be awaited directly (`.run()` still works).

12. A `T` that contains `serde_json::value::RawValue` no longer deserializes;
    use `serde_json::Value`.

13. **No default temperature.** 1.x sent `temperature: 0.0` to OpenAI and
    every OpenAI-compatible API unless you set one; reasoning models reject
    a temperature. 2.0 sends it only when set: add `.with_temperature(0.0)`
    to keep the 1.x request.

14. **Strict schemas carry every type.** For OpenAI strict mode (Chat
    Completions `Schema` and `Tool`, Responses `Schema`) and Anthropic
    structured output, a map is sent as a list of `{"key", "value"}`
    objects, a tuple as an object with keys `"0"`, `"1"`, …, and a root that
    is not an object as `{"value": ..}`; answers are turned back before they
    reach your type. Keywords the provider does not take (`uniqueItems`;
    `minLength` / `maxLength` on OpenAI; bounds and `maxItems` on Anthropic)
    are dropped from the schema and checked on the answer, with the usual
    re-ask. OpenAI schemas now keep the string `format`s OpenAI lists. If you
    read `HttpRequest` bodies or `on_partial` views, expect the sent form. A
    type holding a value of any type (schema `true`) has no strict schema:
    OpenAI gets the plain schema with `strict: false`, Anthropic a tool, and
    the answer is checked here; Gemini gets an empty schema object for that
    field. An Anthropic tool's `input_schema` and a non-strict OpenAI schema
    wrap a root that is not an object in `{"value": ..}` too, since both must
    be objects. Gemini schemas leave out `minLength`,
    `maxLength` and `uniqueItems`, which Gemini ignores, and the answer is
    checked for them.

15. **Anthropic model table.** `Client::anthropic` keeps structured output
    as its default, and models without it (Claude 3, Opus 4 and 4.1, Sonnet
    4) get a tool. The tool is forced unless thinking is on (`Effort` or
    `Budget`; `Off` now forces it, 1.x did not) or the model rejects a
    forced tool (Opus 5.5, Sonnet 5.5, Fable and Mythos 5.1 and 5.2); then
    it is offered with `tool_choice: auto` and the system prompt asks for
    the call. A recursive type's tool is forced on the same terms.

16. **DeepSeek and Qwen.** New `Client::deepseek(key)` and
    `Client::qwen(key)`. Hosts on `deepseek.com` default to `deepseek-flash`
    and hosts on `aliyuncs.com` to `qwen3.7-plus`; 1.x sent the OpenAI
    default, which those APIs reject. Their `Tool` mode offers the
    tool with `tool_choice: auto` unless reasoning is `Off`, since a forced
    tool is rejected in their default thinking mode. `Part::document` on Qwen
    returns `Error::Unsupported` before sending.

17. **Responses API history.** An assistant turn goes as
    `{"role": "assistant", "content": "<text>"}` and every `input_image`
    carries `"detail": "auto"`.

### Added

- `Client::deepseek` and `Client::qwen`, with their own default models.
- Sans-IO core: `Extract::exchange()` returns an `Exchange` state machine
  that yields `Step::Send(HttpRequest)`, `Step::Wait(Duration)` and
  `Step::Done(result)`; feed it responses with `receive` or, streamed,
  `receive_head` / `receive_chunk` / `receive_end`. The core builds for
  `wasm32-unknown-unknown`.
- `Transport` trait and `Client::with_transport`; `ReqwestTransport` with
  `client_builder()` for proxies and custom TLS roots.
- `OutputMode` (`Schema`, `Tool`, `JsonObject`, `Prompt`) with
  `with_output_mode` / `.output_mode`.
- OpenAI Responses API, opt-in: `Client::openai(key).with_api(OpenAiApi::Responses)`.
- `.accept_truncated()`: an answer cut off by the token limit is closed,
  parsed and returned with `truncated: true` when every missing field is
  optional; `extract_many` returns the items that were complete.
- `on_item` for `extract_many`.
- `Part::document` for PDFs on all providers.
- Feature `schemars` with `Client::extract_schemars`.

### Fixed

- Schema properties follow declaration order (1.5.0 sent them sorted).
- An internally tagged enum variant holding a struct
  (`#[serde(tag = "kind")] enum E { A(Inner) }`) lost the struct's fields in
  the schema.
- A refused or failed extraction reports its token usage.
- An error sent inside a stream (Anthropic `error` event, an OpenAI chunk with
  `error`, a Responses `error` event or `response.failed`) and a Responses API
  response with `status: failed` end the extraction with `Error::Api`; they
  were read as an empty answer and re-asked.
- `Usage::reasoning_tokens` counts Anthropic
  `usage.output_tokens_details.thinking_tokens`, and `cache_write_tokens`
  counts OpenAI `cache_write_tokens` (Chat Completions and Responses).
- A streamed Anthropic request that offers the tool without forcing it
  (thinking on, or a recursive type) keeps the model's prose answer when the
  model does not call the tool, as the non-streamed request does; it was
  dropped and the re-ask quoted an empty answer.
- Gemini schemas spell `const` as a one-value `enum`: Gemini's JSON Schema
  subset has no `const` and ignores it, so the tag of a tagged enum was not
  constrained.
- `u128` and `i128` fields deserialize, at full width; the crate's JSON
  reader rejected them. Numbers in a converted strict answer keep their
  text, so integers beyond 2^53 are not rounded.
- Gemini `finishReason: CONTINUATION` (the per-request limit was reached
  before the answer was complete) counts as a truncation, like
  `MAX_TOKENS`.

## [1.5.1] - 2026-11-26

### Added

- Feature `preserve-order`: schema properties are sent in declaration order
  instead of alphabetical order, for every provider. OpenAI generates fields in
  schema order, so a `reasoning` field declared before `answer` is now written
  first. The feature turns on `serde_json/preserve_order`, which affects every
  crate in the build that uses `serde_json::Map`. Without it nothing changes.

## [1.5.0] - 2026-11-25

### Upgrading from 1.4

No public item was removed or changed, so code that compiled against 1.4
compiles against this release. Four requests change on the wire:

- `Client::anthropic` sends the schema as structured output
  (`output_config.format`) instead of a forced tool call. Claude Opus 5.5,
  Sonnet 5.5, Fable 5.1 and later reject forced tool calls with a 400, and
  structured output is available on every current Claude model. To send the
  1.4 request, call `.with_anthropic_tool_use()`. `anthropic_compatible` keeps
  the forced tool call by default, since compatible servers support it more
  widely.
- `extract_many` asks for `{"items": [...]}` instead of a bare array, because
  OpenAI strict mode and Anthropic tools need an object at the root. A bare
  array in the answer is still accepted.
- `with_max_tokens` / `.max_tokens()` now reach OpenAI (as
  `max_completion_tokens`) and OpenAI-compatible APIs (as `max_tokens`).
  1.4 never sent them. When unset, nothing is sent, as before.
- The Gemini API key is sent in the `x-goog-api-key` header instead of the
  `key` query parameter. A proxy that only reads the query parameter needs to
  read the header.

Behaviour that changes without a code change:

- A refusal (OpenAI `refusal`, Anthropic `stop_reason: refusal`, Gemini
  safety stops) and an answer cut off by the token limit are returned at once
  instead of being retried. Check them with `Error::refusal()` and
  `Error::is_truncated()`.
- A fallback client uses its own default model instead of the primary's, and
  the error after all providers fail lists every provider's error.
- Gemini thinking tokens are counted in `usage.output_tokens` and the cost.
- `tiktoken` 5.4 carries the November 2026 price table, so `usage.cost` values
  differ from 1.4 for some models.

### Added

- `Reasoning` (`Effort(Effort)`, `Budget(u32)`, `Off`), set with
  `Client::with_reasoning`, `ExtractBuilder::reasoning` and
  `BatchBuilder::reasoning`, and mapped to each provider's parameter: OpenAI
  `reasoning_effort`, Anthropic `thinking` and `output_config.effort`, Gemini
  `thinkingConfig`, and the DeepSeek, Qwen, Kimi and GLM parameters. Nothing is
  sent unless it is set. Requested in #5; #6, #7 and #8 by Leopard2A5
  implemented the Gemini part first.
- `ExtractBuilder::on_partial`: the JSON parsed so far after each streamed
  delta, as a borrowed `PartialValue`. The `partial` module exposes the
  incremental parser behind it.
- `Error::refusal()` and `Error::is_truncated()`.
- `Client::with_anthropic_tool_use()` to send the schema as a forced tool call.
- `openai_compatible` recognizes DeepSeek, Qwen, Kimi and GLM by their base
  URL host for the reasoning parameters.

### Fixed

- A type containing a recursive type (`struct Doc { root: Node }` with
  `Node { children: Vec<Node> }`) overflowed the stack while the schema was
  prepared. Recursive types are now sent with `$defs`; Anthropic gets a tool
  for them and Gemini `responseJsonSchema`, since their other schema forms
  cannot express recursion.
- A field named `title` or `format` was removed from the schema sent to
  OpenAI and Gemini, so the model was never asked for it.
- `max_tokens` was never sent to OpenAI-family APIs.
- `extract_many` sent a root `array` schema with `strict: true`, which OpenAI
  rejects.
- Streaming replaced a multi-byte UTF-8 character split across two network
  reads with U+FFFD. The SSE decoder now works on bytes.
- OpenAI streaming reported 0 tokens and no cost, because
  `stream_options.include_usage` was not sent.
- A fallback was sent the primary's model id, and the fallback's error was
  dropped.
- Gemini thinking tokens (`thoughtsTokenCount`) were not counted, and only
  the first response part was read; thought parts are now skipped and the
  other parts joined.
- The Gemini API key was part of the request URL, so it could appear in
  error messages.
- An OpenAI refusal (`content: null`) surfaced as a body decode error.
- Anthropic structured output received keywords it rejects (`minimum`,
  `maxLength`, unsupported `format` values, `oneOf`).
- Cost was computed on tokens summed over all attempts; it is now priced per
  request and summed, so a long-context price tier applies per request.
- HTTP backoff, when configured, also retries 500, 502, 504 and 529.

### Changed

- The provider-ready schema is built once per type and kept as serialized
  bytes, so a request copies it instead of transforming it.
- All clients share one HTTP connection pool.
- `tiktoken` 5.1 to 5.4.
- Licence: MIT or Apache-2.0, at your option (was MIT).

## [1.4.1] - 2026-08-13

Documentation only. No code changes.

### Fixed

- The cost-tracking section never said what `cost` being `None` means, nor
  that model ids are resolved rather than matched exactly. 1.4.0 made that the
  whole point of the change — a caller who passes the id they just sent to the
  provider gets a cost — and the README shipped without mentioning it. Noted in
  all three languages.

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
- Maintenance release. No behavior change.
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
- Moved to its own repository, `goliajp/rust-instructor`.
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
