# instructors

[![crates.io](https://img.shields.io/crates/v/instructors?style=flat-square&logo=rust)](https://crates.io/crates/instructors)
[![docs.rs](https://img.shields.io/docsrs/instructors?style=flat-square&logo=docs.rs)](https://docs.rs/instructors)
[![CI](https://img.shields.io/github/actions/workflow/status/goliajp/rust-instructor/ci.yml?branch=develop&style=flat-square&logo=github&label=ci)](https://github.com/goliajp/rust-instructor/actions/workflows/ci.yml)
[![License](https://img.shields.io/crates/l/instructors?style=flat-square)](#license)

**English** | [简体中文](README.zh-CN.md) | [日本語](README.ja.md)

> **Upgrading to 1.5?** The API only grows, but four requests change on the wire. `Client::anthropic` now sends the schema as structured output (`output_config.format`) instead of a forced tool call, because Claude Opus 5.5, Sonnet 5.5 and Fable 5.1 and later reject forced tool calls; `.with_anthropic_tool_use()` restores the old request. `extract_many` asks for `{"items": [...]}` instead of a bare array. A max-tokens setting now reaches OpenAI-family APIs. The Gemini key moves from the URL to the `x-goog-api-key` header. Details: [CHANGELOG](CHANGELOG.md#150---2026-11-25).

Typed structured output from LLMs, in Rust. Derive `Deserialize` and `JsonSchema` on a type, call `extract`, and get a value of that type back. The crate sends the type's JSON schema in the form each provider enforces, repairs malformed JSON, and when parsing or your own validation fails, sends the error back to the model and asks again. It follows the design of Python's [instructor](https://github.com/567-labs/instructor).

On a loopback benchmark with an 8-item invoice, one `extract` call costs 2.3 µs of client CPU time above a bare `reqwest` request, against 25.1 µs for rstructor 0.6.0 and 28.4 µs for rig-core 0.47.1. Streaming partial values for a 1.2 KB answer in 308 deltas costs 5.5 µs in total, against 677 µs for the reparse-per-delta approach rstructor uses. Conditions and full tables are under [Performance](#performance).

## Quick start

```toml
[dependencies]
instructors = "1"
```

```rust
use instructors::prelude::*;

#[derive(Debug, Deserialize, JsonSchema)]
struct Contact {
    name: String,
    email: Option<String>,
}

let client = Client::openai("sk-...");
let contact: Contact = client
    .extract("Contact John Doe at john@example.com")
    .await?
    .value;
```

## Providers

Three wire protocols. The `_compatible` constructors point one of them at another base URL.

| Constructor | Protocol | How the schema is sent |
|---|---|---|
| `Client::openai(key)` | OpenAI Chat Completions | `response_format` with a strict `json_schema` |
| `Client::openai_compatible(key, url)` | same | same; the server must support strict `json_schema` |
| `Client::anthropic(key)` | Anthropic Messages | `output_config.format` (structured output) |
| `Client::anthropic_compatible(key, url)` | same | a tool with forced `tool_choice` |
| `Client::gemini(key)` | Gemini `generateContent` | `generationConfig.responseSchema` |
| `Client::gemini_compatible(key, url)` | same | same |

`openai_compatible` reads the host of the base URL: `api.openai.com` gets `max_completion_tokens`; DeepSeek (`deepseek.com`), Qwen (`aliyuncs.com`), Kimi (`moonshot.cn`, `moonshot.ai`) and GLM (`bigmodel.cn`, `z.ai`) get their own [reasoning parameters](#reasoning); any other host is treated as a generic OpenAI-compatible server.

A type that refers to itself (a tree, a linked list) is sent with `$defs` and `$ref`. Anthropic structured output cannot express recursion, so for such a type the Anthropic request uses a tool with `tool_choice: auto`, and Gemini gets `responseJsonSchema` instead of `responseSchema`.

Default models, used when none is set:

| Provider | Default |
|---|---|
| OpenAI and OpenAI-compatible | `gpt-5.6-terra` |
| Anthropic | `claude-sonnet-5` |
| Gemini | `gemini-3.6-flash` |

Model generations turn over about twice a year and retired ids stop working, so these defaults move with minor releases. Set `.with_model("...")` to keep one model across upgrades.

## Reasoning

One setting for how much the model thinks, on the client or per request:

```rust
use instructors::{Effort, Reasoning};

let client = Client::gemini("AIza...").with_reasoning(Reasoning::Effort(Effort::Low));

let r = client
    .extract::<Invoice>(text)
    .reasoning(Reasoning::Budget(2048))
    .await?;
```

Nothing is sent unless it is set. Each provider gets its own parameter:

| Provider | `Effort(e)` | `Budget(n)` | `Off` |
|---|---|---|---|
| OpenAI, generic OpenAI-compatible | `reasoning_effort: e` | error before sending | `reasoning_effort: "none"` |
| Anthropic | `thinking: {type: "adaptive"}` and `output_config.effort: e` | `thinking: {type: "enabled", budget_tokens: n}` | `thinking: {type: "disabled"}` |
| Gemini | `thinkingConfig.thinkingLevel` | `thinkingConfig.thinkingBudget: n` | `thinkingConfig.thinkingBudget: 0` |
| DeepSeek, Kimi, GLM | `thinking: {type: "enabled"}` and `reasoning_effort` of `low`, `high` or `max` | error before sending | `thinking: {type: "disabled"}` |
| Qwen | `enable_thinking: true` | `enable_thinking: true`, `thinking_budget: n` | `enable_thinking: false` |

`Effort` has six levels: `Minimal`, `Low`, `Medium`, `High`, `XHigh`, `Max`. Anthropic has no `minimal`, so it gets `low`; Gemini has no `MAX`, so it gets `XHIGH`; DeepSeek, Kimi and GLM take three levels, so `Minimal` and `Low` become `low`, `Medium` and `High` become `high`, `XHigh` and `Max` become `max`.

The crate keeps no list of which model accepts which value. A value the model rejects comes back as `Error::Api` with the provider's message. Some examples from the providers' documentation as of November 2026: Claude Opus 5.5 rejects `thinking: disabled` and `budget_tokens`; Claude Sonnet 5.5 rejects `thinking: disabled`; Gemini 4 models cannot turn thinking off.

On Anthropic, a forced tool call cannot be combined with thinking, so when a reasoning setting is present the tool path (`anthropic_compatible`, or a recursive type) uses `tool_choice: auto`.

Thinking tokens are billed as output. They are included in `usage.output_tokens` and the cost estimate for all three protocols.

## Streaming

`on_partial` calls your closure after every streamed delta with the JSON parsed so far:

```rust
let article: Article = client
    .extract("...")
    .on_partial(|v| {
        if let Some(title) = v.get("title").and_then(|t| t.as_str()) {
            println!("title so far: {title}");
        }
    })
    .await?
    .value;
```

The value is a borrowed view into an incremental parser that reads each byte once, so the work for the whole stream grows linearly with its length. Unfinished strings show the characters received so far, and `is_complete()` tells whether a value has closed. `to_value()` copies the view into a `serde_json::Value`. The parser is public as [`instructors::partial::PartialParser`](https://docs.rs/instructors/latest/instructors/partial/) for use outside `extract`.

`on_stream` gives the raw text deltas instead. With either hook the request is streamed; the final result is parsed from the whole answer as usual. For OpenAI the streaming request asks for usage (`stream_options.include_usage`), so token counts and cost are filled in.

## Errors you can act on

| Situation | What you get |
|---|---|
| The provider returned a non-2xx status | `Error::Api { status, message }` with the response body |
| The answer did not parse or failed validation after every retry | `Error::ExtractionFailed` or `Error::ValidationFailed` with the last error |
| The model refused (OpenAI `refusal`, Anthropic `stop_reason: refusal`, Gemini safety stop) | `err.refusal()` returns the refusal text; not retried |
| The answer hit the output token limit (OpenAI `finish_reason: length`, Anthropic `max_tokens`, Gemini `MAX_TOKENS`) and did not parse | `err.is_truncated()` is true; not retried and not repaired |
| The whole extraction took longer than the timeout | `Error::Timeout` |

With fallbacks, the error keeps the primary provider's variant and its message lists each fallback's error.

## Validation and retry

```rust
let user: User = client
    .extract("...")
    .validate(|u: &User| {
        if u.age > 150 { Err("age must be <= 150".into()) } else { Ok(()) }
    })
    .await?
    .value;
```

Or implement `Validate` on the type and call `.validated()`. When parsing or validation fails, the next attempt carries the error message, up to `max_retries` (default 2). Before a retry is spent, malformed JSON (trailing commas, single quotes, unquoted keys, a markdown fence around the JSON) is repaired with [jsonrepair](https://crates.io/crates/jsonrepair).

## Lists, batches, history, images

```rust
// a list: the schema root is {"items": [...]}, the result is Vec<Entity>
let entities: Vec<Entity> = client.extract_many("Tim Cook met Sundar Pichai").await?.value;

// many prompts, at most 5 requests in flight; results keep the input order
let results = client.extract_batch::<Review>(prompts).concurrency(5).run().await;

// conversation history before the prompt
let r = client
    .extract::<Summary>("summarize the above")
    .messages(vec![Message::user("..."), Message::assistant("...")])
    .await?;

// images, by URL or base64
let r = client
    .extract::<Description>("describe this")
    .image(ImageInput::Url("https://example.com/photo.jpg".into()))
    .await?;
```

## Fallback, backoff, timeout

```rust
use std::time::Duration;

let client = Client::openai("sk-...")
    .with_fallback(Client::anthropic("sk-ant-..."))
    .with_retry_backoff(BackoffConfig::default()) // 429, 500, 502, 503, 504, 529
    .with_timeout(Duration::from_secs(120));      // the whole extraction
```

A fallback is tried after the primary provider has used up its retries, with the fallback client's own default model. Backoff is off unless configured; `BackoffConfig::default()` waits 500 ms doubling up to 30 s, with jitter, for at most 3 HTTP retries.

## Settings

Set defaults on the client and override them per request:

```rust
let client = Client::openai("sk-...")
    .with_model("gpt-5.6-luna")
    .with_system("Extract data precisely.")
    .with_temperature(0.0)
    .with_max_tokens(4096)
    .with_max_retries(3);

let r: Contact = client
    .extract("...")
    .model("gpt-5.6-terra")
    .context("extra context appended to the prompt")
    .on_request(|model, prompt| println!("{model}: {} bytes", prompt.len()))
    .on_response(|usage| println!("{} tokens", usage.total_tokens))
    .await?
    .value;
```

| Setting | OpenAI family | Anthropic | Gemini |
|---|---|---|---|
| max tokens | `max_completion_tokens` (OpenAI) or `max_tokens` (compatible); not sent unless set | `max_tokens`, 16384 unless set | `maxOutputTokens`, 16384 unless set |
| temperature | sent; 0.0 by default | never sent: current Claude models reject a non-default value | sent when set |

## Cost

```rust
let r = client.extract::<Contact>("...").await?;
println!("{} in, {} out, {:?} USD", r.usage.input_tokens, r.usage.output_tokens, r.usage.cost);
```

Cost comes from the [tiktoken](https://crates.io/crates/tiktoken) price table (feature `cost-tracking`, on by default; no vocabulary data is linked). It is computed per request and summed over retries, and is `None` when the table does not know the model. Model ids are resolved by tiktoken, so a provider's spelling of an id (dashes, date suffixes, Bedrock or Vertex prefixes) still prices.

```toml
instructors = { version = "1", default-features = false } # no cost tracking
```

## Field order

By default the schema lists a struct's fields in alphabetical order, because schemars stores properties in a `serde_json::Map`, which is sorted unless `serde_json/preserve_order` is enabled. OpenAI generates fields in the order the schema lists them, so a `reasoning` field declared before `answer` is written after it. The `preserve-order` feature keeps declaration order for every provider:

```toml
instructors = { version = "1", features = ["preserve-order"] }
```

It turns on `serde_json/preserve_order`, which applies to every crate in your build that uses `serde_json::Map`: their maps keep insertion order instead of sorting, and `indexmap` is added to the build.

## Performance

Measured on 2026-11-25 on an Apple M4 Pro Mac mini (14 cores) holding an exclusive benchmark lock, release build with LTO and one codegen unit, rustc 1.100.0. The 1-minute load average, sampled every 15 s, stayed between 1.6 and 4.8 for the whole run, builds included. Each figure is the median of 5 to 7 rounds. The 1.4.1 and 1.5 builds were run one after the other, twice; the tables show the first pass, and the second pass agreed within 1.2 µs on the overhead column and within 4 % elsewhere. Payload: an `Invoice` with two nested addresses, an enum, eight line items with optional fields, CJK text and escaped quotes, 1,247 bytes of JSON. Everything runs against a loopback HTTP server; no API is called. The benchmark program is not in this repository.

**One `extract` call, end to end** (sequential calls over a kept-alive HTTP/1.1 connection, 5 × 5,000 calls; the floor is the same `reqwest` client posting a prebuilt body and parsing the reply with `serde_json`, measured in the same process):

| Client | Per call | Above the floor |
|---|---:|---:|
| floor | 50.7 µs | |
| instructors 1.5 | 53.0 µs | 2.3 µs |
| instructors 1.4.1 (its own pass, floor 49.1 µs) | 66.4 µs | 17.3 µs |
| rstructor 0.6.0 | 75.8 µs | 25.1 µs |
| rig-core 0.47.1 (`output_schema`, then `serde_json::from_str`) | 79.1 µs | 28.4 µs |

A real model call takes hundreds of milliseconds or more, so these microseconds matter for CPU cost in large batches and in proxies, not for the latency of one call.

**Schema work per request:**

| What | Time |
|---|---:|
| instructors 1.5: copy the cached, provider-ready schema bytes into the request body | 30 ns |
| instructors 1.4.1: transform the cached schema for OpenAI, then serialize it | 10.8 µs + 1.1 µs |
| rstructor 0.6.0: `Invoice::schema().to_json()` | 7.7 µs |
| `schemars::schema_for!(Invoice)` without any cache, for scale | 3.8 µs |

**Partial values while streaming** (the 1,247-byte answer arriving in 308 deltas of 4 bytes, a partial value after every delta, total for the whole stream):

| Approach | Whole stream |
|---|---:|
| instructors 1.5 `PartialParser`: feed each delta, read the borrowed view | 5.5 µs |
| the same, plus a copy to `serde_json::Value` after every delta | 307 µs |
| reparse the whole buffer after each delta into a `Value` (rstructor 0.6.0's method) | 677 µs |
| partial-json-fixer 0.5.5 on the whole buffer, then parse | 735 µs |
| jsonrepair 0.1.0 on the whole buffer, then parse | 2,035 µs |
| for scale: one `serde_json::from_str::<Invoice>` of the finished answer | 1.6 µs |

The parser does the same work no matter how the bytes are split, so the cost of a stream grows linearly with its length; the reparse approaches grow with the square of the length.

## License

Licensed under either of [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at your option.
