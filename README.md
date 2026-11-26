# instructors

[![crates.io](https://img.shields.io/crates/v/instructors?style=flat-square&logo=rust)](https://crates.io/crates/instructors)
[![docs.rs](https://img.shields.io/docsrs/instructors?style=flat-square&logo=docs.rs)](https://docs.rs/instructors)
[![CI](https://img.shields.io/github/actions/workflow/status/goliajp/rust-instructor/ci.yml?branch=develop&style=flat-square&logo=github&label=ci)](https://github.com/goliajp/rust-instructor/actions/workflows/ci.yml)
[![License](https://img.shields.io/crates/l/instructors?style=flat-square)](#license)

**English** | [简体中文](README.zh-CN.md) | [日本語](README.ja.md)

> **Upgrading from 1.x?** Derive `Schema` instead of `JsonSchema`, add a `_ =>` arm when matching `Error`, and call `usage.total_tokens()` as a method. Schema properties now follow declaration order, HTTP retries are on by default, and a re-ask shows the model its previous answer. The full list with before and after code is in the [CHANGELOG](CHANGELOG.md#upgrading-from-1x).

Typed structured output from LLMs, in Rust. Derive `Deserialize` and `Schema` on a type, call `extract`, and get a value of that type back. The crate sends the type's JSON schema in the form each provider enforces, repairs malformed JSON, and when parsing or your own validation fails, shows the model its answer and the error and asks again. It follows the design of Python's [instructor](https://github.com/567-labs/instructor).

The crate has no third-party runtime dependency of its own: the schema derive, the JSON parser, the repair and the streaming parser are part of it, and serde is the only interface your types need. The HTTP client is a trait; the default implementation is reqwest with rustls. With `default-features = false` the crate builds requests and parses responses without doing any I/O and without an async runtime, and builds for `wasm32-unknown-unknown`.

Parsing an OpenAI response body into a 1.2 KB `Invoice` takes 3.0 µs; with `serde_json` it takes 4.1 µs. One complete `extract` call over loopback HTTP takes about 51 µs, as long as a plain `reqwest` POST and parse in the same process; rstructor 0.6.0 takes 76 µs and rig-core 0.47.1 takes 80 µs. Conditions and all rows are under [Performance](#performance).

## Quick start

```toml
[dependencies]
instructors = "2"
```

```rust
use instructors::prelude::*;

#[derive(Debug, Deserialize, Schema)]
struct Contact {
    /// Full name as written in the text.
    name: String,
    email: Option<String>,
}

let client = Client::openai("sk-...");
let contact: Contact = client
    .extract("Contact John Doe at john@example.com")
    .await?
    .value;
```

Doc comments become `description`s in the schema. Fields keep their declaration order, which matters: OpenAI writes fields in schema order, so a `reasoning` field placed before `answer` is generated first.

## Schema

`#[derive(Schema)]` reads the type's serde attributes, so the schema describes the JSON the `Deserialize` impl accepts: `rename`, `rename_all`, `default`, `skip`, `flatten`, `tag`, `content`, `untagged`, `transparent` and `deny_unknown_fields`. A serde attribute that changes the JSON in a way the derive cannot see (`deserialize_with`, `with`, `try_from`, `from`) is a compile error unless you say what the JSON looks like with `#[schema(like = SomeType)]`.

```rust
#[derive(Deserialize, Schema)]
#[serde(rename_all = "camelCase")]
struct Event {
    #[schema(description = "ISO 8601", format = "date-time")]
    starts_at: String,
    #[schema(min = 1, max = 5)]
    priority: u8,
    #[serde(default)]
    tags: Vec<String>,
}
```

The derive is built on `syn` 3 with exactly the features `serde_derive` uses, so a crate that already derives `Deserialize` compiles no additional crate for it. Types that only implement `schemars::JsonSchema` work through feature `schemars` and `client.extract_schemars::<T>(..)`.

## Providers

| Constructor | Protocol | How the schema is sent by default |
|---|---|---|
| `Client::openai(key)` | OpenAI Chat Completions | `response_format` with a strict `json_schema` |
| `Client::openai(key).with_api(OpenAiApi::Responses)` | OpenAI Responses | `text.format` with a strict `json_schema` |
| `Client::openai_compatible(key, url)` | Chat Completions | strict `json_schema`; JSON mode for DeepSeek, Qwen and GLM hosts |
| `Client::deepseek(key)` | DeepSeek Chat Completions | JSON mode (`response_format: json_object`) |
| `Client::qwen(key)` | Qwen, Model Studio OpenAI-compatible mode (Singapore) | JSON mode |
| `Client::anthropic(key)` | Anthropic Messages | `output_config.format` (structured output) |
| `Client::anthropic_compatible(key, url)` | same | a tool with forced `tool_choice` |
| `Client::gemini(key)`, `gemini_compatible(key, url)` | Gemini `generateContent` | `generationConfig.responseFormat.text` with a JSON Schema |

`with_output_mode` changes how the schema is sent:

| `OutputMode` | What is sent |
|---|---|
| `Schema` | the provider's native structured output (the table above) |
| `Tool` | a tool whose input is the schema, forced where the model allows it; not on Gemini or the Responses API |
| `JsonObject` | JSON mode (`response_format: json_object`) with the schema in the system prompt; Anthropic, which has no JSON mode, gets `Prompt` |
| `Prompt` | no format parameter; the schema in the system prompt, the JSON taken from the answer text |

`openai_compatible` reads the host of the base URL: `api.openai.com` gets `max_completion_tokens`; DeepSeek (`deepseek.com`), Qwen (`aliyuncs.com`), Kimi (`moonshot.cn`, `moonshot.ai`, `kimi.ai`, `kimi.com`) and GLM (`bigmodel.cn`, `z.ai`) get their own [reasoning parameters](#reasoning). DeepSeek, Qwen and GLM default to `JsonObject`, because their documentation describes JSON mode and not strict schemas; any other host is treated as a generic OpenAI-compatible server and gets `Schema`.

A type that refers to itself (a tree, a linked list) is sent with `$defs` and `$ref`. Anthropic structured output cannot express recursion, so for such a type the Anthropic request uses a tool. Gemini unrolls a recursive reference only a few levels and documents it for properties that are not required; a required recursive field such as `children: Vec<Node>` is sent as is and may be rejected or cut short there.

OpenAI strict schemas and Anthropic structured output take a subset of JSON Schema, so the crate sends some types in another form and turns the answer back before it reaches your type: a map (`HashMap`, `BTreeMap`) goes as a list of `{"key": .., "value": ..}` objects, a tuple as an object with the keys `"0"`, `"1"`, …, and a type whose JSON is not an object (an enum, a `Vec`, a number) inside `{"value": ..}`. Constraints the provider does not take (`uniqueItems` everywhere; `minLength` and `maxLength` on OpenAI; bounds and `maxItems` on Anthropic) are left out of its schema and checked on the answer, and an answer that breaks one is asked again like any other invalid answer. `on_item` and the final value always see your type's own JSON; `on_partial` sees the root unwrapped, but maps and tuples inside it in the sent form.

A type that holds a value of any type (a field whose schema is `true`: a hand-written `Schema` returning `Shape::Any`, or `serde_json::Value` through feature `schemars`) has no strict schema. OpenAI then gets the plain schema with `strict: false`, Anthropic gets a tool, and the answer is checked against the schema here; Gemini gets an empty schema object for that field. A tool's `input_schema` and a plain OpenAI schema must be objects too, so a root that is not an object is wrapped in `{"value": ..}` there as well. Gemini ignores `minLength`, `maxLength` and `uniqueItems`, so its schema leaves them out and the answer is checked for them in the same way.

Anthropic models without structured output (Claude 3, Claude Opus 4 and 4.1, Claude Sonnet 4) get the tool instead; any other model, including ones newer than the crate, gets `output_config.format`. Claude Opus 5.5, Sonnet 5.5, Fable 5.1 and 5.2 and Mythos 5.1 and 5.2 reject a forced tool, so on those models, and whenever thinking is on, the tool is offered with `tool_choice: auto` and the system prompt asks for the call. DeepSeek and Qwen reject a forced tool in thinking mode, which their current models use by default, so their `Tool` mode does the same unless reasoning is `Off`. Qwen's OpenAI-compatible mode takes no PDF, so `Part::document` returns `Error::Unsupported` there before anything is sent.

Default models, used when none is set:

| Provider | Default |
|---|---|
| OpenAI and OpenAI-compatible | `gpt-5.6-terra` |
| DeepSeek (`deepseek.com` hosts) | `deepseek-flash` |
| Qwen (`aliyuncs.com` hosts) | `qwen3.7-plus` |
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

On Anthropic, a forced tool call cannot be combined with thinking, so with `Effort` or `Budget` the tool path (`OutputMode::Tool`, the default of `anthropic_compatible`, or a recursive type) uses `tool_choice: auto`; with `Off` the tool is forced where the model allows it.

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

The value is a borrowed view into an incremental parser that reads each byte once, so the work for the whole stream grows linearly with its length. Unfinished strings show the characters received so far, and `is_complete()` tells whether a value has closed. `v.parse::<Snapshot>()` deserializes the value so far into a type of your own (usually one with `Option` fields), and `to_json()` gives its JSON text. The parser is public as [`instructors::partial::PartialParser`](https://docs.rs/instructors/latest/instructors/partial/).

For `extract_many`, `on_item` hands you each list item as soon as it is complete:

```rust
let all: Vec<Entity> = client
    .extract_many("...")
    .on_item(|e: Entity| println!("found {}", e.name))
    .await?
    .value;
```

`on_delta` gives the raw text deltas. With any of these hooks the request is streamed; the final result is parsed from the whole answer as usual, and token counts and cost are filled in for every provider.

## Errors you can act on

| Situation | What you get |
|---|---|
| The provider returned a non-2xx status after the HTTP retries, or sent an error inside the stream | `Error::Api { status, message, retry_after, .. }` |
| The connection failed | `Error::Transport { source, .. }` |
| The answer did not parse, or failed validation, on every attempt | `Error::Parse { attempts, raw, .. }` or `Error::Validation { .. }` |
| The model refused (OpenAI `refusal`, Anthropic `stop_reason: refusal`, Gemini safety stop) | `Error::Refusal { text, .. }`; not retried |
| The answer hit the output token limit and did not parse | `Error::Truncated { reason, raw, .. }`; not retried, unless you called `.accept_truncated()` |
| A setting the provider has no parameter for (`Reasoning::Budget` on OpenAI) | `Error::Unsupported`, before anything is sent |
| The whole extraction took longer than the timeout | `Error::Timeout { after, .. }` |
| Every provider failed | `Error::Fallback`: `chain()` lists each provider's error, newest first, and `source()` links them for anyhow or eyre |

Every error that came after a request carries what the extraction cost: `err.usage()`.

`.accept_truncated()` turns a cut-off answer into a value when it can be done without inventing anything: open strings and containers are closed, an unfinished key or number is dropped, and the result must still deserialize, so it works when the missing fields are `Option` or `#[serde(default)]`. `extract_many` keeps the items that were complete. The result has `truncated: true`.

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

Or implement `Validate` on the type and call `.validated()`. When parsing or validation fails, the next request appends the model's answer and the error to the conversation and asks again, up to `max_attempts` answers (default 3). Before an attempt is spent, malformed JSON (a markdown fence or prose around the value, trailing commas, single quotes, unquoted keys, comments, Python's `True` / `None`) is repaired.

## Lists, batches, history, images, documents

```rust
// a list: the schema root is {"items": [...]}, the result is Vec<Entity>
let entities: Vec<Entity> = client.extract_many("Tim Cook met Sundar Pichai").await?.value;

// many prompts, at most 5 requests in flight; results keep the input order
let results = client.extract_batch::<Review>(prompts).concurrency(5).await;

// conversation history before the prompt
let r = client
    .extract::<Summary>("summarize the above")
    .messages([Message::user("..."), Message::assistant("...")])
    .await?;

// images and PDFs, by URL or base64
let r = client
    .extract::<Invoice>("extract the invoice")
    .part(Part::document("application/pdf", pdf_base64))
    .await?;
```

## Retries, fallback, timeout

```rust
use std::time::Duration;

let client = Client::openai("sk-...")
    .with_fallback(Client::anthropic("sk-ant-..."))
    .with_retry(RetryPolicy::new().max_retries(5))
    .with_timeout(Duration::from_secs(120)); // the whole extraction
```

HTTP retries are on by default: 3 retries on 408, 409, 429, 500, 502, 503, 504, 529 and connection errors, waiting 500 ms doubling up to 30 s with jitter, or what the provider's `retry-after-ms` / `Retry-After` header says (seconds, or an HTTP date counted from the response's `Date` header). An `x-should-retry: true` or `false` header, which OpenAI and Anthropic send, overrides the status. `RetryPolicy::none()` turns them off. A fallback is tried after the primary provider has failed, with the fallback client's own model, output mode and transport.

## Without a runtime, or with your own HTTP client

`.await` drives the request over a `Transport`. The default is reqwest on tokio; any other HTTP client fits through `with_transport`:

```rust
impl Transport for MyHttp {
    fn send(&self, req: HttpRequest) -> BoxFuture<'_, Result<HttpResponse, TransportError>> { ... }
    fn sleep(&self, d: Duration) -> BoxFuture<'static, ()> { ... }
}
let client = Client::openai("sk-...").with_transport(MyHttp::new());
```

Or skip async entirely. `exchange()` returns the extraction as a state machine that says what to send and how long to wait:

```rust
let mut ex = client.extract::<Contact>("...").exchange();
let result = loop {
    match ex.next() {
        Step::Send(req) => {
            let (status, headers, body) = my_http(&req)?; // req.url, req.headers, req.body
            ex.receive(status, &headers, &body);
        }
        Step::Wait(d) => std::thread::sleep(d),
        Step::Done(r) => break r,
    }
};
```

With `default-features = false, features = ["derive"]`, nothing in the build does I/O, and the crate compiles for `wasm32-unknown-unknown`.

## Settings

Set defaults on the client and override them per request:

```rust
let client = Client::openai("sk-...")
    .with_model("gpt-5.6-luna")
    .with_system("Extract data precisely.")
    .with_temperature(0.0)
    .with_max_tokens(4096)
    .with_max_attempts(4);

let r: Contact = client
    .extract("...")
    .model("gpt-5.6-terra")
    .context("extra context appended to the prompt")
    .on_request(|req| println!("{} {} bytes", req.url, req.body.len()))
    .on_response(|usage| println!("{} tokens", usage.total_tokens()))
    .await?
    .value;
```

| Setting | OpenAI family | Anthropic | Gemini |
|---|---|---|---|
| max tokens | `max_completion_tokens` (OpenAI), `max_output_tokens` (Responses) or `max_tokens` (compatible); not sent unless set | `max_tokens`, 16384 unless set | `maxOutputTokens`, 16384 unless set |
| temperature | sent when set | never sent: current Claude models reject a non-default value | sent when set |

## Cost

```rust
let r = client.extract::<Contact>("...").await?;
println!("{} in, {} out, {:?} USD", r.usage.input_tokens, r.usage.output_tokens, r.usage.cost);
```

Cost comes from the [tiktoken](https://crates.io/crates/tiktoken) price table (feature `cost-tracking`, on by default; no vocabulary data is linked). It is computed per request and summed over retries, and is `None` when the table does not know the model. Model ids are resolved by tiktoken, so a provider's spelling of an id (dashes, date suffixes, Bedrock or Vertex prefixes) still prices.

```toml
instructors = { version = "2", default-features = false, features = ["derive", "reqwest"] } # no cost tracking
```

## Performance

Measured on 2026-11-26 on an Apple M4 Pro Mac mini (14 cores) holding an exclusive benchmark lock, release build with LTO and one codegen unit, rustc 1.100.0, serde 1.0.229, serde_json 1.0.154, reqwest 0.13.5. The 1-minute load average, sampled every 15 s, was between 1.5 and 3.1 during the run. Every row was measured in two passes a few seconds apart, and the tables show the median of each pass (5 rounds for the end-to-end rows, 7 or 9 for the others). Payload: an `Invoice` with two nested addresses, an enum, eight line items with optional fields, CJK text and escaped quotes, 1,247 bytes of JSON. All requests go to a loopback HTTP server; no API is called. The JSON and repair rows come from `cargo bench --bench json --features fuzzing`; the program for the other rows is not in this repository.

**One `extract` call, end to end** (OpenAI wire, sequential calls over a kept-alive HTTP/1.1 connection, 5 × 5,000 calls). The floor is the same version of `reqwest` posting a prebuilt body and parsing the response with `serde_json`.

| Client | Per call, pass 1 / pass 2 |
|---|---:|
| floor | 50.5 / 51.9 µs |
| instructors 2.0 | 51.2 / 51.4 µs |
| instructors 1.5.1 | 53.3 / 53.1 µs |
| rstructor 0.6.0 | 76.4 / 76.2 µs |
| rig-core 0.47.1 (`output_schema`, then `serde_json::from_str`) | 79.5 / 79.7 µs |

The floor itself varies by 1.4 µs between passes, so 2.0 and the floor cannot be told apart here; 2.0 was 2.0 and 1.7 µs faster than 1.5.1. For an Anthropic tool answer, 2.0 took 50.3 / 51.6 µs and 1.5.1 took 51.4 / 50.9 µs, no difference. For an answer streamed in 311 events with a partial value after each, 2.0 took 97.2 / 97.1 µs and 1.5.1 took 98.8 / 98.0 µs. A real model call takes hundreds of milliseconds or more, so these times matter for CPU cost in large batches and in proxies.

Without the network, building the request and parsing the answer of one OpenAI extraction takes 3.6 µs, of which building the request is 0.38 / 0.36 µs. The streaming partial parser takes 5.79 / 5.63 µs for the whole answer in 308 deltas of 4 bytes with a view after each; 1.5.1 took 5.66 / 5.66 µs. The parser source is the same in both versions. Run in separate processes, four times each and alternating, the two versions took 5.57 to 5.73 µs (2.0) and 5.65 to 5.88 µs (1.5.1), and on x86_64 Linux the two programs execute the same number of instructions to within 0.001 % (`perf stat`), so the gap in a single run is run-to-run variation, not code.

**Strict answers that need converting.** A type with a map (the `Complex` type of the schema benchmark, three entries) is sent to OpenAI as a list of key and value pairs, and the answer is turned back before it is parsed. The answer is rewritten into the type's JSON in one pass and then parsed. Handling that answer (the full exchange minus building the request) took 2.75 / 2.69 µs; the same type through JSON mode, where the answer is parsed directly, took 1.32 / 1.31 µs. The rewrite alone took 1.26 / 1.19 µs (it was 3.1 µs when the answer was first built into a tree). Only types that need it pay this; a type without a map, tuple, non-object root or dropped constraint is parsed directly.

**The crate's JSON code against serde_json**, pass 1 / pass 2:

| What | serde_json | instructors 2.0 |
|---|---:|---:|
| content → `Invoice` | 1,591 / 1,558 ns | 1,337 / 1,359 ns |
| OpenAI response body → `Invoice`, borrowed envelope | 4,093 / 4,080 ns | 2,969 / 2,961 ns |
| Anthropic tool answer → `Invoice` | 3,057 / 3,058 ns | 2,520 / 2,491 ns |
| one streamed chunk, envelope only | 228 / 229 ns | 185 / 187 ns |
| write a 4,167-byte request into a reused buffer | 1,358 / 1,326 ns | 804 / 805 ns |

**Repair**, input to repaired text, pass 1 / pass 2:

| Input | jsonrepair 0.1.0 | instructors 2.0 |
|---|---:|---:|
| valid JSON | 2.63 / 2.64 µs | 0.84 / 0.85 µs |
| trailing commas | 7.86 / 7.75 µs | 2.92 / 2.94 µs |
| single quotes | 6.22 / 6.17 µs | 3.20 / 3.18 µs |
| markdown fence | 2.66 / 2.64 µs | 0.87 / 0.86 µs |
| cut off at 2/3 of the length | 6.65 / 6.77 µs | 1.97 / 1.96 µs |

In both libraries the first four inputs deserialize into `Invoice` after the repair. The cut-off input does not, because required fields are missing; `accept_truncated` covers types whose missing fields are optional.

**Crates in a build** (`cargo tree -e normal`, duplicates removed, the crate itself included, x86_64 Linux): 89 by default (1.5.1: 111), and 9 with `default-features = false, features = ["derive"]`, of which 7 are serde and what its derive uses. 1.5.1 without default features had 108.

## License

Licensed under either of [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at your option.
