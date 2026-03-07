# instructors

[![Crates.io](https://img.shields.io/crates/v/instructors?style=flat-square&logo=rust)](https://crates.io/crates/instructors)
[![docs.rs](https://img.shields.io/docsrs/instructors?style=flat-square&logo=docs.rs)](https://docs.rs/instructors)
[![License](https://img.shields.io/crates/l/instructors?style=flat-square)](LICENSE)

**English** | [简体中文](README.zh-CN.md) | [日本語](README.ja.md)

Type-safe structured output extraction from LLMs. The Rust [instructor](https://github.com/jxnl/instructor).

Define a Rust struct → instructors generates the JSON Schema → LLM returns valid JSON → you get a typed value. With automatic validation and retry.

## Quick Start

```rust
use instructors::prelude::*;

#[derive(Debug, Deserialize, JsonSchema)]
struct Contact {
    name: String,
    email: Option<String>,
    phone: Option<String>,
}

let client = Client::openai("sk-...");

let result: ExtractResult<Contact> = client
    .extract("Contact John Doe at john@example.com")
    .model("gpt-4o")
    .await?;

println!("{}", result.value.name);      // "John Doe"
println!("{:?}", result.value.email);    // Some("john@example.com")
println!("tokens: {}", result.usage.total_tokens);
```

## Installation

```toml
[dependencies]
instructors = "1"
```

## Providers

| Provider | Constructor | Mechanism |
|---|---|---|
| OpenAI | `Client::openai(key)` | `response_format` strict JSON Schema |
| Anthropic | `Client::anthropic(key)` | `tool_use` with forced tool choice |
| OpenAI-compatible | `Client::openai_compatible(key, url)` | Same as OpenAI (DeepSeek, Together, etc.) |
| Anthropic-compatible | `Client::anthropic_compatible(key, url)` | Same as Anthropic |

```rust
// OpenAI
let client = Client::openai("sk-...");

// Anthropic
let client = Client::anthropic("sk-ant-...");

// DeepSeek, Together, or any OpenAI-compatible API
let client = Client::openai_compatible("sk-...", "https://api.deepseek.com/v1");

// Anthropic-compatible proxy
let client = Client::anthropic_compatible("sk-...", "https://proxy.example.com/v1");
```

## Validation

Validate extracted data with automatic retry — invalid results are fed back to the LLM with error details.

### Closure-based

```rust
let user: User = client.extract("...")
    .validate(|u: &User| {
        if u.age > 150 { Err("age must be <= 150".into()) } else { Ok(()) }
    })
    .await?.value;
```

### Trait-based

```rust
use instructors::prelude::*;

#[derive(Debug, Deserialize, JsonSchema)]
struct Email { address: String }

impl Validate for Email {
    fn validate(&self) -> Result<(), ValidationError> {
        if self.address.contains('@') { Ok(()) }
        else { Err("invalid email".into()) }
    }
}

let email: Email = client.extract("...").validated().await?.value;
```

## List Extraction

Extract multiple items from text with `extract_many`:

```rust
#[derive(Debug, Deserialize, JsonSchema)]
struct Entity {
    name: String,
    entity_type: String,
}

let entities: Vec<Entity> = client
    .extract_many("Apple CEO Tim Cook met Google CEO Sundar Pichai")
    .await?.value;
```

## Batch Processing

Process multiple prompts concurrently with configurable concurrency:

```rust
let prompts = vec!["review 1".into(), "review 2".into(), "review 3".into()];

let results = client
    .extract_batch::<Review>(prompts)
    .concurrency(5)
    .validate(|r: &Review| { /* ... */ Ok(()) })
    .run()
    .await;

// each result is independent — partial failures don't affect others
for result in results {
    match result {
        Ok(r) => println!("{:?}", r.value),
        Err(e) => eprintln!("failed: {e}"),
    }
}
```

## Multi-turn Conversations

Pass message history for context-aware extraction:

```rust
use instructors::Message;

let result = client.extract::<Summary>("summarize the above")
    .messages(vec![
        Message::user("Here is a long document..."),
        Message::assistant("I see the document."),
    ])
    .await?;
```

## Streaming

Stream partial JSON tokens as they arrive:

```rust
let result = client.extract::<Contact>("...")
    .on_stream(|chunk| {
        print!("{chunk}");  // partial JSON fragments
    })
    .await?;
```

Both OpenAI and Anthropic providers support streaming. The final result is assembled from all chunks and deserialized as usual.

## Image Input

Extract structured data from images using vision-capable models:

```rust
use instructors::ImageInput;

// from URL
let result = client.extract::<Description>("Describe this image")
    .image(ImageInput::Url("https://example.com/photo.jpg".into()))
    .model("gpt-4o")
    .await?;

// from base64
let result = client.extract::<Description>("Describe this image")
    .image(ImageInput::Base64 {
        media_type: "image/png".into(),
        data: base64_string,
    })
    .await?;

// multiple images
let result = client.extract::<Comparison>("Compare these images")
    .images(vec![
        ImageInput::Url("https://example.com/a.jpg".into()),
        ImageInput::Url("https://example.com/b.jpg".into()),
    ])
    .await?;
```

## Provider Fallback

Chain multiple providers for automatic failover:

```rust
let client = Client::openai("sk-...")
    .with_fallback(Client::anthropic("sk-ant-..."))
    .with_fallback(Client::openai_compatible("sk-...", "https://api.deepseek.com/v1"));

// tries OpenAI first → Anthropic on failure → DeepSeek as last resort
let result = client.extract::<Contact>("...").await?;
```

Each fallback is tried in order after the primary provider exhausts its retries.

## Lifecycle Hooks

Observe requests and responses:

```rust
let result = client.extract::<Contact>("...")
    .on_request(|model, prompt| {
        println!("[req] model={model}, prompt_len={}", prompt.len());
    })
    .on_response(|usage| {
        println!("[res] tokens={}, cost={:?}", usage.total_tokens, usage.cost);
    })
    .await?;
```

## Classification

Enums work naturally for classification tasks:

```rust
#[derive(Debug, Deserialize, JsonSchema)]
enum Sentiment { Positive, Negative, Neutral }

let sentiment: Sentiment = client
    .extract("This product is amazing!")
    .await?.value;
```

## Nested Types

Complex nested structures with vectors, options, and enums:

```rust
#[derive(Debug, Deserialize, JsonSchema)]
struct Paper {
    title: String,
    authors: Vec<Author>,
    keywords: Vec<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct Author {
    name: String,
    affiliation: Option<String>,
}

let paper: Paper = client.extract(&pdf_text).model("gpt-4o").await?.value;
```

## Configuration

```rust
let result: MyStruct = client
    .extract("input text")
    .model("gpt-4o-mini")            // override model
    .system("You are an expert...")   // custom system prompt
    .temperature(0.0)                 // deterministic output
    .max_tokens(2048)                 // limit output tokens
    .max_retries(3)                   // retry on parse/validation failure
    .context("extra context...")      // append to prompt
    .await?
    .value;
```

## Client Defaults

Set defaults once, override per-request:

```rust
let client = Client::openai("sk-...")
    .with_model("gpt-4o-mini")
    .with_temperature(0.0)
    .with_max_retries(3)
    .with_system("Extract data precisely.");

// all extractions use the defaults above
let a: TypeA = client.extract("...").await?.value;
let b: TypeB = client.extract("...").await?.value;

// override for a specific request
let c: TypeC = client.extract("...").model("gpt-4o").await?.value;
```

## Cost Tracking

Built-in token counting and cost estimation via [tiktoken](https://crates.io/crates/tiktoken):

```rust
let result = client.extract::<Contact>("...").await?;

println!("input:  {} tokens", result.usage.input_tokens);
println!("output: {} tokens", result.usage.output_tokens);
println!("cost:   ${:.6}", result.usage.cost.unwrap_or(0.0));
println!("retries: {}", result.usage.retries);
```

Disable with `default-features = false`:

```toml
[dependencies]
instructors = { version = "1", default-features = false }
```

## How It Works

1. `#[derive(JsonSchema)]` generates a JSON Schema from your Rust type (via [schemars](https://crates.io/crates/schemars))
2. Schema is cached per type (thread-local, zero lock contention)
3. The schema is transformed for the target provider:
   - **OpenAI**: wrapped in `response_format` with strict mode (`additionalProperties: false`, all fields required)
   - **Anthropic**: wrapped as a `tool` with `input_schema`, forced via `tool_choice`
4. LLM is constrained to produce valid JSON matching the schema
5. Response is deserialized with `serde_json::from_str::<T>()`
6. If `Validate` trait or `.validate()` closure is present, validation runs
7. On parse/validation failure, error feedback is sent back and the request is retried

## License

[MIT](LICENSE)
