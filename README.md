# instructors

[![Crates.io](https://img.shields.io/crates/v/instructors?style=flat-square&logo=rust)](https://crates.io/crates/instructors)
[![docs.rs](https://img.shields.io/docsrs/instructors?style=flat-square&logo=docs.rs)](https://docs.rs/instructors)
[![License](https://img.shields.io/crates/l/instructors?style=flat-square)](LICENSE)

Type-safe structured output extraction from LLMs. The Rust [instructor](https://github.com/jxnl/instructor).

Define a Rust struct → instructors generates the JSON Schema → LLM returns valid JSON → you get a typed value. With automatic retry on parse failure.

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
instructors = "0.1"
```

## Providers

| Provider | Constructor | Mechanism |
|---|---|---|
| OpenAI | `Client::openai(key)` | `response_format` strict JSON Schema |
| Anthropic | `Client::anthropic(key)` | `tool_use` with forced tool choice |
| OpenAI-compatible | `Client::openai_compatible(key, url)` | Same as OpenAI (DeepSeek, Together, etc.) |

```rust
// OpenAI
let client = Client::openai("sk-...");

// Anthropic
let client = Client::anthropic("sk-ant-...");

// DeepSeek, Together, or any OpenAI-compatible API
let client = Client::openai_compatible("sk-...", "https://api.deepseek.com/v1");
```

## Classification

Enums work naturally for classification tasks:

```rust
#[derive(Debug, Deserialize, JsonSchema)]
enum Sentiment {
    Positive,
    Negative,
    Neutral,
}

let sentiment: Sentiment = client
    .extract("This product is amazing!")
    .await?
    .value;
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

let paper: Paper = client
    .extract(&pdf_text)
    .model("gpt-4o")
    .await?
    .value;
```

## Configuration

```rust
let result: MyStruct = client
    .extract("input text")
    .model("gpt-4o-mini")            // override model
    .system("You are an expert...")   // custom system prompt
    .temperature(0.0)                 // deterministic output
    .max_tokens(2048)                 // limit output tokens
    .max_retries(3)                   // retry on parse failure
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

Built-in token counting and cost estimation via [tiktoken](https://github.com/goliajp/tokenrs):

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
instructors = { version = "0.1", default-features = false }
```

## How It Works

1. `#[derive(JsonSchema)]` generates a JSON Schema from your Rust type (via [schemars](https://crates.io/crates/schemars))
2. The schema is transformed for the target provider:
   - **OpenAI**: wrapped in `response_format` with strict mode (`additionalProperties: false`, all fields required)
   - **Anthropic**: wrapped as a `tool` with `input_schema`, forced via `tool_choice`
3. LLM is constrained to produce valid JSON matching the schema
4. Response is deserialized with `serde_json::from_str::<T>()`
5. On parse failure, error feedback is sent back and the request is retried

## License

[MIT](LICENSE)
