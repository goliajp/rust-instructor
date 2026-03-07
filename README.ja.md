# instructors

[![Crates.io](https://img.shields.io/crates/v/instructors?style=flat-square&logo=rust)](https://crates.io/crates/instructors)
[![docs.rs](https://img.shields.io/docsrs/instructors?style=flat-square&logo=docs.rs)](https://docs.rs/instructors)
[![License](https://img.shields.io/crates/l/instructors?style=flat-square)](LICENSE)

[English](README.md) | [简体中文](README.zh-CN.md) | **日本語**

LLMからの型安全な構造化出力抽出。Rust版 [instructor](https://github.com/jxnl/instructor)。

Rust構造体を定義 → instructorsがJSON Schemaを自動生成 → LLMが準拠するJSONを返却 → 型付きの値に直接デシリアライズ。パース失敗時は自動リトライ。

## クイックスタート

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

## インストール

```toml
[dependencies]
instructors = "0.1"
```

## 対応プロバイダー

| プロバイダー | コンストラクタ | メカニズム |
|---|---|---|
| OpenAI | `Client::openai(key)` | `response_format` 厳密JSON Schema |
| Anthropic | `Client::anthropic(key)` | `tool_use` 強制ツール選択 |
| OpenAI互換 | `Client::openai_compatible(key, url)` | OpenAIと同じ（DeepSeek、Togetherなど） |

```rust
// OpenAI
let client = Client::openai("sk-...");

// Anthropic
let client = Client::anthropic("sk-ant-...");

// DeepSeek、Together、その他OpenAI互換API
let client = Client::openai_compatible("sk-...", "https://api.deepseek.com/v1");
```

## 分類タスク

enumは分類タスクに自然に適合します：

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

## ネスト型

Vec、Option、enumを含む複雑なネスト構造に対応：

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

## 設定

```rust
let result: MyStruct = client
    .extract("input text")
    .model("gpt-4o-mini")            // モデル指定
    .system("You are an expert...")   // カスタムシステムプロンプト
    .temperature(0.0)                 // 確定的出力
    .max_tokens(2048)                 // 出力トークン数制限
    .max_retries(3)                   // パース失敗時のリトライ回数
    .context("extra context...")      // 追加コンテキスト
    .await?
    .value;
```

## クライアントデフォルト

デフォルト値を一度設定し、個別リクエストで必要に応じてオーバーライド：

```rust
let client = Client::openai("sk-...")
    .with_model("gpt-4o-mini")
    .with_temperature(0.0)
    .with_max_retries(3)
    .with_system("Extract data precisely.");

// すべての抽出リクエストで上記のデフォルト値を使用
let a: TypeA = client.extract("...").await?.value;
let b: TypeB = client.extract("...").await?.value;

// 特定のリクエストでオーバーライド
let c: TypeC = client.extract("...").model("gpt-4o").await?.value;
```

## コスト追跡

[tiktoken](https://crates.io/crates/tiktoken) によるトークンカウントとコスト推定を内蔵：

```rust
let result = client.extract::<Contact>("...").await?;

println!("input:  {} tokens", result.usage.input_tokens);
println!("output: {} tokens", result.usage.output_tokens);
println!("cost:   ${:.6}", result.usage.cost.unwrap_or(0.0));
println!("retries: {}", result.usage.retries);
```

不要な場合は `default-features = false` で無効化：

```toml
[dependencies]
instructors = { version = "0.1", default-features = false }
```

## 仕組み

1. `#[derive(JsonSchema)]` が [schemars](https://crates.io/crates/schemars) を通じてRust型からJSON Schemaを生成
2. Schemaが対象プロバイダー向けに変換される：
   - **OpenAI**: `response_format` にラップ、strictモード有効（`additionalProperties: false`、全フィールドrequired）
   - **Anthropic**: `tool` としてラップ、`tool_choice` で強制実行
3. LLMはSchemaに準拠した有効なJSONのみを出力するよう制約される
4. レスポンスは `serde_json::from_str::<T>()` でデシリアライズ
5. パース失敗時、エラー情報をLLMにフィードバックしてリトライ

## ライセンス

[MIT](LICENSE)
