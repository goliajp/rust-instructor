# instructors

[![Crates.io](https://img.shields.io/crates/v/instructors?style=flat-square&logo=rust)](https://crates.io/crates/instructors)
[![docs.rs](https://img.shields.io/docsrs/instructors?style=flat-square&logo=docs.rs)](https://docs.rs/instructors)
[![License](https://img.shields.io/crates/l/instructors?style=flat-square)](LICENSE)
[![Downloads](https://img.shields.io/crates/d/instructors?style=flat-square)](https://crates.io/crates/instructors)
[![MSRV](https://img.shields.io/badge/MSRV-1.94-blue?style=flat-square)](https://www.rust-lang.org)

[English](README.md) | [简体中文](README.zh-CN.md) | **日本語**

LLM からの型安全な構造化出力抽出ライブラリ。Rust 版 [instructor](https://github.com/jxnl/instructor)。

Rust の構造体を定義 → instructors が JSON Schema を生成 → LLM が有効な JSON を返却 → 型付きの値を取得。自動バリデーションとリトライ機能付き。

## ハイライト

- **6 プロバイダー** — OpenAI、Anthropic、Gemini、DeepSeek、Together、任意の OpenAI/Anthropic/Gemini 互換 API
- **バリデーション + リトライ** — 無効な出力はエラー詳細と共に LLM にフィードバックされ、自動修正
- **JSON 自動修復** — 末尾カンマ、シングルクォート、Markdown フェンスをリトライ前に修復し、トークンとレイテンシを節約
- **プロバイダーフォールバック** — 複数プロバイダーをチェーンして自動フェイルオーバー
- **ストリーミング** — 部分的な JSON トークンをリアルタイム受信（OpenAI、Anthropic、Gemini）
- **バッチ + 並行処理** — `Semaphore` による並行数制御で数百のプロンプトを処理
- **ビジョン** — 画像から構造化データを抽出（URL または base64）
- **コスト追跡** — [tiktoken](https://crates.io/crates/tiktoken) によるリクエストごとのトークンカウントとコスト推定

## なぜ instructors？

| 観点 | 生 API + serde | instructors |
|---|---|---|
| スキーマ強制 | JSON Schema を手書き | `#[derive(JsonSchema)]` から自動生成 |
| パース失敗 | クラッシュまたはデータ損失 | エラーフィードバック付き自動リトライ |
| 不正な JSON | アプリケーションエラー | デシリアライズ前に自動修復 |
| 複数プロバイダー | プロバイダーごとに書き直し | 統一インターフェース、1 行で切替 |
| バリデーション | パース後に手動 if/else | `.validate()` で LLM 対応リトライ |
| コスト追跡 | トークンを自分で計算 | tiktoken による組み込みサポート |

## クイックスタート

```rust
use instructors::prelude::*;

#[derive(Debug, Deserialize, JsonSchema)]
struct Contact { name: String, email: Option<String> }

let client = Client::openai("sk-...");
let contact: Contact = client
    .extract("Contact John Doe at john@example.com")
    .await?.value;
```

## インストール

```toml
[dependencies]
instructors = "1"
```

## プロバイダー

| プロバイダー | コンストラクタ | メカニズム |
|---|---|---|
| OpenAI | `Client::openai(key)` | `response_format` strict JSON Schema |
| Anthropic | `Client::anthropic(key)` | `tool_use` による強制ツール選択 |
| Google Gemini | `Client::gemini(key)` | `response_schema` 構造化 JSON |
| OpenAI 互換 | `Client::openai_compatible(key, url)` | OpenAI と同一方式 (DeepSeek、Together 等) |
| Anthropic 互換 | `Client::anthropic_compatible(key, url)` | Anthropic と同一方式 |
| Gemini 互換 | `Client::gemini_compatible(key, url)` | Gemini と同一方式 |

```rust
// OpenAI
let client = Client::openai("sk-...");

// Anthropic
let client = Client::anthropic("sk-ant-...");

// DeepSeek、Together、その他 OpenAI 互換 API
let client = Client::openai_compatible("sk-...", "https://api.deepseek.com/v1");

// Anthropic 互換プロキシ
let client = Client::anthropic_compatible("sk-...", "https://proxy.example.com/v1");

// Google Gemini
let client = Client::gemini("AIza...");

// Gemini 互換プロキシ
let client = Client::gemini_compatible("AIza...", "https://proxy.example.com/v1beta");
```

## ストリーミング

部分的な JSON トークンをリアルタイムで受信：

```rust
let result = client.extract::<Contact>("...")
    .on_stream(|chunk| {
        print!("{chunk}");  // 部分的な JSON フラグメント
    })
    .await?;
```

3 つのプロバイダー（OpenAI、Anthropic、Gemini）すべてでストリーミングをサポート。最終結果はすべてのチャンクを結合してデシリアライズされます。

## 画像入力

ビジョン対応モデルを使って画像から構造化データを抽出：

```rust
use instructors::ImageInput;

// URL から
let result = client.extract::<Description>("この画像を説明してください")
    .image(ImageInput::Url("https://example.com/photo.jpg".into()))
    .model("gpt-4o")
    .await?;

// base64 から
let result = client.extract::<Description>("この画像を説明してください")
    .image(ImageInput::Base64 {
        media_type: "image/png".into(),
        data: base64_string,
    })
    .await?;

// 複数画像
let result = client.extract::<Comparison>("これらの画像を比較してください")
    .images(vec![
        ImageInput::Url("https://example.com/a.jpg".into()),
        ImageInput::Url("https://example.com/b.jpg".into()),
    ])
    .await?;
```

## プロバイダーフォールバック

複数のプロバイダーをチェーンして自動フェイルオーバーを実現：

```rust
let client = Client::openai("sk-...")
    .with_fallback(Client::anthropic("sk-ant-..."))
    .with_fallback(Client::openai_compatible("sk-...", "https://api.deepseek.com/v1"));

// OpenAI を最初に試行 → 失敗時に Anthropic → 最終手段として DeepSeek
let result = client.extract::<Contact>("...").await?;
```

各フォールバックはプライマリプロバイダーがリトライを使い切った後、順番に試行されます。

## バリデーション

抽出データの自動リトライ付きバリデーション — 無効な結果はエラー詳細と共に LLM にフィードバックされます。

### クロージャベース

```rust
let user: User = client.extract("...")
    .validate(|u: &User| {
        if u.age > 150 { Err("age must be <= 150".into()) } else { Ok(()) }
    })
    .await?.value;
```

### トレイトベース

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

## リスト抽出

`extract_many` でテキストから複数のアイテムを抽出:

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

## バッチ処理

設定可能な並行数で複数のプロンプトを同時処理:

```rust
let prompts = vec!["review 1".into(), "review 2".into(), "review 3".into()];

let results = client
    .extract_batch::<Review>(prompts)
    .concurrency(5)
    .validate(|r: &Review| { /* ... */ Ok(()) })
    .run()
    .await;

// 各結果は独立 — 部分的な失敗は他に影響しません
for result in results {
    match result {
        Ok(r) => println!("{:?}", r.value),
        Err(e) => eprintln!("failed: {e}"),
    }
}
```

## マルチターン会話

メッセージ履歴を渡してコンテキストを考慮した抽出を実行:

```rust
use instructors::Message;

let result = client.extract::<Summary>("summarize the above")
    .messages(vec![
        Message::user("Here is a long document..."),
        Message::assistant("I see the document."),
    ])
    .await?;
```

## 分類

列挙型は分類タスクに自然に対応します:

```rust
#[derive(Debug, Deserialize, JsonSchema)]
enum Sentiment { Positive, Negative, Neutral }

let sentiment: Sentiment = client
    .extract("This product is amazing!")
    .await?.value;
```

## ネストされた型

ベクタ、オプション、列挙型を含む複雑なネスト構造:

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

## リトライとタイムアウト

HTTP 429/503 エラーに対する指数バックオフと、リクエスト全体のタイムアウトを設定：

```rust
use std::time::Duration;
use instructors::BackoffConfig;

let client = Client::openai("sk-...")
    .with_retry_backoff(BackoffConfig::default())  // 500ms ベース, 30s 上限, 3 回リトライ
    .with_timeout(Duration::from_secs(120));        // 全体タイムアウト

// リクエストごとに上書き
let result = client.extract::<Contact>("...")
    .retry_backoff(BackoffConfig {
        base_delay: Duration::from_millis(200),
        max_delay: Duration::from_secs(10),
        jitter: true,
        max_http_retries: 5,
    })
    .timeout(Duration::from_secs(30))
    .await?;
```

バックオフ未設定時、HTTP 429/503 エラーは即座に失敗します（デフォルト動作は変更なし）。

## 設定

```rust
let result: MyStruct = client
    .extract("input text")
    .model("gpt-4o-mini")            // モデルを上書き
    .system("You are an expert...")   // カスタムシステムプロンプト
    .temperature(0.0)                 // 決定的な出力
    .max_tokens(2048)                 // 出力トークン数の上限
    .max_retries(3)                   // パース/バリデーション失敗時のリトライ回数
    .context("extra context...")      // プロンプトに追加コンテキストを付与
    .retry_backoff(BackoffConfig::default()) // HTTP 429/503 バックオフ
    .timeout(Duration::from_secs(30))        // 全体タイムアウト
    .await?
    .value;
```

## クライアントデフォルト

デフォルトを一度設定し、リクエストごとに上書き可能:

```rust
let client = Client::openai("sk-...")
    .with_model("gpt-4o-mini")
    .with_temperature(0.0)
    .with_max_retries(3)
    .with_system("Extract data precisely.");

// すべての抽出で上記のデフォルトが使用されます
let a: TypeA = client.extract("...").await?.value;
let b: TypeB = client.extract("...").await?.value;

// 特定のリクエストでのみ上書き
let c: TypeC = client.extract("...").model("gpt-4o").await?.value;
```

## コスト追跡

[tiktoken](https://crates.io/crates/tiktoken) によるトークンカウントとコスト推定を内蔵:

```rust
let result = client.extract::<Contact>("...").await?;

println!("input:  {} tokens", result.usage.input_tokens);
println!("output: {} tokens", result.usage.output_tokens);
println!("cost:   ${:.6}", result.usage.cost.unwrap_or(0.0));
println!("retries: {}", result.usage.retries);
```

`default-features = false` で無効化:

```toml
[dependencies]
instructors = { version = "1", default-features = false }
```

## JSON 自動修復

LLM が不正な JSON を返した場合 — 末尾カンマ、シングルクォート、引用符なしのキー、Markdown コードフェンスなど — instructors はデシリアライズ前に自動的に出力の修復を試みます。修復に成功した場合、リトライを消費せずに直接パースされるため、トークンとレイテンシを節約できます。これは、軽微なフォーマットエラーが発生しやすい小規模モデルやオープンソースモデルで特に有効です。

修復は透過的に行われ、設定は不要です。すべてのレスポンスに対して `serde_json` パース前に自動実行され、修復できない場合は通常のリトライパスにフォールバックします。

## ライフサイクルフック

リクエストとレスポンスを監視:

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

## 仕組み

1. `#[derive(JsonSchema)]` が Rust の型から JSON Schema を生成 ([schemars](https://crates.io/crates/schemars) を利用)
2. スキーマは型ごとにキャッシュ (スレッドローカル、ロック競合なし)
3. スキーマはターゲットプロバイダー向けに変換:
   - **OpenAI**: `response_format` に strict モードでラップ (`additionalProperties: false`、全フィールド必須)
   - **Anthropic**: `tool` として `input_schema` にラップし、`tool_choice` で強制
   - **Gemini**: `response_schema` として渡し、`response_mime_type: "application/json"` を設定
4. LLM はスキーマに一致する有効な JSON の出力に制約
5. レスポンス JSON が不正な場合 (末尾カンマ、シングルクォート、引用符なしのキー、Markdown フェンス) は自動修復
6. レスポンスは `serde_json::from_str::<T>()` でデシリアライズ
7. `Validate` トレイトまたは `.validate()` クロージャが設定されている場合、バリデーションを実行
8. パースまたはバリデーションの失敗時、エラーフィードバックを送信してリクエストをリトライ

<!-- ECOSYSTEM BEGIN (synced by claws/opensource/scripts/sync-ecosystem.py — edit ecosystem.toml, not this block) -->

## エコシステム

[tiktoken](https://crates.io/crates/tiktoken) · [@goliapkg/tiktoken-wasm](https://www.npmjs.com/package/@goliapkg/tiktoken-wasm) · **instructors** · [chunkedrs](https://crates.io/crates/chunkedrs) · [embedrs](https://crates.io/crates/embedrs)

<!-- ECOSYSTEM END -->

## ライセンス

[MIT](LICENSE)
