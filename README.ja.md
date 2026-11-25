# instructors

[![crates.io](https://img.shields.io/crates/v/instructors?style=flat-square&logo=rust)](https://crates.io/crates/instructors)
[![docs.rs](https://img.shields.io/docsrs/instructors?style=flat-square&logo=docs.rs)](https://docs.rs/instructors)
[![CI](https://img.shields.io/github/actions/workflow/status/goliajp/rust-instructor/ci.yml?branch=develop&style=flat-square&logo=github&label=ci)](https://github.com/goliajp/rust-instructor/actions/workflows/ci.yml)
[![License](https://img.shields.io/crates/l/instructors?style=flat-square)](#ライセンス)

[English](README.md) | [简体中文](README.zh-CN.md) | **日本語**

> **1.5 への更新**：API は追加のみですが、送信するリクエストが 4 か所変わります。`Client::anthropic` はスキーマを強制ツール呼び出しではなく構造化出力（`output_config.format`）で送ります。Claude Opus 5.5、Sonnet 5.5、Fable 5.1 以降は強制ツール呼び出しを拒否するためです。以前のリクエストに戻すには `.with_anthropic_tool_use()` を呼びます。`extract_many` は配列そのものではなく `{"items": [...]}` を求めます。最大出力トークン数の設定が OpenAI 系の API にも送られるようになりました。Gemini の API キーは URL ではなく `x-goog-api-key` ヘッダーで送ります。詳細は [CHANGELOG](CHANGELOG.md#150---2026-11-25) を参照してください。

LLM から型付きの構造化出力を受け取る Rust のライブラリです。型に `Deserialize` と `JsonSchema` を derive して `extract` を呼ぶと、その型の値が返ります。各プロバイダーが強制できる形で型の JSON スキーマを送り、壊れた JSON は修復し、パースや独自の検証に失敗したときはエラーをモデルに返して再度尋ねます。設計は Python の [instructor](https://github.com/567-labs/instructor) にならっています。

8 行の明細を持つ請求書をループバックで計測したところ、`extract` 1 回あたりのクライアント側 CPU 時間は、同じリクエストを `reqwest` で直接送る場合より 2.3 µs 多いだけでした。rstructor 0.6.0 は 25.1 µs、rig-core 0.47.1 は 28.4 µs 多くかかります。1.2 KB の回答が 308 回に分かれてストリーミングされる場合、届くたびに途中の値を取り出す処理の合計は 5.5 µs で、rstructor が使っている「届くたびにバッファ全体を解析し直す」方式では 677 µs かかります。条件と全データは[性能](#性能)にあります。

## クイックスタート

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

## 対応プロバイダー

3 つのプロトコルに対応しています。`_compatible` の付いたコンストラクタは、同じプロトコルを別のベース URL に向けます。

| コンストラクタ | プロトコル | スキーマの送り方 |
|---|---|---|
| `Client::openai(key)` | OpenAI Chat Completions | `response_format`（strict な `json_schema`） |
| `Client::openai_compatible(key, url)` | 同上 | 同上。サーバーが strict な `json_schema` に対応している必要があります |
| `Client::anthropic(key)` | Anthropic Messages | `output_config.format`（構造化出力）|
| `Client::anthropic_compatible(key, url)` | 同上 | ツールを定義し、`tool_choice` で呼び出しを強制 |
| `Client::gemini(key)` | Gemini `generateContent` | `generationConfig.responseSchema` |
| `Client::gemini_compatible(key, url)` | 同上 | 同上 |

`openai_compatible` はベース URL のホスト名を見ます。`api.openai.com` には `max_completion_tokens` を使い、DeepSeek（`deepseek.com`）、Qwen（`aliyuncs.com`）、Kimi（`moonshot.cn`、`moonshot.ai`）、GLM（`bigmodel.cn`、`z.ai`）にはそれぞれの[思考パラメーター](#思考の深さ)を送ります。それ以外のホストは汎用の OpenAI 互換サーバーとして扱います。

自分自身を参照する型（木構造や連結リスト）は `$defs` と `$ref` で送ります。Anthropic の構造化出力は再帰を表現できないため、こうした型では Anthropic へのリクエストをツールと `tool_choice: auto` に切り替え、Gemini には `responseSchema` ではなく `responseJsonSchema` を送ります。

モデルを指定しない場合のデフォルト：

| プロバイダー | デフォルト |
|---|---|
| OpenAI と OpenAI 互換 | `gpt-5.6-terra` |
| Anthropic | `claude-sonnet-5` |
| Gemini | `gemini-3.6-flash` |

モデルはおよそ半年ごとに世代交代し、提供終了した ID は使えなくなります。そのためデフォルトはマイナーリリースごとに変わります。アップグレード後も同じモデルを使うには `.with_model("...")` を指定してください。

## 思考の深さ

モデルにどれだけ考えさせるかは 1 つの設定で指定します。クライアント全体にも、リクエストごとにも設定できます。

```rust
use instructors::{Effort, Reasoning};

let client = Client::gemini("AIza...").with_reasoning(Reasoning::Effort(Effort::Low));

let r = client
    .extract::<Invoice>(text)
    .reasoning(Reasoning::Budget(2048))
    .await?;
```

設定しなければ何も送りません。各プロバイダーには次のパラメーターを送ります。

| プロバイダー | `Effort(e)` | `Budget(n)` | `Off` |
|---|---|---|---|
| OpenAI、汎用の OpenAI 互換 | `reasoning_effort: e` | 送信せずにエラーを返す | `reasoning_effort: "none"` |
| Anthropic | `thinking: {type: "adaptive"}` と `output_config.effort: e` | `thinking: {type: "enabled", budget_tokens: n}` | `thinking: {type: "disabled"}` |
| Gemini | `thinkingConfig.thinkingLevel` | `thinkingConfig.thinkingBudget: n` | `thinkingConfig.thinkingBudget: 0` |
| DeepSeek、Kimi、GLM | `thinking: {type: "enabled"}` と `reasoning_effort`（`low`、`high`、`max` の 3 段階）| 送信せずにエラーを返す | `thinking: {type: "disabled"}` |
| Qwen | `enable_thinking: true` | `enable_thinking: true`、`thinking_budget: n` | `enable_thinking: false` |

`Effort` は `Minimal`、`Low`、`Medium`、`High`、`XHigh`、`Max` の 6 段階です。Anthropic には `minimal` がないので `low` を、Gemini には `MAX` がないので `XHIGH` を送ります。DeepSeek、Kimi、GLM は 3 段階なので、`Minimal` と `Low` は `low`、`Medium` と `High` は `high`、`XHigh` と `Max` は `max` になります。

どのモデルがどの値を受け付けるかの一覧は持っていません。モデルが受け付けない値を送ると、プロバイダーのエラーメッセージがそのまま `Error::Api` で返ります。2026 年 11 月時点の各社ドキュメントによると、たとえば Claude Opus 5.5 は `thinking: disabled` と `budget_tokens` を、Claude Sonnet 5.5 は `thinking: disabled` を拒否し、Gemini 4 系は思考をオフにできません。

Anthropic では強制ツール呼び出しと思考を同時に使えません。そのため思考の設定があるときは、ツールを使う経路（`anthropic_compatible` や再帰型）で `tool_choice: auto` を送ります。

思考に使われたトークンは出力として課金されます。3 つのプロトコルとも、`usage.output_tokens` と費用の見積もりに含めています。

## ストリーミング

`on_partial` は、ストリーミングでデータが届くたびに、その時点までに解析できた JSON を渡してクロージャを呼びます。

```rust
let article: Article = client
    .extract("...")
    .on_partial(|v| {
        if let Some(title) = v.get("title").and_then(|t| t.as_str()) {
            println!("ここまでのタイトル: {title}");
        }
    })
    .await?
    .value;
```

渡される値は、インクリメンタルパーサーの内部状態を借用したビューです。パーサーは各バイトを 1 回しか読まないので、ストリーム全体の処理量は長さに比例します。受信途中の文字列には届いた分の文字が入っており、`is_complete()` で値が閉じたかどうかがわかります。`to_value()` は `serde_json::Value` にコピーします。パーサーは [`instructors::partial::PartialParser`](https://docs.rs/instructors/latest/instructors/partial/) として公開しているので、`extract` の外でも使えます。

`on_stream` は生のテキスト断片を渡します。どちらかを設定するとリクエストはストリーミングになり、最終結果はいつもどおり回答全体から解析します。OpenAI へのストリーミングリクエストには `stream_options.include_usage` を付けるので、トークン数と費用も取得できます。

## 対処できるエラー

| 状況 | 返る内容 |
|---|---|
| プロバイダーが 2xx 以外のステータスを返した | `Error::Api { status, message }`（`message` はレスポンス本文）|
| 再試行をすべて使っても、回答をパースできないか検証に失敗した | `Error::ExtractionFailed` または `Error::ValidationFailed`（最後のエラーつき）|
| モデルが回答を拒否した（OpenAI の `refusal`、Anthropic の `stop_reason: refusal`、Gemini の安全性による停止）| `err.refusal()` が拒否の文面を返す。再試行しない |
| 回答が出力トークンの上限で切れ（OpenAI `finish_reason: length`、Anthropic `max_tokens`、Gemini `MAX_TOKENS`）、パースできなかった | `err.is_truncated()` が true。再試行も修復もしない |
| 抽出全体がタイムアウトを超えた | `Error::Timeout` |

フォールバックを設定している場合、エラーの種類はプライマリーのものを保ち、メッセージに各フォールバックのエラーを並べます。

## 検証と再試行

```rust
let user: User = client
    .extract("...")
    .validate(|u: &User| {
        if u.age > 150 { Err("age must be <= 150".into()) } else { Ok(()) }
    })
    .await?
    .value;
```

型に `Validate` を実装して `.validated()` を呼ぶこともできます。パースや検証に失敗すると、次の試行ではエラーメッセージを付けて尋ね直します。回数は `max_retries`（デフォルト 2）までです。再試行を使う前に、壊れた JSON（末尾のカンマ、シングルクォート、引用符のないキー、markdown のコードフェンス）を [jsonrepair](https://crates.io/crates/jsonrepair) で修復します。

## リスト、バッチ、会話履歴、画像

```rust
// リスト: スキーマのルートは {"items": [...]}、結果は Vec<Entity>
let entities: Vec<Entity> = client.extract_many("Tim Cook met Sundar Pichai").await?.value;

// 複数のプロンプトを同時 5 件までで処理。結果は入力と同じ順序
let results = client.extract_batch::<Review>(prompts).concurrency(5).run().await;

// プロンプトの前に会話履歴を置く
let r = client
    .extract::<Summary>("summarize the above")
    .messages(vec![Message::user("..."), Message::assistant("...")])
    .await?;

// 画像（URL または base64）
let r = client
    .extract::<Description>("describe this")
    .image(ImageInput::Url("https://example.com/photo.jpg".into()))
    .await?;
```

## フォールバック、バックオフ、タイムアウト

```rust
use std::time::Duration;

let client = Client::openai("sk-...")
    .with_fallback(Client::anthropic("sk-ant-..."))
    .with_retry_backoff(BackoffConfig::default()) // 429、500、502、503、504、529
    .with_timeout(Duration::from_secs(120));      // 抽出全体の制限時間
```

フォールバックは、プライマリーが再試行を使い切ってから、フォールバック側のデフォルトモデルで試します。バックオフは設定しない限り無効です。`BackoffConfig::default()` は 500 ms から倍々に最大 30 s まで待ち、ジッターを加え、HTTP の再試行は最大 3 回です。

## 設定

クライアントにデフォルトを設定し、リクエストごとに上書きできます。

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

| 設定 | OpenAI 系 | Anthropic | Gemini |
|---|---|---|---|
| 最大出力トークン数 | OpenAI には `max_completion_tokens`、互換 API には `max_tokens`。設定しなければ送らない | `max_tokens`。未設定なら 16384 | `maxOutputTokens`。未設定なら 16384 |
| temperature | 送る。デフォルトは 0.0 | 送らない。現行の Claude はデフォルト以外の値を拒否する | 設定したときだけ送る |

## 費用

```rust
let r = client.extract::<Contact>("...").await?;
println!("入力 {}、出力 {}、{:?} USD", r.usage.input_tokens, r.usage.output_tokens, r.usage.cost);
```

費用は [tiktoken](https://crates.io/crates/tiktoken) の価格表で計算します（`cost-tracking` feature、デフォルトで有効。語彙データはリンクしません）。リクエストごとに計算して再試行分を合算し、価格表にないモデルでは `None` になります。モデル ID は tiktoken が解決するので、プロバイダーごとの表記（ハイフン区切り、日付の接尾辞、Bedrock や Vertex の接頭辞）でも価格が出ます。

```toml
instructors = { version = "1", default-features = false } # 費用計算なし
```

## フィールドの順序

デフォルトでは、スキーマ内の構造体のフィールドはアルファベット順に並びます。schemars がプロパティを `serde_json::Map` に入れており、この型は `serde_json/preserve_order` が有効でない限りキーをソートするためです。OpenAI はスキーマに並んだ順にフィールドを生成するので、`answer` より前に宣言した `reasoning` が後から書かれます。`preserve-order` feature を有効にすると、どのプロバイダーにも宣言順で送ります。

```toml
instructors = { version = "1", features = ["preserve-order"] }
```

この feature は `serde_json/preserve_order` を有効にします。ビルド内で `serde_json::Map` を使うすべてのクレートに影響し、マップのキーはソートされず挿入順になります。また `indexmap` がビルドに加わります。

## 性能

2026 年 11 月 25 日、Apple M4 Pro の Mac mini（14 コア）で、ベンチマーク用のロックを専有して計測しました。release ビルド、LTO 有効、codegen-units は 1、rustc 1.100.0 です。15 秒ごとに記録した 1 分間のロードアベレージは、ビルドを含む全期間で 1.6 から 4.8 の間でした。各数値は 5〜7 ラウンドの中央値です。1.4.1 と 1.5 のビルドを交互に 2 回ずつ走らせ、表には 1 回目の結果を載せています。2 回目との差は「基準からの増分」の列で 1.2 µs 以内、それ以外で 4% 以内でした。データは `Invoice` で、ネストした住所 2 つ、列挙型 1 つ、省略可能なフィールドを持つ明細 8 行を含み、日中の文字とエスケープされた引用符が入った 1,247 バイトの JSON です。リクエストはすべてローカルのループバック HTTP サーバーに送り、実際の API は呼んでいません。ベンチマークのプログラムはこのリポジトリには含まれていません。

**`extract` 1 回の全体**（キープアライブした HTTP/1.1 接続で逐次呼び出し、5 × 5,000 回。基準は、同じプロセスで同じ `reqwest` クライアントから事前に組み立てた本文を送り、応答を `serde_json` で解析したもの）：

| クライアント | 1 回あたり | 基準からの増分 |
|---|---:|---:|
| 基準 | 50.7 µs | |
| instructors 1.5 | 53.0 µs | 2.3 µs |
| instructors 1.4.1（別の回で計測。その回の基準は 49.1 µs）| 66.4 µs | 17.3 µs |
| rstructor 0.6.0 | 75.8 µs | 25.1 µs |
| rig-core 0.47.1（`output_schema` のあと `serde_json::from_str`）| 79.1 µs | 28.4 µs |

実際のモデル呼び出しは数百ミリ秒以上かかるので、この数マイクロ秒が効くのは大量のバッチ処理やプロキシでの CPU コストで、1 回の呼び出しの待ち時間ではありません。

**リクエストごとのスキーマ処理：**

| 処理 | 時間 |
|---|---:|
| instructors 1.5：キャッシュ済みの送信用スキーマのバイト列をリクエスト本文にコピー | 30 ns |
| instructors 1.4.1：キャッシュしたスキーマを OpenAI 向けに変換してからシリアライズ | 10.8 µs + 1.1 µs |
| rstructor 0.6.0：`Invoice::schema().to_json()` | 7.7 µs |
| 参考：キャッシュなしの `schemars::schema_for!(Invoice)` | 3.8 µs |

**ストリーミング中の途中の値**（1,247 バイトの回答が 4 バイトずつ 308 回に分かれて届き、届くたびに途中の値を取り出す。ストリーム全体の合計）：

| 方式 | ストリーム全体 |
|---|---:|
| instructors 1.5 の `PartialParser`：届いた分を渡し、借用したビューを読む | 5.5 µs |
| 同上に加えて、届くたびに `serde_json::Value` へコピー | 307 µs |
| 届くたびにバッファ全体を `Value` に解析し直す（rstructor 0.6.0 の方式）| 677 µs |
| 届くたびにバッファ全体へ partial-json-fixer 0.5.5 をかけてから解析 | 735 µs |
| 届くたびにバッファ全体へ jsonrepair 0.1.0 をかけてから解析 | 2,035 µs |
| 参考：完成した回答を `serde_json::from_str::<Invoice>` で 1 回解析 | 1.6 µs |

パーサーの処理量はバイト列の分かれ方に左右されないので、コストはストリームの長さに比例します。解析し直す方式では長さの 2 乗に比例して増えます。

## ライセンス

[MIT](LICENSE-MIT) と [Apache-2.0](LICENSE-APACHE) のどちらかを選んで利用できます。
