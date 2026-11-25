# instructors

[![crates.io](https://img.shields.io/crates/v/instructors?style=flat-square&logo=rust)](https://crates.io/crates/instructors)
[![docs.rs](https://img.shields.io/docsrs/instructors?style=flat-square&logo=docs.rs)](https://docs.rs/instructors)
[![CI](https://img.shields.io/github/actions/workflow/status/goliajp/rust-instructor/ci.yml?branch=develop&style=flat-square&logo=github&label=ci)](https://github.com/goliajp/rust-instructor/actions/workflows/ci.yml)
[![License](https://img.shields.io/crates/l/instructors?style=flat-square)](#ライセンス)

[English](README.md) | [简体中文](README.zh-CN.md) | **日本語**

> **1.x からのアップグレード**：型の `JsonSchema` を `Schema` に替え、`Error` の `match` に `_ =>` の腕を足し、`usage.total_tokens` をメソッド呼び出し `usage.total_tokens()` にしてください。このほか、スキーマのフィールドは宣言順に並ぶようになり、HTTP の再試行はデフォルトで有効になり、再質問ではモデルに前回の回答を見せるようになりました。変更点の一覧と書き換え前後のコードは [CHANGELOG](CHANGELOG.md#upgrading-from-1x) にあります。

LLM から型付きの構造化出力を受け取る Rust のライブラリです。型に `Deserialize` と `Schema` を付けて `extract` を呼ぶと、その型の値が返ります。型の JSON スキーマを各プロバイダーが強制できる形で送り、壊れた JSON は修復し、パースや独自の検証に失敗したときはモデルの回答とエラーを送り返してもう一度尋ねます。設計は Python の [instructor](https://github.com/567-labs/instructor) にならっています。

このクレート自身はサードパーティーのランタイム依存を持ちません。スキーマの derive、JSON パーサー、修復、ストリーミング用のパーサーはクレートの中で実装していて、利用者の型が関わるのは serde だけです。HTTP クライアントはトレイトで、デフォルトの実装は rustls を使う reqwest です。`default-features = false` にすると、リクエストの組み立てとレスポンスの解析だけを行い、I/O もしなければ非同期ランタイムも要りません。`wasm32-unknown-unknown` 向けにもビルドできます。

PERF_SUMMARY

## クイックスタート

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

ドキュメントコメントはスキーマの `description` になります。フィールドは宣言順のまま送られます。OpenAI はスキーマに並んだ順にフィールドを生成するので、`answer` より前に置いた `reasoning` が先に書かれます。

## スキーマ

`#[derive(Schema)]` は型に付いた serde の属性を読むので、スキーマは `Deserialize` が実際に受け付ける JSON を表します。対応している属性は `rename`、`rename_all`、`default`、`skip`、`flatten`、`tag`、`content`、`untagged`、`transparent`、`deny_unknown_fields` です。derive から見えない形で JSON を変える属性（`deserialize_with`、`with`、`try_from`、`from`）はコンパイルエラーにします。`#[schema(like = SomeType)]` で JSON の形を指定すれば使えます。

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

derive は `syn` 3 の上に作っていて、有効にしている feature は `serde_derive` とまったく同じです。そのため、すでに `Deserialize` を derive しているプロジェクトで、このために追加でコンパイルされるクレートはありません。`schemars::JsonSchema` しか実装していない型は、`schemars` feature を有効にして `client.extract_schemars::<T>(..)` で使えます。

## 対応プロバイダー

| コンストラクター | プロトコル | デフォルトのスキーマの送り方 |
|---|---|---|
| `Client::openai(key)` | OpenAI Chat Completions | `response_format` で strict な `json_schema` |
| `Client::openai(key).with_api(OpenAiApi::Responses)` | OpenAI Responses | `text.format` で strict な `json_schema` |
| `Client::openai_compatible(key, url)` | Chat Completions | strict な `json_schema`。DeepSeek、Qwen、GLM のホストでは JSON モード |
| `Client::anthropic(key)` | Anthropic Messages | `output_config.format`（構造化出力）|
| `Client::anthropic_compatible(key, url)` | 同上 | ツール 1 つを `tool_choice` で強制 |
| `Client::gemini(key)`、`gemini_compatible(key, url)` | Gemini `generateContent` | `generationConfig.responseJsonSchema` |

スキーマの送り方は `with_output_mode` で変えられます。

| `OutputMode` | 送るもの |
|---|---|
| `Schema` | プロバイダー固有の構造化出力（上の表）|
| `Tool` | スキーマを入力とするツール。モデルが許す場合は呼び出しを強制。Gemini と Responses API では使えない |
| `JsonObject` | JSON モード（`response_format: json_object`）とシステムプロンプト内のスキーマ。JSON モードのない Anthropic では `Prompt` と同じ |
| `Prompt` | フォーマット指定なし。スキーマをシステムプロンプトに書き、回答のテキストから JSON を取り出す |

`openai_compatible` はベース URL のホスト名を見ます。`api.openai.com` には `max_completion_tokens` を送り、DeepSeek（`deepseek.com`）、Qwen（`aliyuncs.com`）、Kimi（`moonshot.cn`、`moonshot.ai`、`kimi.ai`、`kimi.com`）、GLM（`bigmodel.cn`、`z.ai`）にはそれぞれの[思考パラメーター](#思考の深さ)を送ります。DeepSeek、Qwen、GLM のデフォルトは `JsonObject` です。これらのドキュメントには JSON モードの説明があり、strict なスキーマの説明はないためです。それ以外のホストは一般的な OpenAI 互換サーバーとして扱い、`Schema` を使います。

自分自身を含む型（木構造や連結リスト）は `$defs` と `$ref` を使って送ります。Anthropic の構造化出力は再帰を表現できないので、そうした型では Anthropic へのリクエストにツールを使い、`tool_choice: auto` を送ります。

モデルを指定しなかったときのデフォルト：

| プロバイダー | デフォルト |
|---|---|
| OpenAI と OpenAI 互換 | `gpt-5.6-terra` |
| Anthropic | `claude-sonnet-5` |
| Gemini | `gemini-3.6-flash` |

モデルはおよそ半年ごとに世代が替わり、廃止された ID は使えなくなるので、デフォルトはマイナーリリースで更新します。アップグレードしても同じモデルを使い続けたい場合は `.with_model("...")` を設定してください。

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

Anthropic では強制ツール呼び出しと思考を同時に使えません。そのため思考の設定があるときは、ツールを使う経路（`anthropic_compatible` のデフォルトである `OutputMode::Tool` や再帰型）で `tool_choice: auto` を送ります。

思考に使われたトークンは出力として課金されます。3 つのプロトコルとも、`usage.output_tokens` と費用の見積もりに含めています。

## ストリーミング

`on_partial` には、ストリームの差分を受け取るたびに、その時点までにパースできた JSON が渡されます。

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

渡される値は、インクリメンタルパーサーの中身を借用で見せるビューです。パーサーは各バイトを一度しか読まないので、ストリーム全体の処理量は長さに比例します。閉じていない文字列には受信済みの文字が入っていて、値が閉じたかどうかは `is_complete()` でわかります。`v.parse::<Snapshot>()` はその時点の値を自分の型（たいていはフィールドがすべて `Option` の型）にデシリアライズし、`to_json()` は JSON テキストを返します。パーサー自体も [`instructors::partial::PartialParser`](https://docs.rs/instructors/latest/instructors/partial/) として公開しています。

`extract_many` では、`on_item` でリストの各要素を完成した時点で受け取れます。

```rust
let all: Vec<Entity> = client
    .extract_many("...")
    .on_item(|e: Entity| println!("found {}", e.name))
    .await?
    .value;
```

`on_delta` には生のテキスト差分が渡されます。どれか 1 つでも設定するとリクエストはストリーミングになりますが、最終結果はこれまでどおり回答全体からパースし、どのプロバイダーでもトークン数と費用が入ります。

## 対処できるエラー

| 状況 | 返るもの |
|---|---|
| HTTP の再試行を使い切っても 2xx 以外のステータスが返った | `Error::Api { status, message, retry_after, .. }` |
| 接続に失敗した | `Error::Transport { source, .. }` |
| どの回答もパースできなかった、または検証に通らなかった | `Error::Parse { attempts, raw, .. }` または `Error::Validation { .. }` |
| モデルが回答を拒否した（OpenAI の `refusal`、Anthropic の `stop_reason: refusal`、Gemini の安全性による停止）| `Error::Refusal { text, .. }`。再試行しない |
| 回答が出力トークンの上限で途切れ、パースできなかった | `Error::Truncated { reason, raw, .. }`。再試行しない。`.accept_truncated()` を呼んだ場合は別 |
| プロバイダーに対応するパラメーターがない設定（OpenAI への `Reasoning::Budget` など）| `Error::Unsupported`。何も送らない |
| 抽出全体がタイムアウトを超えた | `Error::Timeout { after, .. }` |
| すべてのプロバイダーが失敗した | `Error::Fallback`。`chain()` は最後に試したプロバイダーから順に各エラーを返し、`source()` でもたどれるので anyhow や eyre がそのまま表示できる |

リクエストを 1 回でも送っていれば、エラーにはその抽出にかかった量が入っています（`err.usage()`）。

`.accept_truncated()` は、何もでっち上げずに済む範囲で、途切れた回答を値にします。閉じていない文字列とコンテナを閉じ、書きかけのキーや数値は捨てます。それでもデシリアライズできる必要があるので、足りないフィールドが `Option` か `#[serde(default)]` のときに成功します。`extract_many` では途切れる前に完成していた要素を返します。結果の `truncated` は `true` になります。

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

型に `Validate` を実装して `.validated()` を呼ぶ方法もあります。パースや検証に失敗すると、次のリクエストではモデルの回答とエラーを会話に追加してもう一度尋ねます。回答を受け取る回数の上限は `max_attempts`（デフォルト 3）です。1 回分を使う前に、壊れた JSON（前後の markdown のコードブロックや説明文、末尾のカンマ、シングルクォート、引用符のないキー、コメント、Python の `True` / `None`）を修復します。

## リスト、バッチ、会話履歴、画像、文書

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

## 再試行、フォールバック、タイムアウト

```rust
use std::time::Duration;

let client = Client::openai("sk-...")
    .with_fallback(Client::anthropic("sk-ant-..."))
    .with_retry(RetryPolicy::new().max_retries(5))
    .with_timeout(Duration::from_secs(120)); // the whole extraction
```

HTTP の再試行はデフォルトで有効です。408、429、500、502、503、504、529 と接続エラーで最大 3 回再試行し、待ち時間は 500 ms から倍々に増やして最大 30 s、ランダムな揺らぎを加えます。プロバイダーが `Retry-After` か `retry-after-ms` ヘッダーを返したときはその値だけ待ちます。`RetryPolicy::none()` で無効にできます。フォールバックはメインのプロバイダーが失敗した後に試し、フォールバック側のクライアントのモデル、出力モード、トランスポートを使います。

## ランタイムなしで、または自前の HTTP クライアントで使う

`.await` は `Transport` を通してリクエストを送ります。デフォルトは tokio 上の reqwest で、ほかの HTTP クライアントは `with_transport` でつなげます。

```rust
impl Transport for MyHttp {
    fn send(&self, req: HttpRequest) -> BoxFuture<'_, Result<HttpResponse, TransportError>> { ... }
    fn sleep(&self, d: Duration) -> BoxFuture<'static, ()> { ... }
}
let client = Client::openai("sk-...").with_transport(MyHttp::new());
```

非同期を使わない方法もあります。`exchange()` は 1 回の抽出を状態機械として返し、何を送り、どれだけ待つかを教えてくれます。

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

`default-features = false, features = ["derive"]` では I/O を行うコードは一切ビルドに入らず、`wasm32-unknown-unknown` 向けにもコンパイルできます。

## 設定

クライアントにデフォルトを設定し、リクエストごとに上書きできます。

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

| 設定 | OpenAI 系 | Anthropic | Gemini |
|---|---|---|---|
| 最大出力トークン数 | OpenAI には `max_completion_tokens`、Responses API には `max_output_tokens`、互換 API には `max_tokens`。設定しなければ送らない | `max_tokens`。未設定なら 16384 | `maxOutputTokens`。未設定なら 16384 |
| temperature | 送る。デフォルトは 0.0 | 送らない。現行の Claude はデフォルト以外の値を拒否する | 設定したときだけ送る |

## 費用

```rust
let r = client.extract::<Contact>("...").await?;
println!("入力 {}、出力 {}、{:?} USD", r.usage.input_tokens, r.usage.output_tokens, r.usage.cost);
```

費用は [tiktoken](https://crates.io/crates/tiktoken) の価格表で計算します（`cost-tracking` feature、デフォルトで有効。語彙データはリンクしません）。リクエストごとに計算して再試行分を合算し、価格表にないモデルでは `None` になります。モデル ID は tiktoken が解決するので、プロバイダーごとの表記（ハイフン区切り、日付の接尾辞、Bedrock や Vertex の接頭辞）でも価格が出ます。

```toml
instructors = { version = "2", default-features = false, features = ["derive", "reqwest"] } # 費用計算なし
```

## 性能

PERF_SECTION

## ライセンス

[MIT](LICENSE-MIT) と [Apache-2.0](LICENSE-APACHE) のどちらかを選んで利用できます。
