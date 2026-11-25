# instructors

[![crates.io](https://img.shields.io/crates/v/instructors?style=flat-square&logo=rust)](https://crates.io/crates/instructors)
[![docs.rs](https://img.shields.io/docsrs/instructors?style=flat-square&logo=docs.rs)](https://docs.rs/instructors)
[![CI](https://img.shields.io/github/actions/workflow/status/goliajp/rust-instructor/ci.yml?branch=develop&style=flat-square&logo=github&label=ci)](https://github.com/goliajp/rust-instructor/actions/workflows/ci.yml)
[![License](https://img.shields.io/crates/l/instructors?style=flat-square)](#许可证)

[English](README.md) | **简体中文** | [日本語](README.ja.md)

> **从 1.x 升级**：类型上的 `JsonSchema` 换成 `Schema`；对 `Error` 做 `match` 时加一个 `_ =>` 分支；`usage.total_tokens` 改成方法调用 `usage.total_tokens()`。另外，schema 里的字段现在按声明顺序排列，HTTP 重试默认开启，重新提问时模型能看到它上一次的回答。完整清单和改写前后的代码见 [CHANGELOG](CHANGELOG.md#upgrading-from-1x)。

用 Rust 从 LLM 拿到带类型的结构化输出。给一个类型加上 `Deserialize` 和 `Schema`，调用 `extract`，返回的就是这个类型的值。crate 按各家接口能强制执行的格式发送这个类型的 JSON schema；模型返回的 JSON 有格式错误时先修复；解析失败或者你自己的校验不通过时，把模型的回答和错误一起发回去再问一次。设计上参考了 Python 的 [instructor](https://github.com/567-labs/instructor)。

crate 自己不依赖任何第三方运行时库：schema 派生宏、JSON 解析、修复和流式解析都在 crate 里实现，你的类型只需要和 serde 打交道。HTTP 客户端是一个 trait，默认实现是带 rustls 的 reqwest。用 `default-features = false` 时，crate 只负责生成请求和解析响应，本身不做任何 I/O，也不需要异步运行时，可以编译到 `wasm32-unknown-unknown`。

PERF_SUMMARY

## 快速开始

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

文档注释会变成 schema 里的 `description`。字段保持声明顺序，这一点会影响结果：OpenAI 按 schema 里的顺序生成字段，所以写在 `answer` 前面的 `reasoning` 会先生成。

## Schema

`#[derive(Schema)]` 会读取类型上的 serde 属性，所以 schema 描述的就是 `Deserialize` 实际接受的 JSON。支持 `rename`、`rename_all`、`default`、`skip`、`flatten`、`tag`、`content`、`untagged`、`transparent` 和 `deny_unknown_fields`。有些 serde 属性会以派生宏看不到的方式改变 JSON 的形状（`deserialize_with`、`with`、`try_from`、`from`），遇到它们时编译报错，除非用 `#[schema(like = SomeType)]` 指明 JSON 的样子。

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

派生宏基于 `syn` 3，启用的 feature 和 `serde_derive` 完全一样，所以已经在派生 `Deserialize` 的项目不会因为它多编译任何 crate。只实现了 `schemars::JsonSchema` 的类型，可以开启 `schemars` feature，用 `client.extract_schemars::<T>(..)` 提取。

## 支持的接口

| 构造函数 | 协议 | 默认怎么发送 schema |
|---|---|---|
| `Client::openai(key)` | OpenAI Chat Completions | `response_format`，严格模式的 `json_schema` |
| `Client::openai(key).with_api(OpenAiApi::Responses)` | OpenAI Responses | `text.format`，严格模式的 `json_schema` |
| `Client::openai_compatible(key, url)` | Chat Completions | 严格模式的 `json_schema`；DeepSeek、通义千问、GLM 的地址用 JSON 模式 |
| `Client::anthropic(key)` | Anthropic Messages | `output_config.format`（结构化输出）|
| `Client::anthropic_compatible(key, url)` | 同上 | 一个工具，`tool_choice` 强制调用 |
| `Client::gemini(key)`、`gemini_compatible(key, url)` | Gemini `generateContent` | `generationConfig.responseJsonSchema` |

`with_output_mode` 可以改变发送 schema 的方式：

| `OutputMode` | 发送的内容 |
|---|---|
| `Schema` | 接口原生的结构化输出（上表）|
| `Tool` | 一个以 schema 为输入的工具，模型允许时强制调用；Gemini 和 Responses API 不支持 |
| `JsonObject` | JSON 模式（`response_format: json_object`），schema 写进系统提示词；Anthropic 没有 JSON 模式，按 `Prompt` 处理 |
| `Prompt` | 不发格式参数，schema 写进系统提示词，从回答文本里取出 JSON |

`openai_compatible` 会看 base URL 的主机名：`api.openai.com` 发 `max_completion_tokens`；DeepSeek（`deepseek.com`）、通义千问（`aliyuncs.com`）、Kimi（`moonshot.cn`、`moonshot.ai`、`kimi.ai`、`kimi.com`）和 GLM（`bigmodel.cn`、`z.ai`）各用自己的[思考参数](#思考深度)。DeepSeek、通义千问和 GLM 默认用 `JsonObject`，因为它们的文档只写了 JSON 模式，没有写严格 schema；其他主机按通用的 OpenAI 兼容接口处理，用 `Schema`。

引用自身的类型（树、链表）用 `$defs` 和 `$ref` 发送。Anthropic 的结构化输出表达不了递归，所以这种类型发给 Anthropic 时改用工具，`tool_choice: auto`。

不指定模型时的默认值：

| 接口 | 默认模型 |
|---|---|
| OpenAI 及兼容接口 | `gpt-5.6-terra` |
| Anthropic | `claude-sonnet-5` |
| Gemini | `gemini-3.6-flash` |

模型大约每半年换一代，退役的 ID 会失效，所以这些默认值会随小版本更新。想在升级时固定模型，就设置 `.with_model("...")`。

## 思考深度

控制模型思考多少，只有一个设置，可以设在 client 上，也可以按请求设置：

```rust
use instructors::{Effort, Reasoning};

let client = Client::gemini("AIza...").with_reasoning(Reasoning::Effort(Effort::Low));

let r = client
    .extract::<Invoice>(text)
    .reasoning(Reasoning::Budget(2048))
    .await?;
```

不设置就什么都不发。各家收到的参数如下：

| 接口 | `Effort(e)` | `Budget(n)` | `Off` |
|---|---|---|---|
| OpenAI、通用 OpenAI 兼容接口 | `reasoning_effort: e` | 不发送，直接返回错误 | `reasoning_effort: "none"` |
| Anthropic | `thinking: {type: "adaptive"}` 加 `output_config.effort: e` | `thinking: {type: "enabled", budget_tokens: n}` | `thinking: {type: "disabled"}` |
| Gemini | `thinkingConfig.thinkingLevel` | `thinkingConfig.thinkingBudget: n` | `thinkingConfig.thinkingBudget: 0` |
| DeepSeek、Kimi、GLM | `thinking: {type: "enabled"}` 加 `reasoning_effort`（`low`、`high`、`max` 三档）| 不发送，直接返回错误 | `thinking: {type: "disabled"}` |
| 通义千问 | `enable_thinking: true` | `enable_thinking: true`、`thinking_budget: n` | `enable_thinking: false` |

`Effort` 有六档：`Minimal`、`Low`、`Medium`、`High`、`XHigh`、`Max`。Anthropic 没有 `minimal`，发 `low`；Gemini 没有 `MAX`，发 `XHIGH`；DeepSeek、Kimi、GLM 只有三档，`Minimal` 和 `Low` 发 `low`，`Medium` 和 `High` 发 `high`，`XHigh` 和 `Max` 发 `max`。

crate 不维护「哪个模型接受哪个值」的列表。模型不接受的值，会以 `Error::Api` 返回，内容是接口自己的错误信息。按各家 2026 年 11 月的文档举几个例子：Claude Opus 5.5 不接受 `thinking: disabled` 和 `budget_tokens`；Claude Sonnet 5.5 不接受 `thinking: disabled`；Gemini 4 系列不能关闭思考。

Anthropic 的强制工具调用不能和思考同时使用，所以设置了思考参数时，走工具的请求（`OutputMode::Tool`，也就是 `anthropic_compatible` 的默认方式，或者递归类型）改用 `tool_choice: auto`。

思考用的 token 按输出计费。三种协议都把它算进 `usage.output_tokens` 和费用估算。

## 流式输出

`on_partial` 在每收到一段流式数据后调用你的闭包，传入目前为止解析出的 JSON：

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

传入的值是对增量解析器的借用视图。解析器每个字节只读一遍，所以整个流的开销和长度成正比。没收完的字符串里是已经收到的字符，`is_complete()` 表示一个值是否已经结束。`v.parse::<Snapshot>()` 把目前的值反序列化成你自己的类型（通常字段都是 `Option`），`to_json()` 给出它的 JSON 文本。解析器本身也公开为 [`instructors::partial::PartialParser`](https://docs.rs/instructors/latest/instructors/partial/)。

用 `extract_many` 时，`on_item` 会在列表里的每一项完整之后立刻交给你：

```rust
let all: Vec<Entity> = client
    .extract_many("...")
    .on_item(|e: Entity| println!("found {}", e.name))
    .await?
    .value;
```

`on_delta` 给的是原始的文本片段。设置了这些钩子中的任何一个，请求都会走流式；最终结果照常从完整的回答解析，所有接口都会填好 token 数和费用。

## 可以区分处理的错误

| 情况 | 得到什么 |
|---|---|
| HTTP 重试用完后，接口仍返回非 2xx 状态码 | `Error::Api { status, message, retry_after, .. }` |
| 连接失败 | `Error::Transport { source, .. }` |
| 每次回答都解析失败，或者都没通过校验 | `Error::Parse { attempts, raw, .. }` 或 `Error::Validation { .. }` |
| 模型拒绝回答（OpenAI 的 `refusal`、Anthropic 的 `stop_reason: refusal`、Gemini 的安全拦截）| `Error::Refusal { text, .. }`，不会重试 |
| 回答因为达到输出 token 上限被截断，而且解析不了 | `Error::Truncated { reason, raw, .. }`，不会重试；调用了 `.accept_truncated()` 时另说 |
| 接口没有对应参数的设置（比如对 OpenAI 用 `Reasoning::Budget`）| `Error::Unsupported`，请求不会发出 |
| 整个提取超过了超时时间 | `Error::Timeout { after, .. }` |
| 所有接口都失败了 | `Error::Fallback`：`chain()` 从最后一个接口开始依次列出每个接口的错误，`source()` 把它们串起来，方便 anyhow 或 eyre 打印 |

只要请求发出去过，错误里就带着这次提取的花费：`err.usage()`。

`.accept_truncated()` 会在不编造内容的前提下，把被截断的回答变成一个值：补上没结束的字符串和括号，丢掉写了一半的键或数字，结果仍然要能反序列化，所以缺的字段是 `Option` 或带 `#[serde(default)]` 时才能成功。`extract_many` 保留截断前已经完整的项。返回值里 `truncated` 为 `true`。

## 校验与重试

```rust
let user: User = client
    .extract("...")
    .validate(|u: &User| {
        if u.age > 150 { Err("age must be <= 150".into()) } else { Ok(()) }
    })
    .await?
    .value;
```

也可以给类型实现 `Validate`，再调用 `.validated()`。解析或校验失败时，下一次请求会把模型的回答和错误追加到对话里再问一次，最多取 `max_attempts` 次回答（默认 3 次）。在用掉一次机会之前，会先修复格式有问题的 JSON：JSON 前后的 markdown 代码块或说明文字、多余的逗号、单引号、没加引号的键、注释、Python 的 `True` / `None`。

## 列表、批量、对话历史、图片、文档

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

## 重试、备用接口、超时

```rust
use std::time::Duration;

let client = Client::openai("sk-...")
    .with_fallback(Client::anthropic("sk-ant-..."))
    .with_retry(RetryPolicy::new().max_retries(5))
    .with_timeout(Duration::from_secs(120)); // the whole extraction
```

HTTP 重试默认开启：遇到 408、429、500、502、503、504、529 和连接错误时最多重试 3 次，等待时间从 500 ms 开始翻倍，最多 30 s，带随机抖动；接口返回了 `Retry-After` 或 `retry-after-ms` 请求头时按它等。`RetryPolicy::none()` 关闭重试。主接口失败后才会尝试备用接口，备用接口用它自己的模型、输出方式和 transport。

## 不用运行时，或者用自己的 HTTP 客户端

`.await` 通过一个 `Transport` 发送请求，默认是 tokio 上的 reqwest。别的 HTTP 客户端可以通过 `with_transport` 接入：

```rust
impl Transport for MyHttp {
    fn send(&self, req: HttpRequest) -> BoxFuture<'_, Result<HttpResponse, TransportError>> { ... }
    fn sleep(&self, d: Duration) -> BoxFuture<'static, ()> { ... }
}
let client = Client::openai("sk-...").with_transport(MyHttp::new());
```

也可以完全不用异步。`exchange()` 把一次提取变成一个状态机，由它告诉你该发什么、该等多久：

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

用 `default-features = false, features = ["derive"]` 时，构建里没有任何做 I/O 的代码，crate 可以编译到 `wasm32-unknown-unknown`。

## 设置

在 client 上设默认值，按请求覆盖：

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

| 设置 | OpenAI 系 | Anthropic | Gemini |
|---|---|---|---|
| 最大输出 token 数 | OpenAI 发 `max_completion_tokens`，Responses API 发 `max_output_tokens`，兼容接口发 `max_tokens`；不设置就不发 | `max_tokens`，不设置时为 16384 | `maxOutputTokens`，不设置时为 16384 |
| temperature | 发送，默认 0.0 | 从不发送：现在的 Claude 模型不接受非默认值 | 设置了才发送 |

## 费用

```rust
let r = client.extract::<Contact>("...").await?;
println!("输入 {}，输出 {}，{:?} 美元", r.usage.input_tokens, r.usage.output_tokens, r.usage.cost);
```

费用按 [tiktoken](https://crates.io/crates/tiktoken) 的价格表计算（`cost-tracking` feature，默认开启，不会链接任何词表数据）。每个请求单独计价，重试的费用累加；价格表里没有的模型，`cost` 是 `None`。模型 ID 由 tiktoken 解析，所以各家接口自己的写法（用短横线、带日期后缀、带 Bedrock 或 Vertex 前缀）也能算出价格。

```toml
instructors = { version = "2", default-features = false, features = ["derive", "reqwest"] } # 不计算费用
```

## 性能

PERF_SECTION

## 许可证

MIT 或 Apache-2.0，任选其一：[MIT](LICENSE-MIT)、[Apache-2.0](LICENSE-APACHE)。
