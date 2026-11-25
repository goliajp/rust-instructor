# instructors

[![crates.io](https://img.shields.io/crates/v/instructors?style=flat-square&logo=rust)](https://crates.io/crates/instructors)
[![docs.rs](https://img.shields.io/docsrs/instructors?style=flat-square&logo=docs.rs)](https://docs.rs/instructors)
[![CI](https://img.shields.io/github/actions/workflow/status/goliajp/rust-instructor/ci.yml?branch=develop&style=flat-square&logo=github&label=ci)](https://github.com/goliajp/rust-instructor/actions/workflows/ci.yml)
[![License](https://img.shields.io/crates/l/instructors?style=flat-square)](#许可证)

[English](README.md) | **简体中文** | [日本語](README.ja.md)

> **升级到 1.5？** API 只增不减，但有四处请求内容变了：`Client::anthropic` 改用结构化输出（`output_config.format`）发送 schema，不再强制调用工具，因为 Claude Opus 5.5、Sonnet 5.5 以及 Fable 5.1 之后的模型会拒绝强制工具调用；想要原来的请求，调用 `.with_anthropic_tool_use()`。`extract_many` 要求模型返回 `{"items": [...]}`，不再是裸数组。设置的最大输出 token 数现在会发给 OpenAI 系的接口。Gemini 的 API key 从 URL 移到了 `x-goog-api-key` 请求头。详见 [CHANGELOG](CHANGELOG.md#unreleased)。

用 Rust 从 LLM 拿到带类型的结构化输出。给一个类型加上 `Deserialize` 和 `JsonSchema`，调用 `extract`，返回的就是这个类型的值。crate 按各家接口能强制执行的格式发送这个类型的 JSON schema；模型返回的 JSON 有格式错误时先修复；解析失败或者你自己的校验不通过时，把错误发回给模型再问一次。设计上参考了 Python 的 [instructor](https://github.com/567-labs/instructor)。

在本机回环的基准测试里（一张有 8 个明细行的发票），一次 `extract` 调用比直接用 `reqwest` 发同样的请求多花 2.3 µs 的客户端 CPU 时间；rstructor 0.6.0 多花 25.1 µs，rig-core 0.47.1 多花 28.4 µs。一个 1.2 KB 的回答分 308 段流式到达时，边收边解析出部分值的总开销是 5.5 µs；rstructor 采用的每段都重新解析整个缓冲区的做法是 677 µs。测试条件和完整数据见[性能](#性能)。

## 快速开始

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

## 支持的接口

支持三种协议。带 `_compatible` 的构造函数把同一种协议指向别的 base URL。

| 构造函数 | 协议 | schema 的发送方式 |
|---|---|---|
| `Client::openai(key)` | OpenAI Chat Completions | `response_format`，严格模式的 `json_schema` |
| `Client::openai_compatible(key, url)` | 同上 | 同上；服务端必须支持严格模式的 `json_schema` |
| `Client::anthropic(key)` | Anthropic Messages | `output_config.format`（结构化输出）|
| `Client::anthropic_compatible(key, url)` | 同上 | 一个工具，并用 `tool_choice` 强制调用 |
| `Client::gemini(key)` | Gemini `generateContent` | `generationConfig.responseSchema` |
| `Client::gemini_compatible(key, url)` | 同上 | 同上 |

`openai_compatible` 会看 base URL 的主机名：`api.openai.com` 用 `max_completion_tokens`；DeepSeek（`deepseek.com`）、通义千问（`aliyuncs.com`）、Kimi（`moonshot.cn`、`moonshot.ai`）和智谱 GLM（`bigmodel.cn`、`z.ai`）用各自的[思考参数](#思考深度)；其他主机按通用的 OpenAI 兼容接口处理。

引用自身的类型（树、链表）用 `$defs` 和 `$ref` 发送。Anthropic 的结构化输出表达不了递归，所以遇到这种类型时，Anthropic 请求改用工具并设 `tool_choice: auto`；Gemini 改用 `responseJsonSchema`，不用 `responseSchema`。

不指定模型时使用的默认模型：

| 接口 | 默认模型 |
|---|---|
| OpenAI 及 OpenAI 兼容 | `gpt-5.6-terra` |
| Anthropic | `claude-sonnet-5` |
| Gemini | `gemini-3.6-flash` |

各家的模型大约每半年换一代，下线的模型 ID 就不能再用，所以默认模型会随小版本更新。想在升级后继续用同一个模型，请设置 `.with_model("...")`。

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

Anthropic 的强制工具调用不能和思考同时使用，所以设置了思考参数时，走工具的请求（`anthropic_compatible`，或者递归类型）改用 `tool_choice: auto`。

思考用的 token 按输出计费。三种协议都把它算进 `usage.output_tokens` 和费用估算。

## 流式输出

`on_partial` 在每收到一段流式数据后调用你的闭包，传入到目前为止解析出的 JSON：

```rust
let article: Article = client
    .extract("...")
    .on_partial(|v| {
        if let Some(title) = v.get("title").and_then(|t| t.as_str()) {
            println!("目前的标题：{title}");
        }
    })
    .await?
    .value;
```

传进来的值借用的是一个增量解析器的内部状态。解析器每个字节只读一次，整个流的开销和长度成正比。还没收完的字符串里是已经收到的字符，`is_complete()` 表示这个值是否已经闭合。`to_value()` 把它复制成 `serde_json::Value`。解析器本身也公开了，就是 [`instructors::partial::PartialParser`](https://docs.rs/instructors/latest/instructors/partial/)，可以在 `extract` 之外单独使用。

`on_stream` 给的是原始文本片段。设置了这两个回调中的任何一个，请求都会以流式发送；最终结果和平时一样，用完整的回答解析。对 OpenAI，流式请求会带上 `stream_options.include_usage`，所以 token 数和费用都有值。

## 可以区分处理的错误

| 情况 | 返回什么 |
|---|---|
| 接口返回非 2xx 状态码 | `Error::Api { status, message }`，`message` 是响应体 |
| 重试用完后回答仍然无法解析或者校验不通过 | `Error::ExtractionFailed` 或 `Error::ValidationFailed`，带最后一次的错误 |
| 模型拒绝回答（OpenAI 的 `refusal`、Anthropic 的 `stop_reason: refusal`、Gemini 的安全拦截）| `err.refusal()` 返回拒绝的说明；不重试 |
| 回答撞上了输出 token 上限（OpenAI `finish_reason: length`、Anthropic `max_tokens`、Gemini `MAX_TOKENS`）而且无法解析 | `err.is_truncated()` 为真；不重试，也不尝试修复 |
| 整次提取超过了超时时间 | `Error::Timeout` |

配置了备用接口时，返回的错误保留主接口的错误类型，错误信息里列出每个备用接口的错误。

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

也可以给类型实现 `Validate`，然后调用 `.validated()`。解析或校验失败时，下一次请求会带上错误信息，最多重试 `max_retries` 次（默认 2 次）。在用掉一次重试之前，先用 [jsonrepair](https://crates.io/crates/jsonrepair) 修复格式错误的 JSON（多余的逗号、单引号、键名没加引号、外面包了 markdown 代码块）。

## 列表、批量、对话历史、图片

```rust
// 列表：schema 的根是 {"items": [...]}，结果是 Vec<Entity>
let entities: Vec<Entity> = client.extract_many("Tim Cook met Sundar Pichai").await?.value;

// 一批提示词，同时最多 5 个请求；结果顺序和输入一致
let results = client.extract_batch::<Review>(prompts).concurrency(5).run().await;

// 在提示词前加上对话历史
let r = client
    .extract::<Summary>("summarize the above")
    .messages(vec![Message::user("..."), Message::assistant("...")])
    .await?;

// 图片，URL 或 base64
let r = client
    .extract::<Description>("describe this")
    .image(ImageInput::Url("https://example.com/photo.jpg".into()))
    .await?;
```

## 备用接口、退避重试、超时

```rust
use std::time::Duration;

let client = Client::openai("sk-...")
    .with_fallback(Client::anthropic("sk-ant-..."))
    .with_retry_backoff(BackoffConfig::default()) // 429、500、502、503、504、529
    .with_timeout(Duration::from_secs(120));      // 整次提取的时限
```

主接口用完重试次数后才会试备用接口，用的是备用 client 自己的默认模型。退避重试默认关闭；`BackoffConfig::default()` 从 500 ms 开始每次翻倍，最长 30 s，带随机抖动，HTTP 层最多重试 3 次。

## 设置

在 client 上设默认值，按请求覆盖：

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

| 设置 | OpenAI 系 | Anthropic | Gemini |
|---|---|---|---|
| 最大输出 token 数 | OpenAI 发 `max_completion_tokens`，兼容接口发 `max_tokens`；不设置就不发 | `max_tokens`，不设置时为 16384 | `maxOutputTokens`，不设置时为 16384 |
| temperature | 发送，默认 0.0 | 从不发送：现在的 Claude 模型不接受非默认值 | 设置了才发送 |

## 费用

```rust
let r = client.extract::<Contact>("...").await?;
println!("输入 {}，输出 {}，{:?} 美元", r.usage.input_tokens, r.usage.output_tokens, r.usage.cost);
```

费用按 [tiktoken](https://crates.io/crates/tiktoken) 的价格表计算（`cost-tracking` feature，默认开启，不会链接任何词表数据）。每个请求单独计价，重试的费用累加；价格表里没有的模型，`cost` 是 `None`。模型 ID 由 tiktoken 解析，所以各家接口自己的写法（用短横线、带日期后缀、带 Bedrock 或 Vertex 前缀）也能算出价格。

```toml
instructors = { version = "1", default-features = false } # 不计算费用
```

## 性能

2026 年 11 月 25 日在 Apple M4 Pro Mac mini（14 核）上测量，测量期间独占基准测试锁；release 构建，开启 LTO，codegen-units 为 1，rustc 1.100.0。每 15 秒采样一次的 1 分钟平均负载在整个过程中（包括编译）介于 1.6 和 4.8 之间。每个数字是 5 到 7 轮的中位数。1.4.1 和 1.5 两个构建交替各跑了两遍；表中是第一遍的结果，第二遍和第一遍相比，「比基线多出」一列相差不超过 1.2 µs，其他数字相差不超过 4%。测试数据是一张 `Invoice`：两个嵌套地址、一个枚举、8 个带可选字段的明细行，含中日文字符和转义引号，JSON 共 1,247 字节。所有请求都发给本机回环的 HTTP 服务，不调用任何真实接口。基准测试程序不在本仓库里。

**一次 `extract` 调用的完整开销**（在保持连接的 HTTP/1.1 上顺序调用，5 × 5,000 次；基线是同一个进程里用同一个 `reqwest` client 发送事先构造好的请求体，再用 `serde_json` 解析响应）：

| 客户端 | 每次调用 | 比基线多出 |
|---|---:|---:|
| 基线 | 50.7 µs | |
| instructors 1.5 | 53.0 µs | 2.3 µs |
| instructors 1.4.1（单独一遍，那一遍的基线是 49.1 µs）| 66.4 µs | 17.3 µs |
| rstructor 0.6.0 | 75.8 µs | 25.1 µs |
| rig-core 0.47.1（`output_schema`，再用 `serde_json::from_str`）| 79.1 µs | 28.4 µs |

真实的模型调用要几百毫秒甚至更久，所以这几微秒影响的是大批量处理和代理服务的 CPU 开销，不是单次调用的延迟。

**每次请求花在 schema 上的时间：**

| 做了什么 | 耗时 |
|---|---:|
| instructors 1.5：把缓存好的、可直接发送的 schema 字节复制进请求体 | 30 ns |
| instructors 1.4.1：把缓存的 schema 转换成 OpenAI 的格式，再序列化 | 10.8 µs + 1.1 µs |
| rstructor 0.6.0：`Invoice::schema().to_json()` | 7.7 µs |
| 作为参照：不带缓存的 `schemars::schema_for!(Invoice)` | 3.8 µs |

**流式输出时取部分值**（1,247 字节的回答分 308 段、每段 4 字节到达，每段到达后都取一次部分值，下表是整个流的总耗时）：

| 做法 | 整个流 |
|---|---:|
| instructors 1.5 的 `PartialParser`：喂入每一段，读借用的视图 | 5.5 µs |
| 同上，另外每段之后复制成一个 `serde_json::Value` | 307 µs |
| 每段之后重新解析整个缓冲区，得到 `Value`（rstructor 0.6.0 的做法）| 677 µs |
| 每段之后对整个缓冲区运行 partial-json-fixer 0.5.5，再解析 | 735 µs |
| 每段之后对整个缓冲区运行 jsonrepair 0.1.0，再解析 | 2,035 µs |
| 作为参照：对完整回答做一次 `serde_json::from_str::<Invoice>` | 1.6 µs |

不管字节怎么切分，解析器做的工作都一样，所以开销和流的长度成正比；每段都重新解析的做法，开销和长度的平方成正比。

## 许可证

MIT 或 Apache-2.0，任选其一：[MIT](LICENSE-MIT)、[Apache-2.0](LICENSE-APACHE)。
