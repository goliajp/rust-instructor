# instructors

[![crates.io](https://img.shields.io/crates/v/instructors?style=flat-square&logo=rust)](https://crates.io/crates/instructors)
[![docs.rs](https://img.shields.io/docsrs/instructors?style=flat-square&logo=docs.rs)](https://docs.rs/instructors)
[![CI](https://img.shields.io/github/actions/workflow/status/goliajp/rust-instructor/ci.yml?branch=develop&style=flat-square&logo=github&label=ci)](https://github.com/goliajp/rust-instructor/actions/workflows/ci.yml)
[![License](https://img.shields.io/crates/l/instructors?style=flat-square)](#许可证)

[English](README.md) | **简体中文** | [日本語](README.ja.md)

> **从 1.x 升级**：类型上的 `JsonSchema` 换成 `Schema`；对 `Error` 做 `match` 时加一个 `_ =>` 分支；`usage.total_tokens` 改成方法调用 `usage.total_tokens()`。另外，schema 里的字段现在按声明顺序排列，HTTP 重试默认开启，重新提问时模型能看到它上一次的回答。完整清单和改写前后的代码见 [CHANGELOG](CHANGELOG.md#upgrading-from-1x)。

用 Rust 从 LLM 拿到带类型的结构化输出。给一个类型加上 `Deserialize` 和 `Schema`，调用 `extract`，返回的就是这个类型的值。crate 按各家接口能强制执行的格式发送这个类型的 JSON schema；模型返回的 JSON 有格式错误时先修复；解析失败或者你自己的校验不通过时，把模型的回答和错误一起发回去再问一次。设计上参考了 Python 的 [instructor](https://github.com/567-labs/instructor)。

crate 自己不依赖任何第三方运行时库：schema 派生宏、JSON 解析、修复和流式解析都在 crate 里实现，你的类型只需要和 serde 打交道。HTTP 客户端是一个 trait，默认实现是带 rustls 的 reqwest。用 `default-features = false` 时，crate 只负责生成请求和解析响应，本身不做任何 I/O，也不需要异步运行时，可以编译到 `wasm32-unknown-unknown`。

把 OpenAI 的响应体解析成一个 1.2 KB 的 `Invoice` 要 3.0 µs，用 `serde_json` 要 4.1 µs。通过本机回环 HTTP 完整调用一次 `extract` 要 53 到 54 µs，和同一进程里直接用 `reqwest` 发 POST 再解析差不多；rstructor 0.6.0 要 77 µs，rig-core 0.47.1 要 81 µs。测量条件和全部数据见[性能](#性能)。

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
| `Client::deepseek(key)` | DeepSeek Chat Completions | JSON 模式（`response_format: json_object`）|
| `Client::qwen(key)` | 通义千问，阿里云百炼的 OpenAI 兼容模式（新加坡）| JSON 模式 |
| `Client::anthropic(key)` | Anthropic Messages | `output_config.format`（结构化输出）|
| `Client::anthropic_compatible(key, url)` | 同上 | 一个工具，`tool_choice` 强制调用 |
| `Client::gemini(key)`、`gemini_compatible(key, url)` | Gemini `generateContent` | `generationConfig.responseFormat.text`，带 JSON Schema |

`with_output_mode` 可以改变发送 schema 的方式：

| `OutputMode` | 发送的内容 |
|---|---|
| `Schema` | 接口原生的结构化输出（上表）|
| `Tool` | 一个以 schema 为输入的工具，模型允许时强制调用；Gemini 和 Responses API 不支持 |
| `JsonObject` | JSON 模式（`response_format: json_object`），schema 写进系统提示词；Anthropic 没有 JSON 模式，按 `Prompt` 处理 |
| `Prompt` | 不发格式参数，schema 写进系统提示词，从回答文本里取出 JSON |

`openai_compatible` 会看 base URL 的主机名：`api.openai.com` 发 `max_completion_tokens`；DeepSeek（`deepseek.com`）、通义千问（`aliyuncs.com`）、Kimi（`moonshot.cn`、`moonshot.ai`、`kimi.ai`、`kimi.com`）和 GLM（`bigmodel.cn`、`z.ai`）各用自己的[思考参数](#思考深度)。DeepSeek、通义千问和 GLM 默认用 `JsonObject`，因为它们的文档只写了 JSON 模式，没有写严格 schema；其他主机按通用的 OpenAI 兼容接口处理，用 `Schema`。

引用自身的类型（树、链表）用 `$defs` 和 `$ref` 发送。Anthropic 的结构化输出表达不了递归，所以这种类型发给 Anthropic 时改用工具。Gemini 只把递归引用展开几层，文档里只说可以用在非必填的属性上。`children: Vec<Node>` 这样必填的递归字段会原样发送，Gemini 可能拒绝，也可能只生成到某一层。

OpenAI 的严格模式 schema 和 Anthropic 的结构化输出只接受 JSON Schema 的一个子集，所以有些类型换一种形式发送，收到回答后先转回原来的形式，再交给你的类型。map（`HashMap`、`BTreeMap`）发成由 `{"key": .., "value": ..}` 对象组成的数组；元组发成以 `"0"`、`"1"`、… 为键的对象；JSON 不是对象的类型（enum、`Vec`、数字）包在 `{"value": ..}` 里。接口不接受的约束（两家都不接受 `uniqueItems`，OpenAI 不接受 `minLength` 和 `maxLength`，Anthropic 不接受取值范围和 `maxItems`）不写进发给它的 schema，改为收到回答后检查；违反约束的回答和其他不合格的回答一样，会再问一次。`on_item` 和最终结果拿到的总是你的类型自己的 JSON；`on_partial` 拿到的根已经去掉了外层包装，但里面的 map 和元组还是发送时的形式。

能装任意值的类型（schema 是 `true` 的字段，比如手写的 `Schema` 返回 `Shape::Any`，或者通过 `schemars` feature 使用的 `serde_json::Value`）写不成严格模式的 schema。这时发给 OpenAI 的是普通 schema，`strict: false`；发给 Anthropic 时改用工具；回答由 crate 按 schema 检查。Gemini 会忽略 `minLength`、`maxLength` 和 `uniqueItems`，所以发给它的 schema 不带这几个关键字，回答同样由 crate 检查。

没有结构化输出的 Anthropic 模型（Claude 3、Claude Opus 4 和 4.1、Claude Sonnet 4）改用工具；其他模型，包括比这个 crate 更新的模型，都发 `output_config.format`。Claude Opus 5.5、Sonnet 5.5、Fable 5.1 和 5.2、Mythos 5.1 和 5.2 不接受强制调用工具，所以用这些模型时，以及开启思考时，用 `tool_choice: auto` 提供工具，并在系统提示里要求调用它。DeepSeek 和通义千问在思考模式下不接受强制调用工具，而它们现在的模型默认就是思考模式，所以除非 reasoning 设为 `Off`，`Tool` 模式也这样处理。通义千问的 OpenAI 兼容模式不接受 PDF，传 `Part::document` 时会在发送前返回 `Error::Unsupported`。

不指定模型时的默认值：

| 接口 | 默认模型 |
|---|---|
| OpenAI 及兼容接口 | `gpt-5.6-terra` |
| DeepSeek（`deepseek.com` 的地址）| `deepseek-flash` |
| 通义千问（`aliyuncs.com` 的地址）| `qwen3.7-plus` |
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

Anthropic 的强制工具调用不能和思考同时使用，所以设置了 `Effort` 或 `Budget` 时，走工具的请求（`OutputMode::Tool`，也就是 `anthropic_compatible` 的默认方式，或者递归类型）改用 `tool_choice: auto`；设为 `Off` 时，只要模型允许就强制调用工具。

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
| HTTP 重试用完后，接口仍返回非 2xx 状态码，或者在流式响应中途发来错误 | `Error::Api { status, message, retry_after, .. }` |
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

HTTP 重试默认开启：遇到 408、409、429、500、502、503、504、529 和连接错误时最多重试 3 次，等待时间从 500 ms 开始翻倍，最多 30 s，带随机抖动；接口返回了 `retry-after-ms` 或 `Retry-After` 响应头时按它等。`Retry-After` 可以是秒数，也可以是 HTTP 日期，日期从响应的 `Date` 头开始算。OpenAI 和 Anthropic 会返回 `x-should-retry: true` 或 `false` 响应头，有这个头时以它为准，不看状态码。`RetryPolicy::none()` 关闭重试。主接口失败后才会尝试备用接口，备用接口用它自己的模型、输出方式和 transport。

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
| temperature | 设置了才发送 | 从不发送：现在的 Claude 模型不接受非默认值 | 设置了才发送 |

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

2026 年 11 月 26 日在 Apple M4 Pro Mac mini（14 核）上测量，测量期间独占基准测试锁；release 构建，开启 LTO，codegen-units 为 1，rustc 1.100.0，serde 1.0.229，serde_json 1.0.154，reqwest 0.13.5。每 15 秒采样一次 1 分钟平均负载，测量期间在 1.7 到 2.0 之间。每一行都测了两遍，两遍相隔几秒，表中是每一遍的中位数（端到端的各行 5 轮，其余 7 轮或 9 轮）。测试数据是一张 `Invoice`：两个嵌套地址、一个枚举、8 个带可选字段的明细行，含中日文字符和转义引号，JSON 共 1,247 字节。所有请求都发给本机回环的 HTTP 服务，不调用任何真实接口。JSON 和修复的数据用 `cargo bench --bench json --features fuzzing` 测得，其余各行的测试程序不在本仓库里。

**一次 `extract` 调用的完整耗时**（OpenAI 协议，在保持连接的 HTTP/1.1 上顺序调用，5 × 5,000 次）。基线是用同一版本的 `reqwest` 发送事先构造好的请求体，再用 `serde_json` 解析响应。

| 客户端 | 每次调用，第一遍 / 第二遍 |
|---|---:|
| 基线 | 53.0 / 53.3 µs |
| instructors 2.0 | 53.9 / 52.7 µs |
| instructors 1.5.1 | 55.2 / 53.9 µs |
| rstructor 0.6.0 | 77.4 / 77.0 µs |
| rig-core 0.47.1（`output_schema`，再用 `serde_json::from_str`）| 80.6 / 80.6 µs |

两遍之间，基线相差 0.3 µs，instructors 的两行相差 1.3 µs，所以这里分不出 2.0 和基线谁快；2.0 在两遍里都比 1.5.1 快 1.2 µs。Anthropic 工具调用形式的回答，2.0 要 52.8 / 52.3 µs，1.5.1 要 53.4 / 52.0 µs。回答分 311 个事件流式到达、每次都取部分值时，2.0 要 97.5 / 97.9 µs，1.5.1 要 97.2 / 98.2 µs。这两项看不出差别。真实的模型调用要几百毫秒甚至更久，所以这些时间影响的是大批量处理和代理服务的 CPU 开销。

不算网络，一次 OpenAI 抽取构造请求加解析回答共 3.6 µs，其中构造请求 0.39 µs。流式部分值解析器处理整个回答（308 段、每段 4 字节、每段之后取一次视图）要 5.8 / 5.7 µs，1.5.1 要 5.7 / 5.6 µs，两个版本的解析器源码相同。

**本 crate 的 JSON 代码和 serde_json 对比**，第一遍 / 第二遍：

| 做了什么 | serde_json | instructors 2.0 |
|---|---:|---:|
| 回答内容 → `Invoice` | 1,562 / 1,563 ns | 1,346 / 1,372 ns |
| OpenAI 响应体 → `Invoice`，外层借用 | 4,057 / 4,101 ns | 2,954 / 2,977 ns |
| Anthropic 工具调用回答 → `Invoice` | 3,057 / 3,058 ns | 2,497 / 2,496 ns |
| 一个流式数据块，只解析外层 | 231 / 229 ns | 184 / 185 ns |
| 把 4,167 字节的请求写进复用的缓冲区 | 1,380 / 1,346 ns | 806 / 812 ns |

**修复**，从输入到修复后的文本，第一遍 / 第二遍：

| 输入 | jsonrepair 0.1.0 | instructors 2.0 |
|---|---:|---:|
| 合法 JSON | 2.66 / 2.68 µs | 0.84 / 0.84 µs |
| 多余的尾逗号 | 7.79 / 7.66 µs | 2.90 / 2.90 µs |
| 单引号 | 6.39 / 6.05 µs | 3.18 / 3.22 µs |
| markdown 代码块 | 2.59 / 2.69 µs | 0.87 / 0.87 µs |
| 在 2/3 长度处截断 | 6.55 / 6.76 µs | 1.97 / 1.96 µs |

前四种输入，两个库修复后都能反序列化成 `Invoice`。截断的输入不能，因为缺必填字段；如果缺的字段都是可选的，可以用 `accept_truncated`。

**构建时的 crate 数**（`cargo tree -e normal`，去重，含本 crate，x86_64 Linux）：默认 89 个（1.5.1 是 111 个）；`default-features = false, features = ["derive"]` 时 9 个，其中 7 个是 serde 和它的 derive 用到的。1.5.1 关掉默认 feature 时是 108 个。

## 许可证

MIT 或 Apache-2.0，任选其一：[MIT](LICENSE-MIT)、[Apache-2.0](LICENSE-APACHE)。
