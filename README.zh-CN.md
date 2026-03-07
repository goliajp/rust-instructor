# instructors

[![Crates.io](https://img.shields.io/crates/v/instructors?style=flat-square&logo=rust)](https://crates.io/crates/instructors)
[![docs.rs](https://img.shields.io/docsrs/instructors?style=flat-square&logo=docs.rs)](https://docs.rs/instructors)
[![License](https://img.shields.io/crates/l/instructors?style=flat-square)](LICENSE)

[English](README.md) | **简体中文** | [日本語](README.ja.md)

类型安全的 LLM 结构化输出提取。Rust 版 [instructor](https://github.com/jxnl/instructor)。

定义一个 Rust 结构体 → instructors 自动生成 JSON Schema → LLM 返回合法 JSON → 你得到一个强类型值。支持自动校验与重试。

## 快速开始

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

## 安装

```toml
[dependencies]
instructors = "1"
```

## 服务提供商

| 提供商 | 构造方法 | 机制 |
|---|---|---|
| OpenAI | `Client::openai(key)` | `response_format` 严格 JSON Schema |
| Anthropic | `Client::anthropic(key)` | `tool_use` 强制工具调用 |
| OpenAI 兼容 | `Client::openai_compatible(key, url)` | 与 OpenAI 相同（DeepSeek、Together 等） |
| Anthropic 兼容 | `Client::anthropic_compatible(key, url)` | 与 Anthropic 相同 |

```rust
// OpenAI
let client = Client::openai("sk-...");

// Anthropic
let client = Client::anthropic("sk-ant-...");

// DeepSeek、Together 或任何 OpenAI 兼容 API
let client = Client::openai_compatible("sk-...", "https://api.deepseek.com/v1");

// Anthropic 兼容代理
let client = Client::anthropic_compatible("sk-...", "https://proxy.example.com/v1");
```

## 校验

对提取的数据进行校验并自动重试——校验失败时，会将错误详情反馈给 LLM 重新生成。

### 基于闭包

```rust
let user: User = client.extract("...")
    .validate(|u: &User| {
        if u.age > 150 { Err("age must be <= 150".into()) } else { Ok(()) }
    })
    .await?.value;
```

### 基于 trait

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

## 列表提取

使用 `extract_many` 从文本中提取多个条目：

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

## 批量处理

通过可配置的并发度并行处理多个提示：

```rust
let prompts = vec!["review 1".into(), "review 2".into(), "review 3".into()];

let results = client
    .extract_batch::<Review>(prompts)
    .concurrency(5)
    .validate(|r: &Review| { /* ... */ Ok(()) })
    .run()
    .await;

// 每个结果独立——部分失败不影响其他结果
for result in results {
    match result {
        Ok(r) => println!("{:?}", r.value),
        Err(e) => eprintln!("failed: {e}"),
    }
}
```

## 多轮对话

传入消息历史以实现上下文感知的提取：

```rust
use instructors::Message;

let result = client.extract::<Summary>("summarize the above")
    .messages(vec![
        Message::user("Here is a long document..."),
        Message::assistant("I see the document."),
    ])
    .await?;
```

## 流式输出

实时接收部分 JSON token：

```rust
let result = client.extract::<Contact>("...")
    .on_stream(|chunk| {
        print!("{chunk}");  // 部分 JSON 片段
    })
    .await?;
```

OpenAI 和 Anthropic 提供商均支持流式输出。最终结果由所有片段拼接后反序列化。

## 图片输入

使用视觉模型从图片中提取结构化数据：

```rust
use instructors::ImageInput;

// 通过 URL
let result = client.extract::<Description>("描述这张图片")
    .image(ImageInput::Url("https://example.com/photo.jpg".into()))
    .model("gpt-4o")
    .await?;

// 通过 base64
let result = client.extract::<Description>("描述这张图片")
    .image(ImageInput::Base64 {
        media_type: "image/png".into(),
        data: base64_string,
    })
    .await?;

// 多张图片
let result = client.extract::<Comparison>("对比这些图片")
    .images(vec![
        ImageInput::Url("https://example.com/a.jpg".into()),
        ImageInput::Url("https://example.com/b.jpg".into()),
    ])
    .await?;
```

## 提供商故障转移

链式配置多个提供商实现自动故障转移：

```rust
let client = Client::openai("sk-...")
    .with_fallback(Client::anthropic("sk-ant-..."))
    .with_fallback(Client::openai_compatible("sk-...", "https://api.deepseek.com/v1"));

// 优先尝试 OpenAI → 失败后尝试 Anthropic → 最后尝试 DeepSeek
let result = client.extract::<Contact>("...").await?;
```

每个备选提供商按顺序尝试，仅在主提供商耗尽重试次数后触发。

## 生命周期钩子

观察请求和响应：

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

## 分类

枚举天然适用于分类任务：

```rust
#[derive(Debug, Deserialize, JsonSchema)]
enum Sentiment { Positive, Negative, Neutral }

let sentiment: Sentiment = client
    .extract("This product is amazing!")
    .await?.value;
```

## 嵌套类型

支持包含向量、可选值和枚举的复杂嵌套结构：

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

## 配置

```rust
let result: MyStruct = client
    .extract("input text")
    .model("gpt-4o-mini")            // 覆盖模型
    .system("You are an expert...")   // 自定义系统提示
    .temperature(0.0)                 // 确定性输出
    .max_tokens(2048)                 // 限制输出 token 数
    .max_retries(3)                   // 解析/校验失败时重试
    .context("extra context...")      // 追加到提示
    .await?
    .value;
```

## 客户端默认值

设置一次默认值，按需覆盖：

```rust
let client = Client::openai("sk-...")
    .with_model("gpt-4o-mini")
    .with_temperature(0.0)
    .with_max_retries(3)
    .with_system("Extract data precisely.");

// 所有提取操作使用上述默认值
let a: TypeA = client.extract("...").await?.value;
let b: TypeB = client.extract("...").await?.value;

// 针对特定请求覆盖默认值
let c: TypeC = client.extract("...").model("gpt-4o").await?.value;
```

## 费用追踪

通过 [tiktoken](https://crates.io/crates/tiktoken) 内置 token 计数与费用估算：

```rust
let result = client.extract::<Contact>("...").await?;

println!("input:  {} tokens", result.usage.input_tokens);
println!("output: {} tokens", result.usage.output_tokens);
println!("cost:   ${:.6}", result.usage.cost.unwrap_or(0.0));
println!("retries: {}", result.usage.retries);
```

通过 `default-features = false` 禁用：

```toml
[dependencies]
instructors = { version = "1", default-features = false }
```

## 工作原理

1. `#[derive(JsonSchema)]` 从你的 Rust 类型生成 JSON Schema（基于 [schemars](https://crates.io/crates/schemars)）
2. Schema 按类型缓存（线程本地存储，零锁竞争）
3. Schema 根据目标提供商进行转换：
   - **OpenAI**：封装在 `response_format` 中，启用严格模式（`additionalProperties: false`，所有字段必填）
   - **Anthropic**：封装为 `tool`，设置 `input_schema`，通过 `tool_choice` 强制调用
4. LLM 被约束为生成符合 Schema 的合法 JSON
5. 响应通过 `serde_json::from_str::<T>()` 反序列化
6. 若实现了 `Validate` trait 或设置了 `.validate()` 闭包，则执行校验
7. 解析或校验失败时，将错误反馈发回 LLM 并重试

## 许可证

[MIT](LICENSE)
