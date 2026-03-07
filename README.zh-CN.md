# instructors

[![Crates.io](https://img.shields.io/crates/v/instructors?style=flat-square&logo=rust)](https://crates.io/crates/instructors)
[![docs.rs](https://img.shields.io/docsrs/instructors?style=flat-square&logo=docs.rs)](https://docs.rs/instructors)
[![License](https://img.shields.io/crates/l/instructors?style=flat-square)](LICENSE)

[English](README.md) | **简体中文** | [日本語](README.ja.md)

类型安全的 LLM 结构化输出提取。Rust 版 [instructor](https://github.com/jxnl/instructor)。

定义一个 Rust 结构体 → instructors 自动生成 JSON Schema → LLM 返回合规 JSON → 直接反序列化为类型化的值。解析失败时自动重试。

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
instructors = "0.1"
```

## 支持的供应商

| 供应商 | 构造方法 | 机制 |
|---|---|---|
| OpenAI | `Client::openai(key)` | `response_format` 严格 JSON Schema |
| Anthropic | `Client::anthropic(key)` | `tool_use` 强制工具调用 |
| OpenAI 兼容 | `Client::openai_compatible(key, url)` | 同 OpenAI（DeepSeek、Together 等） |

```rust
// OpenAI
let client = Client::openai("sk-...");

// Anthropic
let client = Client::anthropic("sk-ant-...");

// DeepSeek、Together 或任何 OpenAI 兼容 API
let client = Client::openai_compatible("sk-...", "https://api.deepseek.com/v1");
```

## 分类任务

枚举天然适用于分类场景：

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

## 嵌套类型

支持复杂的嵌套结构——Vec、Option、枚举均可：

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

## 配置

```rust
let result: MyStruct = client
    .extract("input text")
    .model("gpt-4o-mini")            // 指定模型
    .system("You are an expert...")   // 自定义系统提示
    .temperature(0.0)                 // 确定性输出
    .max_tokens(2048)                 // 限制输出 token 数
    .max_retries(3)                   // 解析失败时重试次数
    .context("extra context...")      // 追加上下文
    .await?
    .value;
```

## 客户端默认值

设置一次默认值，按需在单个请求中覆盖：

```rust
let client = Client::openai("sk-...")
    .with_model("gpt-4o-mini")
    .with_temperature(0.0)
    .with_max_retries(3)
    .with_system("Extract data precisely.");

// 所有提取请求使用上述默认值
let a: TypeA = client.extract("...").await?.value;
let b: TypeB = client.extract("...").await?.value;

// 在特定请求中覆盖
let c: TypeC = client.extract("...").model("gpt-4o").await?.value;
```

## 成本追踪

内置 token 计数和成本估算（基于 [tiktoken](https://crates.io/crates/tiktoken)）：

```rust
let result = client.extract::<Contact>("...").await?;

println!("input:  {} tokens", result.usage.input_tokens);
println!("output: {} tokens", result.usage.output_tokens);
println!("cost:   ${:.6}", result.usage.cost.unwrap_or(0.0));
println!("retries: {}", result.usage.retries);
```

不需要时可通过 `default-features = false` 禁用：

```toml
[dependencies]
instructors = { version = "0.1", default-features = false }
```

## 工作原理

1. `#[derive(JsonSchema)]` 通过 [schemars](https://crates.io/crates/schemars) 从 Rust 类型生成 JSON Schema
2. Schema 根据目标供应商进行转换：
   - **OpenAI**：封装为 `response_format`，启用严格模式（`additionalProperties: false`，所有字段 required）
   - **Anthropic**：封装为 `tool`，通过 `tool_choice` 强制调用
3. LLM 被约束只能生成符合 Schema 的有效 JSON
4. 响应通过 `serde_json::from_str::<T>()` 反序列化
5. 解析失败时，错误信息反馈给 LLM 并重试

## 许可证

[MIT](LICENSE)
