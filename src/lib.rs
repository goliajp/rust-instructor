//! # instructors
//!
//! Type-safe structured output extraction from LLMs.
//!
//! Define a Rust struct, and `instructors` will make the LLM return data that
//! deserializes directly into it — with automatic schema generation, validation,
//! and retry on failure.
//!
//! ## Quick start
//!
//! ```rust,no_run
//! use instructors::prelude::*;
//!
//! #[derive(Debug, Deserialize, JsonSchema)]
//! struct Contact {
//!     name: String,
//!     email: Option<String>,
//!     phone: Option<String>,
//! }
//!
//! # async fn run() -> instructors::Result<()> {
//! let client = Client::openai("sk-...");
//! let result: ExtractResult<Contact> = client
//!     .extract("Contact John Doe at john@example.com")
//!     .model("gpt-4o")
//!     .await?;
//!
//! println!("{}: {:?}", result.value.name, result.value.email);
//! println!("tokens: {}, cost: {:?}", result.usage.total_tokens, result.usage.cost);
//! # Ok(())
//! # }
//! ```
//!
//! ## Validation
//!
//! ```rust,no_run
//! use instructors::prelude::*;
//!
//! #[derive(Debug, Deserialize, JsonSchema)]
//! struct User {
//!     name: String,
//!     age: u32,
//! }
//!
//! # async fn run() -> instructors::Result<()> {
//! let client = Client::openai("sk-...");
//!
//! // closure-based validation
//! let user: User = client.extract("...")
//!     .validate(|u: &User| {
//!         if u.age > 150 { Err("age unrealistic".into()) } else { Ok(()) }
//!     })
//!     .await?.value;
//! # Ok(())
//! # }
//! ```
//!
//! ## Features
//!
//! - **Multi-provider** — OpenAI (`response_format` strict), Anthropic (`tool_use`),
//!   plus any OpenAI/Anthropic-compatible API
//! - **List extraction** — `extract_many::<T>()` returns `Vec<T>`
//! - **Batch processing** — `extract_batch::<T>()` with configurable concurrency
//! - **Multi-turn** — `.messages()` for conversation history
//! - **Validation** — closure-based `.validate()` or trait-based `.validated()`
//! - **Lifecycle hooks** — `.on_request()` / `.on_response()`
//! - **Cost tracking** — token counting and cost estimation via `tiktoken` (optional)

mod batch;
mod client;
mod error;
mod provider;
mod schema;
mod usage;
mod validate;

pub use batch::BatchBuilder;
pub use client::{Client, ExtractBuilder, ExtractResult};
pub use error::{Error, Result};
pub use provider::Message;
pub use usage::Usage;
pub use validate::{Validate, ValidationError};

// re-export for user convenience
pub use schemars::JsonSchema;
pub use serde;

/// Common imports for working with instructors.
///
/// ```rust
/// use instructors::prelude::*;
/// ```
pub mod prelude {
    pub use crate::{
        BatchBuilder, Client, ExtractResult, Message, Usage, Validate, ValidationError,
    };
    pub use schemars::JsonSchema;
    pub use serde::Deserialize;
}

#[cfg(test)]
mod tests {
    use super::*;
    use schemars::JsonSchema;
    use serde::Deserialize;

    #[derive(Debug, Deserialize, JsonSchema)]
    struct TestStruct {
        name: String,
        age: u32,
    }

    #[derive(Debug, Deserialize, JsonSchema)]
    struct WithOptional {
        title: String,
        subtitle: Option<String>,
    }

    #[derive(Debug, Deserialize, JsonSchema)]
    enum Category {
        Bug,
        Feature,
        Question,
    }

    #[test]
    fn deserialize_from_json() {
        let json = r#"{"name": "Alice", "age": 30}"#;
        let result: TestStruct = serde_json::from_str(json).unwrap();
        assert_eq!(result.name, "Alice");
        assert_eq!(result.age, 30);
    }

    #[test]
    fn optional_field_present() {
        let json = r#"{"title": "Hello", "subtitle": "World"}"#;
        let result: WithOptional = serde_json::from_str(json).unwrap();
        assert_eq!(result.subtitle, Some("World".into()));
    }

    #[test]
    fn optional_field_null() {
        let json = r#"{"title": "Hello", "subtitle": null}"#;
        let result: WithOptional = serde_json::from_str(json).unwrap();
        assert_eq!(result.subtitle, None);
    }

    #[test]
    fn optional_field_missing() {
        let json = r#"{"title": "Hello"}"#;
        let result: WithOptional = serde_json::from_str(json).unwrap();
        assert_eq!(result.subtitle, None);
    }

    #[test]
    fn enum_deserialize() {
        let json = r#""Bug""#;
        let result: Category = serde_json::from_str(json).unwrap();
        assert!(matches!(result, Category::Bug));

        let json = r#""Feature""#;
        let result: Category = serde_json::from_str(json).unwrap();
        assert!(matches!(result, Category::Feature));

        let json = r#""Question""#;
        let result: Category = serde_json::from_str(json).unwrap();
        assert!(matches!(result, Category::Question));
    }

    #[test]
    fn schema_generation() {
        let schema = schemars::schema_for!(TestStruct);
        let value = serde_json::to_value(&schema).unwrap();
        assert_eq!(value["type"], "object");
        assert!(value["properties"]["name"].is_object());
        assert!(value["properties"]["age"].is_object());
    }

    #[test]
    fn usage_accumulate() {
        let mut usage = Usage::default();
        usage.accumulate(100, 50);
        assert_eq!(usage.input_tokens, 100);
        assert_eq!(usage.output_tokens, 50);
        assert_eq!(usage.total_tokens, 150);

        usage.accumulate(200, 100);
        assert_eq!(usage.input_tokens, 300);
        assert_eq!(usage.output_tokens, 150);
        assert_eq!(usage.total_tokens, 450);
    }

    #[test]
    fn prelude_re_exports() {
        // verify all prelude items are accessible
        fn _check() {
            use crate::prelude::*;
            let _: fn() -> std::result::Result<(), ValidationError> = || Ok(());
            fn _accepts_client(_: &Client) {}
            fn _accepts_usage(_: &Usage) {}
        }
    }

    #[test]
    fn re_exports_available() {
        // verify top-level re-exports
        let _: fn() -> Result<()> = || Ok(());
        fn _check_json_schema<T: JsonSchema>() {}
        _check_json_schema::<TestStruct>();
    }
}
