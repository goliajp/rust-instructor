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
//! ## Providers
//!
//! - **OpenAI** — uses `response_format` with `json_schema` (strict mode)
//! - **Anthropic** — uses `tool_use` with forced tool choice
//! - **OpenAI-compatible** — any API implementing the OpenAI chat completions format

mod client;
mod error;
mod provider;
mod schema;
mod usage;

pub use client::{Client, ExtractBuilder, ExtractResult};
pub use error::{Error, Result};
pub use usage::Usage;

// re-export for user convenience
pub use schemars::JsonSchema;
pub use serde;

/// Common imports for working with instructors.
///
/// ```rust
/// use instructors::prelude::*;
/// ```
pub mod prelude {
    pub use crate::{Client, ExtractResult, Usage};
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
    fn test_deserialize_from_json() {
        let json = r#"{"name": "Alice", "age": 30}"#;
        let result: TestStruct = serde_json::from_str(json).unwrap();
        assert_eq!(result.name, "Alice");
        assert_eq!(result.age, 30);
    }

    #[test]
    fn test_optional_field_present() {
        let json = r#"{"title": "Hello", "subtitle": "World"}"#;
        let result: WithOptional = serde_json::from_str(json).unwrap();
        assert_eq!(result.subtitle, Some("World".into()));
    }

    #[test]
    fn test_optional_field_null() {
        let json = r#"{"title": "Hello", "subtitle": null}"#;
        let result: WithOptional = serde_json::from_str(json).unwrap();
        assert_eq!(result.subtitle, None);
    }

    #[test]
    fn test_optional_field_missing() {
        let json = r#"{"title": "Hello"}"#;
        let result: WithOptional = serde_json::from_str(json).unwrap();
        assert_eq!(result.subtitle, None);
    }

    #[test]
    fn test_enum_deserialize() {
        let json = r#""Bug""#;
        let result: Category = serde_json::from_str(json).unwrap();
        assert!(matches!(result, Category::Bug));
    }

    #[test]
    fn test_schema_generation() {
        let schema = schemars::schema_for!(TestStruct);
        let value = serde_json::to_value(&schema).unwrap();
        assert_eq!(value["type"], "object");
        assert!(value["properties"]["name"].is_object());
        assert!(value["properties"]["age"].is_object());
    }

    #[test]
    fn test_usage_accumulate() {
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
    fn test_error_display() {
        let err = Error::ExtractionFailed {
            retries: 3,
            message: "invalid json".into(),
        };
        assert!(err.to_string().contains("3 retries"));
        assert!(err.to_string().contains("invalid json"));
    }

    #[test]
    fn test_error_api() {
        let err = Error::Api {
            status: 429,
            message: "rate limited".into(),
        };
        assert!(err.to_string().contains("429"));
    }

    #[test]
    fn test_client_default_model() {
        let openai = Client::openai("key");
        match &openai.provider {
            crate::provider::ProviderKind::OpenAi { .. } => {}
            _ => panic!("wrong provider"),
        }

        let anthropic = Client::anthropic("key");
        match &anthropic.provider {
            crate::provider::ProviderKind::Anthropic { .. } => {}
            _ => panic!("wrong provider"),
        }
    }
}
