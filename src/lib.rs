//! # instructors
//!
//! Typed structured output from LLMs.
//!
//! Derive `Deserialize` and `JsonSchema` on a Rust type and call
//! [`Client::extract`]. The crate sends the type's JSON schema in the form
//! each provider enforces, parses the answer into the type, and when parsing
//! or your validation fails, sends the error back and asks again.
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
//!     .model("gpt-5.6-terra")
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
//! let user: User = client.extract("...")
//!     .validate(|u: &User| {
//!         if u.age > 150 { Err("age unrealistic".into()) } else { Ok(()) }
//!     })
//!     .await?.value;
//! # Ok(())
//! # }
//! ```
//!
//! ## What is in the crate
//!
//! - Three wire protocols: OpenAI Chat Completions (`response_format` with a
//!   strict schema), Anthropic Messages (`output_config.format`, or a tool),
//!   Gemini `generateContent` (`responseSchema`); each also works against a
//!   compatible server through the `_compatible` constructors
//! - [`Reasoning`]: one setting for thinking effort or budget, mapped to each
//!   provider's parameter
//! - [`ExtractBuilder::on_partial`]: the value parsed so far after every
//!   streamed delta, from the incremental parser in [`partial`]
//! - [`Client::extract_many`] for lists, [`Client::extract_batch`] for many
//!   prompts with bounded concurrency
//! - validation by closure or the [`Validate`] trait, with the error sent back
//!   to the model on retry
//! - repair of malformed JSON (trailing commas, single quotes, markdown
//!   fences) before a retry is spent
//! - fallback providers, HTTP backoff, an overall timeout, request and
//!   response hooks, image input
//! - token counts and cost estimates through `tiktoken` (feature
//!   `cost-tracking`, on by default); `tracing` spans (feature `tracing`)

mod backoff;
mod batch;
mod client;
mod error;
mod extract;
pub mod partial;
mod provider;
mod reasoning;
mod run;
mod schema;
mod sse;
mod usage;
mod validate;

pub use backoff::BackoffConfig;
pub use batch::BatchBuilder;
pub use client::{Client, ExtractResult};
pub use error::{Error, Result};
pub use extract::ExtractBuilder;
pub use partial::PartialValue;
pub use provider::{ImageInput, Message};
pub use reasoning::{Effort, Reasoning};
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
        BackoffConfig, BatchBuilder, Client, Effort, ExtractResult, ImageInput, Message,
        PartialValue, Reasoning, Usage, Validate, ValidationError,
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

    #[allow(dead_code)]
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
            fn _accepts_backoff(_: &BackoffConfig) {}
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
