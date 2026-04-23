//! Hooks example — observe request/response lifecycle.
//!
//! Run with:
//! ```bash
//! OPENAI_API_KEY=sk-... cargo run --example hooks
//! ```

use instructors::prelude::*;

#[derive(Debug, Deserialize, JsonSchema)]
struct Contact {
    name: String,
    email: Option<String>,
}

#[tokio::main]
async fn main() -> instructors::Result<()> {
    let api_key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set");
    let client = Client::openai(&api_key).with_model("gpt-4o-mini");

    let result = client
        .extract::<Contact>("John Doe, john@example.com")
        .on_request(|model, prompt| {
            println!("[request] model={model}, prompt_len={}", prompt.len());
        })
        .on_response(|usage| {
            println!(
                "[response] tokens={}, retries={}, cost={:?}",
                usage.total_tokens, usage.retries, usage.cost
            );
        })
        .await?;

    println!("\nExtracted: {:?}", result.value);

    Ok(())
}
