//! Chain multiple providers for automatic failover.
//!
//! If the primary provider fails after exhausting retries,
//! the request is retried using fallback providers in order.
//!
//! ```bash
//! OPENAI_API_KEY=sk-... ANTHROPIC_API_KEY=sk-ant-... cargo run --example fallback
//! ```

use instructors::prelude::*;

#[derive(Debug, Deserialize, JsonSchema)]
struct Contact {
    name: String,
    email: Option<String>,
}

#[tokio::main]
async fn main() -> instructors::Result<()> {
    let openai_key = std::env::var("OPENAI_API_KEY").expect("set OPENAI_API_KEY");
    let anthropic_key = std::env::var("ANTHROPIC_API_KEY").expect("set ANTHROPIC_API_KEY");

    let client = Client::openai(&openai_key)
        .with_fallback(Client::anthropic(&anthropic_key));

    // tries OpenAI first; on failure, falls back to Anthropic
    let result = client
        .extract::<Contact>("Contact Jane Smith at jane@example.com")
        .await?;

    println!("{:#?}", result.value);
    Ok(())
}
