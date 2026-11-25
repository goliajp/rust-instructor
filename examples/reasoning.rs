//! Set how much the model may think before answering.
//!
//! ```bash
//! GEMINI_API_KEY=AIza... cargo run --example reasoning
//! ```

use instructors::prelude::*;

#[expect(dead_code, reason = "fields are only filled by deserialization")]
#[derive(Debug, Deserialize, JsonSchema)]
struct Invoice {
    vendor: String,
    total: f64,
    currency: String,
}

#[tokio::main]
async fn main() -> instructors::Result<()> {
    let api_key = std::env::var("GEMINI_API_KEY").expect("set GEMINI_API_KEY");

    // copying fields out of a document needs little reasoning
    let client = Client::gemini(&api_key).with_reasoning(Reasoning::Effort(Effort::Low));

    let result = client
        .extract::<Invoice>("Invoice from ACME GmbH, total due EUR 1,234.50")
        .await?;

    println!("{:#?}", result.value);
    println!(
        "output tokens, thinking included: {}",
        result.usage.output_tokens
    );
    Ok(())
}
