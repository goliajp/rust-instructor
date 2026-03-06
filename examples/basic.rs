//! Basic extraction example.
//!
//! Run with:
//! ```bash
//! OPENAI_API_KEY=sk-... cargo run --example basic
//! ```

use instructors::prelude::*;

#[derive(Debug, Deserialize, JsonSchema)]
struct Contact {
    name: String,
    email: Option<String>,
    phone: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
enum Sentiment {
    Positive,
    Negative,
    Neutral,
}

#[tokio::main]
async fn main() -> instructors::Result<()> {
    let api_key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set");
    let client = Client::openai(&api_key);

    // extract structured data
    let result: ExtractResult<Contact> = client
        .extract("Please contact John Doe at john@example.com or call 555-0123")
        .model("gpt-4o-mini")
        .await?;

    println!("Contact: {:#?}", result.value);
    println!(
        "Usage: {} input + {} output tokens",
        result.usage.input_tokens, result.usage.output_tokens
    );
    if let Some(cost) = result.usage.cost {
        println!("Estimated cost: ${:.6}", cost);
    }

    // classify text
    let sentiment: Sentiment = client
        .extract("This product is absolutely amazing, best purchase ever!")
        .model("gpt-4o-mini")
        .await?
        .value;

    println!("\nSentiment: {:?}", sentiment);

    Ok(())
}
