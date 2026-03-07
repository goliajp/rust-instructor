//! Entity extraction example — extract multiple items from text.
//!
//! Run with:
//! ```bash
//! OPENAI_API_KEY=sk-... cargo run --example entities
//! ```

use instructors::prelude::*;

#[derive(Debug, Deserialize, JsonSchema)]
struct Entity {
    /// The entity name
    name: String,
    /// The entity type (Person, Company, Location, etc.)
    entity_type: String,
}

#[tokio::main]
async fn main() -> instructors::Result<()> {
    let api_key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set");
    let client = Client::openai(&api_key).with_model("gpt-4o-mini");

    let result = client
        .extract_many::<Entity>(
            "Apple CEO Tim Cook met with Google CEO Sundar Pichai \
             in Cupertino to discuss AI partnerships.",
        )
        .await?;

    println!("Found {} entities:", result.value.len());
    for entity in &result.value {
        println!("  - {} ({})", entity.name, entity.entity_type);
    }
    println!(
        "\nTokens: {} input + {} output",
        result.usage.input_tokens, result.usage.output_tokens
    );

    Ok(())
}
