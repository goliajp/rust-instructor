//! Stream partial JSON as the LLM generates it.
//!
//! ```bash
//! OPENAI_API_KEY=sk-... cargo run --example streaming
//! ```

use instructors::prelude::*;

#[derive(Debug, Deserialize, JsonSchema)]
struct Summary {
    title: String,
    key_points: Vec<String>,
}

#[tokio::main]
async fn main() -> instructors::Result<()> {
    let api_key = std::env::var("OPENAI_API_KEY").expect("set OPENAI_API_KEY");
    let client = Client::openai(&api_key);

    let result = client
        .extract::<Summary>("Rust is a systems programming language focused on safety and performance.")
        .model("gpt-4o-mini")
        .on_stream(|chunk| {
            print!("{chunk}");
        })
        .await?;

    println!("\n\nparsed: {:#?}", result.value);
    Ok(())
}
