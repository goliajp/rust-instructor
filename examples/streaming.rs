//! Watch the value fill in while the response streams.
//!
//! ```bash
//! OPENAI_API_KEY=sk-... cargo run --example streaming
//! ```

use instructors::prelude::*;

#[expect(dead_code, reason = "fields are only filled by deserialization")]
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
        .extract::<Summary>(
            "Rust is a systems programming language focused on safety and performance.",
        )
        .model("gpt-5.6-luna")
        .on_partial(|v| {
            let points = v.get("key_points").map_or(0, |p| p.len());
            let title = v.get("title").and_then(|t| t.as_str()).unwrap_or("");
            println!("title so far: {title:?}, key points so far: {points}");
        })
        .await?;

    println!("\n\nparsed: {:#?}", result.value);
    Ok(())
}
