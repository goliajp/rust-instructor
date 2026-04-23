//! Batch extraction example — process multiple prompts concurrently.
//!
//! Run with:
//! ```bash
//! OPENAI_API_KEY=sk-... cargo run --example batch
//! ```

use instructors::prelude::*;

#[derive(Debug, Deserialize, JsonSchema)]
struct ProductReview {
    product: String,
    sentiment: String,
    score: f64,
}

#[tokio::main]
async fn main() -> instructors::Result<()> {
    let api_key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set");
    let client = Client::openai(&api_key).with_model("gpt-4o-mini");

    let reviews = vec![
        "The new MacBook Pro is incredible, best laptop I've ever owned!".into(),
        "This phone case broke after just one week, terrible quality.".into(),
        "The headphones are decent for the price, nothing special.".into(),
        "Amazing camera quality on the new Pixel, highly recommended!".into(),
        "The keyboard is uncomfortable and the keys are too small.".into(),
    ];

    println!("Processing {} reviews concurrently...\n", reviews.len());

    let results = client
        .extract_batch::<ProductReview>(reviews)
        .concurrency(5)
        .validate(|r: &ProductReview| {
            if !(0.0..=10.0).contains(&r.score) {
                Err(format!("score {} must be 0-10", r.score).into())
            } else {
                Ok(())
            }
        })
        .run()
        .await;

    for (i, result) in results.into_iter().enumerate() {
        match result {
            Ok(r) => println!(
                "#{}: {} — {} (score: {:.1})",
                i + 1,
                r.value.product,
                r.value.sentiment,
                r.value.score
            ),
            Err(e) => println!("#{}: error — {e}", i + 1),
        }
    }

    Ok(())
}
