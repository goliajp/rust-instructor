//! Extract structured data from images using vision-capable models.
//!
//! ```bash
//! OPENAI_API_KEY=sk-... cargo run --example image
//! ```

use instructors::prelude::*;

#[derive(Debug, Deserialize, JsonSchema)]
struct ImageDescription {
    subject: String,
    colors: Vec<String>,
    mood: String,
}

#[tokio::main]
async fn main() -> instructors::Result<()> {
    let api_key = std::env::var("OPENAI_API_KEY").expect("set OPENAI_API_KEY");
    let client = Client::openai(&api_key);

    let result = client
        .extract::<ImageDescription>("Describe this image in detail")
        .model("gpt-4o")
        .image(ImageInput::Url(
            "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d5/Rust_programming_language_black_logo.svg/240px-Rust_programming_language_black_logo.svg.png".into(),
        ))
        .await?;

    println!("{:#?}", result.value);
    Ok(())
}
