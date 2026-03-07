//! Validation example — extract with constraints.
//!
//! Run with:
//! ```bash
//! OPENAI_API_KEY=sk-... cargo run --example validation
//! ```

use instructors::prelude::*;

#[derive(Debug, Deserialize, JsonSchema)]
struct UserProfile {
    name: String,
    age: u32,
    email: String,
}

impl Validate for UserProfile {
    fn validate(&self) -> Result<(), ValidationError> {
        if self.name.is_empty() {
            return Err("name must not be empty".into());
        }
        if self.age > 150 {
            return Err(format!("age {} is unrealistic, must be <= 150", self.age).into());
        }
        if !self.email.contains('@') {
            return Err(format!("'{}' is not a valid email address", self.email).into());
        }
        Ok(())
    }
}

#[tokio::main]
async fn main() -> instructors::Result<()> {
    let api_key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set");
    let client = Client::openai(&api_key).with_model("gpt-4o-mini");

    // trait-based validation (auto-retries on failure)
    let user = client
        .extract::<UserProfile>("Alice is 30 years old, her email is alice@example.com")
        .validated()
        .await?;

    println!("User: {:#?}", user.value);
    println!("Retries: {}", user.usage.retries);

    // closure-based validation
    let user = client
        .extract::<UserProfile>("Bob, age 25, bob@company.org")
        .validate(|u: &UserProfile| {
            if u.name.len() < 2 {
                Err("name too short".into())
            } else {
                Ok(())
            }
        })
        .await?;

    println!("\nUser: {:#?}", user.value);

    Ok(())
}
