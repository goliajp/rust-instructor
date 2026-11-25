//! An extraction without an async runtime: the exchange says what to send,
//! and this example answers from a canned response instead of the network.
//!
//! ```bash
//! cargo run --example sans_io
//! ```

use instructors::prelude::*;

#[derive(Deserialize, Schema)]
struct Contact {
    /// Full name as written.
    name: String,
    email: Option<String>,
}

// stands in for any blocking HTTP client
fn http(req: &HttpRequest) -> (u16, Headers, Vec<u8>) {
    println!("{} {} ({} bytes)", req.method, req.url, req.body.len());
    let body = r#"{"choices":[{"message":{"content":"{\"name\":\"John Doe\",\"email\":\"john@example.com\"}"},"finish_reason":"stop"}],"usage":{"prompt_tokens":40,"completion_tokens":12}}"#;
    (200, Headers::new(), body.as_bytes().to_vec())
}

fn main() {
    let client = Client::openai("sk-...");
    let mut ex = client
        .extract::<Contact>("Contact John Doe at john@example.com")
        .exchange();
    let result = loop {
        match ex.next() {
            Step::Send(req) => {
                let (status, headers, body) = http(&req);
                ex.receive(status, &headers, &body);
            }
            Step::Wait(d) => std::thread::sleep(d),
            Step::Done(r) => break r,
        }
    };
    match result {
        Ok(out) => println!(
            "{} <{}>, {} tokens",
            out.value.name,
            out.value.email.as_deref().unwrap_or("-"),
            out.usage.total_tokens()
        ),
        Err(e) => eprintln!("{e}"),
    }
}
