// Byte-exact request bodies per wire and output mode. A change to any body
// fails here; after checking the diff, rewrite the files with
// `GOLDEN_BLESS=1 cargo test --test golden`.

use instructors::{Client, Effort, Message, OpenAiApi, OutputMode, Part, Reasoning, Schema, Step};
use serde::Deserialize;

#[expect(dead_code, reason = "fields are only filled by deserialization")]
#[derive(Debug, Deserialize, Schema)]
/// A person mentioned in the text.
struct Contact {
    /// Full name as written.
    name: String,
    email: Option<String>,
    #[serde(default)]
    tags: Vec<String>,
}

fn body(client: &Client, mode: Option<OutputMode>, stream: bool) -> (String, String) {
    let mut b = client
        .extract::<Contact>("Contact John at john@x.com")
        .system("Extract contacts.")
        .max_tokens(500)
        .reasoning(Reasoning::Effort(Effort::Low))
        .messages([
            Message::user("earlier").with(Part::image_url("https://x/i.png")),
            Message::assistant("noted"),
        ]);
    if let Some(m) = mode {
        b = b.output_mode(m);
    }
    if stream {
        b = b.on_delta(|_| {});
    }
    match b.exchange().next() {
        Step::Send(r) => (r.url, String::from_utf8(r.body).unwrap()),
        other => panic!("{other:?}"),
    }
}

fn check(name: &str, client: Client, mode: Option<OutputMode>, stream: bool) {
    let (url, got) = body(&client, mode, stream);
    let path = format!("{}/tests/golden/{name}.json", env!("CARGO_MANIFEST_DIR"));
    if std::env::var_os("GOLDEN_BLESS").is_some() {
        std::fs::write(&path, format!("{url}\n{got}\n")).unwrap();
        return;
    }
    let want = std::fs::read_to_string(&path).unwrap_or_else(|_| panic!("missing {path}"));
    assert_eq!(format!("{url}\n{got}\n"), want, "{name}");
}

const B: &str = "http://h";

#[test]
fn request_bodies() {
    check(
        "openai_schema",
        Client::openai_compatible("k", B),
        None,
        false,
    );
    check(
        "openai_schema_stream",
        Client::openai_compatible("k", B),
        None,
        true,
    );
    check(
        "openai_tool",
        Client::openai_compatible("k", B),
        Some(OutputMode::Tool),
        false,
    );
    check(
        "openai_json_object",
        Client::openai_compatible("k", B),
        Some(OutputMode::JsonObject),
        false,
    );
    check(
        "openai_prompt",
        Client::openai_compatible("k", B),
        Some(OutputMode::Prompt),
        false,
    );
    let resp = Client::openai_compatible("k", B).with_api(OpenAiApi::Responses);
    check("responses_schema", resp.clone(), None, true);
    check(
        "responses_json_object",
        resp,
        Some(OutputMode::JsonObject),
        false,
    );
    check(
        "anthropic_schema",
        Client::anthropic_compatible("k", B),
        Some(OutputMode::Schema),
        false,
    );
    check(
        "anthropic_tool",
        Client::anthropic_compatible("k", B),
        None,
        true,
    );
    check(
        "anthropic_prompt",
        Client::anthropic_compatible("k", B),
        Some(OutputMode::Prompt),
        false,
    );
    check(
        "gemini_schema",
        Client::gemini_compatible("k", B),
        None,
        false,
    );
    check(
        "gemini_json_object",
        Client::gemini_compatible("k", B),
        Some(OutputMode::JsonObject),
        true,
    );
}
