// Types no strict schema can carry (a value of any type), constraints
// Gemini ignores, and integers beyond 2^53 on the converted-answer path.

use std::collections::{BTreeMap, HashMap, HashSet};

use instructors::Schema;
use instructors::schema::Shape;
use instructors::{Client, Headers, OpenAiApi, OutputMode, Step};
use serde::Deserialize;
use serde::de::DeserializeOwned;
use serde_json::{Value, json};

#[derive(Debug, Deserialize, PartialEq)]
#[serde(transparent)]
struct Anything(Value);

impl Schema for Anything {
    fn schema() -> &'static Shape {
        static S: Shape = Shape::Any;
        &S
    }
}

#[derive(Debug, Deserialize, Schema, PartialEq)]
struct Event {
    name: String,
    #[schema(max = 5)]
    level: u8,
    payload: Anything,
}

#[derive(Debug, Deserialize, Schema, PartialEq)]
struct Big {
    big: u64,
    small: i64,
    // past u64, where the number used to go through f64
    huge: u128,
    counts: HashMap<String, u64>,
}

#[derive(Debug, Deserialize, Schema, PartialEq)]
struct Tags {
    tags: HashSet<String>,
    code: char,
}

#[derive(Clone, Copy, Debug)]
enum Wire {
    ChatSchema,
    ChatTool,
    Responses,
    Anthropic,
    Gemini,
}

const ALL: [Wire; 5] = [
    Wire::ChatSchema,
    Wire::ChatTool,
    Wire::Responses,
    Wire::Anthropic,
    Wire::Gemini,
];

fn client(w: Wire) -> Client {
    match w {
        Wire::ChatSchema => Client::openai("k"),
        Wire::ChatTool => Client::openai("k").with_output_mode(OutputMode::Tool),
        Wire::Responses => Client::openai("k").with_api(OpenAiApi::Responses),
        Wire::Anthropic => Client::anthropic("k"),
        Wire::Gemini => Client::gemini("k"),
    }
}

fn response(w: Wire, answer: &str) -> Vec<u8> {
    let a = answer;
    let v = match w {
        Wire::ChatSchema => json!({"choices": [{"message": {"content": a}}]}),
        Wire::ChatTool => {
            json!({"choices": [{"message": {"tool_calls": [{"function": {"arguments": a}}]}}]})
        }
        Wire::Responses => json!({"status": "completed", "output": [
            {"type": "message", "content": [{"type": "output_text", "text": a}]}]}),
        Wire::Anthropic => json!({"content": [
            {"type": "tool_use", "id": "t", "name": "extract", "input": serde_json::from_str::<Value>(a).unwrap()},
            {"type": "text", "text": a}], "stop_reason": "end_turn"}),
        Wire::Gemini => json!({"candidates": [{"content": {"parts": [{"text": a}]}}]}),
    };
    v.to_string().into_bytes()
}

/// The request body, and the result or the re-ask that followed `answer`.
fn run<T: DeserializeOwned + Schema>(w: Wire, answer: &str) -> (Value, Result<T, String>) {
    let mut ex = client(w).extract::<T>("x").exchange();
    let Step::Send(req) = ex.next() else {
        panic!("{w:?}: no request")
    };
    let body: Value = serde_json::from_slice(&req.body).unwrap();
    ex.receive(200, &Headers::new(), &response(w, answer));
    let out = match ex.next() {
        Step::Done(Ok(x)) => Ok(x.value),
        Step::Send(again) => Err(String::from_utf8(again.body).unwrap()),
        Step::Done(Err(e)) => panic!("{w:?}: {e}"),
        Step::Wait(_) => panic!("{w:?}: wait"),
    };
    (body, out)
}

#[test]
fn a_value_of_any_type_goes_non_strict_and_is_checked_here() {
    let ok = r#"{"name": "boot", "level": 2, "payload": {"any": [1, "x"]}}"#;
    for w in [
        Wire::ChatSchema,
        Wire::ChatTool,
        Wire::Responses,
        Wire::Anthropic,
    ] {
        let (body, got) = run::<Event>(w, ok);
        match w {
            Wire::ChatSchema => assert_eq!(body["response_format"]["json_schema"]["strict"], false),
            Wire::ChatTool => assert_eq!(body["tools"][0]["function"]["strict"], false),
            Wire::Responses => assert_eq!(body["text"]["format"]["strict"], false),
            Wire::Anthropic => {
                assert!(body.get("output_config").is_none(), "{body}");
                assert_eq!(body["tool_choice"]["type"], "tool");
            }
            Wire::Gemini => unreachable!(),
        }
        let s = body.to_string();
        assert!(s.contains(r#""payload":true"#), "{w:?}: {s}");
        assert_eq!(
            got,
            Ok(Event {
                name: "boot".into(),
                level: 2,
                payload: Anything(json!({"any": [1, "x"]})),
            }),
            "{w:?}"
        );
        let (_, got) = run::<Event>(w, r#"{"name": "boot", "level": 9, "payload": null}"#);
        let e = got.unwrap_err();
        assert!(e.contains("greater than the maximum 5"), "{w:?}: {e}");
    }
}

#[test]
fn integers_beyond_2_pow_53_survive_conversion() {
    let answer = format!(
        r#"{{"big": {}, "small": {}, "huge": {}, "counts": [{{"key": "max", "value": {}}}]}}"#,
        u64::MAX,
        i64::MIN,
        (1u128 << 70) + 1,
        u64::MAX
    );
    let plain = format!(
        r#"{{"big": {}, "small": {}, "huge": {}, "counts": {{"max": {}}}}}"#,
        u64::MAX,
        i64::MIN,
        (1u128 << 70) + 1,
        u64::MAX
    );
    let want = Big {
        big: u64::MAX,
        small: i64::MIN,
        huge: (1u128 << 70) + 1,
        counts: HashMap::from([("max".into(), u64::MAX)]),
    };
    for w in ALL {
        let a = if matches!(w, Wire::Gemini) {
            &plain
        } else {
            &answer
        };
        let (_, got) = run::<Big>(w, a);
        assert_eq!(got.as_ref(), Ok(&want), "{w:?}");
    }
    // the same through the list path, item by item
    let mut ex = Client::openai("k")
        .extract_many::<BTreeMap<String, u64>>("x")
        .exchange();
    let Step::Send(_) = ex.next() else { panic!() };
    let a = format!(
        r#"{{"items": [[{{"key": "a", "value": {}}}]]}}"#,
        u64::MAX - 1
    );
    ex.receive(200, &Headers::new(), &response(Wire::ChatSchema, &a));
    let Step::Done(Ok(r)) = ex.next() else {
        panic!()
    };
    assert_eq!(
        r.value,
        vec![BTreeMap::from([("a".to_string(), u64::MAX - 1)])]
    );
}

#[test]
fn gemini_drops_what_it_ignores_and_the_answer_is_checked() {
    let (body, got) = run::<Tags>(Wire::Gemini, r#"{"tags": ["a", "b"], "code": "x"}"#);
    let schema = &body["generationConfig"]["responseFormat"]["text"]["schema"];
    let s = schema.to_string();
    for k in ["uniqueItems", "minLength", "maxLength"] {
        assert!(!s.contains(k), "{k}: {s}");
    }
    assert_eq!(got.unwrap().code, 'x');
    let (_, got) = run::<Tags>(Wire::Gemini, r#"{"tags": ["a", "a"], "code": "x"}"#);
    assert!(got.unwrap_err().contains("repeated items"));
    let (_, got) = run::<Tags>(Wire::Gemini, r#"{"tags": [], "code": "xy"}"#);
    assert!(got.unwrap_err().contains("longer than 1 characters"));
}
