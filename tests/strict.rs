// Types that strict structured output cannot express directly (maps, tuples,
// sets, roots that are not objects), sent to every strict wire and read
// back from answers in the wire form.

use std::collections::{BTreeMap, HashMap, HashSet};

use instructors::{Client, Headers, Message, OpenAiApi, OutputMode, Schema, Step};
use serde::Deserialize;
use serde::de::DeserializeOwned;
use serde_json::{Value, json};

#[derive(Debug, Deserialize, Schema, PartialEq)]
struct Inventory {
    counts: HashMap<String, u32>,
    by_id: BTreeMap<u32, String>,
    pair: (u8, String),
    tags: HashSet<String>,
    flags: Option<BTreeMap<String, bool>>,
    #[schema(min = 1, max = 5)]
    rating: u8,
}

#[derive(Debug, Deserialize, Schema, PartialEq)]
enum Mood {
    Happy,
    Sad,
}

#[derive(Debug, Deserialize, Schema, PartialEq)]
#[serde(tag = "kind")]
enum Figure {
    Circle { r: f64 },
    Square { side: f64 },
}

#[derive(Debug, Deserialize, Schema, PartialEq)]
enum Expr {
    Num(u32),
    Add(Box<Expr>, Box<Expr>),
}

#[derive(Clone, Copy, Debug)]
enum Wire {
    ChatSchema,
    ChatTool,
    Responses,
    Anthropic,
}

const WIRES: [Wire; 4] = [
    Wire::ChatSchema,
    Wire::ChatTool,
    Wire::Responses,
    Wire::Anthropic,
];

fn client(w: Wire) -> Client {
    match w {
        Wire::ChatSchema => Client::openai("k"),
        Wire::ChatTool => Client::openai("k").with_output_mode(OutputMode::Tool),
        Wire::Responses => Client::openai("k").with_api(OpenAiApi::Responses),
        Wire::Anthropic => Client::anthropic("k"),
    }
}

/// The schema the request carries.
fn wire_schema(w: Wire, body: &Value) -> Value {
    match w {
        Wire::ChatSchema => body["response_format"]["json_schema"]["schema"].clone(),
        Wire::ChatTool => body["tools"][0]["function"]["parameters"].clone(),
        Wire::Responses => body["text"]["format"]["schema"].clone(),
        Wire::Anthropic => body["output_config"]["format"]["schema"].clone(),
    }
}

/// A response whose answer is `answer`.
fn response(w: Wire, answer: &Value) -> Vec<u8> {
    let a = answer.to_string();
    let v = match w {
        Wire::ChatSchema => json!({"choices": [{"message": {"content": a}}]}),
        Wire::ChatTool => {
            json!({"choices": [{"message": {"tool_calls": [{"function": {"arguments": a}}]}}]})
        }
        Wire::Responses => json!({"status": "completed", "output": [
            {"type": "message", "content": [{"type": "output_text", "text": a}]}]}),
        Wire::Anthropic => {
            json!({"content": [{"type": "text", "text": a}], "stop_reason": "end_turn"})
        }
    };
    v.to_string().into_bytes()
}

/// Every object closed with all keys required, and no keyword the wire
/// cannot take.
fn assert_strict(w: Wire, v: &Value, path: &str) {
    match v {
        Value::Object(m) => {
            for k in ["prefixItems", "uniqueItems", "minLength", "maxLength"] {
                assert!(!m.contains_key(k), "{w:?} {path}: {k}");
            }
            if matches!(w, Wire::Anthropic) {
                for k in ["minimum", "maximum", "maxItems"] {
                    assert!(!m.contains_key(k), "{w:?} {path}: {k}");
                }
            }
            let is_object = m.get("type") == Some(&json!("object"))
                || m.get("type")
                    .and_then(Value::as_array)
                    .is_some_and(|t| t.contains(&json!("object")));
            if is_object {
                assert_eq!(
                    m.get("additionalProperties"),
                    Some(&json!(false)),
                    "{w:?} {path}"
                );
                let props: Vec<&String> = m["properties"].as_object().unwrap().keys().collect();
                let req: Vec<&str> = m["required"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|r| r.as_str().unwrap())
                    .collect();
                assert_eq!(props.len(), req.len(), "{w:?} {path}");
            }
            for (k, x) in m {
                assert_strict(w, x, &format!("{path}/{k}"));
            }
        }
        Value::Array(a) => a.iter().for_each(|x| assert_strict(w, x, path)),
        _ => {}
    }
}

/// Send the request, answer with `answer`, and return the result or the
/// re-ask that followed.
fn run<T: DeserializeOwned + Schema>(w: Wire, answer: &Value) -> Result<T, String> {
    let mut ex = client(w).extract::<T>("x").exchange();
    let Step::Send(req) = ex.next() else {
        panic!("{w:?}: no request")
    };
    let body: Value = serde_json::from_slice(&req.body).unwrap();
    let schema = wire_schema(w, &body);
    assert_eq!(schema["type"], "object", "{w:?}: root {schema}");
    assert_strict(w, &schema, "");
    ex.receive(200, &Headers::new(), &response(w, answer));
    match ex.next() {
        Step::Done(Ok(x)) => Ok(x.value),
        Step::Send(again) => {
            let body: Value = serde_json::from_slice(&again.body).unwrap();
            Err(body.to_string())
        }
        Step::Done(Err(e)) => panic!("{w:?}: {e}"),
        Step::Wait(_) => panic!("{w:?}: wait"),
    }
}

#[test]
fn maps_tuples_and_sets_round_trip() {
    let answer = json!({
        "counts": [{"key": "apples", "value": 3}, {"key": "pears", "value": 0}],
        "by_id": [{"key": "7", "value": "seven"}],
        "pair": {"0": 2, "1": "two"},
        "tags": ["a", "b"],
        "flags": null,
        "rating": 4
    });
    let want = Inventory {
        counts: HashMap::from([("apples".into(), 3), ("pears".into(), 0)]),
        by_id: BTreeMap::from([(7, "seven".into())]),
        pair: (2, "two".into()),
        tags: HashSet::from(["a".into(), "b".into()]),
        flags: None,
        rating: 4,
    };
    for w in WIRES {
        assert_eq!(run::<Inventory>(w, &answer).as_ref(), Ok(&want), "{w:?}");
        let mut a = answer.clone();
        a["flags"] = json!([{"key": "x", "value": true}]);
        let got = run::<Inventory>(w, &a).unwrap();
        assert_eq!(
            got.flags,
            Some(BTreeMap::from([("x".into(), true)])),
            "{w:?}"
        );
    }
}

#[test]
fn stripped_constraints_are_checked_and_re_asked() {
    let base = json!({
        "counts": [], "by_id": [], "pair": {"0": 1, "1": "x"},
        "tags": ["a"], "flags": null, "rating": 3
    });
    for w in WIRES {
        let mut a = base.clone();
        a["tags"] = json!(["a", "a"]);
        let e = run::<Inventory>(w, &a).unwrap_err();
        assert!(e.contains("repeated items"), "{w:?}: {e}");
        let mut a = base.clone();
        a["rating"] = json!(9);
        let e = run::<Inventory>(w, &a).unwrap_err();
        assert!(e.contains("greater than the maximum 5"), "{w:?}: {e}");
        let mut a = base.clone();
        a["counts"] = json!([{"key": "a", "value": 1}, {"key": "a", "value": 2}]);
        let e = run::<Inventory>(w, &a).unwrap_err();
        assert!(e.contains("repeated key"), "{w:?}: {e}");
    }
}

#[test]
fn roots_that_are_not_objects_are_wrapped() {
    for w in WIRES {
        assert_eq!(
            run::<Mood>(w, &json!({"value": "Sad"})),
            Ok(Mood::Sad),
            "{w:?}"
        );
        assert_eq!(
            run::<Figure>(w, &json!({"value": {"kind": "Square", "side": 2.0}})),
            Ok(Figure::Square { side: 2.0 }),
            "{w:?}"
        );
        assert_eq!(
            run::<Vec<u32>>(w, &json!({"value": [1, 2]})),
            Ok(vec![1, 2])
        );
        assert_eq!(
            run::<(u8, String)>(w, &json!({"value": {"0": 1, "1": "a"}})),
            Ok((1, "a".to_string()))
        );
        assert_eq!(
            run::<BTreeMap<String, u8>>(w, &json!({"value": [{"key": "k", "value": 1}]})),
            Ok(BTreeMap::from([("k".to_string(), 1)]))
        );
        // a compatible server that ignores the schema may send the bare value
        assert_eq!(run::<Mood>(w, &json!("Happy")), Ok(Mood::Happy), "{w:?}");
    }
}

#[test]
fn a_recursive_root_moves_into_defs() {
    // anthropic structured output has no recursion and sends a tool instead
    for w in [Wire::ChatSchema, Wire::ChatTool, Wire::Responses] {
        let answer = json!({"value": {"Add": {"0": {"Num": 1}, "1": {"Num": 2}}}});
        assert_eq!(
            run::<Expr>(w, &answer),
            Ok(Expr::Add(Box::new(Expr::Num(1)), Box::new(Expr::Num(2)))),
            "{w:?}"
        );
    }
    let mut ex = client(Wire::ChatSchema).extract::<Expr>("x").exchange();
    let Step::Send(req) = ex.next() else { panic!() };
    let body: Value = serde_json::from_slice(&req.body).unwrap();
    let schema = &body["response_format"]["json_schema"]["schema"];
    assert_eq!(
        schema["properties"]["value"],
        json!({"$ref": "#/$defs/Root"})
    );
    assert!(!schema.to_string().contains(r##""$ref":"#""##), "{schema}");
}

#[test]
fn lists_stream_items_in_the_type_s_own_form() {
    use std::sync::{Arc, Mutex};
    let seen = Arc::new(Mutex::new(Vec::new()));
    let s2 = seen.clone();
    let client = Client::openai("k");
    let mut ex = client
        .extract_many::<BTreeMap<String, u8>>("x")
        .on_item(move |m| s2.lock().unwrap().push(m))
        .exchange();
    let Step::Send(_) = ex.next() else { panic!() };
    let answer = json!({"items": [[{"key": "a", "value": 1}], [{"key": "b", "value": 2}]]});
    ex.receive_head(200, &Headers::new());
    for piece in answer.to_string().as_bytes().chunks(5) {
        let delta = String::from_utf8(piece.to_vec()).unwrap();
        let event = json!({"choices": [{"delta": {"content": delta}}]});
        ex.receive_chunk(format!("data: {event}\n\n").as_bytes());
    }
    ex.receive_end();
    let Step::Done(Ok(r)) = ex.next() else {
        panic!()
    };
    let want = vec![
        BTreeMap::from([("a".into(), 1)]),
        BTreeMap::from([("b".into(), 2)]),
    ];
    assert_eq!(r.value, want);
    assert_eq!(*seen.lock().unwrap(), want);
}

#[test]
fn other_wires_keep_the_plain_schema() {
    for c in [
        Client::gemini("k"),
        Client::deepseek("k"),
        Client::openai("k").with_output_mode(OutputMode::JsonObject),
    ] {
        let mut ex = c
            .extract::<BTreeMap<String, u8>>("x")
            .messages([Message::user("x")])
            .exchange();
        let Step::Send(req) = ex.next() else { panic!() };
        let body = String::from_utf8(req.body).unwrap();
        assert!(body.contains("additionalProperties"), "{body}");
        assert!(!body.contains(r#""key""#), "{body}");
    }
}

#[derive(Debug, Deserialize, Schema)]
#[expect(dead_code, reason = "only the schema is used")]
struct Formats {
    #[schema(format = "email")]
    mail: String,
    n: u8,
    x: f64,
}

#[test]
fn formats_each_wire_takes() {
    let schema_of = |c: Client, path: &[&str]| {
        let Step::Send(req) = c.extract::<Formats>("x").exchange().next() else {
            panic!()
        };
        let mut v: Value = serde_json::from_slice(&req.body).unwrap();
        for p in path {
            v = v[*p].take();
        }
        v["properties"].clone()
    };
    let g = schema_of(
        Client::gemini("k"),
        &["generationConfig", "responseFormat", "text", "schema"],
    );
    assert_eq!(g["mail"]["format"], "email");
    assert!(
        g["n"].get("format").is_none() && g["x"].get("format").is_none(),
        "{g}"
    );
    assert_eq!(g["n"]["maximum"], 255);
    let o = schema_of(
        Client::openai("k"),
        &["response_format", "json_schema", "schema"],
    );
    assert_eq!(o["mail"]["format"], "email");
    assert!(o["n"].get("format").is_none(), "{o}");
    assert_eq!(o["n"]["maximum"], 255);
    let a = schema_of(
        Client::anthropic("k"),
        &["output_config", "format", "schema"],
    );
    assert_eq!(a["mail"]["format"], "email");
    assert!(a["n"].get("maximum").is_none(), "{a}");
}
