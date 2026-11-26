// Typed snapshots of a streaming answer: the type `#[schema(partial)]`
// generates, read from the partial parser on every wire, with strict maps
// and tuples read back from their wire form.

use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, Mutex};

use instructors::partial::{Partial, PartialOf, PartialParser};
use instructors::{Client, Headers, OpenAiApi, Schema, Step};
use serde::Deserialize;
use serde_json::{Value, json};

#[derive(Debug, Clone, Deserialize, Schema, PartialEq)]
#[schema(partial)]
struct Author {
    name: String,
    born: Option<u16>,
}

#[derive(Debug, Clone, Deserialize, Schema, PartialEq)]
#[schema(partial)]
enum Mood {
    Happy,
    Grim,
}

/// No `Partial`: only ever whole.
#[derive(Debug, Clone, Deserialize, Schema, PartialEq)]
struct Stamp {
    day: u8,
}

#[derive(Debug, Deserialize, Schema, PartialEq)]
#[schema(partial)]
#[serde(rename_all = "camelCase")]
struct Article {
    title: String,
    tag_list: Vec<String>,
    author: Author,
    mood: Mood,
    scores: HashMap<String, u32>,
    pair: (u8, String),
    #[schema(whole)]
    stamp: Stamp,
    editor: Option<Author>,
    #[serde(rename = "kind")]
    r#type: String,
}

#[derive(Debug, Deserialize, Schema, PartialEq)]
#[schema(partial = "PageSoFar")]
struct Page<T> {
    items: Vec<T>,
    next: Option<String>,
}

fn answer() -> Value {
    json!({
        "title": "On streams",
        "tagList": ["rust", "json"],
        "author": {"name": "Ada", "born": 1815},
        "mood": "Happy",
        "scores": {"a": 1, "b": 2},
        "pair": [7, "seven"],
        "stamp": {"day": 3},
        "editor": null,
        "kind": "essay"
    })
}

fn want() -> ArticlePartial {
    ArticlePartial {
        title: Some("On streams".into()),
        tag_list: Some(vec!["rust".into(), "json".into()]),
        author: Some(AuthorPartial {
            name: Some("Ada".into()),
            born: Some(1815),
        }),
        mood: Some(Mood::Happy),
        scores: Some(HashMap::from([("a".into(), 1), ("b".into(), 2)])),
        pair: Some((7, "seven".into())),
        stamp: Some(Stamp { day: 3 }),
        editor: None,
        r#type: Some("essay".into()),
    }
}

/// Equal, the map compared as a map.
fn same(a: &ArticlePartial, b: &ArticlePartial) -> bool {
    let (mut a2, mut b2) = (a.clone(), b.clone());
    a2.scores = None;
    b2.scores = None;
    format!("{a2:?}") == format!("{b2:?}") && a.scores == b.scores
}

#[test]
fn snapshots_grow_with_the_stream() {
    let text = answer().to_string();
    let mut p = PartialParser::new();
    let mut last = ArticlePartial::default();
    for (i, c) in text.char_indices() {
        p.feed(&text[i..i + c.len_utf8()]);
        let Some(snap) = p.root().and_then(|r| r.snapshot::<Article>()) else {
            continue;
        };
        // a string shows what has arrived; a value that is not a string
        // appears only once complete
        if let (Some(old), Some(new)) = (&last.title, &snap.title) {
            assert!(new.starts_with(old.as_str()), "{old} -> {new}");
        }
        if let Some(m) = &snap.mood {
            assert_eq!(*m, Mood::Happy);
        }
        if let Some(a) = &snap.author
            && let Some(b) = a.born
        {
            assert_eq!(b, 1815, "a number is shown only once complete");
        }
        if let Some(s) = &snap.stamp {
            assert_eq!(s.day, 3);
        }
        last = snap;
    }
    assert!(same(&last, &want()), "{last:?}");
}

#[test]
fn the_title_is_shown_while_it_streams() {
    let mut p = PartialParser::new();
    p.feed(r#"{"title": "On str"#);
    let snap = p.root().unwrap().snapshot::<Article>().unwrap();
    assert_eq!(snap.title.as_deref(), Some("On str"));
    assert!(snap.mood.is_none() && snap.author.is_none());
    p.feed(r#"eams", "mood": "Hap"#);
    let snap = p.root().unwrap().snapshot::<Article>().unwrap();
    assert_eq!(snap.title.as_deref(), Some("On streams"));
    assert_eq!(snap.mood, None, "a variant name is not taken half-received");
    p.feed(r#"py", "author": {"name": "A", "born": 18"#);
    let snap = p.root().unwrap().snapshot::<Article>().unwrap();
    assert_eq!(snap.mood, Some(Mood::Happy));
    let author = snap.author.unwrap();
    assert_eq!(author.name.as_deref(), Some("A"));
    assert_eq!(author.born, None, "18 may still become 1815");
}

#[test]
fn generic_types_and_custom_names() {
    let mut p = PartialParser::new();
    p.feed(r#"{"items": [{"name": "Ada", "born": 1815}, {"name": "Gr"#);
    let snap: PageSoFar<Author> = p.root().unwrap().snapshot::<Page<Author>>().unwrap();
    let items = snap.items.unwrap();
    assert_eq!(items.len(), 2);
    assert_eq!(items[1].name.as_deref(), Some("Gr"));
    assert_eq!(snap.next, None);
    let _: Option<PartialOf<Page<Author>>> = None;
    fn is_partial<T: Partial>() {}
    is_partial::<Page<Mood>>();
}

#[derive(Clone, Copy, Debug)]
enum Wire {
    Chat,
    Responses,
    Anthropic,
    Gemini,
    JsonObject,
}

fn client(w: Wire) -> Client {
    match w {
        Wire::Chat => Client::openai("k"),
        Wire::Responses => Client::openai("k").with_api(OpenAiApi::Responses),
        Wire::Anthropic => Client::anthropic("k"),
        Wire::Gemini => Client::gemini("k"),
        Wire::JsonObject => {
            Client::openai_compatible("k", "https://api.deepseek.com").with_model("m")
        }
    }
}

fn event(w: Wire, delta: &str) -> Value {
    match w {
        Wire::Chat | Wire::JsonObject => json!({"choices": [{"delta": {"content": delta}}]}),
        Wire::Responses => json!({"type": "response.output_text.delta", "delta": delta}),
        Wire::Anthropic => {
            json!({"type": "content_block_delta", "delta": {"type": "text_delta", "text": delta}})
        }
        Wire::Gemini => json!({"candidates": [{"content": {"parts": [{"text": delta}]}}]}),
    }
}

/// The answer in the form the wire sends: maps as key and value pairs and
/// tuples as objects for the strict wires.
fn sent(w: Wire) -> Value {
    let mut a = answer();
    if matches!(w, Wire::Chat | Wire::Responses | Wire::Anthropic) {
        a["scores"] = json!([{"key": "a", "value": 1}, {"key": "b", "value": 2}]);
        a["pair"] = json!({"0": 7, "1": "seven"});
    }
    a
}

/// Stream `text` in 3-byte pieces; the snapshots seen and the final value.
fn stream<T, S>(
    w: Wire,
    text: &str,
    ex: instructors::Exchange<T>,
    seen: &Arc<Mutex<Vec<S>>>,
) -> (T, usize)
where
    T: serde::de::DeserializeOwned,
{
    let mut ex = ex;
    let Step::Send(_) = ex.next() else { panic!() };
    ex.receive_head(200, &Headers::new());
    let bytes = text.as_bytes();
    let mut at = 0;
    while at < bytes.len() {
        let mut end = (at + 3).min(bytes.len());
        while !text.is_char_boundary(end) {
            end += 1;
        }
        let e = event(w, &text[at..end]);
        ex.receive_chunk(format!("data: {e}\n\n").as_bytes());
        at = end;
    }
    ex.receive_end();
    let Step::Done(Ok(r)) = ex.next() else {
        panic!("{w:?}: no result")
    };
    (r.value, seen.lock().unwrap().len())
}

#[test]
fn on_snapshot_reads_every_wire_in_the_type_s_own_form() {
    for w in [
        Wire::Chat,
        Wire::Responses,
        Wire::Anthropic,
        Wire::Gemini,
        Wire::JsonObject,
    ] {
        let seen = Arc::new(Mutex::new(Vec::<ArticlePartial>::new()));
        let s2 = seen.clone();
        let ex = client(w)
            .extract::<Article>("x")
            .on_snapshot(move |a| s2.lock().unwrap().push(a))
            .exchange();
        let (value, n) = stream(w, &sent(w).to_string(), ex, &seen);
        assert!(n > 10, "{w:?}: {n} snapshots");
        assert_eq!(value.scores["b"], 2, "{w:?}");
        let last = seen.lock().unwrap().last().cloned().unwrap();
        assert!(same(&last, &want()), "{w:?}: {last:?}");
        // the map filled entry by entry, never in the pair form
        let maps: Vec<usize> = seen
            .lock()
            .unwrap()
            .iter()
            .filter_map(|s| s.scores.as_ref().map(HashMap::len))
            .collect();
        assert_eq!(maps.last(), Some(&2), "{w:?}: {maps:?}");
    }
}

#[test]
fn on_snapshot_for_lists_and_wrapped_roots() {
    let seen = Arc::new(Mutex::new(Vec::<Vec<AuthorPartial>>::new()));
    let s2 = seen.clone();
    let ex = Client::openai("k")
        .extract_many::<Author>("x")
        .on_snapshot(move |a| s2.lock().unwrap().push(a))
        .exchange();
    let text = json!({"items": [{"name": "Ada", "born": 1815}, {"name": "Grace", "born": null}]});
    let (value, _) = stream(Wire::Chat, &text.to_string(), ex, &seen);
    assert_eq!(value.len(), 2);
    let last = seen.lock().unwrap().last().cloned().unwrap();
    assert_eq!(last.len(), 2);
    assert_eq!(last[1].name.as_deref(), Some("Grace"));

    // a root that is not an object is wrapped on strict wires
    let seen = Arc::new(Mutex::new(Vec::<BTreeMap<String, AuthorPartial>>::new()));
    let s2 = seen.clone();
    let ex = Client::openai("k")
        .extract::<BTreeMap<String, Author>>("x")
        .on_snapshot(move |a| s2.lock().unwrap().push(a))
        .exchange();
    let text = json!({"value": [{"key": "first", "value": {"name": "Ada", "born": 1815}}]});
    stream(Wire::Chat, &text.to_string(), ex, &seen);
    let last = seen.lock().unwrap().last().cloned().unwrap();
    assert_eq!(last["first"].born, Some(1815));
}
