// in-house json vs serde_json on the per-request paths:
// `cargo bench --bench json --features fuzzing`

use std::borrow::Cow;
use std::hint::black_box;
use std::time::Instant;

use instructors::__private as ours;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize, Serialize)]
enum Currency {
    Usd,
    Eur,
    Jpy,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct Address {
    street: String,
    city: String,
    postal_code: String,
    country: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct LineItem {
    description: String,
    quantity: u32,
    unit_price: f64,
    sku: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct Invoice {
    invoice_number: String,
    issued_on: String,
    vendor: String,
    vendor_address: Address,
    bill_to: Address,
    currency: Currency,
    items: Vec<LineItem>,
    notes: Option<String>,
    total: f64,
}

fn sample_invoice() -> Invoice {
    let addr = |c: &str| Address {
        street: "1-2-3 Marunouchi".into(),
        city: c.into(),
        postal_code: "100-0005".into(),
        country: "JP".into(),
    };
    Invoice {
        invoice_number: "INV-2026-11-0042".into(),
        issued_on: "2026-11-25".into(),
        vendor: "GOLIA Example Supplies".into(),
        vendor_address: addr("Tokyo"),
        bill_to: addr("Osaka"),
        currency: Currency::Jpy,
        items: (0..8)
            .map(|i| LineItem {
                description: format!("Widget model {i} with \"quoted\" spec 東京"),
                quantity: i + 1,
                unit_price: 1234.5 + i as f64,
                sku: if i % 2 == 0 {
                    Some(format!("SKU-{i:04}"))
                } else {
                    None
                },
            })
            .collect(),
        notes: Some("Payment due in 30 days.\nThank you.".into()),
        total: 98765.25,
    }
}

// openai chat completion, borrowed where possible
#[derive(Deserialize)]
struct OaiResp<'a> {
    #[serde(borrow)]
    choices: Vec<OaiChoice<'a>>,
    usage: Option<OaiUsage>,
}
#[derive(Deserialize)]
struct OaiChoice<'a> {
    #[serde(borrow)]
    message: OaiMsg<'a>,
    #[serde(borrow)]
    finish_reason: Option<&'a str>,
}
#[derive(Deserialize)]
struct OaiMsg<'a> {
    #[serde(borrow)]
    content: Option<Cow<'a, str>>,
    #[serde(borrow)]
    refusal: Option<Cow<'a, str>>,
}
#[derive(Deserialize)]
#[expect(dead_code, reason = "fields are only filled by deserialization")]
struct OaiUsage {
    prompt_tokens: u64,
    completion_tokens: u64,
}

// 1.5's owned envelope
#[derive(Deserialize)]
struct OwnedResp {
    choices: Vec<OwnedChoice>,
}
#[derive(Deserialize)]
struct OwnedChoice {
    message: OwnedMsg,
}
#[derive(Deserialize)]
struct OwnedMsg {
    content: Option<String>,
}

// anthropic tool_use: input kept as raw text
#[derive(Deserialize)]
struct AntResp<'a, R> {
    #[serde(borrow)]
    content: Vec<AntBlock<'a, R>>,
    #[serde(borrow)]
    stop_reason: Option<&'a str>,
}
#[derive(Deserialize)]
struct AntBlock<'a, R> {
    #[serde(rename = "type", borrow)]
    kind: &'a str,
    input: Option<R>,
}

// one streamed chunk
#[derive(Deserialize)]
struct Chunk<'a> {
    #[serde(borrow)]
    choices: Vec<ChunkChoice<'a>>,
}
#[derive(Deserialize)]
struct ChunkChoice<'a> {
    #[serde(borrow)]
    delta: Option<Delta<'a>>,
    #[serde(borrow)]
    finish_reason: Option<&'a str>,
}
#[derive(Deserialize)]
struct Delta<'a> {
    #[serde(borrow)]
    content: Option<Cow<'a, str>>,
}

// the request 1.5 serializes
#[derive(Serialize)]
struct Request<'a> {
    model: &'a str,
    messages: [Msg<'a>; 2],
    response_format: &'a serde_json::value::RawValue,
    temperature: f64,
}
#[derive(Serialize)]
struct Msg<'a> {
    role: &'a str,
    content: &'a str,
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn bench(name: &str, iters: u32, rounds: u32, mut f: impl FnMut()) {
    for _ in 0..iters / 10 + 1 {
        f();
    }
    let mut samples = Vec::new();
    for _ in 0..rounds {
        let t = Instant::now();
        for _ in 0..iters {
            f();
        }
        samples.push(t.elapsed().as_nanos() as f64 / iters as f64);
    }
    let min = samples.iter().cloned().fold(f64::MAX, f64::min);
    println!(
        "{name:<56} median {:>9.1} ns  min {:>9.1} ns",
        median(samples),
        min
    );
}

fn write_request(out: &mut Vec<u8>, model: &str, system: &str, prompt: &str, schema: &str) {
    out.extend_from_slice(b"{\"model\":");
    ours::json_write_str(out, model);
    out.extend_from_slice(b",\"messages\":[{\"role\":\"system\",\"content\":");
    ours::json_write_str(out, system);
    out.extend_from_slice(b"},{\"role\":\"user\",\"content\":");
    ours::json_write_str(out, prompt);
    out.extend_from_slice(b"}],\"response_format\":");
    out.extend_from_slice(schema.as_bytes());
    out.extend_from_slice(b",\"temperature\":");
    ours::json_write_f64(out, 0.0);
    out.push(b'}');
}

fn main() {
    let filter = std::env::args().nth(1).filter(|a| a != "--bench");
    let on = |group: &str| filter.as_deref().is_none_or(|f| group.contains(f));
    let rounds = 9;

    let content = serde_json::to_string(&sample_invoice()).unwrap();
    let oai_body = serde_json::to_vec(&serde_json::json!({
        "id": "chatcmpl-bench", "object": "chat.completion", "created": 1764000000,
        "model": "bench-model",
        "choices": [{"index": 0,
            "message": {"role": "assistant", "content": content, "refusal": null},
            "logprobs": null, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 120, "completion_tokens": 300, "total_tokens": 420},
        "system_fingerprint": "fp_bench"
    }))
    .unwrap();
    let ant_body = format!(
        r#"{{"id":"msg_bench","type":"message","role":"assistant","model":"bench-model","content":[{{"type":"tool_use","id":"toolu_1","name":"extract","input":{content}}}],"stop_reason":"tool_use","stop_sequence":null,"usage":{{"input_tokens":120,"output_tokens":300}}}}"#
    );
    let chunk = r#"{"id":"chatcmpl-bench","object":"chat.completion.chunk","created":1764000000,"model":"bench-model","system_fingerprint":"fp_bench","choices":[{"index":0,"delta":{"content":"\"desc"},"logprobs":null,"finish_reason":null}]}"#;
    println!(
        "content {} bytes, openai body {} bytes, anthropic body {} bytes, chunk {} bytes",
        content.len(),
        oai_body.len(),
        ant_body.len(),
        chunk.len()
    );

    if on("parse") {
        println!("\n# content -> Invoice");
        bench("serde_json from_str::<Invoice>", 20000, rounds, || {
            black_box(serde_json::from_str::<Invoice>(black_box(&content)).unwrap());
        });
        bench("ours       from_str::<Invoice>", 20000, rounds, || {
            black_box(ours::json_from_str::<Invoice>(black_box(&content)).unwrap());
        });
    }

    if on("openai") {
        println!("\n# openai body -> Invoice");
        bench(
            "serde_json owned envelope + Invoice (1.5)",
            20000,
            rounds,
            || {
                let r: OwnedResp = serde_json::from_slice(black_box(&oai_body)).unwrap();
                let c = r.choices[0].message.content.as_deref().unwrap();
                black_box(serde_json::from_str::<Invoice>(c).unwrap());
            },
        );
        bench(
            "serde_json borrowed envelope + Invoice",
            20000,
            rounds,
            || {
                let r: OaiResp<'_> = serde_json::from_slice(black_box(&oai_body)).unwrap();
                let c = r.choices[0].message.content.as_deref().unwrap();
                black_box(serde_json::from_str::<Invoice>(c).unwrap());
                black_box((
                    &r.usage,
                    r.choices[0].finish_reason,
                    &r.choices[0].message.refusal,
                ));
            },
        );
        bench(
            "ours       borrowed envelope + Invoice",
            20000,
            rounds,
            || {
                let r: OaiResp<'_> = ours::json_from_slice(black_box(&oai_body)).unwrap();
                let c = r.choices[0].message.content.as_deref().unwrap();
                black_box(ours::json_from_str::<Invoice>(c).unwrap());
                black_box((
                    &r.usage,
                    r.choices[0].finish_reason,
                    &r.choices[0].message.refusal,
                ));
            },
        );
    }

    if on("anthropic") {
        println!("\n# anthropic tool_use body -> Invoice");
        type SjRaw<'a> = &'a serde_json::value::RawValue;
        bench(
            "serde_json envelope + RawValue + Invoice",
            20000,
            rounds,
            || {
                let r: AntResp<'_, SjRaw<'_>> =
                    serde_json::from_slice(black_box(ant_body.as_bytes())).unwrap();
                let b = &r.content[0];
                let input = b.input.unwrap().get();
                black_box(serde_json::from_str::<Invoice>(input).unwrap());
                black_box((b.kind, r.stop_reason));
            },
        );
        bench("ours       envelope + Raw + Invoice", 20000, rounds, || {
            let r: AntResp<'_, ours::Raw<'_>> =
                ours::json_from_slice(black_box(ant_body.as_bytes())).unwrap();
            let b = &r.content[0];
            let input = b.input.unwrap().0;
            black_box(ours::json_from_str::<Invoice>(input).unwrap());
            black_box((b.kind, r.stop_reason));
        });
    }

    if on("sse") {
        println!("\n# one streamed chunk envelope");
        bench("serde_json chunk", 200000, rounds, || {
            let c: Chunk<'_> = serde_json::from_str(black_box(chunk)).unwrap();
            black_box((
                &c.choices[0].delta.as_ref().unwrap().content,
                c.choices[0].finish_reason,
            ));
        });
        bench("ours       chunk", 200000, rounds, || {
            let c: Chunk<'_> = ours::json_from_str(black_box(chunk)).unwrap();
            black_box((
                &c.choices[0].delta.as_ref().unwrap().content,
                c.choices[0].finish_reason,
            ));
        });
    }

    if on("repair") {
        println!("\n# repair (input -> repaired text)");
        let trailing = content.replace("}]", "},]").replacen('}', ",}", 1);
        let single = content.replace('"', "'");
        let fenced = format!("Here is the JSON:\n```json\n{content}\n```\n");
        let truncated = content[..content.floor_char_boundary(content.len() * 2 / 3)].to_string();
        let inputs = [
            ("valid", content.clone()),
            ("trailing_commas", trailing),
            ("single_quotes", single),
            ("markdown_fence", fenced),
            ("truncated_2/3", truncated),
        ];
        let opts = jsonrepair::Options::default();
        for (name, input) in &inputs {
            let theirs = jsonrepair::repair_json(input, &opts).ok();
            let mine = ours::json_repair(input).map(|r| r.text.into_owned());
            let parses = |s: &Option<String>| {
                s.as_deref()
                    .is_some_and(|t| serde_json::from_str::<Invoice>(t).is_ok())
            };
            println!(
                "  {name}: parses into Invoice after repair: jsonrepair={} ours={}",
                parses(&theirs),
                parses(&mine)
            );
            bench(&format!("jsonrepair 0.1 {name}"), 5000, rounds, || {
                black_box(jsonrepair::repair_json(black_box(input), &opts).ok());
            });
            bench(&format!("ours           {name}"), 5000, rounds, || {
                black_box(ours::json_repair(black_box(input)));
            });
        }
    }

    if on("write") {
        let unit =
            "Contact \"John\" at john@example.com.\nNotes:\ttab, back\\slash, 東京 and more. ";
        let mut prompt = String::new();
        while prompt.len() < 2532 {
            prompt.push_str(unit);
        }
        let prompt = &prompt[..prompt.floor_char_boundary(2532)];
        let schema = invoice_schema();
        let raw = serde_json::value::RawValue::from_string(schema.clone()).unwrap();
        let system = "Extract the requested information from the given text. Return valid JSON matching the provided schema.";
        let req = Request {
            model: "gpt-5.6-terra",
            messages: [
                Msg {
                    role: "system",
                    content: system,
                },
                Msg {
                    role: "user",
                    content: prompt,
                },
            ],
            response_format: &raw,
            temperature: 0.0,
        };
        let a = serde_json::to_vec(&req).unwrap();
        let mut b = Vec::new();
        write_request(&mut b, req.model, system, prompt, &schema);
        assert_eq!(
            serde_json::from_slice::<serde_json::Value>(&a).unwrap(),
            serde_json::from_slice::<serde_json::Value>(&b).unwrap()
        );
        println!(
            "\n# request body, {} bytes (prompt {} bytes, schema {} bytes)",
            a.len(),
            prompt.len(),
            schema.len()
        );
        bench("serde_json to_vec (1.5)", 50000, rounds, || {
            black_box(serde_json::to_vec(black_box(&req)).unwrap());
        });
        let mut buf = Vec::with_capacity(8192);
        bench("serde_json to_writer, reused Vec", 50000, rounds, || {
            buf.clear();
            serde_json::to_writer(&mut buf, black_box(&req)).unwrap();
            black_box(&buf);
        });
        bench("ours       writer, reused Vec", 50000, rounds, || {
            buf.clear();
            write_request(&mut buf, req.model, system, black_box(prompt), &schema);
            black_box(&buf);
        });
    }
}

fn invoice_schema() -> String {
    let addr = serde_json::json!({
        "type": "object",
        "properties": {
            "street": {"type": "string"}, "city": {"type": "string"},
            "postal_code": {"type": "string"}, "country": {"type": "string"}
        },
        "required": ["street", "city", "postal_code", "country"],
        "additionalProperties": false
    });
    let s = serde_json::json!({
        "type": "json_schema",
        "json_schema": {"name": "Invoice", "strict": true, "schema": {
            "type": "object",
            "properties": {
                "invoice_number": {"type": "string"},
                "issued_on": {"type": "string"},
                "vendor": {"type": "string"},
                "vendor_address": addr,
                "bill_to": addr,
                "currency": {"type": "string", "enum": ["Usd", "Eur", "Jpy"]},
                "items": {"type": "array", "items": {
                    "type": "object",
                    "properties": {
                        "description": {"type": "string"},
                        "quantity": {"type": "integer", "minimum": 0},
                        "unit_price": {"type": "number"},
                        "sku": {"type": ["string", "null"]}
                    },
                    "required": ["description", "quantity", "unit_price", "sku"],
                    "additionalProperties": false
                }},
                "notes": {"type": ["string", "null"]},
                "total": {"type": "number"}
            },
            "required": ["invoice_number", "issued_on", "vendor", "vendor_address",
                         "bill_to", "currency", "items", "notes", "total"],
            "additionalProperties": false
        }}
    });
    serde_json::to_string(&s).unwrap()
}
