//! Declaration order and `#[schema(..)]` attributes.

use serde::Deserialize;
use serde_json::Value;

use super::{Invoice, ours};
use crate::Schema;
use crate::partial::PartialParser;
use crate::schema::cache::{Dialect, entry_for};

/// Keys of an object in the order the bytes list them.
pub(super) fn ordered_keys(raw: &str, path: &[&str]) -> Vec<String> {
    let mut p = PartialParser::new();
    p.feed(raw);
    let mut v = p.root().unwrap();
    for k in path {
        v = v.get(k).unwrap();
    }
    v.entries().map(|(k, _)| k.to_owned()).collect()
}

pub(super) fn keys(v: &Value) -> Vec<String> {
    let raw = serde_json::to_string(v).unwrap();
    ordered_keys(&raw, &[])
}

#[test]
fn properties_follow_declaration_order() {
    let want = [
        "invoice_number",
        "issued_on",
        "vendor",
        "vendor_address",
        "bill_to",
        "currency",
        "items",
        "notes",
        "total",
    ];
    for d in [
        Dialect::Strict,
        Dialect::AnthropicOutput,
        Dialect::AnthropicTool,
        Dialect::GeminiJson,
    ] {
        let raw = entry_for::<Invoice>(false).get(d);
        assert_eq!(ordered_keys(raw, &["properties"]), want, "{d:?}");
        let raw = entry_for::<Invoice>(true).get(d);
        let line = ordered_keys(
            raw,
            &[
                "properties",
                "items",
                "items",
                "properties",
                "items",
                "items",
                "properties",
            ],
        );
        assert_eq!(
            line,
            ["description", "quantity", "unit_price", "sku"],
            "{d:?}"
        );
    }
    let raw = entry_for::<Invoice>(false).get(Dialect::Strict);
    let mut p = PartialParser::new();
    p.feed(raw);
    let req: Vec<String> = p
        .root()
        .unwrap()
        .get("required")
        .unwrap()
        .items()
        .map(|v| v.as_str().unwrap().to_owned())
        .collect();
    assert_eq!(req, want);
}

#[test]
fn schema_attributes() {
    #[derive(Deserialize, Schema)]
    struct S {
        #[schema(description = "when", format = "date-time")]
        at: String,
        #[schema(min = 1, max = 5)]
        stars: u8,
        #[schema(min = -1.5)]
        temp: f64,
        #[serde(deserialize_with = "de_flag")]
        #[schema(like = bool)]
        flag: bool,
    }
    fn de_flag<'de, D: serde::Deserializer<'de>>(d: D) -> Result<bool, D::Error> {
        bool::deserialize(d)
    }
    let v = ours::<S>(false, Dialect::GeminiJson);
    assert_eq!(v["properties"]["at"]["description"], "when");
    assert_eq!(v["properties"]["at"]["format"], "date-time");
    assert_eq!(v["properties"]["stars"]["minimum"], 1);
    assert_eq!(v["properties"]["stars"]["maximum"], 5);
    assert_eq!(v["properties"]["temp"]["minimum"], -1.5);
    assert_eq!(v["properties"]["flag"]["type"], "boolean");
}
