//! The names the derive puts in the schema are the names serde_derive
//! accepts: for every rename rule, a JSON value built from the schema's
//! property and variant names must deserialize.

use serde::Deserialize;
use serde_json::{Map, Value};

use crate::Schema;
use crate::schema::cache::{Dialect, entry_for};

fn props<T: Schema>() -> Vec<String> {
    let v: Value = serde_json::from_str(entry_for::<T>(false).get(Dialect::AnthropicTool)).unwrap();
    v["properties"]
        .as_object()
        .unwrap()
        .keys()
        .cloned()
        .collect()
}

fn variants<T: Schema>() -> Vec<String> {
    let v: Value = serde_json::from_str(entry_for::<T>(false).get(Dialect::AnthropicTool)).unwrap();
    v["enum"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_str().unwrap().to_owned())
        .collect()
}

macro_rules! rule {
    ($test:ident, $rule:literal) => {
        #[test]
        fn $test() {
            #[expect(dead_code, reason = "only deserialized")]
            #[derive(Deserialize, Schema)]
            #[serde(rename_all = $rule, deny_unknown_fields)]
            struct S {
                first_name: u8,
                x2_value: u8,
                r#type: u8,
                #[serde(rename(deserialize = "Given", serialize = "ignored"))]
                renamed: u8,
            }
            #[derive(Deserialize, Schema, Debug, PartialEq)]
            #[serde(rename_all = $rule)]
            enum E {
                VeryTasty,
                Z42,
                A,
            }
            let names = props::<S>();
            let obj: Map<String, Value> = names.iter().map(|n| (n.clone(), 1.into())).collect();
            serde_json::from_value::<S>(Value::Object(obj))
                .unwrap_or_else(|e| panic!("{}: {names:?}: {e}", $rule));
            assert!(names.contains(&"Given".to_owned()));
            for v in variants::<E>() {
                serde_json::from_value::<E>(Value::String(v.clone()))
                    .unwrap_or_else(|e| panic!("{}: {v}: {e}", $rule));
            }
        }
    };
}

rule!(lowercase, "lowercase");
rule!(uppercase, "UPPERCASE");
rule!(pascal, "PascalCase");
rule!(camel, "camelCase");
rule!(snake, "snake_case");
rule!(screaming_snake, "SCREAMING_SNAKE_CASE");
rule!(kebab, "kebab-case");
rule!(screaming_kebab, "SCREAMING-KEBAB-CASE");

#[test]
fn rename_all_fields_and_variant_rename_all() {
    #[expect(dead_code, reason = "only deserialized")]
    #[derive(Deserialize, Schema)]
    #[serde(
        tag = "kind",
        rename_all = "snake_case",
        rename_all_fields = "camelCase"
    )]
    enum E {
        FirstThing {
            some_field: u8,
        },
        #[serde(rename_all = "kebab-case")]
        SecondThing {
            other_field: u8,
        },
    }
    let v: Value = serde_json::from_str(entry_for::<E>(false).get(Dialect::AnthropicTool)).unwrap();
    for variant in v["oneOf"].as_array().unwrap() {
        let mut obj = Map::new();
        for (k, p) in variant["properties"].as_object().unwrap() {
            let val = p.get("const").cloned().unwrap_or(1.into());
            obj.insert(k.clone(), val);
        }
        serde_json::from_value::<E>(Value::Object(obj.clone()))
            .unwrap_or_else(|e| panic!("{obj:?}: {e}"));
    }
}
