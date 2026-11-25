use std::sync::{Mutex, OnceLock};

use super::lower::{self, J};
use super::{Schema, Shape, lock};

/// The schema dialects the request builders splice into bodies.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Dialect {
    /// OpenAI strict JSON Schema (the `schema` inside `response_format`,
    /// Responses `text.format` and strict function `parameters`).
    Strict,
    /// Anthropic `output_config.format.schema`.
    AnthropicOutput,
    /// Anthropic tool `input_schema`; also the schema shown in prompts.
    AnthropicTool,
    /// Gemini `responseJsonSchema`.
    GeminiJson,
}

const DIALECTS: usize = 4;

/// Per-type storage for built schemas; lives inside each struct and enum
/// shape so a lookup is a field access, not a map lookup.
pub struct Cache {
    entries: [OnceLock<SchemaEntry>; 2],
}

impl Cache {
    pub(crate) const fn new() -> Self {
        Self {
            entries: [OnceLock::new(), OnceLock::new()],
        }
    }
}

/// One type's schema, lowered once, with each dialect's bytes built on
/// first use.
pub(crate) struct SchemaEntry {
    name: String,
    recursive: bool,
    base: J,
    built: [OnceLock<Box<str>>; DIALECTS],
}

impl SchemaEntry {
    fn new(shape: &'static Shape, list: bool) -> Self {
        let (base, recursive) = lower::canonical(shape, list);
        let mut name = lower::name_of(shape);
        if list {
            name = format!("Array_of_{name}");
        }
        Self {
            name: sanitize_name(&name),
            recursive,
            base,
            built: Default::default(),
        }
    }

    /// True when the schema refers to itself, which some dialects cannot
    /// express.
    pub(crate) fn recursive(&self) -> bool {
        self.recursive
    }

    /// The type's name, as OpenAI's `json_schema.name` wants it.
    pub(crate) fn name(&self) -> &str {
        &self.name
    }

    /// Serialized schema for `dialect`, built on first call.
    pub(crate) fn get(&self, dialect: Dialect) -> &str {
        self.built[dialect as usize].get_or_init(|| lower::build(&self.base, dialect))
    }
}

/// Cached schema entry for `T`. `list` wraps the schema as `{"items": [T]}`.
pub(crate) fn entry_for<T: Schema + ?Sized>(list: bool) -> &'static SchemaEntry {
    let shape = T::schema();
    let cache = match shape {
        Shape::Struct(s) => &s.cache,
        Shape::Enum(e) => &e.cache,
        other => other_cache(other),
    };
    cache.entries[usize::from(list)].get_or_init(|| SchemaEntry::new(shape, list))
}

// caches for roots that are not structs or enums (a `Vec<T>`, a `String`)
fn other_cache(shape: &'static Shape) -> &'static Cache {
    static MAP: OnceLock<Mutex<Vec<(usize, &'static Cache)>>> = OnceLock::new();
    let key = shape as *const Shape as usize;
    let mut m = lock(MAP.get_or_init(Default::default));
    if let Some((_, c)) = m.iter().find(|(k, _)| *k == key) {
        return c;
    }
    let c: &'static Cache = Box::leak(Box::new(Cache::new()));
    m.push((key, c));
    c
}

pub(crate) fn sanitize_name(name: &str) -> String {
    let mut out: String = name
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect();
    out.truncate(64);
    if out.is_empty() {
        out.push_str("output");
    }
    out
}
