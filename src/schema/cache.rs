use std::sync::{Mutex, OnceLock};

use super::dialect;
use super::lower;
use super::tree::J;
#[cfg(any(test, feature = "fuzzing"))]
use crate::json::JsonError;

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
    /// Gemini `responseFormat.text.schema`.
    GeminiJson,
    /// The plain schema with a root that is not an object wrapped as
    /// `{"value": ..}`: Anthropic tool `input_schema`, and OpenAI when no
    /// strict schema can carry the type.
    Plain,
}

const DIALECTS: usize = 5;

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
    reshaped: [OnceLock<bool>; 3],
    has_any: OnceLock<bool>,
    plan: OnceLock<super::guide::Plan>,
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
            reshaped: Default::default(),
            has_any: OnceLock::new(),
            plan: OnceLock::new(),
        }
    }

    #[cfg(feature = "schemars")]
    pub(crate) fn from_parts(name: String, base: J, recursive: bool) -> Self {
        Self {
            name,
            recursive,
            base,
            built: Default::default(),
            reshaped: Default::default(),
            has_any: OnceLock::new(),
            plan: OnceLock::new(),
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

    /// True when answers to the strict `dialect` need [`guide`](Self::guide):
    /// the schema was reshaped or lost a constraint there.
    pub(crate) fn reshaped(&self, dialect: Dialect) -> bool {
        use super::strict::{ANTHROPIC_DROPS, GEMINI_DROPS, OPENAI_DROPS};
        let (i, drops, min_items, reshape) = match dialect {
            Dialect::Strict => (0, &OPENAI_DROPS[..], false, true),
            Dialect::AnthropicOutput => (1, &ANTHROPIC_DROPS[..], true, true),
            Dialect::GeminiJson => (2, &GEMINI_DROPS[..], false, false),
            // nothing is enforced on a plain schema's answer: it is checked
            // here when the type holds any value, and unwrapped when wrapped
            Dialect::Plain => return self.has_any() || self.wrapped(),
            Dialect::AnthropicTool => return false,
        };
        *self.reshaped[i].get_or_init(|| {
            super::strict::reshaped(&self.base, drops, min_items, reshape)
                || (dialect == Dialect::GeminiJson && super::strict::has_fills(&self.base))
        })
    }

    /// The schema accepts any value somewhere, so a strict schema cannot
    /// carry it.
    pub(crate) fn has_any(&self) -> bool {
        *self
            .has_any
            .get_or_init(|| super::strict::has_any(&self.base))
    }

    /// True when a strict schema wraps the root as `{"value": ..}`.
    pub(crate) fn wrapped(&self) -> bool {
        !super::strict::object_root(&self.base)
    }

    /// Reads answers to the dialects [`reshaped`](Self::reshaped) marks.
    pub(crate) fn guide(&self) -> &super::guide::Plan {
        self.plan
            .get_or_init(|| super::guide::Plan::new(&self.base))
    }

    /// The reader for answers in `dialect`'s form.
    pub(crate) fn reader(&'static self, dialect: Dialect) -> Reader {
        Reader {
            plan: self.guide(),
            fill: dialect == Dialect::GeminiJson,
        }
    }

    /// The answer as the type's own JSON text: the reference the guided
    /// reader is tested against.
    #[cfg(any(test, feature = "fuzzing"))]
    pub(crate) fn unwire(&self, answer: &str) -> Result<String, JsonError> {
        super::transcode::answer(&self.base, answer).map_err(JsonError::new)
    }

    #[cfg(test)]
    pub(crate) fn get_15(&self, dialect: Dialect) -> String {
        dialect::build_15(&self.base, dialect)
    }

    /// Serialized schema for `dialect`, built on first call.
    pub(crate) fn get(&self, dialect: Dialect) -> &str {
        self.built[dialect as usize].get_or_init(|| dialect::build(&self.base, dialect))
    }
}

/// Reads answers whose schema was reshaped for the wire straight into the
/// caller's type.
#[derive(Clone, Copy)]
pub(crate) struct Reader {
    pub(crate) plan: &'static super::guide::Plan,
    /// A required recursive list or map the schema left optional (Gemini)
    /// reads as empty when the answer omits it.
    pub(crate) fill: bool,
}

impl Reader {
    pub(crate) fn answer<T: serde::de::DeserializeOwned>(
        &self,
        text: &str,
        list: bool,
    ) -> Result<T, crate::json::JsonError> {
        self.plan.answer(text, list, self.fill)
    }

    pub(crate) fn item<T: serde::de::DeserializeOwned>(
        &self,
        text: &str,
    ) -> Result<T, crate::json::JsonError> {
        self.plan.item(text, self.fill)
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
