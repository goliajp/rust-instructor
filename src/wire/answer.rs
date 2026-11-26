//! Which schema form an answer comes back in, per provider and request.

use super::{OutputMode, Provider, Req, Wire, anthropic};
use crate::schema::{Dialect, Reader};

/// The schema for an OpenAI `json_schema` or function, and whether it is
/// strict. A type that holds a value of any type has no strict schema, so
/// it goes as a plain one and the answer is checked here.
pub(crate) fn openai_schema<'a>(r: &Req<'a>) -> (&'a str, bool) {
    if r.schema.has_any() {
        (r.schema.get(Dialect::Plain), false)
    } else {
        (r.schema.get(Dialect::Strict), true)
    }
}

impl Provider {
    /// The schema dialect the provider enforces on the answer, when it
    /// enforces one: the answer is in that dialect's form, and constraints
    /// the dialect dropped are checked here.
    pub(crate) fn answer_dialect(&self, r: &Req<'_>) -> Option<Dialect> {
        let any = r.schema.has_any();
        match &self.wire {
            Wire::OpenAiChat(_) if matches!(r.mode, OutputMode::Schema | OutputMode::Tool) => {
                Some(if any { Dialect::Plain } else { Dialect::Strict })
            }
            Wire::OpenAiResponses if r.mode == OutputMode::Schema => {
                Some(if any { Dialect::Plain } else { Dialect::Strict })
            }
            Wire::Anthropic if anthropic::output_format(r) => Some(Dialect::AnthropicOutput),
            Wire::Anthropic if anthropic::tool_call(r) => Some(Dialect::Plain),
            Wire::Gemini if r.mode == OutputMode::Schema => Some(Dialect::GeminiJson),
            _ => None,
        }
    }

    /// The reader for an answer whose schema was reshaped or lost a
    /// constraint; `None` when the answer is the type's own JSON.
    pub(crate) fn reader(&self, r: &Req<'_>) -> Option<Reader> {
        let d = self.answer_dialect(r)?;
        r.schema.reshaped(d).then(|| r.schema.reader(d))
    }

    /// The answer's root is wrapped as `{"value": ..}`.
    pub(crate) fn wraps_root(&self, r: &Req<'_>) -> bool {
        matches!(
            self.answer_dialect(r),
            Some(Dialect::Strict | Dialect::AnthropicOutput | Dialect::Plain)
        ) && r.schema.wrapped()
    }
}
