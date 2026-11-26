//! Where an Anthropic answer comes from, per output mode and model.

use crate::reasoning::Reasoning;
use crate::wire::{OutputMode, Req};

/// Where the answer comes from.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Plan {
    /// `output_config.format`: text blocks.
    Output,
    /// A tool; forced unless thinking is on.
    Tool { forced: bool },
    /// The schema in the system prompt: text blocks.
    Text,
}

/// Models without structured output (`output_config.format`); they get the
/// tool instead. Any other model, including ones newer than this list, is
/// assumed to have it.
const NO_OUTPUT_FORMAT: [&str; 6] = [
    "claude-3",
    "claude-opus-4-0",
    "claude-opus-4-1",
    "claude-opus-4-20250514",
    "claude-sonnet-4-0",
    "claude-sonnet-4-20250514",
];

/// Models that reject a forced `tool_choice` on every request.
const NO_FORCED_TOOL: [&str; 6] = [
    "claude-opus-5-5",
    "claude-sonnet-5-5",
    "claude-fable-5-2",
    "claude-mythos-5-2",
    "claude-fable-5-1",
    "claude-mythos-5-1",
];

/// `model` is `id` or a dated or suffixed form of it.
fn is(model: &str, ids: &[&str]) -> bool {
    ids.iter().any(|id| {
        model
            .strip_prefix(id)
            .is_some_and(|rest| rest.is_empty() || rest.starts_with('-'))
    })
}

/// The request uses `output_config.format`.
pub(crate) fn output_format(r: &Req<'_>) -> bool {
    plan(r) == Plan::Output
}

pub(super) fn plan(r: &Req<'_>) -> Plan {
    // manual and adaptive thinking reject or skip a forced tool call
    let thinking = matches!(
        r.reasoning,
        Some(Reasoning::Effort(_) | Reasoning::Budget(_))
    );
    let tool = Plan::Tool {
        forced: !thinking && !is(r.model, &NO_FORCED_TOOL),
    };
    match r.mode {
        // structured output cannot express recursive schemas; a tool can
        OutputMode::Schema if r.schema.recursive() || is(r.model, &NO_OUTPUT_FORMAT) => tool,
        OutputMode::Schema => Plan::Output,
        OutputMode::Tool => tool,
        OutputMode::JsonObject | OutputMode::Prompt => Plan::Text,
    }
}
