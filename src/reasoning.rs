/// How much the model may think before it answers.
///
/// One setting for every provider. Nothing is sent unless you set it, so a
/// client without it sends the same request as before. The crate maps the
/// value to each provider's own parameter and does not keep a list of which
/// model accepts which value: a value a model rejects comes back as
/// [`Error::Api`](crate::Error::Api) with the provider's message.
///
/// | Provider | `Effort(e)` | `Budget(n)` | `Off` |
/// |---|---|---|---|
/// | OpenAI, generic OpenAI-compatible | `reasoning_effort: e` | not sent, returns an error | `reasoning_effort: "none"` |
/// | Anthropic | `thinking: {type: "adaptive"}`, `output_config.effort: e` (`Minimal` is sent as `low`) | `thinking: {type: "enabled", budget_tokens: n}` | `thinking: {type: "disabled"}` |
/// | Gemini | `thinkingConfig.thinkingLevel: E` (`Max` is sent as `XHIGH`) | `thinkingConfig.thinkingBudget: n` | `thinkingConfig.thinkingBudget: 0` |
/// | DeepSeek, Kimi, GLM | `thinking: {type: "enabled"}`, `reasoning_effort` of `low`, `high` or `max` | not sent, returns an error | `thinking: {type: "disabled"}` |
/// | Qwen (DashScope) | `enable_thinking: true` | `enable_thinking: true`, `thinking_budget: n` | `enable_thinking: false` |
///
/// For DeepSeek, Kimi and GLM, `Minimal` and `Low` map to `low`, `Medium` and
/// `High` to `high`, `XHigh` and `Max` to `max`.
///
/// ```rust,no_run
/// use instructors::{Client, Effort, Reasoning};
///
/// let client = Client::openai("sk-...").with_reasoning(Reasoning::Effort(Effort::Low));
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Reasoning {
    /// A relative level, portable across providers.
    Effort(Effort),
    /// A token budget for thinking, where the provider takes one.
    Budget(u32),
    /// Turn thinking off, where the model allows it.
    Off,
}

/// Reasoning effort levels, from least to most.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
#[non_exhaustive]
pub enum Effort {
    Minimal,
    Low,
    Medium,
    High,
    XHigh,
    Max,
}

impl Effort {
    /// The lowercase name used by OpenAI and Anthropic.
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Minimal => "minimal",
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::XHigh => "xhigh",
            Self::Max => "max",
        }
    }

    /// The three-level scale DeepSeek, Kimi and GLM accept.
    pub(crate) fn coarse(self) -> &'static str {
        match self {
            Self::Minimal | Self::Low => "low",
            Self::Medium | Self::High => "high",
            Self::XHigh | Self::Max => "max",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names() {
        assert_eq!(Effort::XHigh.as_str(), "xhigh");
        assert_eq!(Effort::Minimal.coarse(), "low");
        assert_eq!(Effort::Medium.coarse(), "high");
        assert_eq!(Effort::Max.coarse(), "max");
        assert!(Effort::Low < Effort::High);
    }
}
