//! The OpenAI-compatible APIs this crate knows by their base URL.

use crate::reasoning::Reasoning;
use crate::wire::{Fail, Obj, OutputMode, type_obj};

/// Which OpenAI-compatible API is on the other end, from the base URL host.
/// It decides the output-limit field, the reasoning parameters and the
/// default [`OutputMode`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Host {
    OpenAi,
    Generic,
    DeepSeek,
    Qwen,
    Kimi,
    Glm,
}

impl Host {
    pub(crate) fn from_base_url(url: &str) -> Self {
        let host = url
            .split("://")
            .nth(1)
            .unwrap_or(url)
            .split(['/', ':'])
            .next()
            .unwrap_or("")
            .to_ascii_lowercase();
        let is = |domain: &str| host == domain || host.ends_with(&format!(".{domain}"));
        if is("api.openai.com") {
            Self::OpenAi
        } else if is("deepseek.com") {
            Self::DeepSeek
        } else if is("aliyuncs.com") {
            Self::Qwen
        } else if is("moonshot.cn") || is("moonshot.ai") || is("kimi.ai") || is("kimi.com") {
            Self::Kimi
        } else if is("bigmodel.cn") || is("z.ai") {
            Self::Glm
        } else {
            Self::Generic
        }
    }

    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::OpenAi => "openai",
            Self::Generic => "openai-compatible",
            Self::DeepSeek => "deepseek",
            Self::Qwen => "qwen",
            Self::Kimi => "kimi",
            Self::Glm => "glm",
        }
    }

    /// DeepSeek and GLM document only `json_object`; Qwen supports strict
    /// schemas on some models only.
    /// Whether the tool can be forced. DeepSeek and Qwen reject a named
    /// `tool_choice` in thinking mode, which is on by default for their
    /// current models, so they get it only with thinking turned off.
    pub(super) fn forced_tool(self, r: Option<Reasoning>) -> bool {
        match self {
            Self::DeepSeek | Self::Qwen => r == Some(Reasoning::Off),
            _ => true,
        }
    }

    pub(crate) fn default_mode(self) -> OutputMode {
        match self {
            Self::DeepSeek | Self::Glm | Self::Qwen => OutputMode::JsonObject,
            Self::OpenAi | Self::Generic | Self::Kimi => OutputMode::Schema,
        }
    }
}

/// Each host's reasoning parameters.
pub(super) fn reasoning(o: &mut Obj<'_>, host: Host, r: Option<Reasoning>) -> Result<(), Fail> {
    let Some(r) = r else { return Ok(()) };
    let switch = |o: &mut Obj<'_>, on: bool| {
        type_obj(o.key("thinking"), if on { "enabled" } else { "disabled" });
    };
    match (host, r) {
        (Host::OpenAi | Host::Generic, Reasoning::Effort(e)) => {
            o.str("reasoning_effort", e.as_str())
        }
        (Host::OpenAi | Host::Generic, Reasoning::Off) => o.str("reasoning_effort", "none"),
        (Host::Qwen, Reasoning::Effort(_)) => o.bool("enable_thinking", true),
        (Host::Qwen, Reasoning::Budget(n)) => {
            o.bool("enable_thinking", true);
            o.u64("thinking_budget", n.into());
        }
        (Host::Qwen, Reasoning::Off) => o.bool("enable_thinking", false),
        (_, Reasoning::Effort(e)) => {
            switch(o, true);
            o.str("reasoning_effort", e.coarse());
        }
        (_, Reasoning::Off) => switch(o, false),
        (_, Reasoning::Budget(_)) => {
            return Err(Fail::Unsupported(format!(
                "{} has no thinking budget parameter; use Reasoning::Effort",
                host.name()
            )));
        }
    }
    Ok(())
}
