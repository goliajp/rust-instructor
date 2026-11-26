//! The constructors, one per provider.

use super::Client;
use crate::wire::{Host, OutputMode, Wire};

impl Client {
    /// A client for OpenAI (Chat Completions; see [`with_api`](Self::with_api)).
    pub fn openai(api_key: impl Into<String>) -> Self {
        Self::openai_compatible(api_key, "https://api.openai.com/v1")
    }

    /// A client for any OpenAI-compatible Chat Completions API.
    ///
    /// The base URL's host selects details for APIs this crate knows:
    /// `api.openai.com` gets `max_completion_tokens`; DeepSeek
    /// (`deepseek.com`), Qwen (`aliyuncs.com`), Kimi (`moonshot.cn`,
    /// `moonshot.ai`, `kimi.ai`, `kimi.com`) and GLM (`bigmodel.cn`, `z.ai`)
    /// get their own [`Reasoning`](crate::Reasoning) parameters. DeepSeek, Qwen and GLM default
    /// to [`OutputMode::JsonObject`], the others to [`OutputMode::Schema`];
    /// change it with [`with_output_mode`](Self::with_output_mode).
    pub fn openai_compatible(api_key: impl Into<String>, base_url: impl Into<String>) -> Self {
        let base_url = base_url.into();
        let host = Host::from_base_url(&base_url);
        Self::new(
            Wire::OpenAiChat(host),
            api_key.into(),
            base_url,
            host.default_mode(),
        )
    }

    /// A client for DeepSeek (`https://api.deepseek.com`, default model
    /// `deepseek-flash`, [`OutputMode::JsonObject`]).
    pub fn deepseek(api_key: impl Into<String>) -> Self {
        Self::openai_compatible(api_key, "https://api.deepseek.com")
    }

    /// A client for Qwen on Alibaba Cloud Model Studio, through its
    /// OpenAI-compatible mode in the Singapore region
    /// (`https://dashscope-intl.aliyuncs.com/compatible-mode/v1`, default
    /// model `qwen3.7-plus`, [`OutputMode::JsonObject`]). For another region
    /// or a workspace domain, pass its URL to
    /// [`openai_compatible`](Self::openai_compatible); any `aliyuncs.com`
    /// host gets the same defaults.
    pub fn qwen(api_key: impl Into<String>) -> Self {
        Self::openai_compatible(
            api_key,
            "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        )
    }

    /// A client for Anthropic. Uses structured output
    /// (`output_config.format`); a type that refers to itself goes as a tool.
    pub fn anthropic(api_key: impl Into<String>) -> Self {
        Self::new(
            Wire::Anthropic,
            api_key.into(),
            "https://api.anthropic.com/v1".into(),
            OutputMode::Schema,
        )
    }

    /// A client for any Anthropic-compatible API. Sends the schema as a
    /// forced tool call ([`OutputMode::Tool`]), which compatible servers
    /// support more widely than `output_config`.
    pub fn anthropic_compatible(api_key: impl Into<String>, base_url: impl Into<String>) -> Self {
        Self::new(
            Wire::Anthropic,
            api_key.into(),
            base_url.into(),
            OutputMode::Tool,
        )
    }

    /// A client for Google Gemini.
    pub fn gemini(api_key: impl Into<String>) -> Self {
        Self::gemini_compatible(api_key, "https://generativelanguage.googleapis.com/v1beta")
    }

    /// A client for any Gemini-compatible API.
    pub fn gemini_compatible(api_key: impl Into<String>, base_url: impl Into<String>) -> Self {
        Self::new(
            Wire::Gemini,
            api_key.into(),
            base_url.into(),
            OutputMode::Schema,
        )
    }
}
