use schemars::JsonSchema;
use serde::Deserialize;
use serde::de::DeserializeOwned;

use crate::client::{Client, ExtractResult};
use crate::error::{Error, Result};
use crate::extract::ExtractBuilder;
use crate::partial::PartialParser;
use crate::provider::{RawResponse, RequestSpec, Stop};
use crate::schema::{self, SchemaEntry};
use crate::usage::Usage;

const RETRYABLE_STATUS: [u16; 6] = [429, 500, 502, 503, 504, 529];

#[derive(Deserialize)]
struct Items<T> {
    items: T,
}

enum Failure {
    Parse(String),
    Validation(String),
}

impl<T> ExtractBuilder<'_, T>
where
    T: DeserializeOwned + JsonSchema + Send + Sync + 'static,
{
    pub(crate) async fn execute(self) -> Result<ExtractResult<T>> {
        let timeout = self.timeout;
        #[cfg(feature = "tracing")]
        {
            use tracing::Instrument;
            let model = self
                .model
                .as_deref()
                .unwrap_or(self.client.provider.default_model())
                .to_string();
            let span = tracing::info_span!(
                "instructors.extract",
                provider = self.client.provider.kind_name(),
                model = model.as_str(),
                max_retries = self.max_retries,
            );
            tokio::time::timeout(timeout, self.with_fallbacks().instrument(span))
                .await
                .map_err(|_| Error::Timeout(timeout))?
        }
        #[cfg(not(feature = "tracing"))]
        {
            tokio::time::timeout(timeout, self.with_fallbacks())
                .await
                .map_err(|_| Error::Timeout(timeout))?
        }
    }

    async fn with_fallbacks(&self) -> Result<ExtractResult<T>> {
        let entry = schema::entry_for::<T>(self.list);
        let client = self.client;
        let model = self
            .model
            .as_deref()
            .unwrap_or(client.provider.default_model());
        let first = match self.try_provider(client, model, &entry).await {
            Ok(ok) => return Ok(ok),
            Err(e) if client.fallbacks.is_empty() => return Err(e),
            Err(e) => e,
        };
        #[cfg(feature = "tracing")]
        tracing::info!(error = %first, "primary provider failed, trying fallbacks");

        let mut failures = Vec::new();
        for fallback in &client.fallbacks {
            let model = fallback
                .default_model
                .as_deref()
                .unwrap_or(fallback.provider.default_model());
            match self.try_provider(fallback, model, &entry).await {
                Ok(ok) => return Ok(ok),
                Err(e) => {
                    failures.push(format!("{} ({model}): {e}", fallback.provider.kind_name()))
                }
            }
        }
        Err(first.with_fallback_failures(&failures.join("; ")))
    }

    async fn try_provider(
        &self,
        client: &Client,
        model: &str,
        entry: &SchemaEntry,
    ) -> Result<ExtractResult<T>> {
        let mut user_content = self.prompt.clone();
        if let Some(ctx) = &self.context {
            user_content = format!("{user_content}\n\nAdditional context:\n{ctx}");
        }
        let history = self.history.as_deref().unwrap_or_default();
        let mut usage = Usage::default();
        let mut cost = Some(0.0);
        let mut last = Failure::Parse(String::new());

        for attempt in 0..=self.max_retries {
            let prompt = match (attempt, &last) {
                (0, _) => user_content.clone(),
                (_, Failure::Parse(e) | Failure::Validation(e)) => format!(
                    "{user_content}\n\n\
                     IMPORTANT: A previous extraction attempt produced invalid output.\n\
                     Error: {e}\n\
                     Please ensure your response is valid JSON matching the schema exactly."
                ),
            };
            let mut messages = Vec::with_capacity(history.len() + 1);
            messages.extend_from_slice(history);
            messages.push(crate::Message::user_with_images(
                prompt.as_str(),
                self.images.clone(),
            ));
            if let Some(hook) = &self.on_request {
                hook(model, &prompt);
            }
            let spec = RequestSpec {
                model,
                system: self.system.as_deref(),
                messages: &messages,
                schema: entry,
                temperature: self.temperature,
                max_tokens: self.max_tokens,
                reasoning: self.reasoning,
            };
            let raw = self.send(client, &spec).await?;
            usage.accumulate(raw.input_tokens, raw.output_tokens);
            cost = add_cost(cost, model, &raw);

            match self.check(&raw) {
                Ok(value) => {
                    usage.cost = cost;
                    #[cfg(feature = "tracing")]
                    tracing::info!(
                        input_tokens = usage.input_tokens,
                        output_tokens = usage.output_tokens,
                        retries = usage.retries,
                        "extraction succeeded"
                    );
                    if let Some(hook) = &self.on_response {
                        hook(&usage);
                    }
                    return Ok(ExtractResult { value, usage });
                }
                Err(Failure::Parse(_)) if let Stop::Truncated(reason) = &raw.stop => {
                    return Err(Error::truncated(client.provider.kind_name(), reason));
                }
                Err(f) => last = f,
            }
            #[cfg(feature = "tracing")]
            tracing::info!(attempt, "attempt failed, retrying");
            if attempt < self.max_retries {
                usage.retries += 1;
            }
        }

        let retries = self.max_retries;
        Err(match last {
            Failure::Validation(message) => Error::ValidationFailed { retries, message },
            Failure::Parse(message) => Error::ExtractionFailed { retries, message },
        })
    }

    /// Parse and validate one answer.
    fn check(&self, raw: &RawResponse) -> std::result::Result<T, Failure> {
        let value = parse::<T>(&raw.content, self.list, &raw.stop).map_err(Failure::Parse)?;
        if let Some(validator) = &self.validator
            && let Err(e) = validator(&value)
        {
            return Err(Failure::Validation(format!("validation: {}", e.message)));
        }
        Ok(value)
    }

    /// One request, with HTTP-level backoff and streaming hooks.
    async fn send(&self, client: &Client, spec: &RequestSpec<'_>) -> Result<RawResponse> {
        let max_http_retries = self.backoff.as_ref().map_or(0, |b| b.max_http_retries);
        let streaming = self.on_stream.is_some() || self.on_partial.is_some();
        let mut http_attempt = 0;
        loop {
            let mut parser = PartialParser::new();
            let mut on_delta = |delta: &str| {
                if let Some(hook) = &self.on_stream {
                    hook(delta);
                }
                if let Some(hook) = &self.on_partial {
                    parser.feed(delta);
                    let root = parser.root();
                    let view = if self.list {
                        root.and_then(|r| r.get("items"))
                    } else {
                        root
                    };
                    if let Some(v) = view {
                        hook(v);
                    }
                }
            };
            let sink: crate::provider::DeltaSink<'_> = match streaming {
                true => Some(&mut on_delta),
                false => None,
            };
            match client.provider.send(&client.http, spec, sink).await {
                Err(Error::Api { status, .. })
                    if RETRYABLE_STATUS.contains(&status) && http_attempt < max_http_retries =>
                {
                    if let Some(backoff) = &self.backoff {
                        let delay = backoff.delay_for(http_attempt);
                        #[cfg(feature = "tracing")]
                        tracing::warn!(status, attempt = http_attempt, "retryable HTTP error");
                        tokio::time::sleep(delay).await;
                    }
                    http_attempt += 1;
                }
                other => return other,
            }
        }
    }
}

#[cfg_attr(not(feature = "cost-tracking"), allow(unused_variables))]
fn add_cost(total: Option<f64>, model: &str, raw: &RawResponse) -> Option<f64> {
    #[cfg(feature = "cost-tracking")]
    {
        // priced per request: long-context tiers depend on one request's size
        let one = tiktoken::pricing::estimate_cost(
            model,
            raw.input_tokens as u64,
            raw.output_tokens as u64,
        );
        total.zip(one).map(|(a, b)| a + b)
    }
    #[cfg(not(feature = "cost-tracking"))]
    {
        let _ = total;
        None
    }
}

fn parse_direct<T: DeserializeOwned>(s: &str, list: bool) -> serde_json::Result<T> {
    if list {
        // a model that ignored the wrapper may still return the bare list
        serde_json::from_str::<Items<T>>(s)
            .map(|w| w.items)
            .or_else(|e| serde_json::from_str::<T>(s).map_err(|_| e))
    } else {
        serde_json::from_str(s)
    }
}

/// Parse an answer, repairing malformed JSON unless the answer was cut off
/// (a repaired truncation would silently drop fields).
pub(crate) fn parse<T: DeserializeOwned>(
    content: &str,
    list: bool,
    stop: &Stop,
) -> std::result::Result<T, String> {
    let direct_err = match parse_direct(content, list) {
        Ok(v) => return Ok(v),
        Err(e) => e,
    };
    if matches!(stop, Stop::Done)
        && let Some(crate::json::Repaired { text: repaired, .. }) = crate::json::repair(content)
        && repaired != content
    {
        return parse_direct(&repaired, list)
            .map_err(|e| format!("{e} (repaired: {})", truncate(&repaired, 200)));
    }
    Err(format!("{direct_err} (raw: {})", truncate(content, 200)))
}

fn truncate(s: &str, max: usize) -> &str {
    &s[..s.floor_char_boundary(max.min(s.len()))]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Deserialize, PartialEq)]
    struct P {
        a: i32,
    }

    #[test]
    fn truncate_respects_char_boundaries() {
        assert_eq!(truncate("hello", 10), "hello");
        assert_eq!(truncate("hello", 0), "");
        let t = truncate("你好世界", 4);
        assert_eq!(t, "你");
    }

    #[test]
    fn parse_list_wrapper_and_bare_list() {
        let done = Stop::Done;
        assert_eq!(
            parse::<Vec<P>>(r#"{"items":[{"a":1}]}"#, true, &done).unwrap(),
            vec![P { a: 1 }]
        );
        assert_eq!(
            parse::<Vec<P>>(r#"[{"a":2}]"#, true, &done).unwrap(),
            vec![P { a: 2 }]
        );
    }

    #[test]
    fn parse_repairs_only_complete_answers() {
        assert_eq!(
            parse::<P>("{'a': 3,}", false, &Stop::Done).unwrap(),
            P { a: 3 }
        );
        let cut = Stop::Truncated("length".into());
        assert!(parse::<P>("{'a': 3,}", false, &cut).is_err());
    }
}
