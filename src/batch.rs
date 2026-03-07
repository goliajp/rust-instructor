use std::marker::PhantomData;

use schemars::JsonSchema;
use serde::de::DeserializeOwned;

use crate::client::{Client, ExtractResult};
use crate::error::Result;
use crate::validate::ValidationError;

type ValidatorFn<T> = Box<dyn Fn(&T) -> std::result::Result<(), ValidationError> + Send + Sync>;

/// Builder for concurrent batch extraction.
///
/// Created by [`Client::extract_batch`]. Processes multiple prompts concurrently
/// with configurable concurrency limits.
pub struct BatchBuilder<'a, T> {
    client: &'a Client,
    prompts: Vec<String>,
    model: Option<String>,
    system: Option<String>,
    temperature: Option<f64>,
    max_tokens: u32,
    max_retries: u32,
    concurrency: usize,
    validator: Option<ValidatorFn<T>>,
    _phantom: PhantomData<T>,
}

impl<'a, T> BatchBuilder<'a, T>
where
    T: DeserializeOwned + JsonSchema + Send + 'static,
{
    pub(crate) fn new(client: &'a Client, prompts: Vec<String>) -> Self {
        Self {
            client,
            prompts,
            model: None,
            system: None,
            temperature: None,
            max_tokens: 4096,
            max_retries: 2,
            concurrency: 5,
            validator: None,
            _phantom: PhantomData,
        }
    }

    /// Set the model for all extractions in the batch.
    pub fn model(self, model: impl Into<String>) -> Self {
        Self {
            model: Some(model.into()),
            ..self
        }
    }

    /// Set the system prompt for all extractions.
    pub fn system(self, system: impl Into<String>) -> Self {
        Self {
            system: Some(system.into()),
            ..self
        }
    }

    /// Set the temperature for all extractions.
    pub fn temperature(self, temp: f64) -> Self {
        Self {
            temperature: Some(temp),
            ..self
        }
    }

    /// Set max output tokens for each extraction.
    pub fn max_tokens(self, tokens: u32) -> Self {
        Self {
            max_tokens: tokens,
            ..self
        }
    }

    /// Set max retries per extraction.
    pub fn max_retries(self, retries: u32) -> Self {
        Self {
            max_retries: retries,
            ..self
        }
    }

    /// Set the maximum number of concurrent requests.
    ///
    /// Default is 5. Higher values increase throughput but may trigger rate limits.
    pub fn concurrency(self, n: usize) -> Self {
        Self {
            concurrency: n.max(1),
            ..self
        }
    }

    /// Add a validation function applied to each extraction.
    pub fn validate<F>(self, f: F) -> Self
    where
        F: Fn(&T) -> std::result::Result<(), ValidationError> + Send + Sync + 'static,
    {
        Self {
            validator: Some(Box::new(f)),
            ..self
        }
    }

    /// Execute the batch, returning results in the same order as prompts.
    ///
    /// Each element is `Result<ExtractResult<T>>` — individual extractions
    /// can fail independently without affecting others.
    pub async fn run(self) -> Vec<Result<ExtractResult<T>>> {
        let semaphore = std::sync::Arc::new(tokio::sync::Semaphore::new(self.concurrency));
        let client = self.client;
        let model = self.model;
        let system = self.system;
        let temperature = self.temperature;
        let max_tokens = self.max_tokens;
        let max_retries = self.max_retries;
        let validator = self.validator.map(std::sync::Arc::new);

        let handles: Vec<_> = self
            .prompts
            .into_iter()
            .map(|prompt| {
                let sem = semaphore.clone();
                let model = model.clone();
                let system = system.clone();
                let validator = validator.clone();

                async move {
                    let _permit: tokio::sync::OwnedSemaphorePermit =
                        sem.clone().acquire_owned().await.unwrap();
                    let mut builder = client.extract::<T>(prompt);
                    if let Some(m) = model {
                        builder = builder.model(m);
                    }
                    if let Some(s) = system {
                        builder = builder.system(s);
                    }
                    if let Some(t) = temperature {
                        builder = builder.temperature(t);
                    }
                    builder = builder.max_tokens(max_tokens).max_retries(max_retries);
                    if let Some(v) = validator {
                        builder = builder.validate(move |val: &T| v(val));
                    }
                    builder.await
                }
            })
            .collect();

        // run all concurrently, preserving order
        futures::future::join_all(handles).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builder_defaults() {
        let client = Client::openai("key");

        #[derive(serde::Deserialize, JsonSchema)]
        struct D {
            x: i32,
        }

        let builder = BatchBuilder::<D>::new(&client, vec!["a".into(), "b".into()]);
        assert_eq!(builder.concurrency, 5);
        assert_eq!(builder.max_retries, 2);
        assert_eq!(builder.max_tokens, 4096);
        assert!(builder.model.is_none());
        assert!(builder.validator.is_none());
        assert_eq!(builder.prompts.len(), 2);
    }

    #[test]
    fn builder_overrides() {
        let client = Client::openai("key");

        #[derive(serde::Deserialize, JsonSchema)]
        struct D {
            x: i32,
        }

        let builder = BatchBuilder::<D>::new(&client, vec!["a".into()])
            .model("gpt-4o-mini")
            .system("test")
            .temperature(0.5)
            .max_tokens(1024)
            .max_retries(3)
            .concurrency(10);

        assert_eq!(builder.model.as_deref(), Some("gpt-4o-mini"));
        assert_eq!(builder.system.as_deref(), Some("test"));
        assert_eq!(builder.temperature, Some(0.5));
        assert_eq!(builder.max_tokens, 1024);
        assert_eq!(builder.max_retries, 3);
        assert_eq!(builder.concurrency, 10);
    }

    #[test]
    fn concurrency_minimum_one() {
        let client = Client::openai("key");

        #[derive(serde::Deserialize, JsonSchema)]
        struct D {
            x: i32,
        }

        let builder = BatchBuilder::<D>::new(&client, vec![]).concurrency(0);
        assert_eq!(builder.concurrency, 1);
    }
}
