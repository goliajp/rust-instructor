/// Tokens and cost of an extraction, summed over every request it sent
/// (re-asks, HTTP retries and fallbacks included).
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Usage {
    /// All input tokens, cached ones included. Anthropic's three input
    /// counters (uncached, cache read, cache write) are added together.
    pub input_tokens: u64,
    /// Input tokens read from the provider's prompt cache: OpenAI
    /// `cached_tokens`, Anthropic `cache_read_input_tokens`, Gemini
    /// `cachedContentTokenCount`.
    pub cached_input_tokens: u64,
    /// Input tokens written to the prompt cache (Anthropic
    /// `cache_creation_input_tokens`).
    pub cache_write_tokens: u64,
    /// All output tokens, reasoning included.
    pub output_tokens: u64,
    /// Output tokens spent on reasoning: OpenAI `reasoning_tokens`, Gemini
    /// `thoughtsTokenCount`.
    pub reasoning_tokens: u64,
    /// Responses from the model: answers, including rejected ones, and
    /// refusals. HTTP errors are not counted.
    pub requests: u32,
    /// Estimated cost in USD, priced per request and summed (feature
    /// `cost-tracking`). `None` when any request was for a model the price
    /// table does not know.
    pub cost: Option<f64>,
}

impl Usage {
    /// Input plus output tokens.
    pub fn total_tokens(&self) -> u64 {
        self.input_tokens + self.output_tokens
    }

    /// Requests after the first.
    pub fn retries(&self) -> u32 {
        self.requests.saturating_sub(1)
    }

    /// Add one response's counts, priced for `model`.
    pub(crate) fn add(&mut self, one: &Usage, model: &str) {
        let first = self.requests == 0;
        self.input_tokens += one.input_tokens;
        self.cached_input_tokens += one.cached_input_tokens;
        self.cache_write_tokens += one.cache_write_tokens;
        self.output_tokens += one.output_tokens;
        self.reasoning_tokens += one.reasoning_tokens;
        self.requests += 1;
        let priced = price(model, one);
        self.cost = match (first, self.cost, priced) {
            (true, _, p) => p,
            (false, Some(a), Some(b)) => Some(a + b),
            _ => None,
        };
    }

    /// Sum of two extractions' usage (a fallback's onto the primary's).
    pub(crate) fn merge(&mut self, other: &Usage) {
        if other.requests == 0 {
            return;
        }
        if self.requests == 0 {
            *self = other.clone();
            return;
        }
        self.input_tokens += other.input_tokens;
        self.cached_input_tokens += other.cached_input_tokens;
        self.cache_write_tokens += other.cache_write_tokens;
        self.output_tokens += other.output_tokens;
        self.reasoning_tokens += other.reasoning_tokens;
        self.requests += other.requests;
        self.cost = self.cost.zip(other.cost).map(|(a, b)| a + b);
    }
}

#[cfg(feature = "cost-tracking")]
fn price(model: &str, u: &Usage) -> Option<f64> {
    // priced per request: long-context tiers depend on one request's size
    let m = tiktoken::pricing::resolve_model(model)?.model;
    let cached = u.cached_input_tokens.min(u.input_tokens);
    Some(m.estimate_cost_with_cache(u.input_tokens - cached, cached, u.output_tokens))
}

#[cfg(not(feature = "cost-tracking"))]
fn price(_model: &str, _u: &Usage) -> Option<f64> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn one(input: u64, output: u64) -> Usage {
        Usage {
            input_tokens: input,
            output_tokens: output,
            ..Usage::default()
        }
    }

    #[test]
    fn add_sums_counts_and_requests() {
        let mut u = Usage::default();
        u.add(&one(100, 50), "unknown-model-x");
        u.add(&one(200, 100), "unknown-model-x");
        assert_eq!(u.input_tokens, 300);
        assert_eq!(u.output_tokens, 150);
        assert_eq!(u.total_tokens(), 450);
        assert_eq!(u.requests, 2);
        assert_eq!(u.retries(), 1);
        assert_eq!(u.cost, None);
    }

    #[cfg(feature = "cost-tracking")]
    #[test]
    fn cost_is_none_once_any_request_is_unpriced() {
        let mut u = Usage::default();
        u.add(&one(1000, 1000), "gpt-4o");
        assert!(u.cost.is_some_and(|c| c > 0.0));
        u.add(&one(1000, 1000), "unknown-model-x");
        assert_eq!(u.cost, None);
    }

    #[cfg(feature = "cost-tracking")]
    #[test]
    fn cached_tokens_are_cheaper() {
        let mut plain = Usage::default();
        plain.add(&one(10_000, 0), "gpt-4o");
        let mut cached = Usage::default();
        cached.add(
            &Usage {
                cached_input_tokens: 8_000,
                ..one(10_000, 0)
            },
            "gpt-4o",
        );
        assert!(cached.cost.unwrap() < plain.cost.unwrap());
    }

    #[test]
    fn merge_keeps_requests_and_drops_unknown_cost() {
        let mut a = Usage::default();
        a.add(&one(1, 2), "unknown-model-x");
        let mut b = Usage::default();
        b.add(&one(3, 4), "unknown-model-x");
        a.merge(&b);
        assert_eq!(a.requests, 2);
        assert_eq!(a.total_tokens(), 10);
        let mut empty = Usage::default();
        empty.merge(&a);
        assert_eq!(empty, a);
    }
}
