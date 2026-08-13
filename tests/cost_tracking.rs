//! Cost tracking is a thin call into `tiktoken::pricing`. What this crate needs
//! from it is that a model id spelled the way the provider API spells it — which
//! is what a caller has on hand — actually prices, rather than silently costing
//! nothing.
//!
//! tiktoken 4.1 resolves those spellings itself, so this crate carries no
//! normalization of its own. These tests are the contract that keeps that true:
//! if a future tiktoken stops resolving one of these, this fails here rather
//! than showing up as a `None` cost in someone's dashboard.

#![cfg(feature = "cost-tracking")]

/// The defaults in `ProviderKind::default_model` must price, or cost tracking
/// is dead on arrival for anyone who never calls `.with_model()`.
#[test]
fn default_models_price() {
    for id in ["gpt-5.6-terra", "claude-sonnet-5", "gemini-3.6-flash"] {
        let cost = tiktoken::pricing::estimate_cost(id, 1_000, 1_000);
        assert!(
            cost.is_some_and(|c| c > 0.0),
            "no price for default model {id}"
        );
    }
}

/// Ids as the provider APIs spell them: dashes where the price table uses a
/// dot, release-date suffixes, Bedrock and Vertex decoration.
#[test]
fn api_spelled_ids_price() {
    for id in [
        "claude-haiku-4-5",
        "claude-haiku-4-5-20251001",
        "claude-opus-4-6",
        "claude-sonnet-4-6",
        "claude-sonnet-4-20250514",
        "claude-3-5-sonnet-20241022",
        "claude-sonnet-4-0",
        "gpt-4o-2024-08-06",
        "anthropic.claude-opus-5",
        "us.anthropic.claude-opus-5",
    ] {
        let cost = tiktoken::pricing::estimate_cost(id, 1_000, 1_000);
        assert!(cost.is_some_and(|c| c > 0.0), "no price for {id}");
    }
}

/// Resolution must not invent a price for something that is not a model.
#[test]
fn unknown_ids_still_cost_nothing() {
    assert!(tiktoken::pricing::estimate_cost("totally-made-up-9-9", 1_000, 1_000).is_none());
    assert!(tiktoken::pricing::estimate_cost("", 1_000, 1_000).is_none());
}

/// A model addressed two ways is the same model at the same price.
#[test]
fn spellings_agree_on_price() {
    let table = tiktoken::pricing::estimate_cost("claude-haiku-4.5", 1_000, 1_000).unwrap();
    let api = tiktoken::pricing::estimate_cost("claude-haiku-4-5", 1_000, 1_000).unwrap();
    assert_eq!(table, api);
}
