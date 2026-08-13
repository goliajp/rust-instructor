//! Cost estimation, bridging provider model ids to the `tiktoken` price table.
//!
//! The price table keys a model generation with a dot (`claude-haiku-4.5`),
//! while the APIs address it with a dash (`claude-haiku-4-5`) and historically
//! appended a release date (`claude-sonnet-4-20250514`, `gpt-4o-2024-08-06`).
//! Table lookup is an exact match, so passing an API id straight through means
//! a great many real models silently price at zero. This module tries the id as
//! sent first, then progressively normalized forms, and reports `None` only
//! when none of them are in the table.

/// Estimate cost in USD for a model id as the provider spells it.
///
/// Returns `None` when the model is not in the price table.
pub(crate) fn estimate_cost(model: &str, input_tokens: u64, output_tokens: u64) -> Option<f64> {
    candidates(model)
        .into_iter()
        .find_map(|id| tiktoken::pricing::estimate_cost(&id, input_tokens, output_tokens))
}

/// The id forms to try, most literal first.
fn candidates(model: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut push = |id: String| {
        if !id.is_empty() && !out.contains(&id) {
            out.push(id);
        }
    };

    for base in [model.to_string(), strip_release_date(model)] {
        let merged = merge_version_segments(&base);
        push(base);
        push(merged.clone());
        // `claude-sonnet-4-0` → `claude-sonnet-4.0` → `claude-sonnet-4`
        if let Some(trimmed) = merged.strip_suffix(".0") {
            push(trimmed.to_string());
        }
    }

    out
}

/// Drop a trailing release date: `-20250514` or `-2024-08-06`.
fn strip_release_date(model: &str) -> String {
    let parts: Vec<&str> = model.split('-').collect();

    let is_digits = |s: &str, n: usize| s.len() == n && s.bytes().all(|b| b.is_ascii_digit());

    if parts.len() >= 2 && is_digits(parts[parts.len() - 1], 8) {
        return parts[..parts.len() - 1].join("-");
    }
    if parts.len() >= 4
        && is_digits(parts[parts.len() - 3], 4)
        && is_digits(parts[parts.len() - 2], 2)
        && is_digits(parts[parts.len() - 1], 2)
    {
        return parts[..parts.len() - 3].join("-");
    }

    model.to_string()
}

/// Join runs of adjacent numeric segments with dots, which is how the price
/// table spells a generation: `claude-haiku-4-5` → `claude-haiku-4.5`,
/// `claude-3-5-sonnet` → `claude-3.5-sonnet`. A lone number is left alone, so
/// `claude-3-opus` and `gpt-5.6-terra` pass through unchanged.
fn merge_version_segments(model: &str) -> String {
    let parts: Vec<&str> = model.split('-').collect();
    let numeric = |s: &str| !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit());

    let mut out: Vec<String> = Vec::with_capacity(parts.len());
    let mut i = 0;
    while i < parts.len() {
        let mut j = i;
        while j + 1 < parts.len() && numeric(parts[j]) && numeric(parts[j + 1]) {
            j += 1;
        }
        out.push(parts[i..=j].join("."));
        i = j + 1;
    }

    out.join("-")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_compact_release_date() {
        assert_eq!(
            strip_release_date("claude-sonnet-4-20250514"),
            "claude-sonnet-4"
        );
    }

    #[test]
    fn strips_dashed_release_date() {
        assert_eq!(strip_release_date("gpt-4o-2024-08-06"), "gpt-4o");
    }

    #[test]
    fn leaves_undated_ids_alone() {
        assert_eq!(strip_release_date("claude-sonnet-5"), "claude-sonnet-5");
        assert_eq!(strip_release_date("gpt-5.6-terra"), "gpt-5.6-terra");
    }

    #[test]
    fn merges_trailing_version_pair() {
        assert_eq!(
            merge_version_segments("claude-haiku-4-5"),
            "claude-haiku-4.5"
        );
    }

    #[test]
    fn merges_interior_version_pair() {
        assert_eq!(
            merge_version_segments("claude-3-5-sonnet"),
            "claude-3.5-sonnet"
        );
    }

    #[test]
    fn leaves_lone_numbers_alone() {
        assert_eq!(merge_version_segments("claude-3-opus"), "claude-3-opus");
        assert_eq!(merge_version_segments("gpt-5.6-terra"), "gpt-5.6-terra");
        assert_eq!(merge_version_segments("o4-mini"), "o4-mini");
    }

    #[test]
    fn candidate_order_is_literal_first() {
        let c = candidates("claude-haiku-4-5-20251001");
        assert_eq!(c[0], "claude-haiku-4-5-20251001");
        assert!(c.contains(&"claude-haiku-4.5".to_string()));
    }

    #[test]
    fn candidates_are_deduplicated() {
        let c = candidates("claude-sonnet-5");
        assert_eq!(c, vec!["claude-sonnet-5".to_string()]);
    }

    // the current defaults must price, or cost tracking is dead on arrival
    #[test]
    fn default_models_price() {
        for id in ["gpt-5.6-terra", "claude-sonnet-5", "gemini-3.6-flash"] {
            assert!(
                estimate_cost(id, 1_000, 1_000).is_some_and(|c| c > 0.0),
                "no price for default model {id}"
            );
        }
    }

    // the ids that used to silently price at zero
    #[test]
    fn dashed_and_dated_ids_price() {
        for id in [
            "claude-haiku-4-5",
            "claude-haiku-4-5-20251001",
            "claude-opus-4-6",
            "claude-sonnet-4-6",
            "claude-sonnet-4-20250514",
            "claude-3-5-sonnet-20241022",
            "claude-sonnet-4-0",
            "gpt-4o-2024-08-06",
        ] {
            assert!(
                estimate_cost(id, 1_000, 1_000).is_some_and(|c| c > 0.0),
                "no price for {id}"
            );
        }
    }

    // The safety property that makes a normalizing lookup trustworthy: every
    // table id must still resolve to itself, and no id may normalize onto a
    // *different* table entry. Without this, normalization could silently
    // price one model at another's rate.
    #[test]
    fn normalization_never_maps_one_model_onto_another() {
        for model in tiktoken::pricing::all_models() {
            let resolved = candidates(model.id)
                .into_iter()
                .find_map(|id| tiktoken::pricing::get_model(&id))
                .unwrap_or_else(|| panic!("{} no longer resolves", model.id));
            assert_eq!(
                resolved.id, model.id,
                "{} normalized onto {}",
                model.id, resolved.id
            );
        }
    }

    #[test]
    fn normalization_does_not_invent_prices() {
        assert!(estimate_cost("totally-made-up-9-9", 1_000, 1_000).is_none());
    }

    #[test]
    fn exact_ids_still_resolve_directly() {
        let direct = tiktoken::pricing::estimate_cost("claude-opus-5", 1_000, 1_000);
        assert_eq!(estimate_cost("claude-opus-5", 1_000, 1_000), direct);
    }
}
