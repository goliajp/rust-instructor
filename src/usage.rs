/// Token usage and cost information from an extraction request.
#[derive(Debug, Clone, Default)]
pub struct Usage {
    /// Number of input (prompt) tokens consumed.
    pub input_tokens: u32,
    /// Number of output (completion) tokens consumed.
    pub output_tokens: u32,
    /// Total tokens (input + output) across all attempts.
    pub total_tokens: u32,
    /// Number of retry attempts due to parse/validation failures.
    pub retries: u32,
    /// Estimated cost in USD (requires `cost-tracking` feature).
    pub cost: Option<f64>,
}

impl Usage {
    pub(crate) fn accumulate(&mut self, input: u32, output: u32) {
        self.input_tokens += input;
        self.output_tokens += output;
        self.total_tokens = self.input_tokens + self.output_tokens;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_usage() {
        let usage = Usage::default();
        assert_eq!(usage.input_tokens, 0);
        assert_eq!(usage.output_tokens, 0);
        assert_eq!(usage.total_tokens, 0);
        assert_eq!(usage.retries, 0);
        assert_eq!(usage.cost, None);
    }

    #[test]
    fn accumulate_single() {
        let mut usage = Usage::default();
        usage.accumulate(100, 50);
        assert_eq!(usage.input_tokens, 100);
        assert_eq!(usage.output_tokens, 50);
        assert_eq!(usage.total_tokens, 150);
    }

    #[test]
    fn accumulate_multiple() {
        let mut usage = Usage::default();
        usage.accumulate(100, 50);
        usage.accumulate(200, 100);
        assert_eq!(usage.input_tokens, 300);
        assert_eq!(usage.output_tokens, 150);
        assert_eq!(usage.total_tokens, 450);
    }

    #[test]
    fn clone_usage() {
        let mut usage = Usage::default();
        usage.accumulate(10, 5);
        usage.retries = 1;
        let cloned = usage.clone();
        assert_eq!(cloned.input_tokens, 10);
        assert_eq!(cloned.retries, 1);
    }

    #[test]
    fn debug_format() {
        let usage = Usage::default();
        let debug = format!("{usage:?}");
        assert!(debug.contains("Usage"));
        assert!(debug.contains("input_tokens"));
    }
}
