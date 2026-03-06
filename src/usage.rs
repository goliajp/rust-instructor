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
