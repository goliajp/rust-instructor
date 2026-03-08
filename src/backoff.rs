use std::time::Duration;

/// Configuration for exponential backoff on retryable HTTP errors (429, 503).
#[derive(Debug, Clone)]
pub struct BackoffConfig {
    /// Base delay between retries. Default: 500ms.
    pub base_delay: Duration,
    /// Maximum delay cap. Default: 30s.
    pub max_delay: Duration,
    /// Whether to apply random jitter to delays. Default: true.
    pub jitter: bool,
    /// Maximum number of HTTP-level retries. Default: 3.
    pub max_http_retries: u32,
}

impl Default for BackoffConfig {
    fn default() -> Self {
        Self {
            base_delay: Duration::from_millis(500),
            max_delay: Duration::from_secs(30),
            jitter: true,
            max_http_retries: 3,
        }
    }
}

impl BackoffConfig {
    /// Calculate the delay for a given attempt number using exponential backoff.
    ///
    /// Formula: `min(base_delay * 2^attempt, max_delay)`, optionally multiplied
    /// by a random jitter factor in `[0.5, 1.0)`.
    pub fn delay_for(&self, attempt: u32) -> Duration {
        let exp = self.base_delay.saturating_mul(1u32 << attempt.min(16));
        let capped = exp.min(self.max_delay);

        if self.jitter {
            // jitter factor in [0.5, 1.0)
            let factor = 0.5 + rand::random::<f64>() * 0.5;
            capped.mul_f64(factor)
        } else {
            capped
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config() {
        let config = BackoffConfig::default();
        assert_eq!(config.base_delay, Duration::from_millis(500));
        assert_eq!(config.max_delay, Duration::from_secs(30));
        assert!(config.jitter);
        assert_eq!(config.max_http_retries, 3);
    }

    #[test]
    fn delay_exponential() {
        let config = BackoffConfig {
            jitter: false,
            ..Default::default()
        };
        assert_eq!(config.delay_for(0), Duration::from_millis(500));
        assert_eq!(config.delay_for(1), Duration::from_millis(1000));
        assert_eq!(config.delay_for(2), Duration::from_millis(2000));
        assert_eq!(config.delay_for(3), Duration::from_millis(4000));
    }

    #[test]
    fn delay_capped() {
        let config = BackoffConfig {
            jitter: false,
            ..Default::default()
        };
        // attempt 10 → 500ms * 1024 = 512s, but capped at 30s
        let delay = config.delay_for(10);
        assert!(delay <= Duration::from_secs(30));
        assert_eq!(delay, Duration::from_secs(30));
    }

    #[test]
    fn delay_with_jitter() {
        let config = BackoffConfig::default();
        // jitter should produce values in [base/2, base) for attempt 0
        // i.e., [250ms, 500ms)
        for _ in 0..20 {
            let delay = config.delay_for(0);
            assert!(delay >= Duration::from_millis(250));
            assert!(delay < Duration::from_millis(500));
        }
    }

    #[test]
    fn delay_no_jitter() {
        let config = BackoffConfig {
            jitter: false,
            ..Default::default()
        };
        // without jitter, delay is always exact
        assert_eq!(config.delay_for(0), Duration::from_millis(500));
        assert_eq!(config.delay_for(0), Duration::from_millis(500));
    }

    #[test]
    fn clone_and_debug() {
        let config = BackoffConfig::default();
        let cloned = config.clone();
        assert_eq!(cloned.base_delay, config.base_delay);
        let debug = format!("{config:?}");
        assert!(debug.contains("BackoffConfig"));
    }
}
