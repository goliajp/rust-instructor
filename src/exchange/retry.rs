use std::hash::{BuildHasher, RandomState};
use std::time::Duration;

use crate::wire::Headers;

/// When to resend a request that failed at the HTTP level: 408, 429, 500,
/// 502, 503, 504, 529 and connection errors. On by default.
///
/// The delay is `base_delay * 2^n`, capped at `max_delay`, times a random
/// factor in `[0.5, 1)` when `jitter` is on. A `Retry-After` header (seconds)
/// or `retry-after-ms` from the provider replaces the computed delay, up to
/// `max_delay`.
#[derive(Clone, Debug, PartialEq)]
pub struct RetryPolicy {
    pub(crate) max_retries: u32,
    pub(crate) base_delay: Duration,
    pub(crate) max_delay: Duration,
    pub(crate) jitter: bool,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_retries: 3,
            base_delay: Duration::from_millis(500),
            max_delay: Duration::from_secs(30),
            jitter: true,
        }
    }
}

impl RetryPolicy {
    /// The default: 3 retries, 500 ms base delay, 30 s cap, jitter.
    pub fn new() -> Self {
        Self::default()
    }

    /// Never resend.
    pub fn none() -> Self {
        Self {
            max_retries: 0,
            ..Self::default()
        }
    }

    pub fn max_retries(mut self, n: u32) -> Self {
        self.max_retries = n;
        self
    }

    pub fn base_delay(mut self, d: Duration) -> Self {
        self.base_delay = d;
        self
    }

    pub fn max_delay(mut self, d: Duration) -> Self {
        self.max_delay = d;
        self
    }

    pub fn jitter(mut self, on: bool) -> Self {
        self.jitter = on;
        self
    }

    /// Delay before retry number `n` (0-based).
    pub(crate) fn delay(&self, n: u32, server: Option<Duration>, rng: &mut Rng) -> Duration {
        if let Some(d) = server {
            return d.min(self.max_delay);
        }
        let capped = self
            .base_delay
            .saturating_mul(1u32 << n.min(16))
            .min(self.max_delay);
        if self.jitter {
            capped.mul_f64(0.5 + rng.unit() * 0.5)
        } else {
            capped
        }
    }
}

pub(crate) const RETRYABLE: [u16; 7] = [408, 429, 500, 502, 503, 504, 529];

/// `retry-after-ms`, or `Retry-After` in seconds. An HTTP date is ignored.
pub(crate) fn retry_after(h: &Headers) -> Option<Duration> {
    if let Some(ms) = h
        .get("retry-after-ms")
        .and_then(|v| v.trim().parse::<f64>().ok())
        && ms.is_finite()
        && ms >= 0.0
    {
        return Some(Duration::from_secs_f64(ms / 1000.0));
    }
    let s = h.get("retry-after")?.trim().parse::<f64>().ok()?;
    (s.is_finite() && s >= 0.0).then(|| Duration::from_secs_f64(s))
}

/// xorshift64*, seeded from std's per-process random hasher keys.
pub(crate) struct Rng(u64);

impl Rng {
    pub(crate) fn new() -> Self {
        let seed = RandomState::new().hash_one(0x9e37_79b9_7f4a_7c15_u64);
        Self(seed | 1)
    }

    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    /// Uniform in `[0, 1)`.
    pub(crate) fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exponential_and_capped() {
        let p = RetryPolicy::new().jitter(false);
        let mut rng = Rng::new();
        assert_eq!(p.delay(0, None, &mut rng), Duration::from_millis(500));
        assert_eq!(p.delay(2, None, &mut rng), Duration::from_secs(2));
        assert_eq!(p.delay(10, None, &mut rng), Duration::from_secs(30));
        assert_eq!(p.delay(99, None, &mut rng), Duration::from_secs(30));
    }

    #[test]
    fn jitter_stays_in_range() {
        let p = RetryPolicy::new();
        let mut rng = Rng::new();
        for _ in 0..1000 {
            let d = p.delay(1, None, &mut rng);
            assert!(
                d >= Duration::from_millis(500) && d < Duration::from_secs(1),
                "{d:?}"
            );
        }
    }

    #[test]
    fn server_delay_wins_but_is_capped() {
        let p = RetryPolicy::new();
        let mut rng = Rng::new();
        let s = Some(Duration::from_secs(3));
        assert_eq!(p.delay(0, s, &mut rng), Duration::from_secs(3));
        let s = Some(Duration::from_secs(300));
        assert_eq!(p.delay(0, s, &mut rng), Duration::from_secs(30));
    }

    #[test]
    fn retry_after_headers() {
        let mut h = Headers::new();
        assert_eq!(retry_after(&h), None);
        h.push("Retry-After", "2");
        assert_eq!(retry_after(&h), Some(Duration::from_secs(2)));
        let mut h = Headers::new();
        h.push("retry-after-ms", "150");
        h.push("retry-after", "9");
        assert_eq!(retry_after(&h), Some(Duration::from_millis(150)));
        let mut h = Headers::new();
        h.push("retry-after", "Wed, 21 Oct 2026 07:28:00 GMT");
        assert_eq!(retry_after(&h), None);
    }

    #[test]
    fn rng_is_uniform_enough() {
        let mut rng = Rng::new();
        let mean: f64 = (0..10_000).map(|_| rng.unit()).sum::<f64>() / 10_000.0;
        assert!((0.45..0.55).contains(&mean), "{mean}");
    }
}
