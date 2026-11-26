use std::hash::{BuildHasher, RandomState};
use std::time::Duration;

use crate::wire::Headers;

/// When to resend a request that failed at the HTTP level: 408, 409, 429,
/// 500, 502, 503, 504, 529 and connection errors. On by default. An
/// `x-should-retry: true` or `false` response header overrides the status.
///
/// The delay is `base_delay * 2^n`, capped at `max_delay`, times a random
/// factor in `[0.5, 1)` when `jitter` is on. A `retry-after-ms` or
/// `Retry-After` header from the provider replaces the computed delay, up to
/// `max_delay`. A `Retry-After` date is counted from the response's `Date`
/// header, so the client's clock is never read.
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

const RETRYABLE: [u16; 8] = [408, 409, 429, 500, 502, 503, 504, 529];

/// Whether a response with this status and these headers may be resent.
pub(crate) fn retryable(status: u16, h: &Headers) -> bool {
    match h.get("x-should-retry").map(str::trim) {
        Some(v) if v.eq_ignore_ascii_case("true") => true,
        Some(v) if v.eq_ignore_ascii_case("false") => false,
        _ => RETRYABLE.contains(&status),
    }
}

/// `retry-after-ms`, or `Retry-After` in seconds or as an HTTP date.
pub(crate) fn retry_after(h: &Headers) -> Option<Duration> {
    if let Some(ms) = h
        .get("retry-after-ms")
        .and_then(|v| v.trim().parse::<f64>().ok())
        && ms.is_finite()
        && ms >= 0.0
    {
        return Some(Duration::from_secs_f64(ms / 1000.0));
    }
    let v = h.get("retry-after")?.trim();
    if let Ok(s) = v.parse::<f64>() {
        return (s.is_finite() && s >= 0.0).then(|| Duration::from_secs_f64(s));
    }
    let at = http_date(v)?;
    let now = http_date(h.get("date")?.trim())?;
    Some(Duration::from_secs(at.saturating_sub(now)))
}

/// Seconds since the Unix epoch of an RFC 9110 IMF-fixdate such as
/// `Sun, 06 Nov 1994 08:49:37 GMT`.
pub(crate) fn http_date(s: &str) -> Option<u64> {
    const DAYS: [&str; 7] = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
    const MONTHS: [&str; 12] = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ];
    let b = s.as_bytes();
    if b.len() != 29
        || &b[3..5] != b", "
        || b[7] != b' '
        || b[11] != b' '
        || b[16] != b' '
        || b[19] != b':'
        || b[22] != b':'
        || &b[25..] != b" GMT"
    {
        return None;
    }
    let num = |r: std::ops::Range<usize>| -> Option<u64> {
        let t = s.get(r)?;
        t.bytes()
            .all(|c| c.is_ascii_digit())
            .then(|| t.parse().ok())?
    };
    if !DAYS.contains(&s.get(0..3)?) {
        return None;
    }
    let day = num(5..7)?;
    let month = MONTHS.iter().position(|m| Some(*m) == s.get(8..11))? as u64 + 1;
    let year = num(12..16)?;
    let (hh, mm, ss) = (num(17..19)?, num(20..22)?, num(23..25)?);
    if !(1..=31).contains(&day) || hh > 23 || mm > 59 || ss > 60 || year < 1970 {
        return None;
    }
    // days from 1970-01-01 to the date, proleptic Gregorian
    let (y, m) = if month <= 2 {
        (year - 1, month + 9)
    } else {
        (year, month - 3)
    };
    let era = y / 400;
    let yoe = y - era * 400;
    let doy = (153 * m + 2) / 5 + day - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let days = (era * 146_097 + doe).checked_sub(719_468)?;
    Some(days * 86_400 + hh * 3600 + mm * 60 + ss)
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
        assert_eq!(retry_after(&h), None, "no Date header to count from");
        h.push("Date", "Wed, 21 Oct 2026 07:27:45 GMT");
        assert_eq!(retry_after(&h), Some(Duration::from_secs(15)));
        let mut h = Headers::new();
        h.push("retry-after", "Wed, 21 Oct 2026 07:27:00 GMT");
        h.push("date", "Wed, 21 Oct 2026 07:28:00 GMT");
        assert_eq!(retry_after(&h), Some(Duration::ZERO), "a past date");
    }

    #[test]
    fn imf_fixdate() {
        // the examples in RFC 9110 and well-known instants
        assert_eq!(
            http_date("Sun, 06 Nov 1994 08:49:37 GMT"),
            Some(784_111_777)
        );
        assert_eq!(http_date("Thu, 01 Jan 1970 00:00:00 GMT"), Some(0));
        assert_eq!(
            http_date("Tue, 29 Feb 2000 12:00:00 GMT"),
            Some(951_825_600)
        );
        assert_eq!(
            http_date("Thu, 26 Nov 2026 05:21:35 GMT"),
            Some(1_795_670_495)
        );
        for bad in [
            "",
            "Sunday, 06-Nov-94 08:49:37 GMT",
            "Sun Nov  6 08:49:37 1994",
            "Sun, 06 Nov 1994 08:49:37 UTC",
            "Sun, 06 Foo 1994 08:49:37 GMT",
            "Xyz, 06 Nov 1994 08:49:37 GMT",
            "Sun, 32 Nov 1994 08:49:37 GMT",
            "Sun, 06 Nov 1994 24:49:37 GMT",
            "Sun, 06 Nov 1969 08:49:37 GMT",
            "Sun, +6 Nov 1994 08:49:37 GMT",
            "Sun, 06 Nov 1994 08:49:37 GMT ",
            "Sun, 06 Nov 1994 08:4é:37 GM",
        ] {
            assert_eq!(http_date(bad), None, "{bad:?}");
        }
    }

    #[test]
    fn should_retry_header_overrides_the_status() {
        let mut h = Headers::new();
        assert!(retryable(409, &h));
        assert!(retryable(529, &h));
        assert!(!retryable(400, &h));
        h.push("x-should-retry", "true");
        assert!(retryable(400, &h));
        let mut h = Headers::new();
        h.push("X-Should-Retry", "false");
        assert!(!retryable(503, &h));
        let mut h = Headers::new();
        h.push("x-should-retry", "maybe");
        assert!(retryable(503, &h));
    }

    #[test]
    fn rng_is_uniform_enough() {
        let mut rng = Rng::new();
        let mean: f64 = (0..10_000).map(|_| rng.unit()).sum::<f64>() / 10_000.0;
        assert!((0.45..0.55).contains(&mean), "{mean}");
    }
}
