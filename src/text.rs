/// At most `max` bytes of `s`, cut at a character boundary.
pub(crate) fn clip(s: &str, max: usize) -> &str {
    &s[..s.floor_char_boundary(max.min(s.len()))]
}

#[cfg(test)]
mod tests {
    use super::clip;

    #[test]
    fn clip_respects_char_boundaries() {
        assert_eq!(clip("hello", 10), "hello");
        assert_eq!(clip("hello", 0), "");
        assert_eq!(clip("你好世界", 4), "你");
    }
}
