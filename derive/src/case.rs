//! `rename_all` rules, matching serde_derive's conversions exactly.

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub(crate) enum Rule {
    None,
    Lower,
    Upper,
    Pascal,
    Camel,
    Snake,
    ScreamingSnake,
    Kebab,
    ScreamingKebab,
}

const RULES: &[(&str, Rule)] = &[
    ("lowercase", Rule::Lower),
    ("UPPERCASE", Rule::Upper),
    ("PascalCase", Rule::Pascal),
    ("camelCase", Rule::Camel),
    ("snake_case", Rule::Snake),
    ("SCREAMING_SNAKE_CASE", Rule::ScreamingSnake),
    ("kebab-case", Rule::Kebab),
    ("SCREAMING-KEBAB-CASE", Rule::ScreamingKebab),
];

impl Rule {
    pub(crate) fn parse(s: &str) -> Option<Self> {
        RULES.iter().find(|(n, _)| *n == s).map(|(_, r)| *r)
    }

    pub(crate) fn names() -> String {
        RULES
            .iter()
            .map(|(n, _)| format!("{n:?}"))
            .collect::<Vec<_>>()
            .join(", ")
    }

    pub(crate) fn or(self, other: Self) -> Self {
        if self == Self::None { other } else { self }
    }

    pub(crate) fn variant(self, v: &str) -> String {
        match self {
            Self::None | Self::Pascal => v.to_owned(),
            Self::Lower => v.to_ascii_lowercase(),
            Self::Upper => v.to_ascii_uppercase(),
            Self::Camel => v[..1].to_ascii_lowercase() + &v[1..],
            Self::Snake => {
                let mut snake = String::new();
                for (i, ch) in v.char_indices() {
                    if i > 0 && ch.is_uppercase() {
                        snake.push('_');
                    }
                    snake.push(ch.to_ascii_lowercase());
                }
                snake
            }
            Self::ScreamingSnake => Self::Snake.variant(v).to_ascii_uppercase(),
            Self::Kebab => Self::Snake.variant(v).replace('_', "-"),
            Self::ScreamingKebab => Self::ScreamingSnake.variant(v).replace('_', "-"),
        }
    }

    pub(crate) fn field(self, f: &str) -> String {
        match self {
            Self::None | Self::Lower | Self::Snake => f.to_owned(),
            Self::Upper | Self::ScreamingSnake => f.to_ascii_uppercase(),
            Self::Pascal => {
                let mut pascal = String::new();
                let mut capitalize = true;
                for ch in f.chars() {
                    if ch == '_' {
                        capitalize = true;
                    } else if capitalize {
                        pascal.push(ch.to_ascii_uppercase());
                        capitalize = false;
                    } else {
                        pascal.push(ch);
                    }
                }
                pascal
            }
            Self::Camel => {
                let pascal = Self::Pascal.field(f);
                pascal[..1].to_ascii_lowercase() + &pascal[1..]
            }
            Self::Kebab => f.replace('_', "-"),
            Self::ScreamingKebab => Self::ScreamingSnake.field(f).replace('_', "-"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Rule::*;

    #[test]
    fn variants_match_serde() {
        for &(original, lower, upper, camel, snake, screaming, kebab, sk) in &[
            (
                "Outcome", "outcome", "OUTCOME", "outcome", "outcome", "OUTCOME", "outcome",
                "OUTCOME",
            ),
            (
                "VeryTasty",
                "verytasty",
                "VERYTASTY",
                "veryTasty",
                "very_tasty",
                "VERY_TASTY",
                "very-tasty",
                "VERY-TASTY",
            ),
            ("A", "a", "A", "a", "a", "A", "a", "A"),
            ("Z42", "z42", "Z42", "z42", "z42", "Z42", "z42", "Z42"),
        ] {
            assert_eq!(None.variant(original), original);
            assert_eq!(Lower.variant(original), lower);
            assert_eq!(Upper.variant(original), upper);
            assert_eq!(Pascal.variant(original), original);
            assert_eq!(Camel.variant(original), camel);
            assert_eq!(Snake.variant(original), snake);
            assert_eq!(ScreamingSnake.variant(original), screaming);
            assert_eq!(Kebab.variant(original), kebab);
            assert_eq!(ScreamingKebab.variant(original), sk);
        }
    }

    #[test]
    fn fields_match_serde() {
        for &(original, upper, pascal, camel, screaming, kebab, sk) in &[
            (
                "outcome", "OUTCOME", "Outcome", "outcome", "OUTCOME", "outcome", "OUTCOME",
            ),
            (
                "very_tasty",
                "VERY_TASTY",
                "VeryTasty",
                "veryTasty",
                "VERY_TASTY",
                "very-tasty",
                "VERY-TASTY",
            ),
            ("a", "A", "A", "a", "A", "a", "A"),
            ("z42", "Z42", "Z42", "z42", "Z42", "z42", "Z42"),
        ] {
            assert_eq!(None.field(original), original);
            assert_eq!(Upper.field(original), upper);
            assert_eq!(Pascal.field(original), pascal);
            assert_eq!(Camel.field(original), camel);
            assert_eq!(Snake.field(original), original);
            assert_eq!(ScreamingSnake.field(original), screaming);
            assert_eq!(Kebab.field(original), kebab);
            assert_eq!(ScreamingKebab.field(original), sk);
        }
    }
}
