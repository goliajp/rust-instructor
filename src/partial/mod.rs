//! Incremental parser for JSON that arrives in pieces.
//!
//! [`PartialParser`] reads each byte once and grows a value tree in place,
//! so the work for a whole stream is proportional to its length no matter
//! how many deltas it arrives in. After any [`feed`](PartialParser::feed),
//! [`root`](PartialParser::root) gives a borrowed view of everything parsed
//! so far: unfinished strings hold the characters received, unfinished
//! numbers their digits, and containers the members seen.
//!
//! ```
//! use instructors::partial::PartialParser;
//!
//! let mut p = PartialParser::new();
//! p.feed(r#"{"name": "Ada Lov"#);
//! let root = p.root().unwrap();
//! assert_eq!(root.get("name").unwrap().as_str(), Some("Ada Lov"));
//! assert!(!root.is_complete());
//!
//! p.feed(r#"elace", "born": 1815}"#);
//! let root = p.root().unwrap();
//! assert_eq!(root.get("born").unwrap().as_f64(), Some(1815.0));
//! assert!(p.is_complete());
//! ```
//!
//! Text before the first `{` or `[` is skipped, and so is anything after the
//! root value closes, so a reply wrapped in a markdown fence still parses.

mod de;
mod typed;
mod value;

pub use typed::{Partial, PartialOf, whole};
pub use value::PartialValue;

/// Streaming JSON parser. See the [module docs](self).
#[derive(Default)]
pub struct PartialParser {
    pub(super) nodes: Vec<Node>,
    // open containers, innermost last
    stack: Vec<usize>,
    state: State,
    key: String,
    esc: Esc,
    high_surrogate: Option<u32>,
    literal: &'static [u8],
    literal_at: usize,
    root: Option<usize>,
}

pub(super) struct Node {
    pub(super) kind: NodeKind,
    pub(super) complete: bool,
}

pub(super) enum NodeKind {
    Null,
    Bool(bool),
    Number(String),
    String(String),
    Array(Vec<usize>),
    Object(Vec<(String, usize)>),
}

#[derive(Default, Clone, Copy, PartialEq, Eq)]
enum State {
    #[default]
    Start,
    Value,
    ArrayFirst,
    ObjectKey,
    KeyString,
    Colon,
    AfterValue,
    StrValue(usize),
    Number(usize),
    Literal,
    Done,
    Error,
}

#[derive(Default, Clone, Copy)]
enum Esc {
    #[default]
    None,
    Backslash,
    Unicode {
        acc: u32,
        digits: u8,
    },
}

/// The JSON type of a [`PartialValue`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Kind {
    Null,
    Bool,
    Number,
    String,
    Array,
    Object,
}

impl PartialParser {
    /// An empty parser.
    pub fn new() -> Self {
        Self::default()
    }

    /// Clear all state so the parser can read a new document.
    pub fn reset(&mut self) {
        self.nodes.clear();
        self.stack.clear();
        self.key.clear();
        self.state = State::Start;
        self.esc = Esc::None;
        self.high_surrogate = None;
        self.root = None;
    }

    /// The value parsed so far, or `None` before the root `{` or `[`.
    pub fn root(&self) -> Option<PartialValue<'_>> {
        self.root.map(|idx| PartialValue { p: self, idx })
    }

    /// True once the root value has been closed.
    pub fn is_complete(&self) -> bool {
        self.state == State::Done
    }

    /// True if the input stopped being JSON. Later input is ignored; the
    /// tree keeps what was parsed before the error.
    pub fn is_error(&self) -> bool {
        self.state == State::Error
    }

    /// Parse the next piece of input.
    pub fn feed(&mut self, chunk: &str) {
        let b = chunk.as_bytes();
        let mut i = 0;
        while i < b.len() {
            i = match self.state {
                State::Done | State::Error => return,
                State::Start => match b[i..].iter().position(|&c| c == b'{' || c == b'[') {
                    Some(p) => {
                        self.begin_value(b[i + p]);
                        i + p + 1
                    }
                    None => b.len(),
                },
                State::KeyString | State::StrValue(_) => self.string(chunk, i),
                State::Number(idx) => self.number(chunk, i, idx),
                State::Literal => self.literal(b, i),
                _ => {
                    let c = b[i];
                    if !c.is_ascii_whitespace() {
                        self.structural(c);
                    }
                    i + 1
                }
            };
        }
    }

    fn structural(&mut self, c: u8) {
        match (self.state, c) {
            (State::ArrayFirst, b']') | (State::AfterValue, b']' | b'}') => self.close(c),
            (State::ObjectKey, b'}') => self.close(c),
            (State::ObjectKey, b'"') => {
                self.key.clear();
                self.state = State::KeyString;
            }
            (State::Colon, b':') => self.state = State::Value,
            (State::AfterValue, b',') => {
                self.state = match self.top_is_array() {
                    true => State::Value,
                    false => State::ObjectKey,
                }
            }
            (State::Value | State::ArrayFirst, _) => self.begin_value(c),
            _ => self.state = State::Error,
        }
    }

    fn begin_value(&mut self, c: u8) {
        match c {
            b'{' => {
                let idx = self.attach(NodeKind::Object(Vec::new()), false);
                self.stack.push(idx);
                self.state = State::ObjectKey;
            }
            b'[' => {
                let idx = self.attach(NodeKind::Array(Vec::new()), false);
                self.stack.push(idx);
                self.state = State::ArrayFirst;
            }
            b'"' => {
                let idx = self.attach(NodeKind::String(String::new()), false);
                self.state = State::StrValue(idx);
            }
            b'-' | b'0'..=b'9' => {
                let idx = self.attach(NodeKind::Number((c as char).to_string()), false);
                self.state = State::Number(idx);
            }
            b't' | b'f' | b'n' => {
                self.literal = match c {
                    b't' => b"true",
                    b'f' => b"false",
                    _ => b"null",
                };
                self.literal_at = 1;
                self.state = State::Literal;
            }
            _ => self.state = State::Error,
        }
    }

    fn attach(&mut self, kind: NodeKind, complete: bool) -> usize {
        let idx = self.nodes.len();
        self.nodes.push(Node { kind, complete });
        match self.stack.last() {
            None => self.root = Some(idx),
            Some(&parent) => match &mut self.nodes[parent].kind {
                NodeKind::Array(items) => items.push(idx),
                NodeKind::Object(entries) => entries.push((std::mem::take(&mut self.key), idx)),
                _ => unreachable!("only containers are pushed on the stack"),
            },
        }
        idx
    }

    fn top_is_array(&self) -> bool {
        self.stack
            .last()
            .is_some_and(|&t| matches!(self.nodes[t].kind, NodeKind::Array(_)))
    }

    fn close(&mut self, c: u8) {
        let ok = match c {
            b']' => self.top_is_array(),
            _ => !self.top_is_array(),
        };
        match (ok, self.stack.pop()) {
            (true, Some(idx)) => {
                self.nodes[idx].complete = true;
                self.value_done();
            }
            _ => self.state = State::Error,
        }
    }

    fn value_done(&mut self) {
        self.state = if self.stack.is_empty() {
            State::Done
        } else {
            State::AfterValue
        };
    }

    fn literal(&mut self, b: &[u8], mut i: usize) -> usize {
        while i < b.len() && self.literal_at < self.literal.len() {
            if b[i] != self.literal[self.literal_at] {
                self.state = State::Error;
                return i;
            }
            self.literal_at += 1;
            i += 1;
        }
        if self.literal_at == self.literal.len() {
            let kind = match self.literal[0] {
                b't' => NodeKind::Bool(true),
                b'f' => NodeKind::Bool(false),
                _ => NodeKind::Null,
            };
            self.attach(kind, true);
            self.value_done();
        }
        i
    }

    fn number(&mut self, s: &str, i: usize, idx: usize) -> usize {
        let b = s.as_bytes();
        let end = b[i..]
            .iter()
            .position(|c| !matches!(c, b'0'..=b'9' | b'-' | b'+' | b'.' | b'e' | b'E'))
            .map_or(b.len(), |p| i + p);
        if let NodeKind::Number(n) = &mut self.nodes[idx].kind {
            n.push_str(&s[i..end]);
        }
        if end < b.len() {
            // the terminator is handled by the next state
            self.nodes[idx].complete = true;
            self.value_done();
        }
        end
    }

    fn push_text(&mut self, text: &str) {
        match self.state {
            State::StrValue(idx) => {
                if let NodeKind::String(s) = &mut self.nodes[idx].kind {
                    s.push_str(text);
                }
            }
            _ => self.key.push_str(text),
        }
    }

    fn push_char(&mut self, c: char) {
        let mut buf = [0u8; 4];
        self.push_text(c.encode_utf8(&mut buf));
    }

    fn flush_surrogate(&mut self) {
        if self.high_surrogate.take().is_some() {
            self.push_char(char::REPLACEMENT_CHARACTER);
        }
    }

    fn string(&mut self, s: &str, mut i: usize) -> usize {
        let b = s.as_bytes();
        while i < b.len() {
            match self.esc {
                Esc::None => {
                    let run = b[i..]
                        .iter()
                        .position(|&c| c == b'"' || c == b'\\')
                        .map_or(b.len(), |p| i + p);
                    if run > i {
                        self.flush_surrogate();
                        self.push_text(&s[i..run]);
                    }
                    if run == b.len() {
                        return run;
                    }
                    if b[run] == b'\\' {
                        self.esc = Esc::Backslash;
                        i = run + 1;
                        continue;
                    }
                    self.flush_surrogate();
                    match self.state {
                        State::StrValue(idx) => {
                            self.nodes[idx].complete = true;
                            self.value_done();
                        }
                        _ => self.state = State::Colon,
                    }
                    return run + 1;
                }
                Esc::Backslash => {
                    let c = b[i];
                    i += 1;
                    if c == b'u' {
                        self.esc = Esc::Unicode { acc: 0, digits: 0 };
                        continue;
                    }
                    self.flush_surrogate();
                    self.esc = Esc::None;
                    let out = match c {
                        b'"' => '"',
                        b'\\' => '\\',
                        b'/' => '/',
                        b'b' => '\u{8}',
                        b'f' => '\u{c}',
                        b'n' => '\n',
                        b'r' => '\r',
                        b't' => '\t',
                        _ => {
                            self.state = State::Error;
                            return i;
                        }
                    };
                    self.push_char(out);
                }
                Esc::Unicode { acc, digits } => {
                    let Some(d) = (b[i] as char).to_digit(16) else {
                        self.state = State::Error;
                        return i;
                    };
                    i += 1;
                    let acc = acc * 16 + d;
                    if digits < 3 {
                        self.esc = Esc::Unicode {
                            acc,
                            digits: digits + 1,
                        };
                        continue;
                    }
                    self.esc = Esc::None;
                    self.unicode(acc);
                }
            }
        }
        i
    }

    fn unicode(&mut self, code: u32) {
        match code {
            0xD800..=0xDBFF => {
                self.flush_surrogate();
                self.high_surrogate = Some(code);
            }
            0xDC00..=0xDFFF => match self.high_surrogate.take() {
                Some(high) => {
                    let c = 0x10000 + ((high - 0xD800) << 10) + (code - 0xDC00);
                    self.push_char(char::from_u32(c).unwrap_or(char::REPLACEMENT_CHARACTER));
                }
                None => self.push_char(char::REPLACEMENT_CHARACTER),
            },
            _ => {
                self.flush_surrogate();
                self.push_char(char::from_u32(code).unwrap_or(char::REPLACEMENT_CHARACTER));
            }
        }
    }
}

#[cfg(test)]
mod tests;
