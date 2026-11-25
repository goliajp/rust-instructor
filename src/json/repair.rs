//! Turn what a model wrote into JSON, in one pass that copies only when it
//! meets a defect.

use std::borrow::Cow;

mod scalar;

/// The result of [`repair`].
#[derive(Debug, PartialEq, Eq)]
pub struct Repaired<'a> {
    /// The input when it needed no change, else the repaired text.
    pub text: Cow<'a, str>,
    /// The input ended inside a value; open strings and containers were
    /// closed and an unfinished key or number was dropped.
    pub cut: bool,
}

const MAX_DEPTH: usize = 128;

/// Turn what a model wrote into JSON: strip markdown fences and prose around
/// the value, fix trailing commas, single quotes, unquoted keys, comments,
/// Python literals, unescaped quotes and newlines inside strings, and close
/// a value the input stopped in the middle of. Returns `None` when no JSON
/// value can be found.
pub(crate) fn repair(s: &str) -> Option<Repaired<'_>> {
    if super::is_valid(s) {
        return Some(Repaired {
            text: Cow::Borrowed(s),
            cut: false,
        });
    }
    let b = s.as_bytes();
    let first = b.iter().position(|c| !c.is_ascii_whitespace())?;
    let start = match b[first] {
        b'{' | b'[' => first,
        _ => b
            .iter()
            .position(|&c| c == b'{' || c == b'[')
            .unwrap_or(first),
    };
    // prose or a fence around an otherwise valid value
    if start > 0
        && let Some(close) = match b[start] {
            b'{' => Some(b'}'),
            b'[' => Some(b']'),
            _ => None,
        }
        && let Some(end) = b.iter().rposition(|&c| c == close)
        && end > start
        && super::is_valid(&s[start..=end])
    {
        return Some(Repaired {
            text: Cow::Borrowed(&s[start..=end]),
            cut: false,
        });
    }
    let mut r = Repairer {
        s,
        b,
        i: start,
        out: Out {
            src: s,
            buf: None,
            mark: start,
        },
        cut: false,
    };
    // prose before the value
    if start > 0 {
        r.out.buf = Some(String::with_capacity(s.len()));
    }
    let outcome = r.value(0, true)?;
    if outcome == Outcome::Dropped {
        return None;
    }
    let end = r.i;
    let tail = &b[end..];
    if tail.iter().all(u8::is_ascii_whitespace) {
        r.out.keep(b.len());
    } else {
        r.out.skip(b.len());
    }
    let cut = r.cut;
    let text = r.out.finish();
    // a defect this pass does not know can leave invalid output
    super::is_valid(&text).then_some(Repaired { text, cut })
}

/// Output that stays a borrow of the input until something differs.
struct Out<'a> {
    src: &'a str,
    buf: Option<String>,
    // input before `mark` is accounted for
    mark: usize,
}

#[derive(Clone, Copy)]
struct Checkpoint {
    mark: usize,
    len: Option<usize>,
}

impl<'a> Out<'a> {
    fn keep(&mut self, to: usize) {
        if let Some(buf) = &mut self.buf {
            buf.push_str(&self.src[self.mark..to]);
        }
        self.mark = to;
    }

    fn own(&mut self) -> &mut String {
        let (src, mark) = (self.src, self.mark);
        self.buf.get_or_insert_with(|| {
            let mut s = String::with_capacity(src.len() + 16);
            s.push_str(&src[..mark]);
            s
        })
    }

    fn skip(&mut self, to: usize) {
        self.own();
        self.mark = to;
    }

    fn push(&mut self, text: &str) {
        self.own().push_str(text);
    }

    fn checkpoint(&self) -> Checkpoint {
        Checkpoint {
            mark: self.mark,
            len: self.buf.as_ref().map(String::len),
        }
    }

    /// Forget output written since `cp`; input up to `to` is consumed.
    fn rollback(&mut self, cp: Checkpoint, to: usize) {
        match (cp.len, &mut self.buf) {
            (Some(len), Some(buf)) => buf.truncate(len),
            _ => self.buf = Some(self.src[..cp.mark].to_string()),
        }
        self.mark = to;
    }

    fn finish(self) -> Cow<'a, str> {
        match self.buf {
            Some(s) => Cow::Owned(s),
            None => Cow::Borrowed(&self.src[..self.mark]),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Outcome {
    Done,
    /// The input ended inside this value, which was closed.
    Cut,
    /// The input ended before this value was usable; nothing was written.
    Dropped,
}

struct Repairer<'a> {
    s: &'a str,
    b: &'a [u8],
    i: usize,
    out: Out<'a>,
    cut: bool,
}

impl Repairer<'_> {
    fn peek(&self) -> Option<u8> {
        self.b.get(self.i).copied()
    }

    /// Whitespace and comments; comments are dropped.
    fn ws(&mut self) -> Option<u8> {
        loop {
            while let Some(c) = self.peek() {
                if c.is_ascii_whitespace() {
                    self.i += 1;
                } else {
                    break;
                }
            }
            match (self.peek(), self.b.get(self.i + 1)) {
                (Some(b'/'), Some(b'/')) => {
                    self.out.keep(self.i);
                    let end = self.b[self.i..]
                        .iter()
                        .position(|&c| c == b'\n')
                        .map_or(self.b.len(), |p| self.i + p);
                    self.i = end;
                    self.out.skip(end);
                }
                (Some(b'/'), Some(b'*')) => {
                    self.out.keep(self.i);
                    let end = self.s[self.i + 2..]
                        .find("*/")
                        .map_or(self.b.len(), |p| self.i + 2 + p + 2);
                    self.i = end;
                    self.out.skip(end);
                }
                (c, _) => {
                    self.out.keep(self.i);
                    return c;
                }
            }
        }
    }

    fn value(&mut self, depth: usize, root: bool) -> Option<Outcome> {
        if depth > MAX_DEPTH {
            return None;
        }
        let Some(c) = self.ws() else {
            self.cut = true;
            return Some(Outcome::Dropped);
        };
        Some(match c {
            b'{' => self.container(depth, b'}')?,
            b'[' => self.container(depth, b']')?,
            b'"' | b'\'' => self.string(c),
            b'-' | b'+' | b'.' | b'0'..=b'9' => self.number(root),
            c if c.is_ascii_alphabetic() || c == b'_' => self.word(root),
            _ => return None,
        })
    }

    /// An object (`close == b'}'`) or an array.
    fn container(&mut self, depth: usize, close: u8) -> Option<Outcome> {
        let object = close == b'}';
        self.i += 1;
        self.out.keep(self.i);
        // output and input position just after the last complete member
        let mut after_member = self.out.checkpoint();
        let mut empty = true;
        let mut comma = false;
        loop {
            let Some(c) = self.ws() else {
                self.cut = true;
                self.out.rollback(after_member, self.i);
                self.out.push(if object { "}" } else { "]" });
                return Some(Outcome::Cut);
            };
            match c {
                b'}' | b']' => {
                    if comma {
                        // trailing comma
                        self.out.rollback(after_member, self.i);
                    }
                    if c == close {
                        self.i += 1;
                        self.out.keep(self.i);
                    } else {
                        // the closer of an outer container: close this one
                        // and leave it for the outer one
                        self.out.push(if object { "}" } else { "]" });
                    }
                    return Some(Outcome::Done);
                }
                b',' => {
                    self.i += 1;
                    if comma || empty {
                        // a doubled or leading comma
                        self.out.skip(self.i);
                    } else {
                        self.out.keep(self.i);
                        comma = true;
                    }
                    continue;
                }
                _ => {}
            }
            if !empty && !comma {
                self.out.push(",");
            }
            if object {
                match self.key() {
                    Some(true) => {}
                    Some(false) => {
                        self.cut = true;
                        self.out.rollback(after_member, self.i);
                        self.out.push("}");
                        return Some(Outcome::Cut);
                    }
                    None => return None,
                }
                match self.ws() {
                    Some(b':') => {
                        self.i += 1;
                        self.out.keep(self.i);
                    }
                    None => {
                        self.cut = true;
                        self.out.rollback(after_member, self.i);
                        self.out.push("}");
                        return Some(Outcome::Cut);
                    }
                    Some(_) => self.out.push(":"),
                }
            }
            match self.value(depth + 1, false)? {
                Outcome::Done => {}
                Outcome::Cut => {
                    self.out.push(if object { "}" } else { "]" });
                    return Some(Outcome::Cut);
                }
                Outcome::Dropped => {
                    self.out.rollback(after_member, self.i);
                    self.out.push(if object { "}" } else { "]" });
                    return Some(Outcome::Cut);
                }
            }
            after_member = self.out.checkpoint();
            empty = false;
            comma = false;
        }
    }

    /// An object key. `Some(false)` when the input ended inside it.
    fn key(&mut self) -> Option<bool> {
        match self.peek()? {
            q @ (b'"' | b'\'') => Some(self.string(q) == Outcome::Done),
            c if c.is_ascii_alphanumeric() || c == b'_' || c == b'$' => {
                while let Some(c) = self.peek() {
                    if c.is_ascii_alphanumeric() || c == b'_' || c == b'$' || c == b'-' {
                        self.i += 1;
                    } else {
                        break;
                    }
                }
                if self.i == self.b.len() {
                    return Some(false);
                }
                self.out.push("\"");
                self.out.keep(self.i);
                self.out.push("\"");
                Some(true)
            }
            _ => None,
        }
    }
}
