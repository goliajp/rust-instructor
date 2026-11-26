//! An answer that arrived: parse, validate, re-ask or finish.

use serde::de::DeserializeOwned;

use super::{Error, Exchange, parse};
use crate::message::Message;
use crate::wire::{Answer, Fail};

const REASK: &str = "Reply again with only JSON that matches the schema.";

impl<O: DeserializeOwned> Exchange<O> {
    pub(super) fn answer(&mut self, answer: std::result::Result<Answer, Fail>, status: u16) {
        let answer = match answer {
            Ok(a) => a,
            Err(f) => return self.fail_with(f, status),
        };
        let model = self.spec.targets[self.idx].model.clone();
        self.usage.add(&answer.usage, &model);
        let schema = self.spec.schema;
        let unwire = |t: &str| schema.unwire(t);
        let strict = self.spec.targets[self.idx].provider.unwire(&self.req());
        let parsed = parse::parse::<O>(
            &answer.content,
            self.spec.list,
            &answer.stop,
            self.spec.accept_truncated,
            strict.then_some(&unwire as &dyn Fn(&str) -> _),
        );
        let (problem, validation) = match parsed {
            Ok((value, truncated)) => match self.spec.validator.as_ref().map(|v| v(&value)) {
                Some(Err(e)) => (e.message().to_owned(), true),
                _ => return self.succeed(value, truncated),
            },
            Err(parse::Bad::Truncated) => {
                let reason = match &answer.stop {
                    crate::wire::Stop::Truncated(r) => r.clone(),
                    crate::wire::Stop::Done => String::new(),
                };
                return self.fail(Error::Truncated {
                    provider: self.name(),
                    reason,
                    raw: answer.content,
                    usage: Box::default(),
                });
            }
            Err(parse::Bad::Invalid(m)) => (m, false),
        };
        self.attempt += 1;
        if self.attempt < self.spec.max_attempts.max(1) {
            // the model sees what it wrote and why it was rejected
            let raw = if answer.content.is_empty() {
                "(empty answer)".to_owned()
            } else {
                answer.content
            };
            self.spec.messages.push(Message::assistant(raw));
            let what = if validation {
                "failed validation"
            } else {
                "is not valid JSON for the schema"
            };
            self.spec.messages.push(Message::user(format!(
                "That answer {what}: {problem}\n{REASK}"
            )));
            return;
        }
        let attempts = self.attempt;
        let provider = self.name();
        self.fail(if validation {
            Error::Validation {
                provider,
                attempts,
                message: problem,
                usage: Box::default(),
            }
        } else {
            Error::Parse {
                provider,
                attempts,
                message: problem,
                raw: answer.content,
                usage: Box::default(),
            }
        });
    }
}
