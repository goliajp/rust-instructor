//! Responses whose body arrives in pieces.

use super::{Exchange, Hooks, State};
use crate::partial::PartialParser;
use crate::sse::SseDecoder;
use crate::wire::{Headers, Stream};
use serde::de::DeserializeOwned;

pub(super) struct Inbound {
    pub(super) status: u16,
    pub(super) headers: Headers,
    pub(super) buf: Vec<u8>,
    pub(super) sse: Option<Sse>,
}

pub(super) struct Sse {
    pub(super) decoder: SseDecoder,
    pub(super) acc: Stream,
    pub(super) parser: PartialParser,
    pub(super) delivered: usize,
}

impl<O: DeserializeOwned> Exchange<O> {
    /// The status and headers of a response whose body follows in chunks.
    pub fn receive_head(&mut self, status: u16, headers: &Headers) {
        let ok = (200..300).contains(&status);
        let sse = (ok && self.spec.hooks.streaming()).then(|| Sse {
            decoder: SseDecoder::default(),
            acc: self.spec.targets[self.idx].provider.stream(&self.req()),
            parser: PartialParser::new(),
            delivered: 0,
        });
        self.inbound = Some(Inbound {
            status,
            headers: headers.clone(),
            buf: Vec::new(),
            sse,
        });
    }

    /// The next piece of the body.
    pub fn receive_chunk(&mut self, bytes: &[u8]) {
        let Some(inb) = &mut self.inbound else { return };
        let Some(sse) = &mut inb.sse else {
            inb.buf.extend_from_slice(bytes);
            return;
        };
        let hooks = &self.spec.hooks;
        let list = self.spec.list;
        let Sse {
            decoder,
            acc,
            parser,
            delivered,
        } = sse;
        decoder.feed(bytes, |data| {
            acc.event(data, &mut |delta| {
                on_delta(hooks, list, parser, delivered, delta)
            });
        });
    }

    /// The body is complete.
    pub fn receive_end(&mut self) {
        let Some(inb) = self.inbound.take() else {
            return;
        };
        self.state = State::Ready;
        let status = inb.status;
        if !(200..300).contains(&status) {
            return self.http_error(status, &inb.headers, &inb.buf);
        }
        let answer = match inb.sse {
            Some(mut sse) => {
                let hooks = &self.spec.hooks;
                let list = self.spec.list;
                let Sse {
                    decoder,
                    acc,
                    parser,
                    delivered,
                } = &mut sse;
                decoder.finish(|data| {
                    acc.event(data, &mut |delta| {
                        on_delta(hooks, list, parser, delivered, delta)
                    });
                });
                sse.acc.finish()
            }
            None => self.spec.targets[self.idx]
                .provider
                .response(&self.req(), &inb.buf),
        };
        self.answer(answer, status);
    }
}

fn on_delta(
    hooks: &Hooks,
    list: bool,
    parser: &mut PartialParser,
    delivered: &mut usize,
    delta: &str,
) {
    if let Some(h) = &hooks.on_delta {
        h(delta);
    }
    if hooks.on_partial.is_none() && hooks.on_item.is_none() {
        return;
    }
    parser.feed(delta);
    let Some(root) = parser.root() else { return };
    if let Some(h) = &hooks.on_partial {
        let view = if list { root.get("items") } else { Some(root) };
        if let Some(v) = view {
            h(v);
        }
    }
    if let Some(h) = &hooks.on_item {
        let Some(items) = root.get("items").or(Some(root)) else {
            return;
        };
        for item in items.items().skip(*delivered) {
            if !item.is_complete() {
                break;
            }
            h(&item.to_json());
            *delivered += 1;
        }
    }
}
