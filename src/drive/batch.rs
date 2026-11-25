//! Bounded concurrency without an async runtime.

use std::future::poll_fn;
use std::task::Poll;

use crate::transport::BoxFuture;

/// Run `futs` with at most `limit` in flight and return every output in
/// input order.
///
/// Every in-flight future is polled on each wake-up: O(limit) per wake,
/// and `limit` is small.
pub(crate) async fn ordered<'a, T>(futs: Vec<BoxFuture<'a, T>>, limit: usize) -> Vec<T> {
    let limit = limit.max(1);
    let mut out: Vec<Option<T>> = futs.iter().map(|_| None).collect();
    let mut queue = futs.into_iter().enumerate();
    let mut flight: Vec<(usize, BoxFuture<'a, T>)> = Vec::with_capacity(limit);
    poll_fn(|cx| {
        loop {
            while flight.len() < limit {
                match queue.next() {
                    Some(f) => flight.push(f),
                    None => break,
                }
            }
            if flight.is_empty() {
                return Poll::Ready(());
            }
            let before = flight.len();
            let mut i = 0;
            while i < flight.len() {
                match flight[i].1.as_mut().poll(cx) {
                    Poll::Ready(v) => {
                        out[flight[i].0] = Some(v);
                        drop(flight.swap_remove(i));
                    }
                    Poll::Pending => i += 1,
                }
            }
            // started futures need a first poll; otherwise wait for a wake-up
            if flight.len() == before {
                return Poll::Pending;
            }
        }
    })
    .await;
    out.into_iter().flatten().collect()
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::future::Future;
    use std::pin::Pin;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Condvar, Mutex};
    use std::task::{Context, Wake, Waker};

    struct Flag(Mutex<bool>, Condvar);
    impl Wake for Flag {
        fn wake(self: Arc<Self>) {
            *self.0.lock().unwrap() = true;
            self.1.notify_one();
        }
    }

    /// A minimal executor for tests: polls on the calling thread.
    pub(crate) fn block_on<F: Future>(f: F) -> F::Output {
        let flag = Arc::new(Flag(Mutex::new(false), Condvar::new()));
        let waker = Waker::from(flag.clone());
        let mut cx = Context::from_waker(&waker);
        let mut f = std::pin::pin!(f);
        loop {
            if let Poll::Ready(v) = f.as_mut().poll(&mut cx) {
                return v;
            }
            let mut woken = flag.0.lock().unwrap();
            while !*woken {
                woken = flag.1.wait(woken).unwrap();
            }
            *woken = false;
        }
    }

    // ready after `polls` polls, waking itself each time
    struct After {
        polls: usize,
        value: usize,
        live: Arc<AtomicUsize>,
        peak: Arc<AtomicUsize>,
        started: bool,
    }

    impl Future for After {
        type Output = usize;
        fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<usize> {
            if !self.started {
                self.started = true;
                let now = self.live.fetch_add(1, Ordering::SeqCst) + 1;
                self.peak.fetch_max(now, Ordering::SeqCst);
            }
            if self.polls == 0 {
                self.live.fetch_sub(1, Ordering::SeqCst);
                return Poll::Ready(self.value);
            }
            self.polls -= 1;
            cx.waker().wake_by_ref();
            Poll::Pending
        }
    }

    #[test]
    fn results_keep_input_order_and_the_limit_holds() {
        let peak = Arc::new(AtomicUsize::new(0));
        let live = Arc::new(AtomicUsize::new(0));
        let delays = [9, 1, 5, 0, 7, 3, 2, 8, 4, 6];
        let futs: Vec<BoxFuture<'static, usize>> = delays
            .iter()
            .enumerate()
            .map(|(i, &d)| {
                Box::pin(After {
                    polls: d,
                    value: i,
                    live: live.clone(),
                    peak: peak.clone(),
                    started: false,
                }) as BoxFuture<'static, usize>
            })
            .collect();
        assert_eq!(block_on(ordered(futs, 3)), (0..10).collect::<Vec<_>>());
        assert_eq!(peak.load(Ordering::SeqCst), 3);
    }

    #[test]
    fn empty_input() {
        let futs: Vec<BoxFuture<'static, usize>> = Vec::new();
        assert!(block_on(ordered(futs, 4)).is_empty());
    }
}
