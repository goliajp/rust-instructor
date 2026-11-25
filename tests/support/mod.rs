// A loopback HTTP server that records request bodies and writes each reply
// in separate chunks with a pause in between, so the client sees one network
// read per chunk.
// each test crate uses a different subset of these helpers
#![allow(dead_code)]

use std::sync::{Arc, Mutex};
use std::time::Duration;

use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::net::TcpListener;

pub struct Reply {
    pub path_contains: &'static str,
    pub status: u16,
    pub content_type: &'static str,
    pub chunks: Vec<Vec<u8>>,
}

impl Reply {
    pub fn json(path_contains: &'static str, status: u16, body: serde_json::Value) -> Self {
        Self {
            path_contains,
            status,
            content_type: "application/json",
            chunks: vec![serde_json::to_vec(&body).unwrap()],
        }
    }

    pub fn sse(path_contains: &'static str, chunks: Vec<Vec<u8>>) -> Self {
        Self {
            path_contains,
            status: 200,
            content_type: "text/event-stream",
            chunks,
        }
    }
}

#[derive(Clone, Default)]
pub struct Log(Arc<Mutex<Vec<(String, serde_json::Value)>>>);

impl Log {
    /// (request target, parsed body) for every request so far.
    pub fn requests(&self) -> Vec<(String, serde_json::Value)> {
        self.0.lock().unwrap().clone()
    }

    pub fn body(&self, i: usize) -> serde_json::Value {
        self.requests()[i].1.clone()
    }
}

pub async fn start(replies: Vec<Reply>) -> (String, Log) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let log = Log::default();
    let replies = Arc::new(replies);
    let log2 = log.clone();
    tokio::spawn(async move {
        loop {
            let (sock, _) = listener.accept().await.unwrap();
            tokio::spawn(handle(sock, log2.clone(), replies.clone()));
        }
    });
    (base, log)
}

async fn handle(sock: tokio::net::TcpStream, log: Log, replies: Arc<Vec<Reply>>) {
    let (r, mut w) = sock.into_split();
    let mut r = BufReader::new(r);
    let mut line = String::new();
    loop {
        line.clear();
        if r.read_line(&mut line).await.unwrap_or(0) == 0 {
            return;
        }
        let target = line.split_whitespace().nth(1).unwrap_or("").to_string();
        let mut len = 0usize;
        loop {
            line.clear();
            r.read_line(&mut line).await.unwrap();
            if line == "\r\n" {
                break;
            }
            if let Some(v) = line.to_ascii_lowercase().strip_prefix("content-length:") {
                len = v.trim().parse().unwrap();
            }
        }
        let mut buf = vec![0u8; len];
        r.read_exact(&mut buf).await.unwrap();
        let body = serde_json::from_slice(&buf).unwrap_or(serde_json::Value::Null);
        log.0.lock().unwrap().push((target.clone(), body));
        let reply = replies
            .iter()
            .find(|rep| target.contains(rep.path_contains))
            .expect("no reply for path");
        let total: usize = reply.chunks.iter().map(Vec::len).sum();
        let head = format!(
            "HTTP/1.1 {} X\r\ncontent-type: {}\r\ncontent-length: {total}\r\n\r\n",
            reply.status, reply.content_type
        );
        w.write_all(head.as_bytes()).await.unwrap();
        for c in &reply.chunks {
            w.write_all(c).await.unwrap();
            w.flush().await.unwrap();
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    }
}

pub fn openai_ok(content: &str) -> serde_json::Value {
    serde_json::json!({
        "choices": [{"message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5}
    })
}

pub fn anthropic_text(text: &str) -> serde_json::Value {
    serde_json::json!({
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 10, "output_tokens": 5}
    })
}

pub fn gemini_ok(text: &str) -> serde_json::Value {
    serde_json::json!({
        "candidates": [{"content": {"parts": [{"text": text}]}, "finishReason": "STOP"}],
        "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5}
    })
}
