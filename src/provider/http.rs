use std::sync::OnceLock;

use futures::StreamExt;
use serde::Serialize;
use serde::de::DeserializeOwned;

use crate::error::{Error, Result};
use crate::sse::SseDecoder;

/// One connection pool shared by every `Client` in the process.
pub(crate) fn shared_client() -> reqwest::Client {
    static CLIENT: OnceLock<reqwest::Client> = OnceLock::new();
    CLIENT.get_or_init(reqwest::Client::new).clone()
}

/// POST a JSON body; a non-2xx status becomes [`Error::Api`] with the body.
pub(crate) async fn post(
    req: reqwest::RequestBuilder,
    body: &impl Serialize,
) -> Result<reqwest::Response> {
    let bytes = serde_json::to_vec(body)?;
    let resp = req
        .header("content-type", "application/json")
        .body(bytes)
        .send()
        .await?;
    let status = resp.status();
    if !status.is_success() {
        let text = resp.text().await.unwrap_or_default();
        return Err(Error::Api {
            status: status.as_u16(),
            message: text,
        });
    }
    Ok(resp)
}

/// Read and decode a whole JSON response body.
pub(crate) async fn json<T: DeserializeOwned>(resp: reqwest::Response) -> Result<T> {
    let bytes = resp.bytes().await?;
    Ok(serde_json::from_slice(&bytes)?)
}

/// Read an SSE body, handing each `data:` payload to `on_data`.
pub(crate) async fn read_sse(resp: reqwest::Response, mut on_data: impl FnMut(&str)) -> Result<()> {
    let mut decoder = SseDecoder::default();
    let mut stream = resp.bytes_stream();
    while let Some(chunk) = stream.next().await {
        decoder.feed(&chunk?, &mut on_data);
    }
    decoder.finish(&mut on_data);
    Ok(())
}
