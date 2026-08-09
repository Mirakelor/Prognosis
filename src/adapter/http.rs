use std::time::Duration;

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use tokio_util::sync::CancellationToken;

use crate::adapter::config::{RetryConfig, TimeoutConfig};
use crate::adapter::error::AdapterError;
use crate::adapter::retry::retry;

fn log_llm_request<B: Serialize>(url: &str, body: &B) {
    let Ok(path) = std::env::var("PROGNOSIS_LOG_LLM") else {
        return;
    };
    let Ok(json) = serde_json::to_string_pretty(body) else {
        return;
    };
    use std::io::Write;
    let _ = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .and_then(|mut file| {
            writeln!(file, "=== LLM REQUEST {url} ===\n{json}\n=== END ===")
        });
}

pub struct HttpTransport {
    adapter: String,
    api_key: String,
    http: reqwest::Client,
    retry: RetryConfig,
    request_timeout: Duration,
    stream_idle: Duration,
}

impl HttpTransport {
    pub fn new(
        adapter: impl Into<String>,
        api_key: impl Into<String>,
        retry: RetryConfig,
        timeout: TimeoutConfig,
    ) -> Result<Self, AdapterError> {
        let adapter = adapter.into();
        let http = reqwest::Client::builder()
            .connect_timeout(timeout.connect)
            .build()
            .map_err(|err| AdapterError::Config {
                adapter: adapter.clone(),
                message: format!("failed to build http client: {err}"),
            })?;
        Ok(Self {
            adapter,
            api_key: api_key.into(),
            http,
            retry,
            request_timeout: timeout.request,
            stream_idle: timeout.stream_idle,
        })
    }

    pub async fn get(&self, url: &str, cancel: &CancellationToken) -> Result<reqwest::Response, AdapterError> {
        retry(&self.retry, cancel, || {
            let url = url.to_string();
            async move { self.execute(&url, self.get_builder(&url)).await }
        })
        .await
    }

    pub async fn post<B: Serialize + Clone>(
        &self,
        url: &str,
        body: &B,
        cancel: &CancellationToken,
    ) -> Result<reqwest::Response, AdapterError> {
        log_llm_request(url, body);
        retry(&self.retry, cancel, || {
            let url = url.to_string();
            let body = body.clone();
            async move { self.execute(&url, self.post_builder(&url, &body)).await }
        })
        .await
    }

    pub async fn connect_and_first_event<B: Serialize + Clone, T: DeserializeOwned>(
        &self,
        url: &str,
        body: &B,
        cancel: &CancellationToken,
    ) -> Result<(reqwest::Response, Option<T>, String), AdapterError> {
        log_llm_request(url, body);
        retry(&self.retry, cancel, || {
            let url = url.to_string();
            let body = body.clone();
            async move {
                let mut resp = self.execute(&url, self.post_builder(&url, &body)).await?;
                let mut buf = String::new();
                match read_event::<T>(
                    &mut resp,
                    &mut buf,
                    self.stream_idle,
                    &self.adapter,
                    cancel,
                )
                .await?
                {
                    Some(SseEvent::Chunk(chunk)) => Ok((resp, Some(chunk), buf)),
                    Some(SseEvent::Done) | None => Err(AdapterError::Stream {
                        adapter: self.adapter.clone(),
                        message: "stream ended before any chunk".into(),
                    }),
                }
            }
        })
        .await
    }

    fn get_builder(&self, url: &str) -> reqwest::RequestBuilder {
        self.http.get(url).bearer_auth(&self.api_key)
    }

    fn post_builder<B: Serialize>(&self, url: &str, body: &B) -> reqwest::RequestBuilder {
        self.http.post(url).bearer_auth(&self.api_key).json(body)
    }

    async fn execute(
        &self,
        url: &str,
        builder: reqwest::RequestBuilder,
    ) -> Result<reqwest::Response, AdapterError> {
        let resp = tokio::time::timeout(self.request_timeout, builder.send())
            .await
            .map_err(|_| AdapterError::Timeout {
                adapter: self.adapter.clone(),
                message: format!(
                    "request to {url} timed out after {:?}",
                    self.request_timeout
                ),
            })?
            .map_err(|err| network_error(&self.adapter, err))?;
        if resp.status().is_success() {
            Ok(resp)
        } else {
            Err(self.error_from_response(resp).await)
        }
    }

    async fn error_from_response(&self, resp: reqwest::Response) -> AdapterError {
        let status = resp.status().as_u16();
        let request_id = resp
            .headers()
            .get("x-request-id")
            .and_then(|value| value.to_str().ok())
            .map(str::to_string);
        let retry_after = resp
            .headers()
            .get("retry-after")
            .and_then(|value| value.to_str().ok())
            .and_then(parse_retry_after);
        let message = match resp.json::<WireApiError>().await {
            Ok(body) => body.error.message.unwrap_or_default(),
            Err(_) => String::new(),
        };
        let message = if message.is_empty() {
            format!("HTTP {status}")
        } else {
            message
        };
        match status {
            401 => AdapterError::Authentication {
                adapter: self.adapter.clone(),
                message,
                request_id,
            },
            403 => AdapterError::PermissionDenied {
                adapter: self.adapter.clone(),
                message,
                request_id,
            },
            404 => AdapterError::NotFound {
                adapter: self.adapter.clone(),
                message,
                request_id,
            },
            429 => AdapterError::RateLimit {
                adapter: self.adapter.clone(),
                message,
                retry_after,
                request_id,
            },
            400..=499 => AdapterError::InvalidRequest {
                adapter: self.adapter.clone(),
                message,
            },
            500..=599 => AdapterError::ServerError {
                adapter: self.adapter.clone(),
                status,
                message,
                request_id,
            },
            _ => AdapterError::InvalidRequest {
                adapter: self.adapter.clone(),
                message,
            },
        }
    }
}

#[derive(Deserialize)]
struct WireApiError {
    error: WireApiErrorDetail,
}

#[derive(Deserialize)]
struct WireApiErrorDetail {
    #[serde(default)]
    message: Option<String>,
}

pub enum SseEvent<T> {
    Chunk(T),
    Done,
}

pub fn pop_sse_data(buf: &mut String) -> Option<Option<String>> {
    while let Some(pos) = buf.find('\n') {
        let line: String = buf[..pos].trim_end_matches('\r').to_string();
        buf.drain(..=pos);
        if let Some(data) = line.strip_prefix("data:") {
            let data = data.trim_start();
            return Some(if data == "[DONE]" {
                None
            } else {
                Some(data.to_string())
            });
        }
    }
    None
}

pub async fn read_event<T: DeserializeOwned>(
    resp: &mut reqwest::Response,
    buf: &mut String,
    idle: Duration,
    adapter: &str,
    cancel: &CancellationToken,
) -> Result<Option<SseEvent<T>>, AdapterError> {
    loop {
        if let Some(event) = pop_sse_data(buf) {
            return Ok(Some(match event {
                None => SseEvent::Done,
                Some(data) => {
                    let chunk: T = serde_json::from_str(&data).map_err(|err| {
                        AdapterError::Decode {
                            adapter: adapter.into(),
                            message: format!("invalid SSE payload: {err}"),
                        }
                    })?;
                    SseEvent::Chunk(chunk)
                }
            }));
        }
        if cancel.is_cancelled() {
            return Err(AdapterError::cancelled(adapter));
        }
        let bytes = match tokio::time::timeout(idle, resp.chunk()).await {
            Err(_) => {
                return Err(AdapterError::Timeout {
                    adapter: adapter.into(),
                    message: format!("stream idle for more than {idle:?}"),
                });
            }
            Ok(Err(err)) => return Err(network_error(adapter, err)),
            Ok(Ok(None)) => return Ok(None),
            Ok(Ok(Some(bytes))) => bytes,
        };
        buf.push_str(&String::from_utf8_lossy(&bytes));
    }
}

fn network_error(adapter: &str, err: reqwest::Error) -> AdapterError {
    AdapterError::Network {
        adapter: adapter.into(),
        message: err.to_string(),
        source: Box::new(err),
    }
}

pub fn parse_retry_after(value: &str) -> Option<Duration> {
    value.trim().parse::<u64>().ok().map(Duration::from_secs)
}

#[cfg(test)]
pub(crate) mod test_support {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    pub async fn spawn_mock(sse_body: &'static str) -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let (mut sock, _) = listener.accept().await.unwrap();
            let mut buf = vec![0u8; 4096];
            let n = sock.read(&mut buf).await.unwrap();
            let head = String::from_utf8_lossy(&buf[..n]).to_string();
            let path = head
                .split_whitespace()
                .nth(1)
                .unwrap_or_default()
                .to_string();
            if path.contains("/models") {
                let body = r#"{"object":"list","data":[{"id":"gpt-4o","object":"model","created":1700000000,"owned_by":"openai"}]}"#;
                let header = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    body.len()
                );
                sock.write_all(header.as_bytes()).await.unwrap();
                sock.write_all(body.as_bytes()).await.unwrap();
            } else {
                let header = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n";
                sock.write_all(header.as_bytes()).await.unwrap();
                for line in sse_body.split_inclusive('\n') {
                    let size = format!("{:x}\r\n", line.len());
                    sock.write_all(size.as_bytes()).await.unwrap();
                    sock.write_all(line.as_bytes()).await.unwrap();
                    sock.write_all(b"\r\n").await.unwrap();
                }
                sock.write_all(b"0\r\n\r\n").await.unwrap();
            }
            sock.shutdown().await.unwrap();
        });
        format!("http://{addr}")
    }

    pub async fn spawn_mock_recording(
        sse_body: &'static str,
    ) -> (String, std::sync::Arc<std::sync::Mutex<String>>) {
        let recorded = std::sync::Arc::new(std::sync::Mutex::new(String::new()));
        let recorded_for_task = recorded.clone();
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let (mut sock, _) = listener.accept().await.unwrap();
            let mut buf = vec![0u8; 65536];
            let n = sock.read(&mut buf).await.unwrap();
            let head = String::from_utf8_lossy(&buf[..n]).to_string();
            if let Some(body) = head.split("\r\n\r\n").nth(1) {
                *recorded_for_task.lock().unwrap() = body.to_string();
            }
            let header = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n";
            sock.write_all(header.as_bytes()).await.unwrap();
            for line in sse_body.split_inclusive('\n') {
                let size = format!("{:x}\r\n", line.len());
                sock.write_all(size.as_bytes()).await.unwrap();
                sock.write_all(line.as_bytes()).await.unwrap();
                sock.write_all(b"\r\n").await.unwrap();
            }
            sock.write_all(b"0\r\n\r\n").await.unwrap();
            sock.shutdown().await.unwrap();
        });
        (format!("http://{addr}"), recorded)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sse_data_parsing() {
        let mut buf = "data: {\"a\":1}\n\n: keep-alive\n\n".to_string();
        assert_eq!(
            pop_sse_data(&mut buf),
            Some(Some("{\"a\":1}".to_string()))
        );
        assert_eq!(pop_sse_data(&mut buf), None);
    }

    #[test]
    fn sse_done_marker() {
        let mut buf = "data: [DONE]\n".to_string();
        assert_eq!(pop_sse_data(&mut buf), Some(None));
    }

    #[test]
    fn retry_after_parsing() {
        assert_eq!(parse_retry_after("30"), Some(Duration::from_secs(30)));
        assert_eq!(parse_retry_after("not-a-number"), None);
    }
}
