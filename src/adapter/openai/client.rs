use std::pin::Pin;

use async_stream::stream;
use async_trait::async_trait;
use futures::Stream;
use tokio_util::sync::CancellationToken;

use crate::adapter::config::{RetryConfig, TimeoutConfig};
use crate::adapter::error::AdapterError;
use crate::adapter::http::{read_event, HttpTransport, SseEvent};
use crate::adapter::traits::LanguageModelAdapter;
use crate::adapter::types::{AdapterCapabilities, CompletionChunk, CompletionRequest, ModelInfo};

use super::wire::{WireChatCompletionChunk, WireChatCompletionRequest, WireModel, WireModelList};

#[derive(Clone)]
pub struct OpenAIConfig {
    pub api_key: String,
    pub base_url: String,
    pub default_model: String,
    pub retry: RetryConfig,
    pub timeout: TimeoutConfig,
}

impl std::fmt::Debug for OpenAIConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenAIConfig")
            .field("api_key", &"***")
            .field("base_url", &self.base_url)
            .field("default_model", &self.default_model)
            .field("retry", &self.retry)
            .field("timeout", &self.timeout)
            .finish()
    }
}

impl OpenAIConfig {
    fn validate(&self) -> Result<(), AdapterError> {
        reqwest::Url::parse(&self.base_url).map_err(|err| AdapterError::Config {
            adapter: "openai".into(),
            message: format!("invalid base_url {:?}: {err}", self.base_url),
        })?;
        Ok(())
    }
}

#[derive(Debug, Default, Clone)]
pub struct OpenAIConfigBuilder {
    api_key: Option<String>,
    base_url: Option<String>,
    default_model: Option<String>,
    retry: Option<RetryConfig>,
    timeout: Option<TimeoutConfig>,
}

impl OpenAIConfigBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn api_key(mut self, value: impl Into<String>) -> Self {
        self.api_key = Some(value.into());
        self
    }

    pub fn base_url(mut self, value: impl Into<String>) -> Self {
        self.base_url = Some(value.into());
        self
    }

    pub fn default_model(mut self, value: impl Into<String>) -> Self {
        self.default_model = Some(value.into());
        self
    }

    pub fn retry(mut self, value: RetryConfig) -> Self {
        self.retry = Some(value);
        self
    }

    pub fn timeout(mut self, value: TimeoutConfig) -> Self {
        self.timeout = Some(value);
        self
    }

    pub fn build(self) -> Result<OpenAIConfig, AdapterError> {
        let api_key = self
            .api_key
            .or_else(|| env_var("OPENAI_API_KEY"))
            .ok_or_else(|| AdapterError::Config {
                adapter: "openai".into(),
                message: "api key is required: set OPENAI_API_KEY or pass it explicitly".into(),
            })?;
        let base_url = self
            .base_url
            .or_else(|| env_var("OPENAI_BASE_URL"))
            .unwrap_or_else(|| "https://api.openai.com/v1".to_string());
        let default_model = self
            .default_model
            .or_else(|| env_var("OPENAI_DEFAULT_MODEL"))
            .unwrap_or_else(|| "gpt-4o".to_string());
        let config = OpenAIConfig {
            api_key,
            base_url,
            default_model,
            retry: self.retry.unwrap_or_default(),
            timeout: self.timeout.unwrap_or_default(),
        };
        config.validate()?;
        Ok(config)
    }
}

fn env_var(key: &str) -> Option<String> {
    std::env::var(key)
        .ok()
        .filter(|value| !value.trim().is_empty())
}

pub struct OpenAIClient {
    config: OpenAIConfig,
    transport: HttpTransport,
}

impl OpenAIClient {
    pub fn new(config: OpenAIConfig) -> Result<Self, AdapterError> {
        let transport = HttpTransport::new("openai", &config.api_key, config.retry.clone(), config.timeout)?;
        Ok(Self { config, transport })
    }

    pub fn default_model(&self) -> &str {
        &self.config.default_model
    }
}

#[async_trait]
impl LanguageModelAdapter for OpenAIClient {
    fn id(&self) -> &str {
        "openai"
    }

    fn capabilities(&self) -> AdapterCapabilities {
        AdapterCapabilities {
            streaming: true,
            model_listing: true,
            model_retrieval: true,
            tool_calling: true,
            json_mode: true,
            json_schema: true,
            logprobs: true,
            multimodal_text: true,
            multimodal_image: true,
            multimodal_audio: true,
            usage_in_stream: true,
        }
    }

    async fn models(&self) -> Result<Vec<ModelInfo>, AdapterError> {
        let url = format!("{}/models", self.config.base_url);
        let resp = self.transport.get(&url, &CancellationToken::new()).await?;
        let body: WireModelList = resp.json().await.map_err(|err| AdapterError::Decode {
            adapter: "openai".into(),
            message: format!("failed to decode model list: {err}"),
        })?;
        Ok(body.data.into_iter().map(ModelInfo::from).collect())
    }

    async fn model_info(&self, id: &str) -> Result<ModelInfo, AdapterError> {
        let url = format!("{}/models/{id}", self.config.base_url);
        let resp = self.transport.get(&url, &CancellationToken::new()).await?;
        let body: WireModel = resp.json().await.map_err(|err| AdapterError::Decode {
            adapter: "openai".into(),
            message: format!("failed to decode model info: {err}"),
        })?;
        Ok(ModelInfo::from(body))
    }

    async fn stream<'a>(
        &'a self,
        request: CompletionRequest,
        cancel: &'a CancellationToken,
    ) -> Result<
        Pin<Box<dyn Stream<Item = Result<CompletionChunk, AdapterError>> + Send + 'a>>,
        AdapterError,
    > {
        let mut wire_req = WireChatCompletionRequest::from_request(&request)?;
        if wire_req.model.is_empty() {
            wire_req.model = self.config.default_model.clone();
        }
        let url = format!("{}/chat/completions", self.config.base_url);
        let (mut resp, first, mut buf) = self
            .transport
            .connect_and_first_event::<_, WireChatCompletionChunk>(&url, &wire_req, cancel)
            .await?;
        let idle = self.config.timeout.stream_idle;

        let s = stream! {
            if let Some(first) = first {
                for chunk in first.into_chunks() {
                    yield Ok(chunk);
                }
            }
            loop {
                match read_event::<WireChatCompletionChunk>(&mut resp, &mut buf, idle, "openai", cancel).await {
                    Ok(Some(SseEvent::Chunk(wire))) => {
                        for chunk in wire.into_chunks() {
                            yield Ok(chunk);
                        }
                    }
                    Ok(Some(SseEvent::Done)) | Ok(None) => break,
                    Err(err) => {
                        yield Err(err);
                        return;
                    }
                }
            }
        };
        Ok(Box::pin(s))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adapter::http::test_support::spawn_mock;
    use crate::adapter::types::{FinishReason, Message, TokenUsage};

    const SSE_BODY: &str = concat!(
        "data: {\"id\":\"chatcmpl-1\",\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"Hello\"},\"finish_reason\":null}]}\n",
        "data: {\"id\":\"chatcmpl-1\",\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\" world\"},\"finish_reason\":null}]}\n",
        "data: {\"id\":\"chatcmpl-1\",\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":2,\"total_tokens\":7}}\n",
        "data: [DONE]\n",
    );

    #[tokio::test]
    async fn complete_via_mock_server() {
        let base = spawn_mock(SSE_BODY).await;
        let config = OpenAIConfigBuilder::new()
            .api_key("test-key")
            .base_url(format!("{base}/v1"))
            .build()
            .unwrap();
        let client = OpenAIClient::new(config).unwrap();
        let request = CompletionRequest::new("gpt-4o", vec![Message::user("hello")]);
        let response = client
            .complete(&request, &CancellationToken::new())
            .await
            .unwrap();
        assert_eq!(response.choices[0].content, "Hello world");
        assert_eq!(response.choices[0].finish_reason, FinishReason::Stop);
        assert_eq!(
            response.usage,
            Some(TokenUsage {
                prompt_tokens: 5,
                completion_tokens: 2,
                total_tokens: 7,
            })
        );
        assert_eq!(response.request_id.as_deref(), Some("chatcmpl-1"));
    }

    #[tokio::test]
    async fn tool_calling_via_mock_server() {
        const TOOL_SSE: &str = concat!(
            "data: {\"id\":\"chatcmpl-2\",\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":null,\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"arguments\":\"\"}}]},\"finish_reason\":null}]}\n",
            "data: {\"id\":\"chatcmpl-2\",\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{\\\"city\\\":\"}}]},\"finish_reason\":null}]}\n",
            "data: {\"id\":\"chatcmpl-2\",\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"\\\"beijing\\\"}\"}}]},\"finish_reason\":\"tool_calls\"}],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":6,\"total_tokens\":16}}\n",
            "data: [DONE]\n",
        );
        let base = spawn_mock(TOOL_SSE).await;
        let config = OpenAIConfigBuilder::new()
            .api_key("test-key")
            .base_url(format!("{base}/v1"))
            .build()
            .unwrap();
        let client = OpenAIClient::new(config).unwrap();
        let request = CompletionRequest::new("gpt-4o", vec![Message::user("weather in beijing?")]);
        let response = client
            .complete(&request, &CancellationToken::new())
            .await
            .unwrap();
        let choice = &response.choices[0];
        assert!(choice.has_tool_calls());
        assert_eq!(choice.tool_calls[0].id, "call_1");
        assert_eq!(choice.tool_calls[0].name, "get_weather");
        assert_eq!(choice.tool_calls[0].arguments["city"], "beijing");
        assert_eq!(choice.finish_reason, FinishReason::ToolCalls);
        assert_eq!(
            response.usage,
            Some(TokenUsage {
                prompt_tokens: 10,
                completion_tokens: 6,
                total_tokens: 16,
            })
        );
    }

    #[tokio::test]
    async fn models_via_mock_server() {
        let base = spawn_mock(SSE_BODY).await;
        let config = OpenAIConfigBuilder::new()
            .api_key("test-key")
            .base_url(format!("{base}/v1"))
            .build()
            .unwrap();
        let client = OpenAIClient::new(config).unwrap();
        let models = client.models().await.unwrap();
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].id, "gpt-4o");
        assert_eq!(models[0].owned_by.as_deref(), Some("openai"));
    }

    #[test]
    fn config_requires_api_key() {
        let err = OpenAIConfigBuilder::new().build().unwrap_err();
        assert!(matches!(err, AdapterError::Config { .. }));
    }

    #[test]
    fn config_masks_api_key_in_debug() {
        let config = OpenAIConfigBuilder::new()
            .api_key("sk-secret-123")
            .base_url("https://api.openai.com/v1")
            .build()
            .unwrap();
        let debug = format!("{config:?}");
        assert!(!debug.contains("sk-secret-123"));
        assert!(debug.contains("***"));
    }
}
