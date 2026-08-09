use std::pin::Pin;

use async_stream::stream;
use async_trait::async_trait;
use futures::Stream;
use tokio_util::sync::CancellationToken;

use crate::adapter::config::{RetryConfig, TimeoutConfig};
use crate::adapter::error::AdapterError;
use crate::adapter::http::{read_event, HttpTransport, SseEvent};
use crate::adapter::traits::LanguageModelAdapter;
use crate::adapter::types::{
    AdapterCapabilities, CompletionChunk, CompletionRequest, ModelInfo, ReasoningEffort,
};

use super::wire::{
    ThinkingConfig, WireDeepSeekChatCompletionRequest, WireDeepSeekChunk, WireDeepSeekModel,
    WireDeepSeekModelList,
};

#[derive(Clone)]
pub struct DeepSeekConfig {
    pub api_key: String,
    pub base_url: String,
    pub default_model: String,
    pub thinking: Option<ThinkingConfig>,
    pub retry: RetryConfig,
    pub timeout: TimeoutConfig,
}

impl std::fmt::Debug for DeepSeekConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeepSeekConfig")
            .field("api_key", &"***")
            .field("base_url", &self.base_url)
            .field("default_model", &self.default_model)
            .field("thinking", &self.thinking)
            .field("retry", &self.retry)
            .field("timeout", &self.timeout)
            .finish()
    }
}

impl DeepSeekConfig {
    fn validate(&self) -> Result<(), AdapterError> {
        reqwest::Url::parse(&self.base_url).map_err(|err| AdapterError::Config {
            adapter: "deepseek".into(),
            message: format!("invalid base_url {:?}: {err}", self.base_url),
        })?;
        Ok(())
    }
}

#[derive(Debug, Default, Clone)]
pub struct DeepSeekConfigBuilder {
    api_key: Option<String>,
    base_url: Option<String>,
    default_model: Option<String>,
    thinking: Option<ThinkingConfig>,
    retry: Option<RetryConfig>,
    timeout: Option<TimeoutConfig>,
}

impl DeepSeekConfigBuilder {
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

    pub fn thinking(mut self, value: ThinkingConfig) -> Self {
        self.thinking = Some(value);
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

    pub fn build(self) -> Result<DeepSeekConfig, AdapterError> {
        let api_key = self
            .api_key
            .or_else(|| env_var("DEEPSEEK_API_KEY"))
            .ok_or_else(|| AdapterError::Config {
                adapter: "deepseek".into(),
                message: "api key is required: set DEEPSEEK_API_KEY or pass it explicitly".into(),
            })?;
        let base_url = self
            .base_url
            .or_else(|| env_var("DEEPSEEK_BASE_URL"))
            .unwrap_or_else(|| "https://api.deepseek.com".to_string());
        let default_model = self
            .default_model
            .or_else(|| env_var("DEEPSEEK_DEFAULT_MODEL"))
            .unwrap_or_else(|| "deepseek-v4-flash".to_string());
        let config = DeepSeekConfig {
            api_key,
            base_url,
            default_model,
            thinking: self.thinking,
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

pub struct DeepSeekClient {
    config: DeepSeekConfig,
    transport: HttpTransport,
}

impl DeepSeekClient {
    pub fn new(config: DeepSeekConfig) -> Result<Self, AdapterError> {
        let transport =
            HttpTransport::new("deepseek", &config.api_key, config.retry.clone(), config.timeout)?;
        Ok(Self { config, transport })
    }

    pub fn default_model(&self) -> &str {
        &self.config.default_model
    }
}

#[async_trait]
impl LanguageModelAdapter for DeepSeekClient {
    fn id(&self) -> &str {
        "deepseek"
    }

    fn capabilities(&self) -> AdapterCapabilities {
        AdapterCapabilities {
            streaming: true,
            model_listing: true,
            model_retrieval: true,
            tool_calling: true,
            json_mode: true,
            json_schema: false,
            logprobs: true,
            multimodal_text: true,
            multimodal_image: false,
            multimodal_audio: false,
            usage_in_stream: true,
        }
    }

    async fn models(&self) -> Result<Vec<ModelInfo>, AdapterError> {
        let url = format!("{}/models", self.config.base_url);
        let resp = self.transport.get(&url, &CancellationToken::new()).await?;
        let body: WireDeepSeekModelList = resp.json().await.map_err(|err| AdapterError::Decode {
            adapter: "deepseek".into(),
            message: format!("failed to decode model list: {err}"),
        })?;
        Ok(body.data.into_iter().map(ModelInfo::from).collect())
    }

    async fn model_info(&self, id: &str) -> Result<ModelInfo, AdapterError> {
        let url = format!("{}/models/{id}", self.config.base_url);
        let resp = self.transport.get(&url, &CancellationToken::new()).await?;
        let body: WireDeepSeekModel = resp.json().await.map_err(|err| AdapterError::Decode {
            adapter: "deepseek".into(),
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
        let thinking = self.config.thinking.or_else(|| {
            request.params.reasoning_effort.map(|effort| match effort {
                ReasoningEffort::None => ThinkingConfig::disabled(),
                other => ThinkingConfig::enabled().with_effort(other),
            })
        });
        let mut wire_req =
            WireDeepSeekChatCompletionRequest::from_request(&request, thinking.as_ref())?;
        if wire_req.model.is_empty() {
            wire_req.model = self.config.default_model.clone();
        }
        let url = format!("{}/chat/completions", self.config.base_url);
        let (mut resp, first, mut buf) = self
            .transport
            .connect_and_first_event::<_, WireDeepSeekChunk>(&url, &wire_req, cancel)
            .await?;
        let idle = self.config.timeout.stream_idle;

        let s = stream! {
            if let Some(first) = first {
                for chunk in first.into_chunks() {
                    yield Ok(chunk);
                }
            }
            loop {
                match read_event::<WireDeepSeekChunk>(&mut resp, &mut buf, idle, "deepseek", cancel).await {
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
        "data: {\"id\":\"chatcmpl-ds1\",\"model\":\"deepseek-v4-pro\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"reasoning_content\":\"Let me think\"},\"finish_reason\":null}]}\n",
        "data: {\"id\":\"chatcmpl-ds1\",\"model\":\"deepseek-v4-pro\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"The answer\"},\"finish_reason\":null}]}\n",
        "data: {\"id\":\"chatcmpl-ds1\",\"model\":\"deepseek-v4-pro\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":8,\"completion_tokens\":4,\"total_tokens\":12}}\n",
        "data: [DONE]\n",
    );

    #[tokio::test]
    async fn complete_via_mock_server() {
        let base = spawn_mock(SSE_BODY).await;
        let config = DeepSeekConfigBuilder::new()
            .api_key("test-key")
            .base_url(base)
            .build()
            .unwrap();
        let client = DeepSeekClient::new(config).unwrap();
        let request = CompletionRequest::new("deepseek-v4-pro", vec![Message::user("hello")]);
        let response = client
            .complete(&request, &CancellationToken::new())
            .await
            .unwrap();
        assert_eq!(response.choices[0].content, "The answer");
        assert_eq!(response.choices[0].reasoning.as_deref(), Some("Let me think"));
        assert_eq!(response.choices[0].finish_reason, FinishReason::Stop);
        assert_eq!(
            response.usage,
            Some(TokenUsage {
                prompt_tokens: 8,
                completion_tokens: 4,
                total_tokens: 12,
            })
        );
        assert_eq!(response.request_id.as_deref(), Some("chatcmpl-ds1"));
    }

    #[tokio::test]
    async fn tool_calling_via_mock_server() {
        const TOOL_SSE: &str = concat!(
            "data: {\"id\":\"chatcmpl-ds2\",\"model\":\"deepseek-v4-pro\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":null,\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"arguments\":\"\"}}]},\"finish_reason\":null}]}\n",
            "data: {\"id\":\"chatcmpl-ds2\",\"model\":\"deepseek-v4-pro\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{\\\"city\\\":\"}}]},\"finish_reason\":null}]}\n",
            "data: {\"id\":\"chatcmpl-ds2\",\"model\":\"deepseek-v4-pro\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"\\\"beijing\\\"}\"}}]},\"finish_reason\":\"tool_calls\"}],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":6,\"total_tokens\":16}}\n",
            "data: [DONE]\n",
        );
        let base = spawn_mock(TOOL_SSE).await;
        let config = DeepSeekConfigBuilder::new()
            .api_key("test-key")
            .base_url(base)
            .build()
            .unwrap();
        let client = DeepSeekClient::new(config).unwrap();
        let request = CompletionRequest::new("deepseek-v4-pro", vec![Message::user("weather in beijing?")]);
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
    }

    #[tokio::test]
    async fn models_via_mock_server() {
        let base = spawn_mock(SSE_BODY).await;
        let config = DeepSeekConfigBuilder::new()
            .api_key("test-key")
            .base_url(base)
            .build()
            .unwrap();
        let client = DeepSeekClient::new(config).unwrap();
        let models = client.models().await.unwrap();
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].id, "gpt-4o");
    }

    #[tokio::test]
    async fn explicit_no_reasoning_disables_thinking() {
        use crate::adapter::http::test_support::spawn_mock_recording;
        let (base, recorded) = spawn_mock_recording(SSE_BODY).await;
        let config = DeepSeekConfigBuilder::new()
            .api_key("test-key")
            .base_url(base)
            .build()
            .unwrap();
        let client = DeepSeekClient::new(config).unwrap();
        let mut request = CompletionRequest::new("deepseek-v4-pro", vec![Message::user("hi")]);
        request.params.reasoning_effort = Some(ReasoningEffort::None);
        request.params.temperature = Some(crate::adapter::types::Temperature::new(0.0).unwrap());
        client
            .complete(&request, &CancellationToken::new())
            .await
            .unwrap();
        let body: serde_json::Value =
            serde_json::from_str(&recorded.lock().unwrap()).unwrap();
        assert_eq!(body["thinking"]["type"], "disabled");
        assert!(
            body.get("temperature").is_some(),
            "sampling params must be sent when thinking is disabled"
        );
    }

    #[tokio::test]
    async fn explicit_reasoning_enables_thinking_with_effort() {
        use crate::adapter::http::test_support::spawn_mock_recording;
        let (base, recorded) = spawn_mock_recording(SSE_BODY).await;
        let config = DeepSeekConfigBuilder::new()
            .api_key("test-key")
            .base_url(base)
            .build()
            .unwrap();
        let client = DeepSeekClient::new(config).unwrap();
        let mut request = CompletionRequest::new("deepseek-v4-pro", vec![Message::user("hi")]);
        request.params.reasoning_effort = Some(ReasoningEffort::High);
        client
            .complete(&request, &CancellationToken::new())
            .await
            .unwrap();
        let body: serde_json::Value =
            serde_json::from_str(&recorded.lock().unwrap()).unwrap();
        assert_eq!(body["thinking"]["type"], "enabled");
        assert_eq!(body["thinking"]["reasoning_effort"], "high");
        assert!(
            body.get("temperature").is_none(),
            "sampling params must be dropped while thinking is enabled"
        );
    }

    #[tokio::test]
    async fn unset_effort_leaves_thinking_to_config() {
        use crate::adapter::http::test_support::spawn_mock_recording;
        let (base, recorded) = spawn_mock_recording(SSE_BODY).await;
        let config = DeepSeekConfigBuilder::new()
            .api_key("test-key")
            .base_url(base)
            .build()
            .unwrap();
        let client = DeepSeekClient::new(config).unwrap();
        let request = CompletionRequest::new("deepseek-v4-pro", vec![Message::user("hi")]);
        client
            .complete(&request, &CancellationToken::new())
            .await
            .unwrap();
        let body: serde_json::Value =
            serde_json::from_str(&recorded.lock().unwrap()).unwrap();
        assert!(
            body.get("thinking").is_none(),
            "unset effort must not inject thinking (vendor default applies)"
        );
    }

    #[test]
    fn config_requires_api_key() {
        let err = DeepSeekConfigBuilder::new().build().unwrap_err();
        assert!(matches!(err, AdapterError::Config { .. }));
    }

    #[test]
    fn config_masks_api_key_in_debug() {
        let config = DeepSeekConfigBuilder::new()
            .api_key("sk-secret-456")
            .base_url("https://api.deepseek.com")
            .build()
            .unwrap();
        let debug = format!("{config:?}");
        assert!(!debug.contains("sk-secret-456"));
        assert!(debug.contains("***"));
    }
}
