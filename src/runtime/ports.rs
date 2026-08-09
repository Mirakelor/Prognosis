use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use futures::Stream;
use tokio_util::sync::CancellationToken;

use crate::adapter::error::AdapterError;
use crate::adapter::traits::LanguageModelAdapter;
use crate::adapter::types::{CompletionChunk, CompletionParams, CompletionRequest};
use crate::runtime::types::GenerateRequest;

#[async_trait]
pub trait LlmPort: Send + Sync {
    async fn generate<'a>(
        &'a self,
        request: &'a GenerateRequest,
        cancel: &'a CancellationToken,
    ) -> Result<
        Pin<Box<dyn Stream<Item = Result<CompletionChunk, AdapterError>> + Send + 'a>>,
        AdapterError,
    >;
}

pub struct LlmAdapter {
    inner: Arc<dyn LanguageModelAdapter>,
}

impl LlmAdapter {
    pub fn new(inner: Arc<dyn LanguageModelAdapter>) -> Self {
        Self { inner }
    }
}

#[async_trait]
impl LlmPort for LlmAdapter {
    async fn generate<'a>(
        &'a self,
        request: &'a GenerateRequest,
        cancel: &'a CancellationToken,
    ) -> Result<
        Pin<Box<dyn Stream<Item = Result<CompletionChunk, AdapterError>> + Send + 'a>>,
        AdapterError,
    > {
        let modulation = &request.modulation;
        let params = CompletionParams {
            temperature: modulation.temperature,
            top_p: modulation.top_p,
            reasoning_effort: modulation.reasoning_effort,
            n: modulation.n,
            ..Default::default()
        };
        let mut messages = request.messages.clone();
        messages.extend(modulation.injected_messages.iter().cloned());
        let mut completion = CompletionRequest::new(
            modulation.model.clone().unwrap_or_default(),
            messages,
        )
        .with_params(params);
        if let Some(tools) = &request.tools {
            completion.params.tools = Some(tools.clone());
        }
        self.inner.stream(completion, cancel).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adapter::types::{ChunkDelta, FinishReason, Message, ReasoningEffort, Role, Temperature};
    use crate::runtime::types::ModulationContext;
    use futures::StreamExt;
    use std::sync::{Arc, Mutex};

    struct RecordingAdapter {
        last_request: Arc<Mutex<Option<CompletionRequest>>>,
    }

    #[async_trait]
    impl LanguageModelAdapter for RecordingAdapter {
        fn id(&self) -> &str {
            "recording"
        }

        fn capabilities(&self) -> crate::adapter::types::AdapterCapabilities {
            crate::adapter::types::AdapterCapabilities::default()
        }

        async fn stream<'a>(
            &'a self,
            request: CompletionRequest,
            _cancel: &'a CancellationToken,
        ) -> Result<
            Pin<Box<dyn Stream<Item = Result<CompletionChunk, AdapterError>> + Send + 'a>>,
            AdapterError,
        > {
            *self.last_request.lock().unwrap() = Some(request.clone());
            let chunk = CompletionChunk {
                model: request.model.clone(),
                index: 0,
                delta: ChunkDelta {
                    role: None,
                    content: Some("ok".into()),
                    tool_calls: vec![],
                    logprobs: None,
                    reasoning: None,
                },
                finish_reason: Some(FinishReason::Stop),
                usage: None,
                request_id: None,
            };
            Ok(Box::pin(futures::stream::iter(vec![Ok(chunk)])))
        }
    }

    #[tokio::test]
    async fn modulation_context_is_consumed_by_request_build() {
        let last_request = Arc::new(Mutex::new(None));
        let adapter = RecordingAdapter {
            last_request: last_request.clone(),
        };
        let port = LlmAdapter::new(Arc::new(adapter));
        let modulation = ModulationContext {
            temperature: Some(Temperature::new(0.2).unwrap()),
            reasoning_effort: Some(ReasoningEffort::Max),
            n: Some(2),
            injected_messages: vec![Message::system("correct yourself")],
            ..Default::default()
        };
        let request = GenerateRequest {
            messages: vec![Message::user("hello")],
            modulation,
            tools: None,
        };
        let cancel = CancellationToken::new();
        let mut stream = port.generate(&request, &cancel).await.unwrap();
        stream.next().await.unwrap().unwrap();

        let recorded = last_request.lock().unwrap().clone().unwrap();
        assert_eq!(recorded.params.temperature.map(|t| t.get()), Some(0.2));
        assert_eq!(recorded.params.reasoning_effort, Some(ReasoningEffort::Max));
        assert_eq!(recorded.params.n, Some(2));
        assert_eq!(recorded.messages.len(), 2);
        assert_eq!(recorded.messages[1].role, Role::System);
    }
}
