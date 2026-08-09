use std::pin::Pin;

use async_trait::async_trait;
use futures::{Stream, StreamExt};
use serde_json::Value;
use tokio_util::sync::CancellationToken;

use crate::adapter::error::AdapterError;
use crate::adapter::types::{
    AdapterCapabilities, CompletionChunk, CompletionChoice, CompletionRequest, CompletionResponse,
    ContentLogprobs, FinishReason, ModelInfo, TokenUsage, ToolCall, ToolCallDelta,
};

#[async_trait]
pub trait LanguageModelAdapter: Send + Sync {
    fn id(&self) -> &str;

    fn capabilities(&self) -> AdapterCapabilities;

    async fn models(&self) -> Result<Vec<ModelInfo>, AdapterError> {
        Err(AdapterError::Unsupported {
            adapter: self.id().into(),
            feature: "model listing".into(),
        })
    }

    async fn model_info(&self, id: &str) -> Result<ModelInfo, AdapterError> {
        Err(AdapterError::Unsupported {
            adapter: self.id().into(),
            feature: format!("model retrieval for {id}"),
        })
    }

    async fn stream<'a>(
        &'a self,
        request: CompletionRequest,
        cancel: &'a CancellationToken,
    ) -> Result<
        Pin<Box<dyn Stream<Item = Result<CompletionChunk, AdapterError>> + Send + 'a>>,
        AdapterError,
    >;

    async fn complete<'a>(
        &'a self,
        request: &'a CompletionRequest,
        cancel: &'a CancellationToken,
    ) -> Result<CompletionResponse, AdapterError> {
        let mut stream = self.stream(request.clone(), cancel).await?;
        aggregate(self.id(), request, stream.as_mut(), cancel).await
    }
}

async fn aggregate(
    adapter: &str,
    request: &CompletionRequest,
    mut stream: Pin<&mut (dyn Stream<Item = Result<CompletionChunk, AdapterError>> + Send)>,
    cancel: &CancellationToken,
) -> Result<CompletionResponse, AdapterError> {
    let mut choices: Vec<ChoiceAccumulator> = Vec::new();
    let mut usage: Option<TokenUsage> = None;
    let mut request_id: Option<String> = None;
    let mut saw_chunk = false;

    while let Some(chunk) = stream.next().await {
        if cancel.is_cancelled() {
            return Err(AdapterError::cancelled(adapter));
        }
        let chunk = chunk?;
        saw_chunk = true;
        let index = chunk.index as usize;
        if choices.len() <= index {
            choices.resize_with(index + 1, ChoiceAccumulator::default);
        }
        let acc = &mut choices[index];
        if let Some(delta) = &chunk.delta.content {
            acc.content.push_str(delta);
        }
        if let Some(part) = &chunk.delta.reasoning {
            acc.reasoning.push_str(part);
        }
        for delta in &chunk.delta.tool_calls {
            merge_tool_call_delta(&mut acc.tool_deltas, delta);
        }
        if let Some(reason) = chunk.finish_reason {
            acc.finish_reason = reason;
        }
        if chunk.delta.logprobs.is_some() {
            acc.logprobs = chunk.delta.logprobs;
        }
        if chunk.usage.is_some() {
            usage = chunk.usage;
        }
        if chunk.request_id.is_some() {
            request_id = chunk.request_id;
        }
    }

    if !saw_chunk {
        return Err(AdapterError::Stream {
            adapter: adapter.into(),
            message: "stream ended without any chunk".into(),
        });
    }

    let choices = choices
        .into_iter()
        .map(|acc| CompletionChoice {
            content: acc.content,
            tool_calls: acc
                .tool_deltas
                .into_iter()
                .map(|delta| ToolCall {
                    id: delta.id.unwrap_or_default(),
                    name: delta.name.unwrap_or_default(),
                    arguments: parse_arguments(delta.arguments_delta.as_deref()),
                })
                .collect(),
            finish_reason: acc.finish_reason,
            logprobs: acc.logprobs,
            reasoning: if acc.reasoning.is_empty() {
                None
            } else {
                Some(acc.reasoning)
            },
        })
        .collect();

    Ok(CompletionResponse {
        model: request.model.clone(),
        choices,
        usage,
        request_id,
    })
}

#[derive(Default)]
struct ChoiceAccumulator {
    content: String,
    tool_deltas: Vec<ToolCallDelta>,
    finish_reason: FinishReason,
    logprobs: Option<ContentLogprobs>,
    reasoning: String,
}

fn merge_tool_call_delta(acc: &mut Vec<ToolCallDelta>, delta: &ToolCallDelta) {
    match acc.iter_mut().find(|existing| existing.index == delta.index) {
        Some(existing) => {
            if let Some(id) = &delta.id {
                existing.id = Some(id.clone());
            }
            if let Some(name) = &delta.name {
                existing.name = Some(name.clone());
            }
            if let Some(arg) = &delta.arguments_delta {
                existing
                    .arguments_delta
                    .get_or_insert_with(String::new)
                    .push_str(arg);
            }
        }
        None => acc.push(delta.clone()),
    }
}

fn parse_arguments(delta: Option<&str>) -> Value {
    match delta {
        Some(text) => serde_json::from_str(text).unwrap_or_else(|_| Value::String(text.into())),
        None => Value::Null,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adapter::types::{ChunkDelta, ContentLogprobs, LogprobToken, Message};
    use futures::stream;

    struct TestAdapter {
        chunks: Vec<Result<CompletionChunk, AdapterError>>,
    }

    impl TestAdapter {
        fn with_chunks(chunks: Vec<Result<CompletionChunk, AdapterError>>) -> Self {
            Self { chunks }
        }
    }

    #[async_trait]
    impl LanguageModelAdapter for TestAdapter {
        fn id(&self) -> &str {
            "test"
        }

        fn capabilities(&self) -> AdapterCapabilities {
            AdapterCapabilities::default()
        }

        async fn stream<'a>(
            &'a self,
            _request: CompletionRequest,
            _cancel: &'a CancellationToken,
        ) -> Result<
            Pin<Box<dyn Stream<Item = Result<CompletionChunk, AdapterError>> + Send + 'a>>,
            AdapterError,
        > {
            Ok(Box::pin(stream::iter(self.chunks.clone())))
        }
    }

    fn chunk(
        content: Option<&str>,
        tool_calls: Vec<ToolCallDelta>,
        finish_reason: Option<FinishReason>,
        usage: Option<TokenUsage>,
    ) -> Result<CompletionChunk, AdapterError> {
        Ok(CompletionChunk {
            model: "test-model".into(),
            index: 0,
            delta: ChunkDelta {
                role: None,
                content: content.map(Into::into),
                tool_calls,
                logprobs: None,
                reasoning: None,
            },
            finish_reason,
            usage,
            request_id: None,
        })
    }

    fn tool_delta(
        index: u32,
        id: Option<&str>,
        name: Option<&str>,
        args: Option<&str>,
    ) -> ToolCallDelta {
        ToolCallDelta {
            index,
            id: id.map(Into::into),
            name: name.map(Into::into),
            arguments_delta: args.map(Into::into),
        }
    }

    #[tokio::test]
    async fn complete_aggregates_content_and_usage() {
        let adapter = TestAdapter::with_chunks(vec![
            chunk(Some("hello "), vec![], None, None),
            chunk(
                Some("world"),
                vec![],
                Some(FinishReason::Stop),
                Some(TokenUsage::new(5, 3)),
            ),
        ]);
        let request = CompletionRequest::new("test-model", vec![Message::user("hi")]);
        let response = adapter
            .complete(&request, &CancellationToken::new())
            .await
            .unwrap();
        assert_eq!(response.choices[0].content, "hello world");
        assert_eq!(response.choices[0].finish_reason, FinishReason::Stop);
        assert_eq!(response.usage, Some(TokenUsage::new(5, 3)));
        assert_eq!(response.model, "test-model");
    }

    #[tokio::test]
    async fn complete_merges_tool_call_deltas() {
        let adapter = TestAdapter::with_chunks(vec![
            chunk(
                None,
                vec![tool_delta(0, Some("call_1"), Some("get_weather"), Some("{\"city\":")),],
                None,
                None,
            ),
            chunk(
                None,
                vec![
                    tool_delta(0, None, None, Some("\"beijing\"}")),
                    tool_delta(1, Some("call_2"), Some("get_time"), Some("{}")),
                ],
                Some(FinishReason::ToolCalls),
                None,
            ),
        ]);
        let request = CompletionRequest::new("test-model", vec![Message::user("hi")]);
        let response = adapter
            .complete(&request, &CancellationToken::new())
            .await
            .unwrap();
        assert_eq!(response.choices[0].tool_calls.len(), 2);
        assert_eq!(response.choices[0].tool_calls[0].id, "call_1");
        assert_eq!(response.choices[0].tool_calls[0].name, "get_weather");
        assert_eq!(response.choices[0].tool_calls[0].arguments["city"], "beijing");
        assert_eq!(response.choices[0].tool_calls[1].arguments, serde_json::json!({}));
        assert_eq!(response.choices[0].finish_reason, FinishReason::ToolCalls);
    }

    #[tokio::test]
    async fn complete_errors_on_empty_stream() {
        let adapter = TestAdapter::with_chunks(vec![]);
        let request = CompletionRequest::new("test-model", vec![Message::user("hi")]);
        let err = adapter
            .complete(&request, &CancellationToken::new())
            .await
            .unwrap_err();
        assert!(matches!(err, AdapterError::Stream { .. }));
    }

    #[tokio::test]
    async fn complete_stops_on_cancellation() {
        let cancel = CancellationToken::new();
        cancel.cancel();
        let adapter = TestAdapter::with_chunks(vec![chunk(Some("hi"), vec![], None, None)]);
        let request = CompletionRequest::new("test-model", vec![Message::user("hi")]);
        let err = adapter.complete(&request, &cancel).await.unwrap_err();
        assert!(matches!(err, AdapterError::Cancelled { .. }));
    }

    #[tokio::test]
    async fn complete_aggregates_logprobs() {
        let adapter = TestAdapter::with_chunks(vec![Ok(CompletionChunk {
            model: "test-model".into(),
            index: 0,
            delta: ChunkDelta {
                role: None,
                content: Some("x".into()),
                tool_calls: vec![],
                logprobs: Some(ContentLogprobs {
                    tokens: vec![LogprobToken {
                        token: "x".into(),
                        logprob: -0.5,
                    }],
                }),
                reasoning: None,
            },
            finish_reason: Some(FinishReason::Stop),
            usage: None,
            request_id: None,
        })]);
        let request = CompletionRequest::new("test-model", vec![Message::user("hi")]);
        let response = adapter
            .complete(&request, &CancellationToken::new())
            .await
            .unwrap();
        assert_eq!(
            response.choices[0].logprobs.as_ref().unwrap().tokens[0].logprob,
            -0.5
        );
    }

    #[tokio::test]
    async fn complete_aggregates_reasoning() {
        let adapter = TestAdapter::with_chunks(vec![
            Ok(CompletionChunk {
                model: "test-model".into(),
                index: 0,
                delta: ChunkDelta {
                    role: None,
                    content: Some("answer".into()),
                    tool_calls: vec![],
                    logprobs: None,
                    reasoning: Some("step one ".into()),
                },
                finish_reason: None,
                usage: None,
                request_id: None,
            }),
            Ok(CompletionChunk {
                model: "test-model".into(),
                index: 0,
                delta: ChunkDelta {
                    role: None,
                    content: None,
                    tool_calls: vec![],
                    logprobs: None,
                    reasoning: Some("step two".into()),
                },
                finish_reason: Some(FinishReason::Stop),
                usage: None,
                request_id: None,
            }),
        ]);
        let request = CompletionRequest::new("test-model", vec![Message::user("hi")]);
        let response = adapter
            .complete(&request, &CancellationToken::new())
            .await
            .unwrap();
        assert_eq!(response.choices[0].reasoning.as_deref(), Some("step one step two"));
        assert_eq!(response.choices[0].content, "answer");
    }

    #[tokio::test]
    async fn complete_aggregates_multiple_choices() {
        let adapter = TestAdapter::with_chunks(vec![
            Ok(CompletionChunk {
                model: "test-model".into(),
                index: 0,
                delta: ChunkDelta {
                    role: None,
                    content: Some("first".into()),
                    tool_calls: vec![],
                    logprobs: None,
                    reasoning: None,
                },
                finish_reason: None,
                usage: None,
                request_id: None,
            }),
            Ok(CompletionChunk {
                model: "test-model".into(),
                index: 1,
                delta: ChunkDelta {
                    role: None,
                    content: Some("second".into()),
                    tool_calls: vec![],
                    logprobs: None,
                    reasoning: None,
                },
                finish_reason: None,
                usage: None,
                request_id: None,
            }),
            Ok(CompletionChunk {
                model: "test-model".into(),
                index: 0,
                delta: ChunkDelta {
                    role: None,
                    content: Some("!".into()),
                    tool_calls: vec![],
                    logprobs: None,
                    reasoning: None,
                },
                finish_reason: Some(FinishReason::Stop),
                usage: Some(TokenUsage::new(5, 5)),
                request_id: None,
            }),
            Ok(CompletionChunk {
                model: "test-model".into(),
                index: 1,
                delta: ChunkDelta {
                    role: None,
                    content: Some("?".into()),
                    tool_calls: vec![],
                    logprobs: None,
                    reasoning: None,
                },
                finish_reason: Some(FinishReason::Stop),
                usage: None,
                request_id: None,
            }),
        ]);
        let request = CompletionRequest::new("test-model", vec![Message::user("hi")]);
        let response = adapter
            .complete(&request, &CancellationToken::new())
            .await
            .unwrap();
        assert_eq!(response.choices.len(), 2);
        assert_eq!(response.choices[0].content, "first!");
        assert_eq!(response.choices[1].content, "second?");
        assert_eq!(response.choices[1].finish_reason, FinishReason::Stop);
        assert_eq!(response.usage, Some(TokenUsage::new(5, 5)));
    }

    #[tokio::test]
    async fn unsupported_defaults() {
        let adapter = TestAdapter::with_chunks(vec![]);
        assert!(adapter.models().await.is_err());
        assert!(adapter.model_info("some-model").await.is_err());
    }
}
