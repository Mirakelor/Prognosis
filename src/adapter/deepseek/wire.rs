use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::adapter::error::AdapterError;
use crate::adapter::types::{
    ChunkDelta, CompletionChunk, CompletionRequest, ContentLogprobs, ContentPart, FinishReason,
    LogprobToken, Message, MessageContent, ModelInfo, ReasoningEffort, ResponseFormat, Role,
    Temperature, TokenUsage, ToolCallDelta, ToolChoice, ToolDefinition, TopP,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ThinkingConfig {
    pub enabled: bool,
    pub reasoning_effort: Option<ReasoningEffort>,
}

impl ThinkingConfig {
    pub fn enabled() -> Self {
        Self {
            enabled: true,
            reasoning_effort: None,
        }
    }

    pub fn disabled() -> Self {
        Self {
            enabled: false,
            reasoning_effort: None,
        }
    }

    pub fn with_effort(mut self, effort: ReasoningEffort) -> Self {
        self.reasoning_effort = Some(effort);
        self
    }
}

fn deepseek_effort(effort: ReasoningEffort) -> Option<&'static str> {
    match effort {
        ReasoningEffort::None => None,
        ReasoningEffort::Minimal | ReasoningEffort::Low => Some("low"),
        ReasoningEffort::Medium | ReasoningEffort::High | ReasoningEffort::XHigh => Some("high"),
        ReasoningEffort::Max => Some("max"),
    }
}

#[derive(Debug, Serialize, Clone)]
pub struct WireDeepSeekChatCompletionRequest {
    pub model: String,
    pub messages: Vec<WireDeepSeekMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<WireThinking>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<WireDeepSeekResponseFormat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<WireDeepSeekTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<WireDeepSeekToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_logprobs: Option<u8>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user_id: Option<String>,
    pub stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<WireDeepSeekStreamOptions>,
}

#[derive(Debug, Serialize, Clone)]
pub struct WireThinking {
    pub r#type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<String>,
}

impl WireDeepSeekChatCompletionRequest {
    pub fn from_request(
        request: &CompletionRequest,
        thinking: Option<&ThinkingConfig>,
    ) -> Result<Self, AdapterError> {
        let params = &request.params;
        let messages = request
            .messages
            .iter()
            .map(WireDeepSeekMessage::from_message)
            .collect::<Result<Vec<_>, _>>()?;
        let response_format = match params.response_format.as_ref() {
            None | Some(ResponseFormat::Text) => None,
            Some(ResponseFormat::JsonObject) => Some(WireDeepSeekResponseFormat::JsonObject),
            Some(ResponseFormat::JsonSchema { .. }) => {
                return Err(AdapterError::invalid_request(
                    "deepseek",
                    "json_schema response format is not supported by deepseek; use JsonObject",
                ));
            }
        };
        let sampling_allowed = thinking.is_some_and(|config| !config.enabled);
        Ok(Self {
            model: request.model.clone(),
            messages,
            thinking: thinking.map(|config| WireThinking {
                r#type: if config.enabled {
                    "enabled".into()
                } else {
                    "disabled".into()
                },
                reasoning_effort: config
                    .reasoning_effort
                    .and_then(deepseek_effort)
                    .map(Into::into),
            }),
            max_tokens: params.max_tokens,
            temperature: if sampling_allowed {
                params.temperature.map(Temperature::get)
            } else {
                None
            },
            top_p: if sampling_allowed {
                params.top_p.map(TopP::get)
            } else {
                None
            },
            stop: params.stop.clone(),
            response_format,
            tools: params
                .tools
                .as_ref()
                .map(|tools| tools.iter().map(WireDeepSeekTool::from_definition).collect()),
            tool_choice: params
                .tool_choice
                .as_ref()
                .map(WireDeepSeekToolChoice::from_choice),
            logprobs: params.logprobs,
            top_logprobs: params.top_logprobs,
            user_id: params.user.clone(),
            stream: true,
            stream_options: Some(WireDeepSeekStreamOptions { include_usage: true }),
        })
    }
}

#[derive(Debug, Serialize, Clone)]
pub struct WireDeepSeekMessage {
    pub role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<WireDeepSeekRequestToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

impl WireDeepSeekMessage {
    fn from_message(message: &Message) -> Result<Self, AdapterError> {
        let content = match &message.content {
            MessageContent::Text(text) if text.is_empty() => None,
            MessageContent::Text(text) => Some(text.clone()),
            MessageContent::Parts(parts) => {
                let mut text = String::new();
                for part in parts {
                    match part {
                        ContentPart::Text { text: chunk } => text.push_str(chunk),
                        _ => {
                            return Err(AdapterError::invalid_request(
                                "deepseek",
                                "multimodal content parts are not supported by deepseek",
                            ));
                        }
                    }
                }
                if text.is_empty() {
                    None
                } else {
                    Some(text)
                }
            }
        };
        let tool_calls = message
            .tool_calls
            .as_ref()
            .map(|calls| {
                calls
                    .iter()
                    .map(|call| -> Result<WireDeepSeekRequestToolCall, AdapterError> {
                        let arguments = serde_json::to_string(&call.arguments).map_err(|e| {
                            AdapterError::invalid_request(
                                "deepseek",
                                format!("tool call arguments are not valid JSON: {e}"),
                            )
                        })?;
                        Ok(WireDeepSeekRequestToolCall {
                            id: call.id.clone(),
                            kind: "function".into(),
                            function: WireDeepSeekFunctionCall {
                                name: call.name.clone(),
                                arguments,
                            },
                        })
                    })
                    .collect()
            })
            .transpose()?;
        Ok(Self {
            role: message.role.to_string(),
            name: message.name.clone(),
            content,
            reasoning_content: message.reasoning.clone(),
            tool_calls,
            tool_call_id: message.tool_call_id.clone(),
        })
    }
}

#[derive(Debug, Serialize, Clone)]
pub struct WireDeepSeekRequestToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: String,
    pub function: WireDeepSeekFunctionCall,
}

#[derive(Debug, Serialize, Clone)]
pub struct WireDeepSeekFunctionCall {
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Serialize, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum WireDeepSeekResponseFormat {
    JsonObject,
}

#[derive(Debug, Serialize, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum WireDeepSeekTool {
    Function {
        function: WireDeepSeekFunction,
    },
}

#[derive(Debug, Serialize, Clone)]
pub struct WireDeepSeekFunction {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub parameters: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
}

impl WireDeepSeekTool {
    fn from_definition(tool: &ToolDefinition) -> Self {
        match tool {
            ToolDefinition::Function { function } => Self::Function {
                function: WireDeepSeekFunction {
                    name: function.name.clone(),
                    description: function.description.clone(),
                    parameters: function.parameters.schema.clone(),
                    strict: function.strict,
                },
            },
        }
    }
}

#[derive(Debug, Serialize, Clone)]
#[serde(untagged)]
pub enum WireDeepSeekToolChoice {
    Mode(WireDeepSeekToolChoiceMode),
    Function {
        r#type: String,
        function: WireDeepSeekNamedFunction,
    },
}

#[derive(Debug, Serialize, Clone)]
#[serde(rename_all = "snake_case")]
pub enum WireDeepSeekToolChoiceMode {
    Auto,
    None,
    Required,
}

#[derive(Debug, Serialize, Clone)]
pub struct WireDeepSeekNamedFunction {
    pub name: String,
}

impl WireDeepSeekToolChoice {
    fn from_choice(choice: &ToolChoice) -> Self {
        match choice {
            ToolChoice::Auto => Self::Mode(WireDeepSeekToolChoiceMode::Auto),
            ToolChoice::None => Self::Mode(WireDeepSeekToolChoiceMode::None),
            ToolChoice::Required => Self::Mode(WireDeepSeekToolChoiceMode::Required),
            ToolChoice::Named { kind, function } => Self::Function {
                r#type: kind.clone(),
                function: WireDeepSeekNamedFunction {
                    name: function.name.clone(),
                },
            },
        }
    }
}

#[derive(Debug, Serialize, Clone)]
pub struct WireDeepSeekStreamOptions {
    pub include_usage: bool,
}

#[derive(Deserialize)]
pub struct WireDeepSeekChunk {
    pub id: String,
    pub model: String,
    #[serde(default)]
    pub choices: Vec<WireDeepSeekChunkChoice>,
    #[serde(default)]
    pub usage: Option<WireDeepSeekUsage>,
}

#[derive(Deserialize, Clone)]
pub struct WireDeepSeekChunkChoice {
    #[serde(default)]
    pub index: u32,
    pub delta: WireDeepSeekDelta,
    #[serde(default)]
    pub finish_reason: Option<String>,
}

#[derive(Deserialize, Clone, Default)]
pub struct WireDeepSeekDelta {
    #[serde(default)]
    pub role: Option<String>,
    #[serde(default)]
    pub content: Option<String>,
    #[serde(default)]
    pub reasoning_content: Option<String>,
    #[serde(default)]
    pub tool_calls: Vec<WireDeepSeekToolCallDelta>,
    #[serde(default)]
    pub logprobs: Option<WireDeepSeekChunkLogprobs>,
}

#[derive(Deserialize, Clone)]
pub struct WireDeepSeekToolCallDelta {
    pub index: u32,
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub function: Option<WireDeepSeekFunctionDelta>,
}

#[derive(Deserialize, Clone, Default)]
pub struct WireDeepSeekFunctionDelta {
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub arguments: Option<String>,
}

#[derive(Deserialize, Clone, Default)]
pub struct WireDeepSeekChunkLogprobs {
    #[serde(default)]
    pub content: Vec<WireDeepSeekLogprobToken>,
}

#[derive(Deserialize, Clone)]
pub struct WireDeepSeekLogprobToken {
    pub token: String,
    pub logprob: f32,
}

#[derive(Deserialize, Clone, Copy)]
pub struct WireDeepSeekUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

impl From<WireDeepSeekUsage> for TokenUsage {
    fn from(usage: WireDeepSeekUsage) -> Self {
        Self {
            prompt_tokens: usage.prompt_tokens,
            completion_tokens: usage.completion_tokens,
            total_tokens: usage.total_tokens,
        }
    }
}

impl WireDeepSeekChunk {
    pub fn into_chunks(self) -> Vec<CompletionChunk> {
        let model = self.model;
        let usage = self.usage.map(TokenUsage::from);
        let request_id = Some(self.id);
        if self.choices.is_empty() {
            return vec![CompletionChunk {
                model,
                index: 0,
                delta: ChunkDelta::default(),
                finish_reason: None,
                usage,
                request_id,
            }];
        }
        self.choices
            .into_iter()
            .map(|choice| {
                let delta = choice.delta;
                let logprobs = delta.logprobs.as_ref().map(|logprobs| ContentLogprobs {
                    tokens: logprobs
                        .content
                        .iter()
                        .map(|token| LogprobToken {
                            token: token.token.clone(),
                            logprob: token.logprob,
                        })
                        .collect(),
                });
                CompletionChunk {
                    model: model.clone(),
                    index: choice.index,
                    delta: ChunkDelta {
                        role: delta.role.as_deref().and_then(parse_role),
                        content: delta.content,
                        tool_calls: delta
                            .tool_calls
                            .iter()
                            .map(|tool_call| ToolCallDelta {
                                index: tool_call.index,
                                id: tool_call.id.clone(),
                                name: tool_call
                                    .function
                                    .as_ref()
                                    .and_then(|function| function.name.clone()),
                                arguments_delta: tool_call
                                    .function
                                    .as_ref()
                                    .and_then(|function| function.arguments.clone()),
                            })
                            .collect(),
                        logprobs,
                        reasoning: delta.reasoning_content,
                    },
                    finish_reason: choice
                        .finish_reason
                        .as_deref()
                        .and_then(parse_finish_reason),
                    usage,
                    request_id: request_id.clone(),
                }
            })
            .collect()
    }
}

#[derive(Deserialize)]
pub struct WireDeepSeekModelList {
    pub data: Vec<WireDeepSeekModel>,
}

#[derive(Deserialize)]
pub struct WireDeepSeekModel {
    pub id: String,
    #[serde(default)]
    pub object: Option<String>,
    #[serde(default)]
    pub owned_by: Option<String>,
    #[serde(default)]
    pub created: Option<u64>,
}

impl From<WireDeepSeekModel> for ModelInfo {
    fn from(model: WireDeepSeekModel) -> Self {
        Self {
            id: model.id,
            object: model.object,
            owned_by: model.owned_by,
            created_at: model.created,
        }
    }
}

fn parse_role(value: &str) -> Option<Role> {
    match value {
        "system" => Some(Role::System),
        "user" => Some(Role::User),
        "assistant" => Some(Role::Assistant),
        "tool" => Some(Role::Tool),
        _ => None,
    }
}

fn parse_finish_reason(value: &str) -> Option<FinishReason> {
    match value {
        "stop" => Some(FinishReason::Stop),
        "length" => Some(FinishReason::Length),
        "content_filter" => Some(FinishReason::ContentFilter),
        "tool_calls" => Some(FinishReason::ToolCalls),
        "function_call" => Some(FinishReason::FunctionCall),
        _ => Some(FinishReason::Unknown),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adapter::types::{
        CompletionParams, ContentPart, MessageContent, ResponseFormat, ToolChoice, ToolDefinition,
    };

    #[test]
    fn request_serialization_shape() {
        let request = CompletionRequest::new("deepseek-v4-pro", vec![Message::user("hi")])
            .with_params(
                CompletionParams::builder()
                    .temperature(0.5)
                    .unwrap()
                    .max_tokens(512)
                    .user("user-1")
                    .logprobs(true, Some(3))
                    .tools(vec![ToolDefinition::function(
                        "get_weather",
                        Some("weather lookup".to_string()),
                        serde_json::json!({"type": "object"}),
                    )])
                    .tool_choice(ToolChoice::named("get_weather"))
                    .build(),
            );
        let thinking = ThinkingConfig::enabled().with_effort(ReasoningEffort::Max);
        let wire =
            WireDeepSeekChatCompletionRequest::from_request(&request, Some(&thinking)).unwrap();
        let json = serde_json::to_value(&wire).unwrap();
        assert_eq!(json["model"], "deepseek-v4-pro");
        assert_eq!(json["stream"], true);
        assert_eq!(json["stream_options"]["include_usage"], true);
        assert_eq!(json["thinking"]["type"], "enabled");
        assert_eq!(json["thinking"]["reasoning_effort"], "max");
        assert_eq!(json["max_tokens"], 512);
        assert_eq!(json["user_id"], "user-1");
        assert_eq!(json["logprobs"], true);
        assert_eq!(json["messages"][0]["content"], "hi");
        assert_eq!(json["tools"][0]["function"]["name"], "get_weather");
        assert_eq!(
            json["tool_choice"],
            serde_json::json!({"type": "function", "function": {"name": "get_weather"}})
        );
        assert!(json.get("temperature").is_none());
        assert!(json.get("top_p").is_none());
        assert!(json.get("seed").is_none());
        assert!(json.get("presence_penalty").is_none());
    }

    #[test]
    fn thinking_disabled_serialization() {
        let request = CompletionRequest::new("deepseek-v4-flash", vec![Message::user("hi")]);
        let wire =
            WireDeepSeekChatCompletionRequest::from_request(&request, Some(&ThinkingConfig::disabled()))
                .unwrap();
        let json = serde_json::to_value(&wire).unwrap();
        assert_eq!(json["thinking"]["type"], "disabled");
        assert!(json["thinking"].get("reasoning_effort").is_none());
    }

    #[test]
    fn effort_mapping_mirrors_server_compat() {
        let effort = |level: ReasoningEffort| {
            let request = CompletionRequest::new("deepseek-v4-flash", vec![Message::user("hi")]);
            let thinking = ThinkingConfig::enabled().with_effort(level);
            let wire =
                WireDeepSeekChatCompletionRequest::from_request(&request, Some(&thinking)).unwrap();
            let json = serde_json::to_value(&wire).unwrap();
            json["thinking"]["reasoning_effort"].as_str().unwrap_or("").to_string()
        };
        assert_eq!(effort(ReasoningEffort::Minimal), "low");
        assert_eq!(effort(ReasoningEffort::Low), "low");
        assert_eq!(effort(ReasoningEffort::Medium), "high");
        assert_eq!(effort(ReasoningEffort::High), "high");
        assert_eq!(
            effort(ReasoningEffort::XHigh), "high",
            "DeepSeek maps xhigh to high server-side; mirror that"
        );
        assert_eq!(effort(ReasoningEffort::Max), "max");
    }

    #[test]
    fn sampling_params_sent_only_when_thinking_disabled() {
        let request = CompletionRequest::new("deepseek-v4-pro", vec![Message::user("hi")])
            .with_params(CompletionParams::builder().temperature(0.5).unwrap().build());

        let thinking_enabled =
            WireDeepSeekChatCompletionRequest::from_request(&request, Some(&ThinkingConfig::enabled()))
                .unwrap();
        let json = serde_json::to_value(&thinking_enabled).unwrap();
        assert!(json.get("temperature").is_none());
        assert!(json.get("top_p").is_none());

        let thinking_disabled =
            WireDeepSeekChatCompletionRequest::from_request(&request, Some(&ThinkingConfig::disabled()))
                .unwrap();
        let json = serde_json::to_value(&thinking_disabled).unwrap();
        assert_eq!(json["temperature"], 0.5);

        let default_thinking =
            WireDeepSeekChatCompletionRequest::from_request(&request, None).unwrap();
        let json = serde_json::to_value(&default_thinking).unwrap();
        assert!(json.get("temperature").is_none());
    }

    #[test]
    fn json_schema_rejected() {
        let request = CompletionRequest::new("deepseek-v4-pro", vec![Message::user("hi")])
            .with_params(
                CompletionParams::builder()
                    .response_format(ResponseFormat::JsonSchema {
                        name: "x".into(),
                        schema: serde_json::json!({}),
                        strict: None,
                    })
                    .build(),
            );
        let err = WireDeepSeekChatCompletionRequest::from_request(&request, None).unwrap_err();
        assert!(matches!(err, AdapterError::InvalidRequest { .. }));
    }

    #[test]
    fn multimodal_parts_rejected() {
        let request = CompletionRequest::new(
            "deepseek-v4-pro",
            vec![Message::user(MessageContent::parts(vec![
                ContentPart::text("look"),
                ContentPart::image_url("https://example.com/a.png", crate::adapter::types::ImageDetail::Low),
            ]))],
        );
        let err = WireDeepSeekChatCompletionRequest::from_request(&request, None).unwrap_err();
        assert!(matches!(err, AdapterError::InvalidRequest { .. }));
    }

    #[test]
    fn text_parts_joined() {
        let request = CompletionRequest::new(
            "deepseek-v4-pro",
            vec![Message::user(MessageContent::parts(vec![
                ContentPart::text("hello "),
                ContentPart::text("world"),
            ]))],
        );
        let wire = WireDeepSeekChatCompletionRequest::from_request(&request, None).unwrap();
        let json = serde_json::to_value(&wire).unwrap();
        assert_eq!(json["messages"][0]["content"], "hello world");
    }

    #[test]
    fn chunk_with_reasoning_deserialization() {
        let payload = r#"{"id":"chatcmpl-ds1","model":"deepseek-v4-pro","choices":[{"index":0,"delta":{"role":"assistant","reasoning_content":"Let me think","content":"Answer"},"finish_reason":null}]}"#;
        let wire: WireDeepSeekChunk = serde_json::from_str(payload).unwrap();
        let chunk = wire.into_chunks().into_iter().next().unwrap();
        assert_eq!(chunk.content(), Some("Answer"));
        assert_eq!(chunk.delta.reasoning.as_deref(), Some("Let me think"));
        assert_eq!(chunk.request_id.as_deref(), Some("chatcmpl-ds1"));
    }
}
