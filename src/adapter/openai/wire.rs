use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::adapter::error::AdapterError;
use crate::adapter::types::{
    AudioFormat, ChunkDelta, CompletionChunk, CompletionRequest, ContentLogprobs, ContentPart,
    FinishReason, ImageDetail, LogprobToken, Message, MessageContent, ModelInfo, Penalty,
    ReasoningEffort, ResponseFormat, Role, Temperature, TokenUsage, ToolCallDelta, ToolChoice,
    ToolDefinition, TopP,
};

fn openai_effort(effort: ReasoningEffort) -> Option<&'static str> {
    match effort {
        ReasoningEffort::None => None,
        other => Some(other.as_str()),
    }
}

fn supports_sampling(model: &str) -> bool {
    model.starts_with("gpt-4")
}

#[derive(Serialize, Clone)]
pub struct WireChatCompletionRequest {
    pub model: String,
    pub messages: Vec<WireMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_completion_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<WireResponseFormat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<WireTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<WireToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_logprobs: Option<u8>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n: Option<u32>,
    pub stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<WireStreamOptions>,
}

impl WireChatCompletionRequest {
    pub fn from_request(request: &CompletionRequest) -> Result<Self, AdapterError> {
        let params = &request.params;
        let messages = request
            .messages
            .iter()
            .map(WireMessage::from_message)
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            model: request.model.clone(),
            messages,
            temperature: if supports_sampling(&request.model) {
                params.temperature.map(Temperature::get)
            } else {
                None
            },
            top_p: if supports_sampling(&request.model) {
                params.top_p.map(TopP::get)
            } else {
                None
            },
            max_completion_tokens: params.max_tokens,
            seed: params.seed,
            stop: params.stop.clone(),
            presence_penalty: params.presence_penalty.map(Penalty::get),
            frequency_penalty: params.frequency_penalty.map(Penalty::get),
            user: if supports_sampling(&request.model) {
                params.user.clone()
            } else {
                None
            },
            response_format: params.response_format.as_ref().and_then(|format| {
                match format {
                    ResponseFormat::Text => None,
                    ResponseFormat::JsonObject => Some(WireResponseFormat::JsonObject),
                    ResponseFormat::JsonSchema { name, schema, strict } => {
                        Some(WireResponseFormat::JsonSchema {
                            json_schema: WireJsonSchema {
                                name: name.clone(),
                                schema: schema.clone(),
                                strict: *strict,
                            },
                        })
                    }
                }
            }),
            tools: params
                .tools
                .as_ref()
                .map(|tools| tools.iter().map(WireTool::from_definition).collect()),
            tool_choice: params
                .tool_choice
                .as_ref()
                .map(WireToolChoice::from_choice),
            logprobs: params.logprobs,
            top_logprobs: params.top_logprobs,
            reasoning_effort: params.reasoning_effort.and_then(openai_effort).map(String::from),
            parallel_tool_calls: params.parallel_tool_calls,
            n: params.n,
            stream: true,
            stream_options: Some(WireStreamOptions { include_usage: true }),
        })
    }
}

#[derive(Serialize, Clone)]
pub struct WireMessage {
    pub role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<WireContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<WireRequestToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

impl WireMessage {
    fn from_message(message: &Message) -> Result<Self, AdapterError> {
        let content = match &message.content {
            MessageContent::Text(text) if text.is_empty() => None,
            MessageContent::Text(text) => Some(WireContent::Text(text.clone())),
            MessageContent::Parts(parts) => Some(WireContent::Parts(
                parts.iter().map(WireContentPart::from_part).collect(),
            )),
        };
        let tool_calls = message
            .tool_calls
            .as_ref()
            .map(|calls| {
                calls
                    .iter()
                    .map(|call| -> Result<WireRequestToolCall, AdapterError> {
                        let arguments = serde_json::to_string(&call.arguments).map_err(|e| {
                            AdapterError::invalid_request(
                                "openai",
                                format!("tool call arguments are not valid JSON: {e}"),
                            )
                        })?;
                        Ok(WireRequestToolCall {
                            id: call.id.clone(),
                            kind: "function".into(),
                            function: WireFunctionCall {
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
            tool_calls,
            tool_call_id: message.tool_call_id.clone(),
        })
    }
}

#[derive(Serialize, Clone)]
#[serde(untagged)]
pub enum WireContent {
    Text(String),
    Parts(Vec<WireContentPart>),
}

#[derive(Serialize, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum WireContentPart {
    Text { text: String },
    ImageUrl { image_url: WireImageUrl },
    InputAudio { input_audio: WireInputAudio },
}

#[derive(Serialize, Clone)]
pub struct WireImageUrl {
    pub url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

#[derive(Serialize, Clone)]
pub struct WireInputAudio {
    pub data: String,
    pub format: String,
}

impl WireContentPart {
    fn from_part(part: &ContentPart) -> Self {
        match part {
            ContentPart::Text { text } => Self::Text { text: text.clone() },
            ContentPart::ImageUrl { url, detail } => Self::ImageUrl {
                image_url: WireImageUrl {
                    url: url.clone(),
                    detail: Some(image_detail(*detail)),
                },
            },
            ContentPart::InputAudio { data, format } => Self::InputAudio {
                input_audio: WireInputAudio {
                    data: data.clone(),
                    format: audio_format(*format),
                },
            },
        }
    }
}

#[derive(Serialize, Clone)]
pub struct WireRequestToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: String,
    pub function: WireFunctionCall,
}

#[derive(Serialize, Clone)]
pub struct WireFunctionCall {
    pub name: String,
    pub arguments: String,
}

#[derive(Serialize, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum WireResponseFormat {
    JsonObject,
    JsonSchema { json_schema: WireJsonSchema },
}

#[derive(Serialize, Clone)]
pub struct WireJsonSchema {
    pub name: String,
    pub schema: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
}

#[derive(Serialize, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum WireTool {
    Function { function: WireFunction },
}

#[derive(Serialize, Clone)]
pub struct WireFunction {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub parameters: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
}

impl WireTool {
    fn from_definition(tool: &ToolDefinition) -> Self {
        match tool {
            ToolDefinition::Function { function } => Self::Function {
                function: WireFunction {
                    name: function.name.clone(),
                    description: function.description.clone(),
                    parameters: function.parameters.schema.clone(),
                    strict: function.strict,
                },
            },
        }
    }
}

#[derive(Serialize, Clone)]
#[serde(untagged)]
pub enum WireToolChoice {
    Mode(WireToolChoiceMode),
    Function {
        r#type: String,
        function: WireNamedFunction,
    },
}

#[derive(Serialize, Clone)]
#[serde(rename_all = "snake_case")]
pub enum WireToolChoiceMode {
    Auto,
    None,
    Required,
}

#[derive(Serialize, Clone)]
pub struct WireNamedFunction {
    pub name: String,
}

impl WireToolChoice {
    fn from_choice(choice: &ToolChoice) -> Self {
        match choice {
            ToolChoice::Auto => Self::Mode(WireToolChoiceMode::Auto),
            ToolChoice::None => Self::Mode(WireToolChoiceMode::None),
            ToolChoice::Required => Self::Mode(WireToolChoiceMode::Required),
            ToolChoice::Named { kind, function } => Self::Function {
                r#type: kind.clone(),
                function: WireNamedFunction {
                    name: function.name.clone(),
                },
            },
        }
    }
}

#[derive(Serialize, Clone)]
pub struct WireStreamOptions {
    pub include_usage: bool,
}

#[derive(Deserialize)]
pub struct WireChatCompletionChunk {
    pub id: String,
    pub model: String,
    #[serde(default)]
    pub choices: Vec<WireChunkChoice>,
    #[serde(default)]
    pub usage: Option<WireUsage>,
}

#[derive(Deserialize, Clone)]
pub struct WireChunkChoice {
    #[serde(default)]
    pub index: u32,
    pub delta: WireDelta,
    #[serde(default)]
    pub finish_reason: Option<String>,
}

#[derive(Deserialize, Clone, Default)]
pub struct WireDelta {
    #[serde(default)]
    pub role: Option<String>,
    #[serde(default)]
    pub content: Option<String>,
    #[serde(default)]
    pub tool_calls: Vec<WireToolCallDelta>,
    #[serde(default)]
    pub logprobs: Option<WireChunkLogprobs>,
}

#[derive(Deserialize, Clone)]
pub struct WireToolCallDelta {
    pub index: u32,
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub function: Option<WireFunctionDelta>,
}

#[derive(Deserialize, Clone, Default)]
pub struct WireFunctionDelta {
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub arguments: Option<String>,
}

#[derive(Deserialize, Clone, Default)]
pub struct WireChunkLogprobs {
    #[serde(default)]
    pub content: Vec<WireLogprobToken>,
}

#[derive(Deserialize, Clone)]
pub struct WireLogprobToken {
    pub token: String,
    pub logprob: f32,
}

#[derive(Deserialize, Clone, Copy)]
pub struct WireUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

impl From<WireUsage> for TokenUsage {
    fn from(usage: WireUsage) -> Self {
        Self {
            prompt_tokens: usage.prompt_tokens,
            completion_tokens: usage.completion_tokens,
            total_tokens: usage.total_tokens,
        }
    }
}

impl WireChatCompletionChunk {
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
                        reasoning: None,
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
pub struct WireModelList {
    pub data: Vec<WireModel>,
}

#[derive(Deserialize)]
pub struct WireModel {
    pub id: String,
    #[serde(default)]
    pub object: Option<String>,
    #[serde(default)]
    pub owned_by: Option<String>,
    #[serde(default)]
    pub created: Option<u64>,
}

impl From<WireModel> for ModelInfo {
    fn from(model: WireModel) -> Self {
        Self {
            id: model.id,
            object: model.object,
            owned_by: model.owned_by,
            created_at: model.created,
        }
    }
}

fn image_detail(detail: ImageDetail) -> String {
    match detail {
        ImageDetail::Low => "low".into(),
        ImageDetail::High => "high".into(),
        ImageDetail::Auto => "auto".into(),
    }
}

fn audio_format(format: AudioFormat) -> String {
    match format {
        AudioFormat::Wav => "wav".into(),
        AudioFormat::Mp3 => "mp3".into(),
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
        CompletionParams, ContentPart, FunctionDefinition, FunctionParameters, MessageContent,
        ReasoningEffort, ResponseFormat, ToolChoice, ToolDefinition,
    };

    #[test]
    fn request_serialization_shape() {
        let request = CompletionRequest::new("gpt-4o", vec![Message::user("hi")]).with_params(
            CompletionParams::builder()
                .temperature(0.5)
                .unwrap()
                .logprobs(true, Some(3))
                .tools(vec![ToolDefinition::function(
                    "get_weather",
                    Some("weather lookup".to_string()),
                    serde_json::json!({"type": "object"}),
                )])
                .tool_choice(ToolChoice::named("get_weather"))
                .response_format(ResponseFormat::JsonSchema {
                    name: "weather".into(),
                    schema: serde_json::json!({"type": "object"}),
                    strict: Some(true),
                })
                .build(),
        );
        let wire = WireChatCompletionRequest::from_request(&request).unwrap();
        let json = serde_json::to_value(&wire).unwrap();
        assert_eq!(json["model"], "gpt-4o");
        assert_eq!(json["stream"], true);
        assert_eq!(json["stream_options"]["include_usage"], true);
        assert_eq!(json["messages"][0]["role"], "user");
        assert_eq!(json["messages"][0]["content"], "hi");
        assert_eq!(json["temperature"], 0.5);
        assert_eq!(json["logprobs"], true);
        assert_eq!(json["top_logprobs"], 3);
        assert_eq!(json["tools"][0]["type"], "function");
        assert_eq!(json["tools"][0]["function"]["name"], "get_weather");
        assert_eq!(
            json["tool_choice"],
            serde_json::json!({"type": "function", "function": {"name": "get_weather"}})
        );
        assert_eq!(json["response_format"]["type"], "json_schema");
        assert_eq!(json["response_format"]["json_schema"]["strict"], true);
    }

    #[test]
    fn advanced_params_serialization() {
        let request = CompletionRequest::new("gpt-4o", vec![Message::user("hi")]).with_params(
            CompletionParams::builder()
                .reasoning_effort(ReasoningEffort::XHigh)
                .parallel_tool_calls(false)
                .n(3)
                .build(),
        );
        let wire = WireChatCompletionRequest::from_request(&request).unwrap();
        let json = serde_json::to_value(&wire).unwrap();
        assert_eq!(
            json["reasoning_effort"], "xhigh",
            "xhigh is a valid OpenAI effort value and must pass through"
        );
        assert_eq!(json["parallel_tool_calls"], false);
        assert_eq!(json["n"], 3);
    }

    #[test]
    fn effort_passes_through_full_enum() {
        let effort = |level: ReasoningEffort| {
            let mut request = CompletionRequest::new("gpt-5", vec![Message::user("hi")]);
            request.params.reasoning_effort = Some(level);
            let wire = WireChatCompletionRequest::from_request(&request).unwrap();
            let json = serde_json::to_value(&wire).unwrap();
            json["reasoning_effort"].as_str().unwrap_or("").to_string()
        };
        assert_eq!(effort(ReasoningEffort::Minimal), "minimal");
        assert_eq!(effort(ReasoningEffort::Low), "low");
        assert_eq!(effort(ReasoningEffort::Medium), "medium");
        assert_eq!(effort(ReasoningEffort::High), "high");
        assert_eq!(effort(ReasoningEffort::XHigh), "xhigh");
        assert_eq!(effort(ReasoningEffort::Max), "max");
    }

    #[test]
    fn sampling_params_omitted_for_reasoning_models() {
        let build = |model: &str| {
            let mut request = CompletionRequest::new(model, vec![Message::user("hi")]);
            request.params.temperature = Temperature::new(0.5).ok();
            request.params.top_p = TopP::new(0.9).ok();
            request.params.user = Some("u1".into());
            let wire = WireChatCompletionRequest::from_request(&request).unwrap();
            serde_json::to_value(&wire).unwrap()
        };
        let legacy = build("gpt-4o");
        assert_eq!(legacy["temperature"], 0.5);
        assert!((legacy["top_p"].as_f64().unwrap() - 0.9).abs() < 1e-6);
        assert_eq!(legacy["user"], "u1");
        for model in ["gpt-5", "gpt-5.1", "o3", "o4-mini", "chatgpt-5"] {
            let json = build(model);
            assert!(
                json.get("temperature").is_none(),
                "{model} must not send temperature"
            );
            assert!(json.get("top_p").is_none(), "{model} must not send top_p");
            assert!(json.get("user").is_none(), "{model} must not send user");
        }
    }

    #[test]
    fn explicit_no_reasoning_omits_effort_field() {
        let mut request = CompletionRequest::new("gpt-4o", vec![Message::user("hi")]);
        request.params.reasoning_effort = Some(ReasoningEffort::None);
        let wire = WireChatCompletionRequest::from_request(&request).unwrap();
        let json = serde_json::to_value(&wire).unwrap();
        assert!(
            json.get("reasoning_effort").is_none(),
            "\"none\" is not a valid OpenAI reasoning_effort value; it must be omitted"
        );
    }

    #[test]
    fn strict_tool_passthrough() {
        let tool = ToolDefinition::Function {
            function: FunctionDefinition {
                name: "get_weather".into(),
                description: None,
                parameters: FunctionParameters::new(serde_json::json!({"type": "object"})),
                strict: Some(true),
            },
        };
        let request = CompletionRequest::new("gpt-4o", vec![Message::user("hi")]).with_params(
            CompletionParams::builder().tools(vec![tool]).build(),
        );
        let wire = WireChatCompletionRequest::from_request(&request).unwrap();
        let json = serde_json::to_value(&wire).unwrap();
        assert_eq!(json["tools"][0]["function"]["strict"], true);
    }

    #[test]
    fn multimodal_message_serialization() {
        let request = CompletionRequest::new(
            "gpt-4o",
            vec![Message::user(MessageContent::parts(vec![
                ContentPart::text("what is this"),
                ContentPart::image_url("https://example.com/a.png", ImageDetail::High),
                ContentPart::input_audio("AQID", AudioFormat::Wav),
            ]))],
        );
        let wire = WireChatCompletionRequest::from_request(&request).unwrap();
        let json = serde_json::to_value(&wire).unwrap();
        assert_eq!(json["messages"][0]["content"][0]["type"], "text");
        assert_eq!(json["messages"][0]["content"][1]["type"], "image_url");
        assert_eq!(json["messages"][0]["content"][1]["image_url"]["detail"], "high");
        assert_eq!(json["messages"][0]["content"][2]["type"], "input_audio");
        assert_eq!(json["messages"][0]["content"][2]["input_audio"]["format"], "wav");
    }

    #[test]
    fn tool_call_message_serialization() {
        let request = CompletionRequest::new(
            "gpt-4o",
            vec![Message::assistant_with_tool_calls(vec![crate::adapter::types::ToolCall {
                id: "call_1".into(),
                name: "get_weather".into(),
                arguments: serde_json::json!({"city": "beijing"}),
            }])],
        );
        let wire = WireChatCompletionRequest::from_request(&request).unwrap();
        let json = serde_json::to_value(&wire).unwrap();
        assert_eq!(json["messages"][0]["role"], "assistant");
        assert_eq!(json["messages"][0]["tool_calls"][0]["id"], "call_1");
        assert_eq!(json["messages"][0]["tool_calls"][0]["type"], "function");
        assert_eq!(json["messages"][0]["tool_calls"][0]["function"]["name"], "get_weather");
        assert_eq!(
            json["messages"][0]["tool_calls"][0]["function"]["arguments"],
            r#"{"city":"beijing"}"#
        );
    }

    #[test]
    fn chunk_deserialization() {
        let payload = r#"{"id":"chatcmpl-1","model":"gpt-4o","choices":[{"index":0,"delta":{"role":"assistant","content":"Hello","tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"get_weather","arguments":"{\"city\":"}}]},"finish_reason":null}],"usage":null}"#;
        let wire: WireChatCompletionChunk = serde_json::from_str(payload).unwrap();
        let chunk = wire.into_chunks().into_iter().next().unwrap();
        assert_eq!(chunk.content(), Some("Hello"));
        assert_eq!(chunk.delta.tool_calls[0].index, 0);
        assert_eq!(chunk.delta.tool_calls[0].id.as_deref(), Some("call_1"));
        assert_eq!(chunk.delta.tool_calls[0].name.as_deref(), Some("get_weather"));
        assert_eq!(
            chunk.delta.tool_calls[0].arguments_delta.as_deref(),
            Some("{\"city\":")
        );
        assert_eq!(chunk.request_id.as_deref(), Some("chatcmpl-1"));
        assert_eq!(chunk.finish_reason, None);
    }

    #[test]
    fn chunk_logprobs_deserialization() {
        let payload = r#"{"id":"chatcmpl-1","model":"gpt-4o","choices":[{"index":0,"delta":{"content":"hi","logprobs":{"content":[{"token":"hi","logprob":-0.3,"bytes":[104,105],"top_logprobs":[]}]}},"finish_reason":null}]}"#;
        let wire: WireChatCompletionChunk = serde_json::from_str(payload).unwrap();
        let chunk = wire.into_chunks().into_iter().next().unwrap();
        let logprobs = chunk.delta.logprobs.unwrap();
        assert_eq!(logprobs.tokens[0].token, "hi");
        assert_eq!(logprobs.tokens[0].logprob, -0.3);
    }

    #[test]
    fn model_list_deserialization() {
        let payload = r#"{"object":"list","data":[{"id":"gpt-4o","object":"model","created":1700000000,"owned_by":"openai"}]}"#;
        let list: WireModelList = serde_json::from_str(payload).unwrap();
        let info = ModelInfo::from(list.data.into_iter().next().unwrap());
        assert_eq!(info.id, "gpt-4o");
        assert_eq!(info.owned_by.as_deref(), Some("openai"));
        assert_eq!(info.created_at, Some(1700000000));
    }
}
