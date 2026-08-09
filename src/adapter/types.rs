use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

impl std::fmt::Display for Role {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::Tool => "tool",
        };
        f.write_str(s)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[derive(Default)]
pub enum ImageDetail {
    Low,
    High,
    #[default]
    Auto,
}


#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AudioFormat {
    Wav,
    Mp3,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart {
    Text {
        text: String,
    },
    ImageUrl {
        url: String,
        #[serde(default)]
        detail: ImageDetail,
    },
    InputAudio {
        data: String,
        format: AudioFormat,
    },
}

impl ContentPart {
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text { text: text.into() }
    }
    pub fn image_url(url: impl Into<String>, detail: ImageDetail) -> Self {
        Self::ImageUrl {
            url: url.into(),
            detail,
        }
    }
    pub fn input_audio(data: impl Into<String>, format: AudioFormat) -> Self {
        Self::InputAudio {
            data: data.into(),
            format,
        }
    }
    pub fn as_text(&self) -> Option<&str> {
        match self {
            Self::Text { text } => Some(text),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

impl MessageContent {
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text(text.into())
    }
    pub fn parts(parts: Vec<ContentPart>) -> Self {
        Self::Parts(parts)
    }
    pub fn to_plain_text(&self) -> String {
        match self {
            Self::Text(text) => text.clone(),
            Self::Parts(parts) => parts
                .iter()
                .filter_map(|part| part.as_text())
                .collect::<Vec<_>>()
                .join("\n"),
        }
    }
    pub fn is_empty(&self) -> bool {
        match self {
            Self::Text(text) => text.trim().is_empty(),
            Self::Parts(parts) => parts.is_empty(),
        }
    }
}

impl From<String> for MessageContent {
    fn from(value: String) -> Self {
        Self::Text(value)
    }
}

impl From<&str> for MessageContent {
    fn from(value: &str) -> Self {
        Self::Text(value.to_string())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FunctionParameters {
    #[serde(flatten)]
    pub schema: Value,
}

impl FunctionParameters {
    pub fn new(schema: Value) -> Self {
        Self { schema }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ReasoningEffort {
    None,
    Minimal,
    Low,
    Medium,
    High,
    XHigh,
    Max,
}

impl ReasoningEffort {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Minimal => "minimal",
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::XHigh => "xhigh",
            Self::Max => "max",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FunctionDefinition {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub parameters: FunctionParameters,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolDefinition {
    Function {
        function: FunctionDefinition,
    },
}

impl ToolDefinition {
    pub fn function(
        name: impl Into<String>,
        description: Option<String>,
        parameters: Value,
    ) -> Self {
        Self::Function {
            function: FunctionDefinition {
                name: name.into(),
                description,
                parameters: FunctionParameters::new(parameters),
                strict: None,
            },
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: Value,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolCallDelta {
    pub index: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arguments_delta: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: MessageContent,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<String>,
}

impl Message {
    pub fn new(role: Role, content: impl Into<MessageContent>) -> Self {
        Self {
            role,
            content: content.into(),
            name: None,
            tool_calls: None,
            tool_call_id: None,
            reasoning: None,
        }
    }

    pub fn system(content: impl Into<MessageContent>) -> Self {
        Self::new(Role::System, content)
    }

    pub fn user(content: impl Into<MessageContent>) -> Self {
        Self::new(Role::User, content)
    }

    pub fn assistant(content: impl Into<MessageContent>) -> Self {
        Self::new(Role::Assistant, content)
    }

    pub fn tool(tool_call_id: impl Into<String>, content: impl Into<MessageContent>) -> Self {
        Self {
            role: Role::Tool,
            content: content.into(),
            name: None,
            tool_calls: None,
            tool_call_id: Some(tool_call_id.into()),
            reasoning: None,
        }
    }

    pub fn assistant_with_tool_calls(tool_calls: Vec<ToolCall>) -> Self {
        Self::assistant_with_tool_calls_and_reasoning(String::new(), tool_calls, None)
    }

    pub fn assistant_with_tool_calls_and_reasoning(
        content: String,
        tool_calls: Vec<ToolCall>,
        reasoning: Option<String>,
    ) -> Self {
        Self {
            role: Role::Assistant,
            content: MessageContent::Text(content),
            name: None,
            tool_calls: Some(tool_calls),
            tool_call_id: None,
            reasoning,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Temperature(f32);

impl Temperature {
    pub const MIN: f32 = 0.0;
    pub const MAX: f32 = 2.0;

    pub fn new(value: f32) -> Result<Self, crate::adapter::error::AdapterError> {
        if value.is_finite() && (Self::MIN..=Self::MAX).contains(&value) {
            Ok(Self(value))
        } else {
            Err(crate::adapter::error::AdapterError::invalid_request(
                "adapter",
                format!(
                    "temperature must be finite and within [{}, {}], got {value}",
                    Self::MIN,
                    Self::MAX
                ),
            ))
        }
    }

    pub fn get(self) -> f32 {
        self.0
    }
}

impl Default for Temperature {
    fn default() -> Self {
        Self(0.7)
    }
}

impl TryFrom<f32> for Temperature {
    type Error = crate::adapter::error::AdapterError;

    fn try_from(value: f32) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

impl std::fmt::Display for Temperature {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:.2}", self.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct TopP(f32);

impl TopP {
    pub const MIN: f32 = 0.0;
    pub const MAX: f32 = 1.0;

    pub fn new(value: f32) -> Result<Self, crate::adapter::error::AdapterError> {
        if value.is_finite() && (Self::MIN..=Self::MAX).contains(&value) {
            Ok(Self(value))
        } else {
            Err(crate::adapter::error::AdapterError::invalid_request(
                "adapter",
                format!(
                    "top_p must be finite and within [{}, {}], got {value}",
                    Self::MIN,
                    Self::MAX
                ),
            ))
        }
    }

    pub fn get(self) -> f32 {
        self.0
    }
}

impl Default for TopP {
    fn default() -> Self {
        Self(1.0)
    }
}

impl TryFrom<f32> for TopP {
    type Error = crate::adapter::error::AdapterError;

    fn try_from(value: f32) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

impl std::fmt::Display for TopP {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:.2}", self.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Penalty(f32);

impl Penalty {
    pub const MIN: f32 = -2.0;
    pub const MAX: f32 = 2.0;

    pub fn new(value: f32) -> Result<Self, crate::adapter::error::AdapterError> {
        if value.is_finite() && (Self::MIN..=Self::MAX).contains(&value) {
            Ok(Self(value))
        } else {
            Err(crate::adapter::error::AdapterError::invalid_request(
                "adapter",
                format!(
                    "penalty must be finite and within [{}, {}], got {value}",
                    Self::MIN,
                    Self::MAX
                ),
            ))
        }
    }

    pub fn get(self) -> f32 {
        self.0
    }
}

impl Default for Penalty {
    fn default() -> Self {
        Self(0.0)
    }
}

impl TryFrom<f32> for Penalty {
    type Error = crate::adapter::error::AdapterError;

    fn try_from(value: f32) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[derive(Default)]
pub enum ResponseFormat {
    #[default]
    Text,
    JsonObject,
    JsonSchema {
        name: String,
        schema: Value,
        #[serde(skip_serializing_if = "Option::is_none")]
        strict: Option<bool>,
    },
}


#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolChoice {
    Auto,
    None,
    Required,
    Named {
        kind: String,
        function: NamedFunction,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NamedFunction {
    pub name: String,
}

impl ToolChoice {
    pub fn named(name: impl Into<String>) -> Self {
        Self::Named {
            kind: "function".to_string(),
            function: NamedFunction { name: name.into() },
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
#[derive(Default)]
pub struct CompletionParams {
    pub temperature: Option<Temperature>,
    pub top_p: Option<TopP>,
    pub max_tokens: Option<u32>,
    pub seed: Option<u64>,
    pub stop: Option<Vec<String>>,
    pub presence_penalty: Option<Penalty>,
    pub frequency_penalty: Option<Penalty>,
    pub user: Option<String>,
    pub response_format: Option<ResponseFormat>,
    pub tools: Option<Vec<ToolDefinition>>,
    pub tool_choice: Option<ToolChoice>,
    pub logprobs: Option<bool>,
    pub top_logprobs: Option<u8>,
    pub reasoning_effort: Option<ReasoningEffort>,
    pub parallel_tool_calls: Option<bool>,
    pub n: Option<u32>,
}


impl CompletionParams {
    pub fn builder() -> CompletionParamsBuilder {
        CompletionParamsBuilder::default()
    }
}

#[derive(Debug, Default)]
pub struct CompletionParamsBuilder {
    inner: CompletionParams,
}

impl CompletionParamsBuilder {
    pub fn temperature(mut self, value: f32) -> Result<Self, crate::adapter::error::AdapterError> {
        self.inner.temperature = Some(Temperature::new(value)?);
        Ok(self)
    }

    pub fn max_tokens(mut self, value: u32) -> Self {
        self.inner.max_tokens = Some(value);
        self
    }

    pub fn user(mut self, value: impl Into<String>) -> Self {
        self.inner.user = Some(value.into());
        self
    }

    pub fn response_format(mut self, value: ResponseFormat) -> Self {
        self.inner.response_format = Some(value);
        self
    }

    pub fn tools(mut self, value: Vec<ToolDefinition>) -> Self {
        self.inner.tools = Some(value);
        self
    }

    pub fn tool_choice(mut self, value: ToolChoice) -> Self {
        self.inner.tool_choice = Some(value);
        self
    }

    pub fn logprobs(mut self, enabled: bool, top_logprobs: Option<u8>) -> Self {
        self.inner.logprobs = Some(enabled);
        self.inner.top_logprobs = top_logprobs;
        self
    }

    pub fn reasoning_effort(mut self, value: ReasoningEffort) -> Self {
        self.inner.reasoning_effort = Some(value);
        self
    }

    pub fn parallel_tool_calls(mut self, enabled: bool) -> Self {
        self.inner.parallel_tool_calls = Some(enabled);
        self
    }

    pub fn n(mut self, value: u32) -> Self {
        self.inner.n = Some(value);
        self
    }

    pub fn build(self) -> CompletionParams {
        self.inner
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CompletionRequest {
    pub model: String,
    pub messages: Vec<Message>,
    pub params: CompletionParams,
}

impl CompletionRequest {
    pub fn new(model: impl Into<String>, messages: Vec<Message>) -> Self {
        Self {
            model: model.into(),
            messages,
            params: CompletionParams::default(),
        }
    }

    pub fn with_params(mut self, params: CompletionParams) -> Self {
        self.params = params;
        self
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[derive(Default)]
pub enum FinishReason {
    Stop,
    Length,
    ContentFilter,
    ToolCalls,
    FunctionCall,
    #[default]
    Unknown,
}


#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

impl TokenUsage {
    pub fn new(prompt_tokens: u32, completion_tokens: u32) -> Self {
        Self {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens.saturating_add(completion_tokens),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LogprobToken {
    pub token: String,
    pub logprob: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContentLogprobs {
    pub tokens: Vec<LogprobToken>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CompletionChoice {
    pub content: String,
    pub tool_calls: Vec<ToolCall>,
    pub finish_reason: FinishReason,
    pub logprobs: Option<ContentLogprobs>,
    pub reasoning: Option<String>,
}

impl CompletionChoice {
    pub fn has_tool_calls(&self) -> bool {
        !self.tool_calls.is_empty()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CompletionResponse {
    pub model: String,
    pub choices: Vec<CompletionChoice>,
    pub usage: Option<TokenUsage>,
    pub request_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct ChunkDelta {
    pub role: Option<Role>,
    pub content: Option<String>,
    pub tool_calls: Vec<ToolCallDelta>,
    pub logprobs: Option<ContentLogprobs>,
    pub reasoning: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CompletionChunk {
    pub model: String,
    pub index: u32,
    pub delta: ChunkDelta,
    pub finish_reason: Option<FinishReason>,
    pub usage: Option<TokenUsage>,
    pub request_id: Option<String>,
}

impl CompletionChunk {
    pub fn content(&self) -> Option<&str> {
        self.delta.content.as_deref()
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelInfo {
    pub id: String,
    pub object: Option<String>,
    pub owned_by: Option<String>,
    pub created_at: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AdapterCapabilities {
    pub streaming: bool,
    pub model_listing: bool,
    pub model_retrieval: bool,
    pub tool_calling: bool,
    pub json_mode: bool,
    pub json_schema: bool,
    pub logprobs: bool,
    pub multimodal_text: bool,
    pub multimodal_image: bool,
    pub multimodal_audio: bool,
    pub usage_in_stream: bool,
}

impl Default for AdapterCapabilities {
    fn default() -> Self {
        Self {
            streaming: false,
            model_listing: false,
            model_retrieval: false,
            tool_calling: false,
            json_mode: false,
            json_schema: false,
            logprobs: false,
            multimodal_text: true,
            multimodal_image: false,
            multimodal_audio: false,
            usage_in_stream: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn message_content_plain_text() {
        let content = MessageContent::parts(vec![
            ContentPart::text("hello"),
            ContentPart::text("world"),
        ]);
        assert_eq!(content.to_plain_text(), "hello\nworld");
    }

    #[test]
    fn temperature_validation() {
        assert!(Temperature::new(0.7).is_ok());
        assert!(Temperature::new(2.1).is_err());
        assert!(Temperature::new(f32::NAN).is_err());
    }

    #[test]
    fn top_p_validation() {
        assert!(TopP::new(1.0).is_ok());
        assert!(TopP::new(1.1).is_err());
    }

    #[test]
    fn penalty_validation() {
        assert!(Penalty::new(0.0).is_ok());
        assert!(Penalty::new(-2.0).is_ok());
        assert!(Penalty::new(2.0).is_ok());
        assert!(Penalty::new(2.1).is_err());
    }

    #[test]
    fn tool_choice_serialization() {
        let choice = ToolChoice::named("get_weather");
        let json = serde_json::to_value(&choice).unwrap();
        assert_eq!(json["type"], "named");
        assert_eq!(json["function"]["name"], "get_weather");
    }

    #[test]
    fn token_usage_total() {
        let usage = TokenUsage::new(10, 20);
        assert_eq!(usage.total_tokens, 30);
    }
}
