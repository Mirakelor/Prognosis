use serde_json::Value;

use crate::adapter::types::{Message, ReasoningEffort, Temperature, ToolDefinition, TopP};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CycleId(pub u64);

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ModulatorState {
    pub dopamine: f32,
    pub norepinephrine: f32,
    pub acetylcholine: f32,
    pub serotonin: f32,
}

impl Default for ModulatorState {
    fn default() -> Self {
        Self {
            dopamine: 0.0,
            norepinephrine: 0.5,
            acetylcholine: 0.5,
            serotonin: 0.5,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CognitiveMode {
    Automatic,
    Controlled,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct ModulationContext {
    pub temperature: Option<Temperature>,
    pub top_p: Option<TopP>,
    pub reasoning_effort: Option<ReasoningEffort>,
    pub n: Option<u32>,
    pub model: Option<String>,
    pub injected_messages: Vec<Message>,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct GenerateRequest {
    pub messages: Vec<Message>,
    pub modulation: ModulationContext,
    pub tools: Option<Vec<ToolDefinition>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PerceptionSource {
    User,
    ToolResult,
    System,
    Internal,
    Scheduled,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntentKind {
    Question,
    Command,
    Statement,
    Smalltalk,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PerceptionPayload {
    pub source: PerceptionSource,
    pub content: String,
    pub salience: f32,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct PerceptionFeatures {
    pub topic_hints: Vec<String>,
    pub emotional_tone: f32,
    pub length: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PredictionTrajectory {
    pub topics: Vec<String>,
    pub key_elements: Vec<String>,
    pub direction: f32,
    pub intent: IntentKind,
    pub intent_candidates: Vec<IntentKind>,
    pub reaction: String,
    pub reaction_sentiment: f32,
}

impl Default for PredictionTrajectory {
    fn default() -> Self {
        Self {
            topics: Vec::new(),
            key_elements: Vec::new(),
            direction: 0.0,
            intent: IntentKind::Statement,
            intent_candidates: Vec::new(),
            reaction: String::new(),
            reaction_sentiment: 0.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PredictionError {
    pub semantic: f32,
    pub confidence: f32,
    pub precision: f32,
}

impl PredictionError {
    pub fn weighted(&self) -> f32 {
        (self.semantic + self.confidence) * self.precision
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RpeSignal(pub f32);

#[derive(Debug, Clone, PartialEq)]
pub struct AttentionFocus {
    pub payload: PerceptionPayload,
    pub salience: f32,
    pub relevance: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct InhibitionSignal {
    pub targets: Vec<String>,
    pub strength: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EmotionState {
    pub valence: f32,
    pub arousal: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MetaState {
    pub uncertainty: f32,
    pub conflict: f32,
    pub confidence: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct DriveState {
    pub homeostatic: f32,
    pub curiosity: f32,
    pub salience: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TaskSetState {
    pub goal: String,
    pub priority: f32,
    pub progress: f32,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct DialogueTurn {
    pub user: String,
    pub assistant: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct WorkingMemorySlot {
    pub id: u64,
    pub content: String,
    pub source: String,
    pub activation: f32,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct WorkingMemorySnapshot {
    pub slots: Vec<WorkingMemorySlot>,
    pub dialogue: Vec<DialogueTurn>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RuleContext {
    pub name: String,
    pub rule: String,
    pub description: String,
    pub globs: String,
    pub regex: String,
    pub always_apply: Option<bool>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SkillContext {
    pub name: String,
    pub description: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EpisodicMemory {
    pub id: u64,
    pub cycle_id: CycleId,
    pub summary: String,
    pub emotion: EmotionState,
    pub strength: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SemanticMemory {
    pub id: u64,
    pub content: String,
    pub strength: f32,
    pub belief: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemoryKind {
    Episodic,
    Semantic,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MemoryRetrieval {
    pub episodic: Vec<EpisodicMemory>,
    pub semantic: Vec<SemanticMemory>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ActionCandidate {
    Respond {
        content: String,
    },
    CallTool {
        name: String,
        arguments: Value,
        tool_call_id: Option<String>,
        reasoning: Option<String>,
    },
    AskClarification {
        question: String,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct ActionDecision {
    pub candidate: ActionCandidate,
    pub confidence: f32,
    pub go: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ToolResult {
    pub name: String,
    pub output: String,
    pub tool_call_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct CycleSummary {
    pub rpe: Option<f32>,
    pub error: Option<f32>,
    pub uncertainty: Option<f32>,
    pub decision: Option<ActionDecision>,
    pub modulation: Option<ModulationContext>,
    pub user_input: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TraceRecord {
    pub cycle_id: CycleId,
    pub modulation: Option<ModulationContext>,
    pub error_before: Option<f32>,
    pub error_after: Option<f32>,
    pub decision: Option<ActionDecision>,
    pub retrieval: Option<String>,
    pub memory_writes: Vec<String>,
    pub prediction_direction: Option<f32>,
    pub prediction_sentiment: Option<f32>,
    pub prediction_reaction: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prediction_error_weighting() {
        let error = PredictionError {
            semantic: 0.3,
            confidence: 0.2,
            precision: 2.0,
        };
        assert_eq!(error.weighted(), 1.0);
    }

    #[test]
    fn modulation_context_defaults() {
        let ctx = ModulationContext::default();
        assert!(ctx.temperature.is_none());
        assert!(ctx.injected_messages.is_empty());
    }

    #[test]
    fn modulator_state_defaults() {
        let state = ModulatorState::default();
        assert_eq!(state.dopamine, 0.0);
        assert_eq!(state.norepinephrine, 0.5);
    }
}
