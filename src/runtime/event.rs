use crate::adapter::types::{CompletionChunk, TokenUsage};
use crate::runtime::types::{
    ActionDecision, AttentionFocus, CognitiveMode, CycleId, CycleSummary, DialogueTurn,
    DriveState, EmotionState, GenerateRequest, InhibitionSignal, IntentKind, MemoryKind,
    MemoryRetrieval, MetaState, ModulationContext, ModulatorState, PerceptionFeatures,
    PerceptionPayload, PredictionError, PredictionTrajectory, RpeSignal, RuleContext,
    SkillContext, TaskSetState, ToolResult, WorkingMemorySnapshot,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EventMeta {
    pub cycle_id: CycleId,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EventKind {
    Perception,
    Sensed,
    Attention,
    Inhibition,
    Prediction,
    Generation,
    Error,
    Modulation,
    WorkingMemory,
    Memory,
    Drive,
    TaskSet,
    Action,
    Cycle,
    Time,
    Language,
    Context,
    State,
}

impl Event {
    pub fn kind(&self) -> EventKind {
        match self {
            Event::Perception { .. } => EventKind::Perception,
            Event::Sensed { .. } => EventKind::Sensed,
            Event::Attention { .. } => EventKind::Attention,
            Event::Inhibition { .. } => EventKind::Inhibition,
            Event::Prediction { .. } => EventKind::Prediction,
            Event::Chunk { .. }
            | Event::StreamEnd { .. }
            | Event::Generate { .. }
            | Event::CancelGeneration { .. }
            | Event::GenerationError { .. } => EventKind::Generation,
            Event::ErrorComputed { .. } | Event::Rpe { .. } => EventKind::Error,
            Event::ModulatorUpdate { .. }
            | Event::EmotionUpdate { .. }
            | Event::MetaUpdate { .. }
            | Event::Modulate { .. } => EventKind::Modulation,
            Event::WorkingMemoryUpdate { .. } | Event::RestoreDialogue { .. } => EventKind::WorkingMemory,
            Event::MemoryRetrieved { .. } | Event::MemoryWrite { .. } => EventKind::Memory,
            Event::DriveUpdate { .. } => EventKind::Drive,
            Event::TaskSetUpdate { .. } => EventKind::TaskSet,
            Event::ActionSelected { .. } | Event::ToolResult { .. } => EventKind::Action,
            Event::CycleStart { .. } | Event::CycleComplete { .. } => EventKind::Cycle,
            Event::Tick { .. } => EventKind::Time,
            Event::LanguageInsight { .. } => EventKind::Language,
            Event::CompactContext { .. } | Event::ContextUpdate { .. } | Event::ConversationCleared { .. } => EventKind::Context,
            Event::RequestState { .. } | Event::StateResponse { .. } => EventKind::State,
            Event::Shutdown => EventKind::State,
        }
    }

    pub fn meta(&self) -> Option<EventMeta> {
        match self {
            Event::Perception { meta, .. }
            | Event::Sensed { meta, .. }
            | Event::Attention { meta, .. }
            | Event::Inhibition { meta, .. }
            | Event::Prediction { meta, .. }
            | Event::Chunk { meta, .. }
            | Event::StreamEnd { meta, .. }
            | Event::Generate { meta, .. }
            | Event::Modulate { meta, .. }
            | Event::CancelGeneration { meta }
            | Event::GenerationError { meta, .. }
            | Event::ErrorComputed { meta, .. }
            | Event::Rpe { meta, .. }
            | Event::ModulatorUpdate { meta, .. }
            | Event::EmotionUpdate { meta, .. }
            | Event::MetaUpdate { meta, .. }
            | Event::WorkingMemoryUpdate { meta, .. }
            | Event::RestoreDialogue { meta, .. }
            | Event::MemoryRetrieved { meta, .. }
            | Event::MemoryWrite { meta, .. }
            | Event::DriveUpdate { meta, .. }
            | Event::TaskSetUpdate { meta, .. }
            | Event::ActionSelected { meta, .. }
            | Event::ToolResult { meta, .. }
            | Event::LanguageInsight { meta, .. }
            | Event::CompactContext { meta, .. }
            | Event::ContextUpdate { meta, .. }
            | Event::ConversationCleared { meta }
            | Event::CycleStart { meta }
            | Event::CycleComplete { meta, .. }
            | Event::Tick { meta }
            | Event::RequestState { meta, .. }
            | Event::StateResponse { meta, .. } => Some(*meta),
            Event::Shutdown => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Event {
    Perception {
        meta: EventMeta,
        payload: PerceptionPayload,
    },
    Sensed {
        meta: EventMeta,
        payload: PerceptionPayload,
        features: PerceptionFeatures,
        salience: f32,
    },
    Attention {
        meta: EventMeta,
        focus: AttentionFocus,
    },
    Inhibition {
        meta: EventMeta,
        signal: InhibitionSignal,
    },
    Prediction {
        meta: EventMeta,
        trajectory: PredictionTrajectory,
    },
    Chunk {
        meta: EventMeta,
        chunk: CompletionChunk,
    },
    StreamEnd {
        meta: EventMeta,
        usage: Option<TokenUsage>,
    },
    Generate {
        meta: EventMeta,
        request: GenerateRequest,
    },
    Modulate {
        meta: EventMeta,
        modulation: ModulationContext,
    },
    CancelGeneration {
        meta: EventMeta,
    },
    GenerationError {
        meta: EventMeta,
        error: String,
    },
    ErrorComputed {
        meta: EventMeta,
        error: PredictionError,
    },
    Rpe {
        meta: EventMeta,
        rpe: RpeSignal,
    },
    ModulatorUpdate {
        meta: EventMeta,
        state: ModulatorState,
        mode: CognitiveMode,
    },
    EmotionUpdate {
        meta: EventMeta,
        emotion: EmotionState,
    },
    MetaUpdate {
        meta: EventMeta,
        meta_state: MetaState,
    },
    WorkingMemoryUpdate {
        meta: EventMeta,
        snapshot: WorkingMemorySnapshot,
    },
    RestoreDialogue {
        meta: EventMeta,
        turns: Vec<DialogueTurn>,
    },
    MemoryRetrieved {
        meta: EventMeta,
        retrieval: MemoryRetrieval,
    },
    MemoryWrite {
        meta: EventMeta,
        kind: MemoryKind,
        content: String,
        strength: f32,
    },
    DriveUpdate {
        meta: EventMeta,
        drives: DriveState,
    },
    TaskSetUpdate {
        meta: EventMeta,
        task_set: TaskSetState,
    },
    ActionSelected {
        meta: EventMeta,
        decision: ActionDecision,
    },
    ToolResult {
        meta: EventMeta,
        result: ToolResult,
        verdict: Option<String>,
    },
    LanguageInsight {
        meta: EventMeta,
        intent: IntentKind,
        quality: f32,
    },
    CompactContext {
        meta: EventMeta,
        summary: String,
    },
    ConversationCleared {
        meta: EventMeta,
    },
    ContextUpdate {
        meta: EventMeta,
        rules: Vec<RuleContext>,
        skills: Vec<SkillContext>,
    },
    CycleStart {
        meta: EventMeta,
    },
    CycleComplete {
        meta: EventMeta,
        summary: CycleSummary,
    },
    Tick {
        meta: EventMeta,
    },
    RequestState {
        meta: EventMeta,
        request: StateRequest,
        correlation_id: u64,
    },
    StateResponse {
        meta: EventMeta,
        response: StateResponse,
        correlation_id: u64,
    },
    Shutdown,
}

#[derive(Debug, Clone, PartialEq)]
pub enum StateRequest {
    TaskSet,
    Modulator,
    Emotion,
    Meta,
    WorkingMemory,
    Prediction,
    Motivation,
    MemoryRetrieval {
        query: String,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub enum StateResponse {
    TaskSet(TaskSetState),
    Modulator(ModulatorState),
    Emotion(EmotionState),
    Meta(MetaState),
    WorkingMemory(WorkingMemorySnapshot),
    Prediction(PredictionTrajectory),
    Motivation(DriveState),
    MemoryRetrieval(MemoryRetrieval),
}
