use async_trait::async_trait;

use crate::adapter::types::ToolCallDelta;
use crate::runtime::actor::{ActorContext, CognitiveActor};
use crate::runtime::event::{Event, EventKind, StateRequest, StateResponse};
use crate::runtime::types::{
    ActionCandidate, ActionDecision, DriveState, IntentKind, TaskSetState,
};

const GO_THRESHOLD: f32 = 0.4;

pub struct ActionSelectionActor {
    accumulated: String,
    reasoning_acc: String,
    tool_deltas: Vec<ToolCallDelta>,
    last_intent: IntentKind,
}

impl ActionSelectionActor {
    pub fn new() -> Self {
        Self {
            accumulated: String::new(),
            reasoning_acc: String::new(),
            tool_deltas: Vec::new(),
            last_intent: IntentKind::Statement,
        }
    }

    async fn select(&mut self, ctx: &mut ActorContext, meta: crate::runtime::event::EventMeta) -> Vec<Event> {
        let task_set = match ctx.request_state(StateRequest::TaskSet).await {
            Some(StateResponse::TaskSet(task_set)) => task_set,
            _ => TaskSetState {
                goal: String::new(),
                priority: 1.0,
                progress: 0.0,
            },
        };
        let drive = match ctx.request_state(StateRequest::Motivation).await {
            Some(StateResponse::Motivation(drive)) => drive,
            _ => DriveState::default(),
        };
        let serotonin = match ctx.request_state(StateRequest::Modulator).await {
            Some(StateResponse::Modulator(state)) => state.serotonin,
            _ => 0.5,
        };
        let dopamine = match ctx.request_state(StateRequest::Modulator).await {
            Some(StateResponse::Modulator(state)) => state.dopamine,
            _ => 0.5,
        };
        let emotion = match ctx.request_state(StateRequest::Emotion).await {
            Some(StateResponse::Emotion(emotion)) => emotion,
            _ => crate::runtime::types::EmotionState {
                valence: 0.0,
                arousal: 0.0,
            },
        };

        let go_threshold = (GO_THRESHOLD - (dopamine - 0.5) * 0.3).clamp(0.2, 0.6);

        let content_ready = !self.accumulated.trim().is_empty();
        let negative_valence = (-emotion.valence).max(0.0);
        let intent_boost = match self.last_intent {
            IntentKind::Question => 0.1,
            IntentKind::Command => 0.1,
            _ => 0.0,
        };

        if !self.tool_deltas.is_empty() {
            let mut decisions = Vec::new();
            for delta in std::mem::take(&mut self.tool_deltas) {
                let arguments = parse_arguments(delta.arguments_delta.as_deref());
                let reasoning = if self.reasoning_acc.trim().is_empty() {
                    None
                } else {
                    Some(self.reasoning_acc.clone())
                };
                decisions.push(Event::ActionSelected {
                    meta,
                    decision: ActionDecision {
                        candidate: ActionCandidate::CallTool {
                            name: delta.name.unwrap_or_default(),
                            arguments,
                            tool_call_id: delta.id,
                            reasoning,
                        },
                        confidence: 0.9,
                        go: true,
                    },
                });
            }
            return decisions;
        }

        if !content_ready {
            let score = 0.3
                + 0.5 * drive.curiosity
                + 0.2 * drive.salience
                + 0.2 * negative_valence;
            let decision = ActionDecision {
                candidate: ActionCandidate::AskClarification {
                    question: "I don't have enough context to answer accurately yet. Could you provide more details?"
                        .into(),
                },
                confidence: score.clamp(0.0, 1.0),
                go: score >= go_threshold,
            };
            return vec![Event::ActionSelected { meta, decision }];
        }

        let score = 0.5
            + 0.3 * task_set.progress
            + 0.1 * serotonin
            + 0.1 * drive.homeostatic
            + intent_boost;
        let decision = ActionDecision {
            candidate: ActionCandidate::Respond {
                content: self.accumulated.trim().to_string(),
            },
            confidence: score.clamp(0.0, 1.0),
            go: score >= go_threshold,
        };
        vec![Event::ActionSelected { meta, decision }]
    }
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

fn parse_arguments(delta: Option<&str>) -> serde_json::Value {
    match delta {
        Some(text) => serde_json::from_str(text)
            .unwrap_or_else(|_| serde_json::Value::String(text.into())),
        None => serde_json::Value::Null,
    }
}

impl Default for ActionSelectionActor {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl CognitiveActor for ActionSelectionActor {
    fn id(&self) -> &str {
        "action_selection"
    }

    fn subscriptions(&self) -> Vec<EventKind> {
        vec![EventKind::Generation, EventKind::Language]
    }

    async fn handle(&mut self, event: &Event, ctx: &mut ActorContext) -> Vec<Event> {
        match event {
            Event::LanguageInsight { intent, .. } => {
                self.last_intent = *intent;
                vec![]
            }
            Event::Chunk { chunk, .. } => {
                if let Some(content) = chunk.content() {
                    self.accumulated.push_str(content);
                }
                if let Some(reasoning) = &chunk.delta.reasoning {
                    self.reasoning_acc.push_str(reasoning);
                }
                for delta in &chunk.delta.tool_calls {
                    merge_tool_call_delta(&mut self.tool_deltas, delta);
                }
                vec![]
            }
            Event::StreamEnd { meta, .. } => {
                let events = self.select(ctx, *meta).await;
                self.accumulated.clear();
                self.reasoning_acc.clear();
                self.tool_deltas.clear();
                events
            }
            _ => vec![],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adapter::types::{ChunkDelta, CompletionChunk};
    use crate::runtime::actor::spawn_actor;
    use crate::runtime::bus::EventBus;
    use crate::runtime::event::EventMeta;
    use crate::runtime::types::{
        CycleId, IntentKind, ModulatorState, RpeSignal, TaskSetState,
    };
    use std::time::Duration;

    fn meta() -> EventMeta {
        EventMeta {
            cycle_id: CycleId(1),
            timestamp: 0,
        }
    }

    fn chunk_event(content: &str) -> Event {
        Event::Chunk {
            meta: meta(),
            chunk: CompletionChunk {
                model: "test".into(),
                index: 0,
                delta: ChunkDelta {
                    role: None,
                    content: Some(content.into()),
                    tool_calls: vec![],
                    logprobs: None,
                    reasoning: None,
                },
                finish_reason: None,
                usage: None,
                request_id: None,
            },
        }
    }

    async fn selected_decision(bus: &EventBus) -> ActionDecision {
        let mut rx = bus.subscribe();
        bus.publish(Event::StreamEnd {
            meta: meta(),
            usage: None,
        });
        tokio::time::timeout(Duration::from_secs(2), async {
            loop {
                match rx.recv().await.unwrap() {
                    Event::ActionSelected { decision, .. } => return decision,
                    _ => continue,
                }
            }
        })
        .await
        .unwrap()
    }

    async fn spawn_state_actors(bus: &EventBus) -> Vec<tokio::task::JoinHandle<()>> {
        let mut handles = Vec::new();
        let (h, ready) = spawn_actor(
            bus.clone(),
            crate::runtime::task_set::TaskSetActor::new(),
        );
        handles.push(h);
        ready.await.unwrap();
        let (h, ready) = spawn_actor(bus.clone(), crate::runtime::motivation::MotivationActor::new());
        handles.push(h);
        ready.await.unwrap();
        let (h, ready) = spawn_actor(bus.clone(), crate::runtime::modulator::ModulatorActor::new());
        handles.push(h);
        ready.await.unwrap();
        handles
    }

    fn seed_state(bus: &EventBus) {
        bus.publish(Event::TaskSetUpdate {
            meta: meta(),
            task_set: TaskSetState {
                goal: "answer".into(),
                priority: 1.0,
                progress: 0.5,
            },
        });
        bus.publish(Event::ModulatorUpdate {
            meta: meta(),
            state: ModulatorState {
                dopamine: 0.0,
                norepinephrine: 0.5,
                acetylcholine: 0.5,
                serotonin: 0.5,
            },
            mode: crate::runtime::types::CognitiveMode::Automatic,
        });
        bus.publish(Event::Rpe {
            meta: meta(),
            rpe: RpeSignal(0.3),
        });
    }

    #[tokio::test]
    async fn low_error_selects_respond_with_content() {
        let bus = EventBus::new(64);
        let _state = spawn_state_actors(&bus).await;
        let (_h, ready) = spawn_actor(bus.clone(), ActionSelectionActor::new());
        ready.await.unwrap();
        seed_state(&bus);
        tokio::time::sleep(Duration::from_millis(50)).await;

        bus.publish(chunk_event("the answer is here"));

        let decision = selected_decision(&bus).await;
        assert!(decision.go);
        match decision.candidate {
            ActionCandidate::Respond { content } => {
                assert_eq!(content, "the answer is here");
            }
            _ => panic!("expected respond action"),
        }
    }

    #[tokio::test]
    async fn uncertain_without_content_asks_clarification() {
        let bus = EventBus::new(64);
        let _state = spawn_state_actors(&bus).await;
        let (_h, ready) = spawn_actor(bus.clone(), ActionSelectionActor::new());
        ready.await.unwrap();
        seed_state(&bus);
        tokio::time::sleep(Duration::from_millis(50)).await;

        let decision = selected_decision(&bus).await;
        assert!(matches!(
            decision.candidate,
            ActionCandidate::AskClarification { .. }
        ));
    }

    #[tokio::test]
    async fn tool_calls_become_call_tool_candidate() {
        let bus = EventBus::new(64);
        let _state = spawn_state_actors(&bus).await;
        let (_h, ready) = spawn_actor(bus.clone(), ActionSelectionActor::new());
        ready.await.unwrap();
        seed_state(&bus);
        tokio::time::sleep(Duration::from_millis(50)).await;

        let first = CompletionChunk {
            model: "test".into(),
            index: 0,
            delta: crate::adapter::types::ChunkDelta {
                role: None,
                content: None,
                tool_calls: vec![crate::adapter::types::ToolCallDelta {
                    index: 0,
                    id: Some("call_1".into()),
                    name: Some("time".into()),
                    arguments_delta: Some("{}".into()),
                }],
                logprobs: None,
                reasoning: None,
            },
            finish_reason: None,
            usage: None,
            request_id: None,
        };
        bus.publish(Event::Chunk {
            meta: meta(),
            chunk: first,
        });

        let decision = selected_decision(&bus).await;
        match decision.candidate {
            ActionCandidate::CallTool {
                name,
                arguments,
                tool_call_id,
                ..
            } => {
                assert_eq!(name, "time");
                assert_eq!(arguments, serde_json::json!({}));
                assert_eq!(tool_call_id.as_deref(), Some("call_1"));
            }
            _ => panic!("expected call tool action"),
        }
    }

    #[tokio::test]
    async fn parallel_tool_deltas_emit_one_action_each() {
        let bus = EventBus::new(64);
        let _state = spawn_state_actors(&bus).await;
        let (_h, ready) = spawn_actor(bus.clone(), ActionSelectionActor::new());
        ready.await.unwrap();
        seed_state(&bus);
        tokio::time::sleep(Duration::from_millis(50)).await;

        let make = |index: u32, id: &str, name: &str| CompletionChunk {
            model: "test".into(),
            index: 0,
            delta: crate::adapter::types::ChunkDelta {
                role: None,
                content: None,
                tool_calls: vec![crate::adapter::types::ToolCallDelta {
                    index,
                    id: Some(id.into()),
                    name: Some(name.into()),
                    arguments_delta: Some("{}".into()),
                }],
                logprobs: None,
                reasoning: None,
            },
            finish_reason: None,
            usage: None,
            request_id: None,
        };
        bus.publish(Event::Chunk {
            meta: meta(),
            chunk: make(0, "call_a", "ls"),
        });
        bus.publish(Event::Chunk {
            meta: meta(),
            chunk: make(1, "call_b", "read_file"),
        });
        bus.publish(Event::StreamEnd {
            meta: meta(),
            usage: None,
        });

        let mut rx = bus.subscribe();
        let mut names = Vec::new();
        tokio::time::timeout(Duration::from_secs(2), async {
            loop {
                match rx.recv().await.unwrap() {
                    Event::ActionSelected { decision, .. } => {
                        if let ActionCandidate::CallTool { name, .. } = &decision.candidate {
                            names.push(name.clone());
                            if names.len() == 2 {
                                break;
                            }
                        }
                    }
                    _ => continue,
                }
            }
        })
        .await
        .expect("two actions must be emitted");
        assert_eq!(names, vec!["ls".to_string(), "read_file".to_string()]);
    }

    #[tokio::test]
    async fn negative_valence_boosts_clarification_confidence() {
        let bus = EventBus::new(64);
        let _state = spawn_state_actors(&bus).await;
        let (_h, ready) = spawn_actor(bus.clone(), ActionSelectionActor::new());
        ready.await.unwrap();
        seed_state(&bus);
        tokio::time::sleep(Duration::from_millis(50)).await;

        let baseline = selected_decision(&bus).await;

        let (_em, em_ready) = spawn_actor(
            bus.clone(),
            crate::runtime::emotion::EmotionActor::new(),
        );
        em_ready.await.unwrap();
        for _ in 0..8 {
            bus.publish(Event::Rpe {
                meta: meta(),
                rpe: RpeSignal(-0.9),
            });
        }
        let boosted = selected_decision(&bus).await;

        assert!(matches!(
            boosted.candidate,
            ActionCandidate::AskClarification { .. }
        ));
        assert!(
            boosted.confidence > baseline.confidence,
            "negative valence should boost clarification confidence"
        );
    }

    #[tokio::test]
    async fn question_intent_boosts_respond_confidence() {
        let bus = EventBus::new(64);
        let _state = spawn_state_actors(&bus).await;
        let (_h, ready) = spawn_actor(bus.clone(), ActionSelectionActor::new());
        ready.await.unwrap();
        seed_state(&bus);
        tokio::time::sleep(Duration::from_millis(50)).await;

        bus.publish(chunk_event("the answer is 42"));
        let baseline = selected_decision(&bus).await;

        bus.publish(Event::LanguageInsight {
            meta: meta(),
            intent: IntentKind::Question,
            quality: 0.0,
        });
        bus.publish(chunk_event("the answer is 42"));
        let boosted = selected_decision(&bus).await;

        assert!(matches!(boosted.candidate, ActionCandidate::Respond { .. }));
        assert!(
            boosted.confidence > baseline.confidence,
            "question intent should boost respond confidence"
        );
    }
}
