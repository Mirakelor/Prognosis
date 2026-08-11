use async_trait::async_trait;

use crate::runtime::actor::{ActorContext, CognitiveActor};
use crate::runtime::event::{Event, EventKind, StateRequest, StateResponse};
use crate::runtime::types::{
    AttentionFocus, DialogueTurn, PerceptionSource, WorkingMemorySlot, WorkingMemorySnapshot,
};

const CAPACITY: usize = 5;
const RPE_GATE_THRESHOLD: f32 = 0.3;
const ACTIVATION_DECAY: f32 = 0.9;
const MAX_DIALOGUE: usize = 200;

pub struct WorkingMemoryActor {
    slots: Vec<WorkingMemorySlot>,
    next_id: u64,
    pending: Option<AttentionFocus>,
    pending_gated: Option<AttentionFocus>,
    pending_user: Option<String>,
    dialogue: Vec<DialogueTurn>,
}

impl WorkingMemoryActor {
    pub fn new() -> Self {
        Self {
            slots: Vec::with_capacity(CAPACITY),
            next_id: 0,
            pending: None,
            pending_gated: None,
            pending_user: None,
            dialogue: Vec::new(),
        }
    }

    pub fn snapshot(&self) -> WorkingMemorySnapshot {
        WorkingMemorySnapshot {
            slots: self.slots.clone(),
            dialogue: self.dialogue.clone(),
        }
    }

    fn gate_open(rpe: f32) -> bool {
        rpe.abs() >= RPE_GATE_THRESHOLD
    }

    fn decay(&mut self) {
        for slot in &mut self.slots {
            slot.activation *= ACTIVATION_DECAY;
        }
    }

    fn insert(&mut self, focus: AttentionFocus) {
        let slot = WorkingMemorySlot {
            id: self.next_id,
            content: focus.payload.content.clone(),
            source: format!("{:?}", focus.payload.source),
            activation: 1.0,
        };
        self.next_id += 1;
        if self.slots.len() < CAPACITY {
            self.slots.push(slot);
        } else if let Some(min_idx) = self
            .slots
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.activation.total_cmp(&b.1.activation))
            .map(|(idx, _)| idx)
        {
            self.slots[min_idx] = slot;
        }
    }
}

impl Default for WorkingMemoryActor {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl CognitiveActor for WorkingMemoryActor {
    fn id(&self) -> &str {
        "working_memory"
    }

    fn subscriptions(&self) -> Vec<EventKind> {
        vec![
            EventKind::Attention,
            EventKind::Error,
            EventKind::State,
            EventKind::Time,
            EventKind::Cycle,
            EventKind::Context,
            EventKind::WorkingMemory,
        ]
    }

    async fn handle(&mut self, event: &Event, _ctx: &mut ActorContext) -> Vec<Event> {
        match event {
            Event::Attention { focus, .. } => {
                self.pending = Some(focus.clone());
                if focus.payload.source == PerceptionSource::User {
                    self.pending_user = Some(focus.payload.content.clone());
                }
                vec![]
            }
            Event::CompactContext { meta, .. } | Event::ConversationCleared { meta } => {
                self.dialogue.clear();
                vec![Event::WorkingMemoryUpdate {
                    meta: *meta,
                    snapshot: self.snapshot(),
                }]
            }
            Event::RestoreDialogue { meta, turns, .. } => {
                self.dialogue = turns.clone();
                if self.dialogue.len() > MAX_DIALOGUE {
                    self.dialogue.drain(..self.dialogue.len() - MAX_DIALOGUE);
                }
                vec![Event::WorkingMemoryUpdate {
                    meta: *meta,
                    snapshot: self.snapshot(),
                }]
            }
            Event::CycleComplete { meta, summary } => {
                self.pending_gated = self.pending.take();
                if let (Some(user), Some(decision)) =
                    (&self.pending_user, &summary.decision)
                {
                    let assistant = match &decision.candidate {
                        crate::runtime::types::ActionCandidate::Respond { content }
                        | crate::runtime::types::ActionCandidate::AskClarification { question: content } => {
                            content.clone()
                        }
                        crate::runtime::types::ActionCandidate::CallTool {
                            name,
                            arguments,
                            ..
                        } => format!("(Tool call in progress: {name} {arguments} — the tool is running; continue only when its result arrives)"),
                    };
                    if !user.trim().is_empty() && !assistant.trim().is_empty() {
                        if let Some(last) = self.dialogue.last_mut()
                            && last.user == *user
                        {
                            last.assistant.push('\n');
                            last.assistant.push_str(&assistant);
                        } else {
                            self.dialogue.push(DialogueTurn {
                                user: user.clone(),
                                assistant,
                            });
                            if self.dialogue.len() > MAX_DIALOGUE {
                                self.dialogue.remove(0);
                            }
                        }
                        if matches!(
                            decision.candidate,
                            crate::runtime::types::ActionCandidate::Respond { .. }
                                | crate::runtime::types::ActionCandidate::AskClarification { .. }
                        ) {
                            self.pending_user = None;
                        }
                        vec![Event::WorkingMemoryUpdate {
                            meta: *meta,
                            snapshot: self.snapshot(),
                        }]
                    } else {
                        vec![]
                    }
                } else {
                    vec![]
                }
            }
            Event::Tick { meta } => {
                self.decay();
                vec![Event::WorkingMemoryUpdate {
                    meta: *meta,
                    snapshot: self.snapshot(),
                }]
            }
            Event::Rpe { meta, rpe } => {
                let gate = Self::gate_open(rpe.0);
                let pending = self.pending_gated.take();
                if gate
                    && let Some(focus) = pending {
                        self.insert(focus);
                    }
                if gate {
                    vec![Event::WorkingMemoryUpdate {
                        meta: *meta,
                        snapshot: self.snapshot(),
                    }]
                } else {
                    vec![]
                }
            }
            Event::RequestState {
                meta,
                request: StateRequest::WorkingMemory,
                correlation_id,
            } => vec![Event::StateResponse {
                meta: *meta,
                response: StateResponse::WorkingMemory(self.snapshot()),
                correlation_id: *correlation_id,
            }],
            _ => vec![],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::actor::spawn_actor;
    use crate::runtime::bus::EventBus;
    use crate::runtime::event::EventMeta;
    use crate::runtime::types::{
        CycleId, PerceptionPayload, PerceptionSource, RpeSignal,
    };
    use futures::StreamExt;
    use std::time::Duration;

    fn meta() -> EventMeta {
        EventMeta {
            cycle_id: CycleId(1),

        }
    }

    fn attention_event(content: &str) -> Event {
        Event::Attention {
            meta: meta(),
            focus: AttentionFocus {
                payload: PerceptionPayload {
                    source: PerceptionSource::User,
                    content: content.into(),
                    salience: 0.8,
                },
                salience: 0.8,
                relevance: 0.8,
            },
        }
    }

    fn rpe_event(value: f32) -> Event {
        Event::Rpe {
            meta: meta(),
            rpe: RpeSignal(value),
        }
    }

    fn cycle_complete_event() -> Event {
        Event::CycleComplete {
            meta: meta(),
            summary: crate::runtime::types::CycleSummary {
                rpe: None,
                error: None,
                uncertainty: None,
                decision: None,
                modulation: None,
                user_input: None,
            },
        }
    }

    #[tokio::test]
    async fn high_rpe_gates_focus_into_memory() {
        let bus = EventBus::new(16);
        let (_h, ready) = spawn_actor(bus.clone(), WorkingMemoryActor::new());
        ready.await.unwrap();
        let mut rx = Box::pin(bus.subscribe_kinds(&[EventKind::WorkingMemory]));

        bus.publish(attention_event("important topic"));
        bus.publish(cycle_complete_event());
        bus.publish(rpe_event(0.6));

        let update = tokio::time::timeout(Duration::from_secs(2), rx.next())
            .await
            .unwrap()
            .unwrap();
        match update {
            Event::WorkingMemoryUpdate { snapshot, .. } => {
                assert_eq!(snapshot.slots.len(), 1);
                assert_eq!(snapshot.slots[0].content, "important topic");
            }
            _ => panic!("expected working memory update"),
        }
    }

    #[tokio::test]
    async fn low_rpe_keeps_memory_unchanged() {
        let bus = EventBus::new(16);
        let (_h, ready) = spawn_actor(bus.clone(), WorkingMemoryActor::new());
        ready.await.unwrap();
        let mut rx = Box::pin(bus.subscribe_kinds(&[EventKind::WorkingMemory]));

        bus.publish(attention_event("ignored topic"));
        bus.publish(cycle_complete_event());
        bus.publish(rpe_event(0.1));

        let stray = tokio::time::timeout(Duration::from_millis(300), rx.next()).await;
        assert!(stray.is_err(), "no update expected when the gate is closed");
    }

    #[tokio::test]
    async fn ticks_decay_slot_activation() {
        let bus = EventBus::new(16);
        let (_h, ready) = spawn_actor(bus.clone(), WorkingMemoryActor::new());
        ready.await.unwrap();
        let mut rx = Box::pin(bus.subscribe_kinds(&[EventKind::WorkingMemory]));

        bus.publish(attention_event("fresh topic"));
        bus.publish(cycle_complete_event());
        bus.publish(rpe_event(0.6));

        let update = tokio::time::timeout(Duration::from_secs(2), rx.next())
            .await
            .unwrap()
            .unwrap();
        let before = match update {
            Event::WorkingMemoryUpdate { snapshot, .. } => snapshot.slots[0].activation,
            _ => panic!("expected working memory update"),
        };
        assert_eq!(before, 1.0);

        bus.publish(Event::Tick { meta: meta() });
        let update = tokio::time::timeout(Duration::from_secs(2), rx.next())
            .await
            .unwrap()
            .unwrap();
        let after = match update {
            Event::WorkingMemoryUpdate { snapshot, .. } => snapshot.slots[0].activation,
            _ => panic!("expected working memory update"),
        };
        assert!(after < before, "ticks should decay slot activation");
    }

    #[tokio::test]
    async fn capacity_replaces_least_active_slot() {
        let bus = EventBus::new(16);
        let (_h, ready) = spawn_actor(bus.clone(), WorkingMemoryActor::new());
        ready.await.unwrap();
        let mut rx = Box::pin(bus.subscribe_kinds(&[EventKind::WorkingMemory]));

        for i in 0..CAPACITY {
            bus.publish(attention_event(&format!("topic {i}")));
            bus.publish(cycle_complete_event());
            bus.publish(rpe_event(0.6));
        }
        while {
            match tokio::time::timeout(Duration::from_secs(2), rx.next())
                .await
                .unwrap()
                .unwrap()
            {
                Event::WorkingMemoryUpdate { snapshot, .. } => snapshot.slots.len() < CAPACITY,
                _ => true,
            }
        } {}

        bus.publish(attention_event("overflow topic"));
        bus.publish(cycle_complete_event());
        bus.publish(rpe_event(0.6));

        loop {
            match tokio::time::timeout(Duration::from_secs(2), rx.next())
                .await
                .unwrap()
                .unwrap()
            {
                Event::WorkingMemoryUpdate { snapshot, .. } => {
                    assert_eq!(snapshot.slots.len(), CAPACITY);
                    break;
                }
                _ => continue,
            }
        }
    }

    #[tokio::test]
    async fn restore_dialogue_loads_history_through_bus() {
        let bus = EventBus::new(16);
        let (_h, ready) = spawn_actor(bus.clone(), WorkingMemoryActor::new());
        ready.await.unwrap();
        let mut rx = Box::pin(bus.subscribe_kinds(&[EventKind::WorkingMemory]));

        bus.publish(Event::RestoreDialogue {
            meta: meta(),
            turns: vec![crate::runtime::types::DialogueTurn {
                user: "question".into(),
                assistant: "answer".into(),
            }],
            tools: vec!["ls({\"dirPath\":\".\"}) -> src/ [allowed]".into()],
        });

        let update = tokio::time::timeout(Duration::from_secs(2), async {
            loop {
                match rx.next().await.unwrap() {
                    Event::WorkingMemoryUpdate { snapshot, .. } => return snapshot,
                    _ => continue,
                }
            }
        })
        .await
        .unwrap();
        assert_eq!(update.dialogue.len(), 1);
        assert_eq!(update.dialogue[0].user, "question");
        assert_eq!(update.dialogue[0].assistant, "answer");
    }

    #[tokio::test]
    async fn non_user_focus_does_not_pollute_dialogue() {
        let bus = EventBus::new(64);
        let mut actor = WorkingMemoryActor::new();
        let mut ctx = ActorContext::new(bus.clone(), meta());

        let scheduled = Event::Attention {
            meta: meta(),
            focus: AttentionFocus {
                payload: PerceptionPayload {
                    source: PerceptionSource::Scheduled,
                    content: "(Scheduled task #1 — delay)".into(),
                    salience: 0.8,
                },
                salience: 0.8,
                relevance: 0.8,
            },
        };
        actor.handle(&scheduled, &mut ctx).await;

        let tool_result = Event::Attention {
            meta: meta(),
            focus: AttentionFocus {
                payload: PerceptionPayload {
                    source: PerceptionSource::ToolResult,
                    content: "(Tool read_file result)\nfn main() {}".into(),
                    salience: 0.8,
                },
                salience: 0.8,
                relevance: 0.8,
            },
        };
        actor.handle(&tool_result, &mut ctx).await;

        let complete = Event::CycleComplete {
            meta: meta(),
            summary: crate::runtime::types::CycleSummary {
                user_input: None,
                decision: Some(crate::runtime::types::ActionDecision {
                    candidate: crate::runtime::types::ActionCandidate::Respond {
                        content: "task done".into(),
                    },
                    confidence: 0.9,
                    go: true,
                }),
                rpe: Some(0.5),
                error: None,
                uncertainty: None,
                modulation: None,
            },
        };
        actor.handle(&complete, &mut ctx).await;
        assert!(actor.dialogue.is_empty());
    }

    #[tokio::test]
    async fn tool_rounds_enter_dialogue_and_merge_with_same_user() {
        let bus = EventBus::new(64);
        let mut actor = WorkingMemoryActor::new();
        let mut ctx = ActorContext::new(bus.clone(), meta());

        let user_attention = Event::Attention {
            meta: meta(),
            focus: AttentionFocus {
                payload: PerceptionPayload {
                    source: PerceptionSource::User,
                    content: "test all tools".into(),
                    salience: 0.5,
                },
                salience: 0.5,
                relevance: 0.5,
            },
        };
        actor.handle(&user_attention, &mut ctx).await;

        let tool_call = Event::CycleComplete {
            meta: meta(),
            summary: crate::runtime::types::CycleSummary {
                user_input: None,
                decision: Some(crate::runtime::types::ActionDecision {
                    candidate: crate::runtime::types::ActionCandidate::CallTool {
                        name: "ls".into(),
                        arguments: serde_json::json!({"dirPath": "."}),
                        tool_call_id: None,
                        reasoning: None,
                    },
                    confidence: 0.9,
                    go: true,
                }),
                rpe: Some(0.5),
                error: None,
                uncertainty: None,
                modulation: None,
            },
        };
        actor.handle(&tool_call, &mut ctx).await;

        let respond = Event::CycleComplete {
            meta: meta(),
            summary: crate::runtime::types::CycleSummary {
                user_input: None,
                decision: Some(crate::runtime::types::ActionDecision {
                    candidate: crate::runtime::types::ActionCandidate::Respond {
                        content: "done".into(),
                    },
                    confidence: 0.9,
                    go: true,
                }),
                rpe: Some(0.5),
                error: None,
                uncertainty: None,
                modulation: None,
            },
        };
        actor.handle(&respond, &mut ctx).await;

        assert_eq!(actor.dialogue.len(), 1, "tool round and respond must merge into one turn");
        assert!(actor.dialogue[0].user.contains("test all tools"));
        assert!(
            actor.dialogue[0].assistant.contains("(Tool call in progress: ls"),
            "tool round must be recorded: {}",
            actor.dialogue[0].assistant
        );
        assert!(actor.dialogue[0].assistant.contains("done"));
    }
}
