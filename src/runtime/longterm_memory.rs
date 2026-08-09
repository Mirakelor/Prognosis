use std::sync::{Arc, Mutex};

use async_trait::async_trait;

use crate::runtime::actor::{ActorContext, CognitiveActor};
use crate::runtime::error_comparator::similarity;
use crate::runtime::event::{Event, EventKind, StateRequest, StateResponse};
use crate::runtime::types::{
    ActionCandidate, ActionDecision, EmotionState, EpisodicMemory, MemoryRetrieval, SemanticMemory,
};

const CONSOLIDATE_THRESHOLD: f32 = 0.7;
const RETRIEVAL_TOP_N: usize = 3;
const BELIEF_INITIAL: f32 = 0.5;
const BELIEF_DOWNGRADE: f32 = 0.8;
const BELIEF_BOOST: f32 = 0.05;
const BELIEF_FLOOR: f32 = 0.1;
const ERROR_DOWNGRADE: f32 = 0.6;
const ERROR_BOOST: f32 = 0.2;

pub struct LongTermMemoryActor {
    episodic: Arc<Mutex<Vec<EpisodicMemory>>>,
    semantic: Arc<Mutex<Vec<SemanticMemory>>>,
    next_id: u64,
    last_used_semantic: Vec<u64>,
}

impl LongTermMemoryActor {
    pub fn new() -> Self {
        Self {
            episodic: Arc::new(Mutex::new(Vec::new())),
            semantic: Arc::new(Mutex::new(Vec::new())),
            next_id: 0,
            last_used_semantic: Vec::new(),
        }
    }

    pub fn episodic(&self) -> Arc<Mutex<Vec<EpisodicMemory>>> {
        self.episodic.clone()
    }

    pub fn semantic(&self) -> Arc<Mutex<Vec<SemanticMemory>>> {
        self.semantic.clone()
    }

    fn summary_text(user: &Option<String>, decision: &Option<ActionDecision>) -> String {
        let assistant = match decision {
            Some(decision) => match &decision.candidate {
                ActionCandidate::Respond { content } => content.clone(),
                ActionCandidate::AskClarification { question } => question.clone(),
                ActionCandidate::CallTool { name, .. } => format!("tool call {name}"),
            },
            None => String::new(),
        };
        match user {
            Some(user) if !user.trim().is_empty() => format!("user: {user}\nassistant: {assistant}"),
            _ => assistant,
        }
    }

    fn consolidate(&mut self, memory: &EpisodicMemory) -> Option<(String, f32)> {
        let similar_episode = {
            let episodic = self.episodic.lock().unwrap();
            episodic.iter().rev().any(|existing| {
                existing.id != memory.id
                    && similarity(&existing.summary, &memory.summary) > CONSOLIDATE_THRESHOLD
            })
        };
        if !similar_episode {
            return None;
        }
        let mut semantic = self.semantic.lock().unwrap();
        match semantic
            .iter_mut()
            .find(|entry| similarity(&entry.content, &memory.summary) > CONSOLIDATE_THRESHOLD)
        {
            Some(entry) => {
                if memory.summary.chars().count() > entry.content.chars().count() {
                    entry.content = memory.summary.clone();
                }
                entry.strength += memory.strength * 0.5;
                Some((entry.content.clone(), entry.strength))
            }
            None => {
                let entry = SemanticMemory {
                    id: self.next_id,
                    content: memory.summary.clone(),
                    strength: memory.strength * 0.5,
                    belief: BELIEF_INITIAL,
                };
                self.next_id += 1;
                let strength = entry.strength;
                let content = entry.content.clone();
                semantic.push(entry);
                Some((content, strength))
            }
        }
    }

    fn adjust_beliefs(&mut self, error: f32) {
        let mut semantic = self.semantic.lock().unwrap();
        for id in &self.last_used_semantic {
            if let Some(entry) = semantic.iter_mut().find(|entry| entry.id == *id) {
                if error >= ERROR_DOWNGRADE {
                    entry.belief = (entry.belief * BELIEF_DOWNGRADE).max(BELIEF_FLOOR);
                } else if error <= ERROR_BOOST {
                    entry.belief = (entry.belief + BELIEF_BOOST).min(1.0);
                }
            }
        }
    }

    fn retrieve(&mut self, query: &str) -> MemoryRetrieval {
        let episodic = self.episodic.lock().unwrap();
        let semantic = self.semantic.lock().unwrap();
        let mut episodic_ranked: Vec<(f32, EpisodicMemory)> = episodic
            .iter()
            .map(|memory| (similarity(query, &memory.summary), memory.clone()))
            .collect();
        episodic_ranked.sort_by(|a, b| b.0.total_cmp(&a.0));
        let mut semantic_ranked: Vec<(f32, SemanticMemory)> = semantic
            .iter()
            .map(|memory| (similarity(query, &memory.content), memory.clone()))
            .collect();
        semantic_ranked.sort_by(|a, b| b.0.total_cmp(&a.0));
        self.last_used_semantic = semantic_ranked
            .iter()
            .take(RETRIEVAL_TOP_N)
            .map(|(_, memory)| memory.id)
            .collect();
        MemoryRetrieval {
            episodic: episodic_ranked
                .into_iter()
                .take(RETRIEVAL_TOP_N)
                .map(|(_, memory)| memory)
                .collect(),
            semantic: semantic_ranked
                .into_iter()
                .take(RETRIEVAL_TOP_N)
                .map(|(_, memory)| memory)
                .collect(),
        }
    }
}

impl Default for LongTermMemoryActor {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl CognitiveActor for LongTermMemoryActor {
    fn id(&self) -> &str {
        "longterm_memory"
    }

    fn subscriptions(&self) -> Vec<EventKind> {
        vec![EventKind::Cycle, EventKind::State, EventKind::Error]
    }

    async fn handle(&mut self, event: &Event, _ctx: &mut ActorContext) -> Vec<Event> {
        match event {
            Event::ErrorComputed { error, .. } => {
                self.adjust_beliefs(error.weighted());
                vec![]
            }
            Event::Rpe { rpe, .. } => {
                let mut episodic = self.episodic.lock().unwrap();
                if let Some(last) = episodic.last_mut() {
                    last.emotion = EmotionState {
                        valence: rpe.0.clamp(-1.0, 1.0),
                        arousal: rpe.0.abs(),
                    };
                    last.strength = (0.2 + rpe.0.abs()).max(last.strength);
                }
                vec![]
            }
            Event::CycleComplete { meta, summary } => {
                let rpe = summary.rpe.unwrap_or(0.0);
                let surprise = summary.error.unwrap_or(0.0) * 0.3
                    + summary.uncertainty.unwrap_or(0.0) * 0.2;
                let memory = EpisodicMemory {
                    id: self.next_id,
                    cycle_id: meta.cycle_id,
                    summary: Self::summary_text(&summary.user_input, &summary.decision),
                    emotion: EmotionState {
                        valence: rpe.clamp(-1.0, 1.0),
                        arousal: rpe.abs(),
                    },
                    strength: (0.2 + rpe.abs() + surprise).clamp(0.0, 1.0),
                };
                self.next_id += 1;
                let consolidated = self.consolidate(&memory);
                let episodic_strength = memory.strength;
                let episodic_content = memory.summary.clone();
                self.episodic.lock().unwrap().push(memory);
                let mut events = vec![Event::MemoryWrite {
                    meta: *meta,
                    kind: crate::runtime::types::MemoryKind::Episodic,
                    content: episodic_content,
                    strength: episodic_strength,
                }];
                if let Some((content, strength)) = consolidated {
                    events.push(Event::MemoryWrite {
                        meta: *meta,
                        kind: crate::runtime::types::MemoryKind::Semantic,
                        content,
                        strength,
                    });
                }
                events
            }
            Event::RequestState {
                meta,
                request: StateRequest::MemoryRetrieval { query },
                correlation_id,
            } => {
                let retrieval = self.retrieve(query);
                vec![
                    Event::StateResponse {
                        meta: *meta,
                        response: StateResponse::MemoryRetrieval(retrieval.clone()),
                        correlation_id: *correlation_id,
                    },
                    Event::MemoryRetrieved {
                        meta: *meta,
                        retrieval,
                    },
                ]
            }
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
    use crate::runtime::types::{CycleId, CycleSummary, PredictionError};
    use std::time::Duration;

    fn meta(cycle: u64) -> EventMeta {
        EventMeta {
            cycle_id: CycleId(cycle),

        }
    }

    fn cycle_complete(cycle: u64, rpe: f32, content: &str) -> Event {
        Event::CycleComplete {
            meta: meta(cycle),
            summary: CycleSummary {
                rpe: Some(rpe),
                error: None,
                uncertainty: None,
                decision: Some(ActionDecision {
                    candidate: ActionCandidate::Respond {
                        content: content.into(),
                    },
                    confidence: 0.9,
                    go: true,
                }),
                modulation: None,
                user_input: Some("user asked something".into()),
            },
        }
    }

    #[tokio::test]
    async fn cycle_complete_writes_episodic_memory() {
        let bus = EventBus::new(16);
        let actor = LongTermMemoryActor::new();
        let episodic_handle = actor.episodic();
        let (_h, ready) = spawn_actor(bus.clone(), actor);
        ready.await.unwrap();

        bus.publish(cycle_complete(1, 0.8, "the weather is sunny"));
        bus.publish(cycle_complete(2, 0.2, "nothing special happened"));

        tokio::time::sleep(Duration::from_millis(100)).await;
        let episodic = episodic_handle.lock().unwrap();
        assert_eq!(episodic.len(), 2);
        assert!(episodic[0].strength > episodic[1].strength);
        assert_eq!(episodic[0].emotion.valence, 0.8);
    }

    #[tokio::test]
    async fn error_and_uncertainty_boost_memory_strength() {
        let bus = EventBus::new(16);
        let actor = LongTermMemoryActor::new();
        let episodic_handle = actor.episodic();
        let (_h, ready) = spawn_actor(bus.clone(), actor);
        ready.await.unwrap();

        bus.publish(cycle_complete(1, 0.0, "calm round"));
        bus.publish(Event::CycleComplete {
            meta: meta(2),
            summary: CycleSummary {
                rpe: Some(0.0),
                error: Some(1.0),
                uncertainty: Some(1.0),
                decision: Some(ActionDecision {
                    candidate: ActionCandidate::Respond {
                        content: "surprising round".into(),
                    },
                    confidence: 0.9,
                    go: true,
                }),
                modulation: None,
                user_input: Some("unexpected input".into()),
            },
        });

        tokio::time::sleep(Duration::from_millis(100)).await;
        let episodic = episodic_handle.lock().unwrap();
        assert_eq!(episodic.len(), 2);
        assert!(
            episodic[1].strength > episodic[0].strength,
            "surprise must strengthen encoding: {} vs {}",
            episodic[1].strength,
            episodic[0].strength
        );
        assert!(
            (episodic[1].strength - 0.7).abs() < 0.001,
            "0.2 base + 0.3 error + 0.2 uncertainty = 0.7, got {}",
            episodic[1].strength
        );
    }

    #[tokio::test]
    async fn repeated_similar_episodes_consolidate_into_semantic() {
        let bus = EventBus::new(16);
        let actor = LongTermMemoryActor::new();
        let semantic_handle = actor.semantic();
        let (_h, ready) = spawn_actor(bus.clone(), actor);
        ready.await.unwrap();

        bus.publish(cycle_complete(1, 0.5, "user asks about weather forecast"));
        bus.publish(cycle_complete(2, 0.5, "user asks about the weather forecast"));

        tokio::time::sleep(Duration::from_millis(100)).await;
        let semantic = semantic_handle.lock().unwrap();
        assert_eq!(semantic.len(), 1);
        assert!(semantic[0].strength > 0.0);
    }

    #[tokio::test]
    async fn retrieval_returns_matching_memories() {
        let bus = EventBus::new(16);
        let (_h, ready) = spawn_actor(bus.clone(), LongTermMemoryActor::new());
        ready.await.unwrap();
        let mut rx = bus.subscribe();

        bus.publish(cycle_complete(1, 0.5, "user asks about weather forecast"));
        bus.publish(cycle_complete(2, 0.5, "user asks about the weather forecast"));
        bus.publish(Event::RequestState {
            meta: meta(3),
            request: StateRequest::MemoryRetrieval {
                query: "weather forecast".into(),
            },
            correlation_id: 99,
        });

        let retrieval = tokio::time::timeout(Duration::from_secs(2), async {
            loop {
                match rx.recv().await.unwrap() {
                    Event::StateResponse {
                        response: StateResponse::MemoryRetrieval(retrieval),
                        correlation_id: 99,
                        ..
                    } => return retrieval,
                    _ => continue,
                }
            }
        })
        .await
        .unwrap();
        assert!(!retrieval.episodic.is_empty());
        assert!(!retrieval.semantic.is_empty());
    }

    #[tokio::test]
    async fn cycle_complete_broadcasts_memory_writes() {
        let bus = EventBus::new(16);
        let (_h, ready) = spawn_actor(bus.clone(), LongTermMemoryActor::new());
        ready.await.unwrap();
        let mut rx = bus.subscribe();

        bus.publish(cycle_complete(1, 0.5, "user asks about weather forecast"));
        bus.publish(cycle_complete(2, 0.5, "user asks about the weather forecast"));

        let mut writes = Vec::new();
        tokio::time::timeout(Duration::from_secs(2), async {
            while writes.len() < 3 {
                match rx.recv().await.unwrap() {
                    Event::MemoryWrite { kind, .. } => writes.push(kind),
                    _ => continue,
                }
            }
        })
        .await
        .unwrap();
        assert_eq!(writes[0], crate::runtime::types::MemoryKind::Episodic);
        assert!(writes.contains(&crate::runtime::types::MemoryKind::Semantic));
    }

    #[tokio::test]
    async fn high_error_downgrades_retrieved_belief() {
        let bus = EventBus::new(16);
        let actor = LongTermMemoryActor::new();
        let semantic_handle = actor.semantic();
        let (_h, ready) = spawn_actor(bus.clone(), actor);
        ready.await.unwrap();
        let mut rx = bus.subscribe();

        bus.publish(cycle_complete(1, 0.5, "user asks about weather forecast"));
        bus.publish(cycle_complete(2, 0.5, "user asks about the weather forecast"));
        bus.publish(Event::RequestState {
            meta: meta(3),
            request: StateRequest::MemoryRetrieval {
                query: "weather forecast".into(),
            },
            correlation_id: 5,
        });
        tokio::time::timeout(Duration::from_secs(2), async {
            loop {
                match rx.recv().await.unwrap() {
                    Event::StateResponse { correlation_id: 5, .. } => break,
                    _ => continue,
                }
            }
        })
        .await
        .unwrap();

        let before = semantic_handle.lock().unwrap()[0].belief;
        bus.publish(Event::ErrorComputed {
            meta: meta(4),
            error: PredictionError {
                semantic: 0.8,
                confidence: 0.2,
                precision: 1.0,
            },
        });
        tokio::time::sleep(Duration::from_millis(100)).await;
        let after = semantic_handle.lock().unwrap()[0].belief;
        assert!(after < before, "high error should downgrade the retrieved belief");
    }
}
