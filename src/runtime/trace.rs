use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;

use crate::runtime::actor::{ActorContext, CognitiveActor};
use crate::runtime::event::{Event, EventKind};
use crate::runtime::types::{
    ActionDecision, CycleId, CycleSummary, ModulationContext, TraceRecord,
};

#[derive(Default)]
struct CycleAccumulator {
    rpe: Option<f32>,
    error: Option<f32>,
    uncertainty: Option<f32>,
    decision: Option<ActionDecision>,
    modulation: Option<ModulationContext>,
    error_at_intervention: Option<f32>,
    user_input: Option<String>,
    retrieval: Option<String>,
    prediction_direction: Option<f32>,
    prediction_sentiment: Option<f32>,
    prediction_reaction: Option<String>,
    stream_ended: bool,
    action_selected: bool,
    chunk_count: usize,
}

pub struct TraceActor {
    cycles: HashMap<CycleId, CycleAccumulator>,
    records: Arc<Mutex<Vec<TraceRecord>>>,
}

impl TraceActor {
    pub fn new() -> Self {
        Self {
            cycles: HashMap::new(),
            records: Arc::new(Mutex::new(Vec::new())),
        }
    }

    pub fn records(&self) -> Arc<Mutex<Vec<TraceRecord>>> {
        self.records.clone()
    }

    fn retrieval_summary(retrieval: &crate::runtime::types::MemoryRetrieval) -> String {
        let mut parts = Vec::new();
        for memory in retrieval.episodic.iter().take(1) {
            parts.push(format!("episodic: {}", memory.summary.chars().take(60).collect::<String>()));
        }
        for memory in retrieval.semantic.iter().take(1) {
            parts.push(format!(
                "semantic: {} (belief {:.2})",
                memory.content.chars().take(60).collect::<String>(),
                memory.belief
            ));
        }
        if parts.is_empty() {
            "none".into()
        } else {
            parts.join(" | ")
        }
    }
}

impl Default for TraceActor {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl CognitiveActor for TraceActor {
    fn id(&self) -> &str {
        "trace"
    }

    fn subscriptions(&self) -> Vec<EventKind> {
        vec![
            EventKind::Generation,
            EventKind::Error,
            EventKind::Modulation,
            EventKind::Action,
            EventKind::Perception,
            EventKind::Memory,
            EventKind::Prediction,
        ]
    }

    async fn handle(&mut self, event: &Event, _ctx: &mut ActorContext) -> Vec<Event> {
        let Some(meta) = event.meta() else {
            return vec![];
        };
        let cycle = self.cycles.entry(meta.cycle_id).or_default();
        match event {
            Event::Rpe { rpe, .. } => cycle.rpe = Some(rpe.0),
            Event::ErrorComputed { error, .. } => {
                let weighted = error.weighted();
                cycle.error = Some(weighted);
                if cycle.error_at_intervention.is_none() && cycle.modulation.is_some() {
                    cycle.error_at_intervention = Some(weighted);
                }
            }
            Event::MetaUpdate { meta_state, .. } => {
                cycle.uncertainty = Some(meta_state.uncertainty);
            }
            Event::Modulate { modulation, .. } => {
                cycle.modulation = Some(modulation.clone());
            }
            Event::Chunk { .. } => cycle.chunk_count += 1,
            Event::StreamEnd { .. } => cycle.stream_ended = true,
            Event::ActionSelected { decision, .. } => {
                cycle.decision = Some(decision.clone());
                cycle.action_selected = true;
            }
            Event::Perception { payload, .. } => {
                if payload.source == crate::runtime::types::PerceptionSource::User {
                    cycle.user_input = Some(payload.content.clone());
                }
            }
            Event::MemoryRetrieved { retrieval, .. } => {
                cycle.retrieval = Some(Self::retrieval_summary(retrieval));
            }
            Event::Prediction { trajectory, .. } => {
                cycle.prediction_direction = Some(trajectory.direction);
                cycle.prediction_sentiment = Some(trajectory.reaction_sentiment);
                cycle.prediction_reaction = Some(trajectory.reaction.clone());
            }
            _ => {}
        }

        if cycle.stream_ended && cycle.action_selected {
            let summary = CycleSummary {
                rpe: cycle.rpe,
                error: cycle.error,
                uncertainty: cycle.uncertainty,
                decision: cycle.decision.clone(),
                modulation: cycle.modulation.clone(),
                user_input: cycle.user_input.clone(),
            };
            self.records.lock().unwrap().push(TraceRecord {
                cycle_id: meta.cycle_id,
                modulation: cycle.modulation.clone(),
                error_before: cycle.error_at_intervention,
                error_after: cycle.error,
                decision: cycle.decision.clone(),
                retrieval: cycle.retrieval.clone(),
                prediction_direction: cycle.prediction_direction,
                prediction_sentiment: cycle.prediction_sentiment,
                prediction_reaction: cycle.prediction_reaction.clone(),
            });
            self.cycles.remove(&meta.cycle_id);
            vec![Event::CycleComplete { meta, summary }]
        } else {
            vec![]
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
        ActionCandidate, CycleId, EmotionState, EpisodicMemory, MemoryRetrieval, MetaState,
        PredictionError, RpeSignal,
    };
    use std::time::Duration;

    fn meta() -> EventMeta {
        EventMeta {
            cycle_id: CycleId(7),
            timestamp: 0,
        }
    }

    fn error_event(meta: EventMeta) -> Event {
        Event::ErrorComputed {
            meta,
            error: PredictionError {
                semantic: 0.4,
                confidence: 0.1,
                precision: 1.0,
            },
        }
    }

    fn action_event(meta: EventMeta) -> Event {
        Event::ActionSelected {
            meta,
            decision: ActionDecision {
                candidate: ActionCandidate::Respond {
                    content: "done".into(),
                },
                confidence: 0.9,
                go: true,
            },
        }
    }

    #[tokio::test]
    async fn cycle_completes_when_stream_end_and_action_selected() {
        let bus = EventBus::new(32);
        let (_h, ready) = spawn_actor(bus.clone(), TraceActor::new());
        ready.await.unwrap();
        let mut rx = bus.subscribe();

        bus.publish(Event::Rpe {
            meta: meta(),
            rpe: RpeSignal(0.5),
        });
        bus.publish(error_event(meta()));
        bus.publish(Event::MetaUpdate {
            meta: meta(),
            meta_state: MetaState {
                uncertainty: 0.3,
                conflict: 0.2,
                confidence: 0.8,
            },
        });
        bus.publish(Event::StreamEnd {
            meta: meta(),
            usage: None,
        });
        bus.publish(action_event(meta()));

        let summary = tokio::time::timeout(Duration::from_secs(2), async {
            loop {
                match rx.recv().await.unwrap() {
                    Event::CycleComplete { summary, .. } => return summary,
                    _ => continue,
                }
            }
        })
        .await
        .unwrap();

        assert_eq!(summary.rpe, Some(0.5));
        assert_eq!(summary.error, Some(0.5));
        assert_eq!(summary.uncertainty, Some(0.3));
        assert!(summary.decision.is_some());
    }

    #[tokio::test]
    async fn record_keeps_memory_retrieval_summary() {
        let bus = EventBus::new(32);
        let trace = TraceActor::new();
        let records = trace.records();
        let (_h, ready) = spawn_actor(bus.clone(), trace);
        ready.await.unwrap();

        bus.publish(Event::MemoryRetrieved {
            meta: meta(),
            retrieval: MemoryRetrieval {
                episodic: vec![EpisodicMemory {
                    id: 1,
                    cycle_id: meta().cycle_id,
                    summary: "user asked about the weather forecast".into(),
                    emotion: EmotionState {
                        valence: 0.5,
                        arousal: 0.3,
                    },
                    strength: 0.6,
                }],
                semantic: vec![],
            },
        });
        bus.publish(Event::StreamEnd {
            meta: meta(),
            usage: None,
        });
        bus.publish(action_event(meta()));

        tokio::time::sleep(Duration::from_millis(100)).await;
        let records = records.lock().unwrap();
        assert_eq!(records.len(), 1);
        let retrieval = records[0].retrieval.as_deref().unwrap_or("");
        assert!(retrieval.contains("episodic: user asked about the weather"), "{retrieval}");
    }

    #[tokio::test]
    async fn record_keeps_prediction_fields() {
        let bus = EventBus::new(32);
        let trace = TraceActor::new();
        let records = trace.records();
        let (_h, ready) = spawn_actor(bus.clone(), trace);
        ready.await.unwrap();

        bus.publish(Event::Prediction {
            meta: meta(),
            trajectory: crate::runtime::types::PredictionTrajectory {
                topics: vec!["t".into()],
                key_elements: vec!["k".into()],
                direction: 0.7,
                intent: crate::runtime::types::IntentKind::Question,
                intent_candidates: vec![crate::runtime::types::IntentKind::Question],
                reaction: "the user asks for details".into(),
                reaction_sentiment: 0.4,
            },
        });
        bus.publish(Event::StreamEnd {
            meta: meta(),
            usage: None,
        });
        bus.publish(action_event(meta()));

        tokio::time::sleep(Duration::from_millis(100)).await;
        let records = records.lock().unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].prediction_direction, Some(0.7));
        assert_eq!(records[0].prediction_sentiment, Some(0.4));
        assert_eq!(
            records[0].prediction_reaction.as_deref(),
            Some("the user asks for details")
        );
    }
}
