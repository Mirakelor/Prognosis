use std::collections::HashSet;

use async_trait::async_trait;

use crate::runtime::actor::{ActorContext, CognitiveActor};
use crate::runtime::event::{Event, EventKind};
use crate::runtime::types::{InhibitionSignal, PredictionTrajectory};

const ERROR_THRESHOLD: f32 = 0.8;
const MAX_INHIBITED: usize = 16;

pub struct InhibitionActor {
    inhibited: HashSet<String>,
    last_trajectory: PredictionTrajectory,
}

impl InhibitionActor {
    pub fn new() -> Self {
        Self {
            inhibited: HashSet::new(),
            last_trajectory: PredictionTrajectory::default(),
        }
    }
}

impl Default for InhibitionActor {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl CognitiveActor for InhibitionActor {
    fn id(&self) -> &str {
        "inhibition"
    }

    fn subscriptions(&self) -> Vec<EventKind> {
        vec![EventKind::Prediction, EventKind::Error]
    }

    async fn handle(&mut self, event: &Event, _ctx: &mut ActorContext) -> Vec<Event> {
        match event {
            Event::Prediction { trajectory, .. } => {
                self.last_trajectory = trajectory.clone();
                vec![]
            }
            Event::ErrorComputed { meta, error } => {
                if error.weighted() < ERROR_THRESHOLD {
                    return vec![];
                }
                let mut targets = Vec::new();
                for element in self
                    .last_trajectory
                    .key_elements
                    .iter()
                    .chain(self.last_trajectory.topics.iter())
                {
                    if self.inhibited.insert(element.clone()) {
                        targets.push(element.clone());
                    }
                }
                if self.inhibited.len() > MAX_INHIBITED {
                    let excess: Vec<String> =
                        self.inhibited.iter().take(self.inhibited.len() - MAX_INHIBITED).cloned().collect();
                    for element in excess {
                        self.inhibited.remove(&element);
                    }
                }
                if targets.is_empty() {
                    vec![]
                } else {
                    vec![Event::Inhibition {
                        meta: *meta,
                        signal: InhibitionSignal {
                            targets,
                            strength: error.weighted(),
                        },
                    }]
                }
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
    use crate::runtime::types::{CycleId, PredictionError};
    use futures::StreamExt;
    use std::time::Duration;

    fn meta() -> EventMeta {
        EventMeta {
            cycle_id: CycleId(1),

        }
    }

    #[tokio::test]
    async fn high_error_inhibits_predicted_elements() {
        let bus = EventBus::new(16);
        let actor = InhibitionActor::new();
        let (_h, ready) = spawn_actor(bus.clone(), actor);
        ready.await.unwrap();
        let mut rx = Box::pin(bus.subscribe_kinds(&[EventKind::Inhibition]));

        bus.publish(Event::Prediction {
            meta: meta(),
            trajectory: PredictionTrajectory {
                topics: vec!["weather".into()],
                key_elements: vec!["sunny".into(), "temperature".into()],
                direction: 0.5,
                intent: crate::runtime::types::IntentKind::Statement,
                intent_candidates: vec![crate::runtime::types::IntentKind::Statement],
                reaction: String::new(),
                reaction_sentiment: 0.0,
            },
        });
        bus.publish(Event::ErrorComputed {
            meta: meta(),
            error: PredictionError {
                semantic: 0.8,
                confidence: 0.2,
                precision: 1.0,
            },
        });

        let signal = tokio::time::timeout(Duration::from_secs(2), rx.next())
            .await
            .unwrap()
            .unwrap();
        match signal {
            Event::Inhibition { signal, .. } => {
                assert!(signal.targets.contains(&"sunny".to_string()));
                assert!(signal.targets.contains(&"temperature".to_string()));
                assert_eq!(signal.strength, 1.0);
            }
            _ => panic!("expected inhibition event"),
        }
    }

    #[tokio::test]
    async fn low_error_does_not_inhibit() {
        let bus = EventBus::new(16);
        let (_h, ready) = spawn_actor(bus.clone(), InhibitionActor::new());
        ready.await.unwrap();
        let mut rx = Box::pin(bus.subscribe_kinds(&[EventKind::Inhibition]));

        bus.publish(Event::Prediction {
            meta: meta(),
            trajectory: PredictionTrajectory {
                topics: vec!["weather".into()],
                key_elements: vec!["sunny".into()],
                direction: 0.5,
                intent: crate::runtime::types::IntentKind::Statement,
                intent_candidates: vec![crate::runtime::types::IntentKind::Statement],
                reaction: String::new(),
                reaction_sentiment: 0.0,
            },
        });
        bus.publish(Event::ErrorComputed {
            meta: meta(),
            error: PredictionError {
                semantic: 0.1,
                confidence: 0.1,
                precision: 1.0,
            },
        });

        let result = tokio::time::timeout(Duration::from_millis(300), rx.next()).await;
        assert!(result.is_err(), "no inhibition expected for low error");
    }
}
