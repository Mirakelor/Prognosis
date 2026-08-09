use async_trait::async_trait;

use crate::runtime::actor::{ActorContext, CognitiveActor};
use crate::runtime::event::{Event, EventKind, StateRequest, StateResponse};
use crate::runtime::types::MetaState;

const UNCERTAINTY_SMOOTH: f32 = 0.8;
const CONFLICT_SMOOTH: f32 = 0.8;
const CHANGE_THRESHOLD: f32 = 0.05;

pub struct MetacognitionActor {
    uncertainty: f32,
    conflict: f32,
    last_emitted: MetaState,
}

impl MetacognitionActor {
    pub fn new() -> Self {
        Self {
            uncertainty: 0.0,
            conflict: 0.0,
            last_emitted: MetaState {
                uncertainty: 0.0,
                conflict: 0.0,
                confidence: 1.0,
            },
        }
    }

    fn meta_state(&self) -> MetaState {
        MetaState {
            uncertainty: self.uncertainty,
            conflict: self.conflict,
            confidence: 1.0 - self.uncertainty,
        }
    }
}

impl Default for MetacognitionActor {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl CognitiveActor for MetacognitionActor {
    fn id(&self) -> &str {
        "metacognition"
    }

    fn subscriptions(&self) -> Vec<EventKind> {
        vec![
            EventKind::Error,
            EventKind::Generation,
            EventKind::Language,
            EventKind::Modulation,
            EventKind::State,
        ]
    }

    async fn handle(&mut self, event: &Event, _ctx: &mut ActorContext) -> Vec<Event> {
        let mut events = Vec::new();
        match event {
            Event::Rpe { rpe, .. } => {
                self.conflict = self.conflict * CONFLICT_SMOOTH
                    + rpe.0.abs() * (1.0 - CONFLICT_SMOOTH);
            }
            Event::ErrorComputed { error, .. } => {
                let weighted = error.weighted();
                self.conflict = self.conflict * CONFLICT_SMOOTH
                    + weighted * (1.0 - CONFLICT_SMOOTH);
            }
            Event::ModulatorUpdate { state, .. } => {
                self.uncertainty = self.uncertainty * UNCERTAINTY_SMOOTH
                    + state.acetylcholine * (1.0 - UNCERTAINTY_SMOOTH);
            }
            Event::LanguageInsight { quality, .. } => {
                self.uncertainty = self.uncertainty * UNCERTAINTY_SMOOTH
                    + quality * (1.0 - UNCERTAINTY_SMOOTH);
            }
            Event::StreamEnd { meta, .. } => {
                let changed = (self.meta_state().uncertainty - self.last_emitted.uncertainty).abs()
                    >= CHANGE_THRESHOLD
                    || (self.meta_state().conflict - self.last_emitted.conflict).abs()
                        >= CHANGE_THRESHOLD;
                if changed {
                    self.last_emitted = self.meta_state();
                    events.push(Event::MetaUpdate {
                        meta: *meta,
                        meta_state: self.meta_state(),
                    });
                }
            }
            Event::RequestState {
                meta,
                request: StateRequest::Meta,
                correlation_id,
            } => {
                events.push(Event::StateResponse {
                    meta: *meta,
                    response: StateResponse::Meta(self.meta_state()),
                    correlation_id: *correlation_id,
                });
            }
            _ => {}
        }
        events
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::actor::spawn_actor;
    use crate::runtime::bus::EventBus;
    use crate::runtime::event::EventMeta;
    use crate::runtime::types::{
        CognitiveMode, CycleId, ModulatorState, PredictionError,
    };
    use futures::StreamExt;
    use std::time::Duration;

    fn meta() -> EventMeta {
        EventMeta {
            cycle_id: CycleId(1),

        }
    }

    fn stream_end() -> Event {
        Event::StreamEnd {
            meta: meta(),
            usage: None,
        }
    }

    #[tokio::test]
    async fn meta_state_reflects_signals() {
        let bus = EventBus::new(32);
        let (_h, ready) = spawn_actor(bus.clone(), MetacognitionActor::new());
        ready.await.unwrap();
        let mut rx = Box::pin(bus.subscribe_kinds(&[EventKind::Modulation]));

        for _ in 0..4 {
            bus.publish(Event::ModulatorUpdate {
                meta: meta(),
                state: ModulatorState {
                    dopamine: 0.0,
                    norepinephrine: 0.3,
                    acetylcholine: 0.9,
                    serotonin: 0.5,
                },
                mode: CognitiveMode::Automatic,
            });
        }
        for _ in 0..4 {
            bus.publish(Event::ErrorComputed {
                meta: meta(),
                error: PredictionError {
                    semantic: 0.9,
                    confidence: 0.0,
                    precision: 1.0,
                },
            });
        }
        bus.publish(stream_end());

        let meta_state = tokio::time::timeout(Duration::from_secs(2), async {
            loop {
                match rx.next().await.unwrap() {
                    Event::MetaUpdate { meta_state, .. } => return meta_state,
                    _ => continue,
                }
            }
        })
        .await
        .unwrap();
        assert!(meta_state.uncertainty > 0.4, "ACh should drive uncertainty");
        assert!(meta_state.conflict > 0.4, "error should drive conflict");
        assert!(meta_state.confidence < 0.6);
    }

    #[tokio::test]
    async fn meta_state_is_queryable() {
        let bus = EventBus::new(16);
        let (_h, ready) = spawn_actor(bus.clone(), MetacognitionActor::new());
        ready.await.unwrap();
        let mut rx = bus.subscribe();

        bus.publish(Event::RequestState {
            meta: meta(),
            request: StateRequest::Meta,
            correlation_id: 11,
        });

        let meta_state = tokio::time::timeout(Duration::from_secs(2), async {
            loop {
                match rx.recv().await.unwrap() {
                    Event::StateResponse {
                        response: StateResponse::Meta(meta_state),
                        correlation_id: 11,
                        ..
                    } => return meta_state,
                    _ => continue,
                }
            }
        })
        .await
        .unwrap();
        assert_eq!(meta_state.confidence, 1.0);
    }
}
