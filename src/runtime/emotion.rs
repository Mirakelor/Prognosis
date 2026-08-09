use async_trait::async_trait;

use crate::runtime::actor::{ActorContext, CognitiveActor};
use crate::runtime::event::{Event, EventKind, StateRequest, StateResponse};
use crate::runtime::types::EmotionState;

const VALENCE_SMOOTH: f32 = 0.7;
const AROUSAL_SMOOTH: f32 = 0.7;
const CHANGE_THRESHOLD: f32 = 0.05;

pub struct EmotionActor {
    emotion: EmotionState,
    last_emitted: EmotionState,
}

impl EmotionActor {
    pub fn new() -> Self {
        Self {
            emotion: EmotionState {
                valence: 0.0,
                arousal: 0.0,
            },
            last_emitted: EmotionState {
                valence: 0.0,
                arousal: 0.0,
            },
        }
    }
}

impl Default for EmotionActor {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl CognitiveActor for EmotionActor {
    fn id(&self) -> &str {
        "emotion"
    }

    fn subscriptions(&self) -> Vec<EventKind> {
        vec![EventKind::Error, EventKind::State]
    }

    async fn handle(&mut self, event: &Event, _ctx: &mut ActorContext) -> Vec<Event> {
        match event {
            Event::Rpe { rpe, .. } => {
                self.emotion.valence = (self.emotion.valence * VALENCE_SMOOTH
                    + rpe.0 * (1.0 - VALENCE_SMOOTH))
                    .clamp(-1.0, 1.0);
                vec![]
            }
            Event::ErrorComputed { meta, error } => {
                self.emotion.arousal = (self.emotion.arousal * AROUSAL_SMOOTH
                    + error.weighted() * (1.0 - AROUSAL_SMOOTH))
                    .clamp(0.0, 1.0);
                let changed = (self.emotion.valence - self.last_emitted.valence).abs()
                    >= CHANGE_THRESHOLD
                    || (self.emotion.arousal - self.last_emitted.arousal).abs()
                        >= CHANGE_THRESHOLD;
                if changed {
                    self.last_emitted = self.emotion;
                    vec![Event::EmotionUpdate {
                        meta: *meta,
                        emotion: self.emotion,
                    }]
                } else {
                    vec![]
                }
            }
            Event::RequestState {
                meta,
                request: StateRequest::Emotion,
                correlation_id,
            } => vec![Event::StateResponse {
                meta: *meta,
                response: StateResponse::Emotion(self.emotion),
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
    use crate::runtime::types::{CycleId, PredictionError, RpeSignal};
    use futures::StreamExt;
    use std::time::Duration;

    fn meta() -> EventMeta {
        EventMeta {
            cycle_id: CycleId(1),

        }
    }

    fn rpe_event(value: f32) -> Event {
        Event::Rpe {
            meta: meta(),
            rpe: RpeSignal(value),
        }
    }

    fn error_event(weighted: f32) -> Event {
        Event::ErrorComputed {
            meta: meta(),
            error: PredictionError {
                semantic: weighted,
                confidence: 0.0,
                precision: 1.0,
            },
        }
    }

    #[tokio::test]
    async fn positive_rpe_raises_valence() {
        let bus = EventBus::new(16);
        let (_h, ready) = spawn_actor(bus.clone(), EmotionActor::new());
        ready.await.unwrap();
        let mut rx = Box::pin(bus.subscribe_kinds(&[EventKind::Modulation]));

        bus.publish(rpe_event(0.8));
        bus.publish(error_event(0.1));

        let emotion = tokio::time::timeout(Duration::from_secs(2), async {
            loop {
                match rx.next().await.unwrap() {
                    Event::EmotionUpdate { emotion, .. } => return emotion,
                    _ => continue,
                }
            }
        })
        .await
        .unwrap();
        assert!(emotion.valence > 0.0);
    }

    #[tokio::test]
    async fn high_error_raises_arousal() {
        let bus = EventBus::new(16);
        let (_h, ready) = spawn_actor(bus.clone(), EmotionActor::new());
        ready.await.unwrap();
        let mut rx = Box::pin(bus.subscribe_kinds(&[EventKind::Modulation]));

        bus.publish(rpe_event(0.0));
        for _ in 0..3 {
            bus.publish(error_event(0.9));
        }

        let emotion = tokio::time::timeout(Duration::from_secs(2), async {
            loop {
                match rx.next().await.unwrap() {
                    Event::EmotionUpdate { emotion, .. } if emotion.arousal > 0.5 => {
                        return emotion;
                    }
                    _ => continue,
                }
            }
        })
        .await
        .unwrap();
        assert!(emotion.arousal > 0.5);
    }
}
