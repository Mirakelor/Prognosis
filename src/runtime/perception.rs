use std::collections::VecDeque;

use async_trait::async_trait;

use crate::runtime::actor::{ActorContext, CognitiveActor};
use crate::runtime::event::{Event, EventKind};
use crate::runtime::types::PerceptionFeatures;

const BUFFER_CAPACITY: usize = 8;

pub struct PerceptionActor {
    buffer: VecDeque<String>,
}

impl PerceptionActor {
    pub fn new() -> Self {
        Self {
            buffer: VecDeque::with_capacity(BUFFER_CAPACITY),
        }
    }

    fn novelty(&self, content: &str) -> f32 {
        if self.buffer.iter().any(|seen| seen == content) {
            0.1
        } else {
            1.0
        }
    }

    fn extract_features(content: &str) -> PerceptionFeatures {
        let length = content.chars().count();
        let lower = content.to_lowercase();
        let mut topic_hints = Vec::new();
        for hint in ["weather", "time", "price", "code", "test", "help", "predict", "tool"] {
            if lower.contains(hint) {
                topic_hints.push(hint.to_string());
            }
        }
        let emotional_tone = if lower.contains("thank") || lower.contains("great") {
            0.5
        } else if lower.contains("error") || lower.contains("fail") || lower.contains("bad") {
            -0.5
        } else {
            0.0
        };
        PerceptionFeatures {
            topic_hints,
            emotional_tone,
            length,
        }
    }

    fn salience(features: &PerceptionFeatures, novelty: f32, base: f32) -> f32 {
        let intensity = (features.length as f32 / 100.0).min(1.0);
        let emotion = features.emotional_tone.abs();
        ((base + intensity + emotion) / 3.0).clamp(0.0, 1.0) * novelty
    }
}

impl Default for PerceptionActor {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl CognitiveActor for PerceptionActor {
    fn id(&self) -> &str {
        "perception"
    }

    fn subscriptions(&self) -> Vec<EventKind> {
        vec![EventKind::Perception]
    }

    async fn handle(&mut self, event: &Event, _ctx: &mut ActorContext) -> Vec<Event> {
        match event {
            Event::Perception { meta, payload } => {
                let novelty = self.novelty(&payload.content);
                let features = Self::extract_features(&payload.content);
                let salience = Self::salience(&features, novelty, payload.salience);
                if self.buffer.len() >= BUFFER_CAPACITY {
                    self.buffer.pop_front();
                }
                self.buffer.push_back(payload.content.clone());
                let sensed = Event::Sensed {
                    meta: *meta,
                    payload: payload.clone(),
                    features,
                    salience,
                };
                vec![sensed]
            }
            _ => vec![],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::types::PerceptionPayload;
    use crate::runtime::actor::spawn_actor;
    use crate::runtime::bus::EventBus;
    use crate::runtime::event::EventMeta;
    use crate::runtime::types::{CycleId, PerceptionSource};
    use futures::StreamExt;
    use std::time::Duration;

    fn meta() -> EventMeta {
        EventMeta {
            cycle_id: CycleId(1),
            timestamp: 0,
        }
    }

    fn perception_event(content: &str) -> Event {
        Event::Perception {
            meta: meta(),
            payload: PerceptionPayload {
                source: PerceptionSource::User,
                content: content.into(),
                salience: 0.5,
            },
        }
    }

    #[tokio::test]
    async fn perception_extracts_features_and_publishes_sensed() {
        let bus = EventBus::new(16);
        let (_h, ready) = spawn_actor(bus.clone(), PerceptionActor::new());
        ready.await.unwrap();
        let mut rx = Box::pin(bus.subscribe_kinds(&[EventKind::Sensed]));

        bus.publish(perception_event("what is the weather in beijing?"));
        let sensed = tokio::time::timeout(Duration::from_secs(2), rx.next())
            .await
            .unwrap()
            .unwrap();
        match sensed {
            Event::Sensed {
                payload,
                features,
                salience,
                ..
            } => {
                assert_eq!(payload.content, "what is the weather in beijing?");
                assert!(features.topic_hints.contains(&"weather".to_string()));
                assert!(features.length > 0);
                assert!(salience > 0.0);
            }
            _ => panic!("expected sensed event"),
        }
    }

    #[tokio::test]
    async fn repeated_input_loses_salience() {
        let bus = EventBus::new(16);
        let (_h, ready) = spawn_actor(bus.clone(), PerceptionActor::new());
        ready.await.unwrap();
        let mut rx = Box::pin(bus.subscribe_kinds(&[EventKind::Sensed]));

        bus.publish(perception_event("hello"));
        let first = tokio::time::timeout(Duration::from_secs(2), rx.next())
            .await
            .unwrap()
            .unwrap();
        let first_salience = match first {
            Event::Sensed { salience, .. } => salience,
            _ => panic!("expected sensed event"),
        };
        bus.publish(perception_event("hello"));
        let second = tokio::time::timeout(Duration::from_secs(2), rx.next())
            .await
            .unwrap()
            .unwrap();
        let second_salience = match second {
            Event::Sensed { salience, .. } => salience,
            _ => panic!("expected sensed event"),
        };
        assert!(second_salience < first_salience);
    }
}
