use async_trait::async_trait;

use crate::adapter::types::LogprobToken;
use crate::runtime::actor::{ActorContext, CognitiveActor};
use crate::runtime::event::{Event, EventKind};
const LOW_CONFIDENCE_LOGPROB: f32 = -2.0;
const QUALITY_SMOOTH: f32 = 0.8;

pub struct LanguageActor {
    low_confidence_tokens: usize,
    total_tokens: usize,
    quality: f32,
}

impl LanguageActor {
    pub fn new() -> Self {
        Self {
            low_confidence_tokens: 0,
            total_tokens: 0,
            quality: 0.0,
        }
    }

    fn observe_tokens(&mut self, tokens: &[LogprobToken]) {
        self.total_tokens += tokens.len();
        self.low_confidence_tokens += tokens
            .iter()
            .filter(|token| token.logprob < LOW_CONFIDENCE_LOGPROB)
            .count();
        if self.total_tokens > 0 {
            let density = self.low_confidence_tokens as f32 / self.total_tokens as f32;
            self.quality = self.quality * QUALITY_SMOOTH + density * (1.0 - QUALITY_SMOOTH);
        }
    }
}

impl Default for LanguageActor {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl CognitiveActor for LanguageActor {
    fn id(&self) -> &str {
        "language"
    }

    fn subscriptions(&self) -> Vec<EventKind> {
        vec![EventKind::Prediction, EventKind::Generation]
    }

    async fn handle(&mut self, event: &Event, _ctx: &mut ActorContext) -> Vec<Event> {
        match event {
            Event::Prediction { meta, trajectory } => {
                vec![Event::LanguageInsight {
                    meta: *meta,
                    intent: trajectory.intent,
                    quality: self.quality,
                }]
            }
            Event::Chunk { meta, chunk } => {
                if let Some(logprobs) = &chunk.delta.logprobs {
                    self.observe_tokens(&logprobs.tokens);
                }
                let _ = meta;
                vec![]
            }
            _ => vec![],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adapter::types::{ChunkDelta, CompletionChunk, ContentLogprobs, LogprobToken};
    use crate::runtime::actor::spawn_actor;
    use crate::runtime::bus::EventBus;
    use crate::runtime::event::EventMeta;
    use crate::runtime::types::{CycleId, IntentKind, PredictionTrajectory};
    use futures::StreamExt;
    use std::time::Duration;

    fn meta() -> EventMeta {
        EventMeta {
            cycle_id: CycleId(1),
            timestamp: 0,
        }
    }

    fn prediction_event(intent: IntentKind) -> Event {
        Event::Prediction {
            meta: meta(),
            trajectory: PredictionTrajectory {
                topics: vec!["a".into()],
                key_elements: vec!["b".into()],
                direction: 0.5,
                intent,
                intent_candidates: vec![intent],
                reaction: "the user asks for details".into(),
                reaction_sentiment: 0.2,
            },
        }
    }

    fn chunk_event(logprob: f32) -> Event {
        Event::Chunk {
            meta: meta(),
            chunk: CompletionChunk {
                model: "test".into(),
                index: 0,
                delta: ChunkDelta {
                    role: None,
                    content: Some("x".into()),
                    tool_calls: vec![],
                    logprobs: Some(ContentLogprobs {
                        tokens: vec![LogprobToken {
                            token: "x".into(),
                            logprob,
                        }],
                    }),
                    reasoning: None,
                },
                finish_reason: None,
                usage: None,
                request_id: None,
            },
        }
    }

    async fn next_insight(rx: &mut (impl futures::Stream<Item = Event> + std::marker::Unpin)) -> (IntentKind, f32) {
        loop {
            match tokio::time::timeout(Duration::from_secs(2), rx.next())
                .await
                .unwrap()
                .unwrap()
            {
                Event::LanguageInsight { intent, quality, .. } => return (intent, quality),
                _ => continue,
            }
        }
    }

    #[tokio::test]
    async fn intent_comes_from_semantic_prediction() {
        let bus = EventBus::new(16);
        let (_h, ready) = spawn_actor(bus.clone(), LanguageActor::new());
        ready.await.unwrap();
        let mut rx = Box::pin(bus.subscribe_kinds(&[EventKind::Language]));

        bus.publish(prediction_event(IntentKind::Question));
        let (intent, _) = next_insight(&mut rx).await;
        assert_eq!(intent, IntentKind::Question);

        bus.publish(prediction_event(IntentKind::Command));
        let (intent, _) = next_insight(&mut rx).await;
        assert_eq!(intent, IntentKind::Command);
    }

    #[tokio::test]
    async fn low_confidence_tokens_raise_quality_signal() {
        let bus = EventBus::new(16);
        let (_h, ready) = spawn_actor(bus.clone(), LanguageActor::new());
        ready.await.unwrap();
        let mut rx = Box::pin(bus.subscribe_kinds(&[EventKind::Language]));

        bus.publish(chunk_event(-0.3));
        bus.publish(chunk_event(-5.0));
        bus.publish(prediction_event(IntentKind::Statement));

        let (_, quality) = next_insight(&mut rx).await;
        assert!(quality > 0.0, "low-confidence tokens should raise the quality signal");
    }
}
