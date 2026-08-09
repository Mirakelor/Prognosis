pub mod action_selection;
pub mod actor;
pub mod attention;
pub mod bus;
pub mod emotion;
pub mod error_comparator;
pub mod event;
pub mod inhibition;
pub mod language;
pub mod llm_actor;
pub mod longterm_memory;
pub mod metacognition;
pub mod modulator;
pub mod motivation;
pub mod perception;
pub mod ports;
pub mod prediction;
pub mod task_set;
pub mod time;
pub mod trace;
pub mod types;
pub mod working_memory;

use std::sync::{Arc, Mutex};
use std::time::Duration;

use tokio::sync::oneshot;

use crate::runtime::actor::{spawn_actor, CognitiveActor};
use crate::runtime::bus::EventBus;

use crate::adapter::types::ToolDefinition;
use crate::runtime::event::Event;
use crate::runtime::types::TraceRecord;
use crate::runtime::ports::LlmPort;

pub struct Runtime {
    bus: EventBus,
    ready: Vec<oneshot::Receiver<()>>,
    trace_records: Arc<Mutex<Vec<TraceRecord>>>,
}

impl Runtime {
    pub fn new(port: Arc<dyn LlmPort>, tick_interval: Duration, tools: Vec<ToolDefinition>) -> Self {
        let bus = EventBus::new(4096);
        let mut ready = Vec::new();
        let trace_actor = trace::TraceActor::new();
        let trace_records = trace_actor.records();
        let mut actors: Vec<Box<dyn CognitiveActor>> = vec![
            Box::new(llm_actor::LlmActor::new(port.clone())),
            Box::new(time::TimeActor::new(tick_interval)),
            Box::new(trace_actor),
            Box::new(perception::PerceptionActor::new()),
            Box::new(attention::AttentionActor::new()),
            Box::new(inhibition::InhibitionActor::new()),
            Box::new(prediction::PredictionActor::new(port.clone()).with_tools(tools)),
            Box::new(error_comparator::ErrorComparatorActor::new(port.clone())),
            Box::new(modulator::ModulatorActor::new()),
            Box::new(emotion::EmotionActor::new()),
            Box::new(working_memory::WorkingMemoryActor::new()),
            Box::new(longterm_memory::LongTermMemoryActor::new()),
            Box::new(motivation::MotivationActor::new()),
            Box::new(task_set::TaskSetActor::new()),
            Box::new(action_selection::ActionSelectionActor::new()),
            Box::new(language::LanguageActor::new()),
            Box::new(metacognition::MetacognitionActor::new()),
        ];
        for actor in actors.drain(..) {
            let (_, ready_rx) = spawn_actor(bus.clone(), actor);
            ready.push(ready_rx);
        }
        Self {
            bus,
            ready,
            trace_records,
        }
    }

    pub fn bus(&self) -> EventBus {
        self.bus.clone()
    }

    pub fn publish(&self, event: Event) {
        self.bus.publish(event);
    }

    pub async fn wait_ready(&mut self) {
        for rx in self.ready.drain(..) {
            let _ = rx.await;
        }
    }

    pub fn shutdown(&self) {
        self.bus.publish(Event::Shutdown);
    }

    pub fn trace_records(&self) -> Arc<Mutex<Vec<TraceRecord>>> {
        self.trace_records.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adapter::error::AdapterError;
    use crate::adapter::types::{ChunkDelta, CompletionChunk, FinishReason};
    use crate::runtime::event::EventMeta;
    use crate::runtime::types::{
        ActionCandidate, CycleId, PerceptionPayload, PerceptionSource, TaskSetState,
    };
    use std::pin::Pin;

    struct FakePort;

    #[async_trait::async_trait]
    impl LlmPort for FakePort {
        async fn generate<'a>(
            &'a self,
            request: &'a crate::runtime::types::GenerateRequest,
            _cancel: &'a tokio_util::sync::CancellationToken,
        ) -> Result<
            Pin<Box<dyn futures::Stream<Item = Result<CompletionChunk, AdapterError>> + Send + 'a>>,
            AdapterError,
        > {
            let is_prediction = request.messages[0]
                .content
                .to_plain_text()
                .contains("You are a predictor for a cognitive agent");
            let chunks = if is_prediction {
                vec![chunk(
                    r#"{"topics":["weather"],"key_elements":["sunny"],"direction":0.5}"#,
                )]
            } else {
                vec![
                    chunk("The weather"),
                    chunk(" is sunny"),
                    chunk(" with usage"),
                ]
            };
            Ok(Box::pin(futures::stream::iter(chunks.into_iter().map(Ok))))
        }
    }

    fn chunk(content: &str) -> CompletionChunk {
        CompletionChunk {
            model: "fake".into(),
            index: 0,
            delta: ChunkDelta {
                role: None,
                content: Some(content.into()),
                tool_calls: vec![],
                logprobs: None,
                reasoning: None,
            },
            finish_reason: Some(FinishReason::Stop),
            usage: None,
            request_id: None,
        }
    }

    fn meta() -> EventMeta {
        EventMeta {
            cycle_id: CycleId(1),

        }
    }

    #[tokio::test]
    async fn full_cycle_smoke_test() {
        let mut runtime = Runtime::new(Arc::new(FakePort), Duration::from_millis(10), vec![]);
        runtime.wait_ready().await;
        let mut rx = runtime.bus().subscribe();

        runtime.publish(Event::TaskSetUpdate {
            meta: meta(),
            task_set: TaskSetState {
                goal: "answer the user".into(),
                priority: 1.0,
                progress: 0.0,
            },
        });
        runtime.publish(Event::CycleStart { meta: meta() });
        runtime.publish(Event::Perception {
            meta: meta(),
            payload: PerceptionPayload {
                source: PerceptionSource::User,
                content: "what is the weather?".into(),
                salience: 0.5,
            },
        });

        let summary = tokio::time::timeout(Duration::from_secs(10), async {
            loop {
                match rx.recv().await.unwrap() {
                    Event::CycleComplete { summary, .. } => return summary,
                    _ => continue,
                }
            }
        })
        .await
        .expect("the full cognitive cycle should complete");

        assert!(summary.decision.is_some());
        match summary.decision.unwrap().candidate {
            ActionCandidate::Respond { content } => {
                assert!(content.contains("The weather is sunny"));
            }
            _ => panic!("expected a respond action in the smoke cycle"),
        }
    }
}
