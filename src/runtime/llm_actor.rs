use std::sync::Arc;

use async_trait::async_trait;
use futures::StreamExt;
use tokio_util::sync::CancellationToken;

use crate::runtime::actor::{ActorContext, CognitiveActor};
use crate::runtime::event::{Event, EventKind, EventMeta};
use crate::runtime::ports::LlmPort;
use crate::runtime::types::{GenerateRequest, ModulationContext};

struct ActiveGeneration {
    cancel: CancellationToken,
}

pub struct LlmActor {
    port: Arc<dyn LlmPort>,
    pending_modulation: ModulationContext,
    active: Option<ActiveGeneration>,
}

impl LlmActor {
    pub fn new(port: Arc<dyn LlmPort>) -> Self {
        Self {
            port,
            pending_modulation: ModulationContext::default(),
            active: None,
        }
    }

    fn merge_modulation(base: &mut ModulationContext, pending: &ModulationContext) {
        if let Some(temperature) = pending.temperature {
            base.temperature = Some(temperature);
        }
        if let Some(top_p) = pending.top_p {
            base.top_p = Some(top_p);
        }
        if let Some(reasoning_effort) = pending.reasoning_effort {
            base.reasoning_effort = Some(reasoning_effort);
        }
        if let Some(n) = pending.n {
            base.n = Some(n);
        }
        if let Some(model) = &pending.model {
            base.model = Some(model.clone());
        }
        if base.injected_messages.is_empty() {
            base.injected_messages = pending.injected_messages.clone();
        }
    }

    fn start_generation(&mut self, meta: EventMeta, mut request: GenerateRequest, bus: crate::runtime::bus::EventBus) {
        if let Some(active) = &self.active {
            active.cancel.cancel();
        }
        Self::merge_modulation(&mut request.modulation, &self.pending_modulation);
        let cancel = CancellationToken::new();
        let port = self.port.clone();
        let task_cancel = cancel.clone();
        let _handle = tokio::spawn(async move {
            let stream = match port.generate(&request, &task_cancel).await {
                Ok(stream) => stream,
                Err(err) => {
                    bus.publish(Event::GenerationError {
                        meta,
                        error: err.to_string(),
                    });
                    bus.publish(Event::StreamEnd { meta, usage: None });
                    return;
                }
            };
            let mut stream = stream;
            let mut usage = None;
            loop {
                if task_cancel.is_cancelled() {
                    break;
                }
                match stream.next().await {
                    Some(Ok(chunk)) => {
                        if chunk.usage.is_some() {
                            usage = chunk.usage;
                        }
                        bus.publish(Event::Chunk { meta, chunk });
                    }
                    Some(Err(err)) => {
                        bus.publish(Event::GenerationError {
                            meta,
                            error: err.to_string(),
                        });
                        break;
                    }
                    None => break,
                }
            }
            bus.publish(Event::StreamEnd { meta, usage });
        });
        self.active = Some(ActiveGeneration { cancel });
    }
}

#[async_trait]
impl CognitiveActor for LlmActor {
    fn id(&self) -> &str {
        "llm"
    }

    fn subscriptions(&self) -> Vec<EventKind> {
        vec![EventKind::Generation, EventKind::Modulation]
    }

    async fn handle(&mut self, event: &Event, ctx: &mut ActorContext) -> Vec<Event> {
        match event {
            Event::Generate { meta, request } => {
                self.start_generation(*meta, request.clone(), ctx.bus());
            }
            Event::Modulate { modulation, .. } => {
                self.pending_modulation = modulation.clone();
            }
            Event::CancelGeneration { .. } => {
                if let Some(active) = &self.active {
                    active.cancel.cancel();
                }
            }
            _ => {}
        }
        vec![]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adapter::types::{ChunkDelta, CompletionChunk, Message};
    use crate::runtime::actor::spawn_actor;
    use crate::runtime::bus::EventBus;
    use crate::runtime::event::EventMeta;
    use crate::adapter::types::{ReasoningEffort, Temperature};
    use crate::runtime::types::CycleId;
    use std::sync::Mutex;

    fn meta() -> EventMeta {
        EventMeta {
            cycle_id: CycleId(1),
            timestamp: 0,
        }
    }

    fn chunk(content: &str) -> CompletionChunk {
        CompletionChunk {
            model: "test-model".into(),
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
        }
    }

    struct FakePort {
        received: Arc<Mutex<Vec<GenerateRequest>>>,
        chunks: Vec<CompletionChunk>,
    }

    #[async_trait]
    impl LlmPort for FakePort {
        async fn generate<'a>(
            &'a self,
            request: &'a GenerateRequest,
            cancel: &'a CancellationToken,
        ) -> Result<
            std::pin::Pin<
                Box<
                    dyn futures::Stream<
                            Item = Result<CompletionChunk, crate::adapter::error::AdapterError>,
                        > + Send
                        + 'a,
                >,
            >,
            crate::adapter::error::AdapterError,
        > {
            self.received.lock().unwrap().push(request.clone());
            let chunks = self.chunks.clone();
            let cancel = cancel.clone();
            let stream = async_stream::stream! {
                for chunk in chunks {
                    if cancel.is_cancelled() {
                        break;
                    }
                    tokio::task::yield_now().await;
                    yield Ok(chunk);
                }
            };
            Ok(Box::pin(stream))
        }
    }

    fn generate_event(meta: EventMeta, modulation: ModulationContext) -> Event {
        Event::Generate {
            meta,
            request: GenerateRequest {
                messages: vec![Message::user("hello")],
                modulation,
                tools: None,
            },
        }
    }

    #[tokio::test]
    async fn generate_publishes_chunks_and_stream_end() {
        let bus = EventBus::new(32);
        let received = Arc::new(Mutex::new(Vec::new()));
        let port = Arc::new(FakePort {
            received,
            chunks: vec![chunk("hi"), chunk(" there")],
        });
        let (_h, ready) = spawn_actor(bus.clone(), LlmActor::new(port.clone()));
        ready.await.unwrap();
        let mut rx = bus.subscribe();

        bus.publish(generate_event(meta(), ModulationContext::default()));

        let mut contents = vec![];
        loop {
            match rx.recv().await.unwrap() {
                Event::Chunk { chunk, .. } => contents.push(chunk.content().unwrap_or("").to_string()),
                Event::StreamEnd { .. } => break,
                _ => continue,
            }
        }
        assert_eq!(contents, vec!["hi", " there"]);
    }

    #[tokio::test]
    async fn cancel_generation_stops_stream() {
        let bus = EventBus::new(32);
        let port = Arc::new(FakePort {
            received: Arc::new(Mutex::new(Vec::new())),
            chunks: vec![chunk("a"), chunk("b"), chunk("c"), chunk("d")],
        });
        let (_h, ready) = spawn_actor(bus.clone(), LlmActor::new(port.clone()));
        ready.await.unwrap();
        let mut rx = bus.subscribe();

        bus.publish(generate_event(meta(), ModulationContext::default()));
        bus.publish(Event::CancelGeneration { meta: meta() });

        let mut count = 0;
        loop {
            match rx.recv().await.unwrap() {
                Event::Chunk { .. } => count += 1,
                Event::StreamEnd { .. } => break,
                _ => continue,
            }
        }
        assert!(count < 4);
    }

    #[tokio::test]
    async fn modulate_merges_into_next_generate() {
        let bus = EventBus::new(32);
        let received = Arc::new(Mutex::new(Vec::new()));
        let port = Arc::new(FakePort {
            received,
            chunks: vec![chunk("ok")],
        });
        let (_h, ready) = spawn_actor(bus.clone(), LlmActor::new(port.clone()));
        ready.await.unwrap();
        let mut rx = bus.subscribe();

        let modulation = ModulationContext {
            temperature: Some(Temperature::new(0.1).unwrap()),
            ..Default::default()
        };
        bus.publish(Event::Modulate {
            meta: meta(),
            modulation,
        });
        bus.publish(generate_event(meta(), ModulationContext::default()));

        loop {
            match rx.recv().await.unwrap() {
                Event::StreamEnd { .. } => break,
                _ => continue,
            }
        }
        let received = port.received.lock().unwrap();
        let request = received.first().unwrap();
        assert_eq!(request.modulation.temperature.map(|t| t.get()), Some(0.1));
    }

    #[tokio::test]
    async fn modulate_max_overrides_generation_default_high() {
        let bus = EventBus::new(32);
        let received = Arc::new(Mutex::new(Vec::new()));
        let port = Arc::new(FakePort {
            received,
            chunks: vec![chunk("ok")],
        });
        let (_h, ready) = spawn_actor(bus.clone(), LlmActor::new(port.clone()));
        ready.await.unwrap();
        let mut rx = bus.subscribe();

        let modulation = ModulationContext {
            reasoning_effort: Some(ReasoningEffort::Max),
            ..Default::default()
        };
        bus.publish(Event::Modulate {
            meta: meta(),
            modulation,
        });
        bus.publish(generate_event(
            meta(),
            ModulationContext {
                reasoning_effort: Some(ReasoningEffort::High),
                ..Default::default()
            },
        ));

        loop {
            match rx.recv().await.unwrap() {
                Event::StreamEnd { .. } => break,
                _ => continue,
            }
        }
        let received = port.received.lock().unwrap();
        let request = received.first().unwrap();
        assert_eq!(
            request.modulation.reasoning_effort,
            Some(ReasoningEffort::Max),
            "modulator boost must override the generation default"
        );
    }
}
