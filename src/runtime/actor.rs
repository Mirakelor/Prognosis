use std::time::Duration;

use async_trait::async_trait;
use futures::StreamExt;

use crate::runtime::bus::EventBus;
use crate::runtime::event::{Event, EventKind, EventMeta, StateRequest, StateResponse};

#[async_trait]
pub trait CognitiveActor: Send {
    fn id(&self) -> &str;

    fn subscriptions(&self) -> Vec<EventKind>;

    async fn handle(&mut self, event: &Event, ctx: &mut ActorContext) -> Vec<Event>;
}

#[async_trait]
impl CognitiveActor for Box<dyn CognitiveActor> {
    fn id(&self) -> &str {
        (**self).id()
    }

    fn subscriptions(&self) -> Vec<EventKind> {
        (**self).subscriptions()
    }

    async fn handle(&mut self, event: &Event, ctx: &mut ActorContext) -> Vec<Event> {
        (**self).handle(event, ctx).await
    }
}

pub struct ActorContext {
    bus: EventBus,
    current_meta: EventMeta,
    next_correlation: u64,
}

impl ActorContext {
    pub fn new(bus: EventBus, initial_meta: EventMeta) -> Self {
        Self {
            bus,
            current_meta: initial_meta,
            next_correlation: 0,
        }
    }

    pub fn meta(&self) -> EventMeta {
        self.current_meta
    }

    pub fn bus(&self) -> EventBus {
        self.bus.clone()
    }

    pub fn set_meta(&mut self, meta: EventMeta) {
        self.current_meta = meta;
    }

    pub fn publish(&self, event: Event) {
        self.bus.publish(event);
    }

    pub async fn request_state(&mut self, request: StateRequest) -> Option<StateResponse> {
        let correlation_id = self.next_correlation;
        self.next_correlation += 1;
        self.bus.publish(Event::RequestState {
            meta: self.current_meta,
            request,
            correlation_id,
        });
        let mut responses = Box::pin(self.bus.subscribe_kinds(&[EventKind::State]));
        tokio::time::timeout(Duration::from_millis(500), async {
            loop {
                match responses.next().await? {
                    Event::StateResponse {
                        correlation_id: id,
                        response,
                        ..
                    } if id == correlation_id => return Some(response),
                    _ => continue,
                }
            }
        })
        .await
        .ok()
        .flatten()
    }
}

pub fn spawn_actor<A: CognitiveActor + 'static>(
    bus: EventBus,
    actor: A,
) -> (tokio::task::JoinHandle<()>, tokio::sync::oneshot::Receiver<()>) {
    let (ready_tx, ready_rx) = tokio::sync::oneshot::channel();
    let handle = tokio::spawn(async move {
        let mut actor = actor;
        let subscriptions = actor.subscriptions();
        let mut events = Box::pin(bus.subscribe_kinds(&subscriptions));
        let _ = ready_tx.send(());
        let mut ctx = ActorContext::new(
            bus.clone(),
            EventMeta {
                cycle_id: crate::runtime::types::CycleId(0),
                timestamp: 0,
            },
        );
        while let Some(event) = events.next().await {
            if matches!(event, Event::Shutdown) {
                break;
            }
            if let Some(meta) = event.meta() {
                ctx.set_meta(meta);
            }
            let outputs = actor.handle(&event, &mut ctx).await;
            for output in outputs {
                ctx.publish(output);
            }
        }
    });
    (handle, ready_rx)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::event::EventMeta;
    use crate::runtime::types::{
        ActionCandidate, ActionDecision, CycleId, PerceptionPayload, PerceptionSource,
        TaskSetState,
    };

    fn meta() -> EventMeta {
        EventMeta {
            cycle_id: CycleId(1),
            timestamp: 0,
        }
    }

    fn perception_event() -> Event {
        Event::Perception {
            meta: meta(),
            payload: PerceptionPayload {
                source: PerceptionSource::User,
                content: "hello".into(),
                salience: 0.5,
            },
        }
    }

    struct StateProvider;

    #[async_trait]
    impl CognitiveActor for StateProvider {
        fn id(&self) -> &str {
            "state"
        }

        fn subscriptions(&self) -> Vec<EventKind> {
            vec![EventKind::State]
        }

        async fn handle(&mut self, event: &Event, _ctx: &mut ActorContext) -> Vec<Event> {
            match event {
                Event::RequestState {
                    meta,
                    request,
                    correlation_id,
                } => {
                    let response = match request {
                        StateRequest::TaskSet => StateResponse::TaskSet(TaskSetState {
                            goal: "answer".into(),
                            priority: 1.0,
                            progress: 0.0,
                        }),
                        _ => return vec![],
                    };
                    vec![Event::StateResponse {
                        meta: *meta,
                        response,
                        correlation_id: *correlation_id,
                    }]
                }
                _ => vec![],
            }
        }
    }

    struct EchoActor;

    #[async_trait]
    impl CognitiveActor for EchoActor {
        fn id(&self) -> &str {
            "echo"
        }

        fn subscriptions(&self) -> Vec<EventKind> {
            vec![EventKind::Perception, EventKind::State]
        }

        async fn handle(&mut self, event: &Event, ctx: &mut ActorContext) -> Vec<Event> {
            match event {
                Event::Perception { .. } => {
                    let goal = match ctx.request_state(StateRequest::TaskSet).await {
                        Some(StateResponse::TaskSet(task_set)) => task_set.goal,
                        _ => "none".into(),
                    };
                    vec![Event::ActionSelected {
                        meta: ctx.meta(),
                        decision: ActionDecision {
                            candidate: ActionCandidate::Respond {
                                content: format!("goal={goal}"),
                            },
                            confidence: 1.0,
                            go: true,
                        },
                    }]
                }
                _ => vec![],
            }
        }
    }

    #[tokio::test]
    async fn actor_requests_state_across_actors() {
        let bus = EventBus::new(16);
        let (_state, state_ready) = spawn_actor(bus.clone(), StateProvider);
        let mut out_rx = bus.subscribe();
        let (_echo, echo_ready) = spawn_actor(bus.clone(), EchoActor);
        state_ready.await.unwrap();
        echo_ready.await.unwrap();

        bus.publish(perception_event());

        let decision = tokio::time::timeout(Duration::from_secs(2), async {
            loop {
                match out_rx.recv().await {
                    Ok(Event::ActionSelected { decision, .. }) => return decision,
                    Ok(_) => continue,
                    Err(_) => continue,
                }
            }
        })
        .await
        .unwrap();

        match decision.candidate {
            ActionCandidate::Respond { content } => {
                assert_eq!(content, "goal=answer");
            }
            _ => panic!("expected respond action"),
        }
    }
}
