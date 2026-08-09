use async_trait::async_trait;

use crate::runtime::actor::{ActorContext, CognitiveActor};
use crate::runtime::event::{Event, EventKind, StateRequest, StateResponse};
use crate::runtime::types::TaskSetState;

const PROGRESS_STEP: f32 = 0.1;

pub struct TaskSetActor {
    task_set: TaskSetState,
}

impl TaskSetActor {
    pub fn new() -> Self {
        Self {
            task_set: TaskSetState {
                goal: String::new(),
                priority: 1.0,
                progress: 0.0,
            },
        }
    }
}

impl Default for TaskSetActor {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl CognitiveActor for TaskSetActor {
    fn id(&self) -> &str {
        "task_set"
    }

    fn subscriptions(&self) -> Vec<EventKind> {
        vec![EventKind::TaskSet, EventKind::Cycle, EventKind::State]
    }

    async fn handle(&mut self, event: &Event, _ctx: &mut ActorContext) -> Vec<Event> {
        match event {
            Event::TaskSetUpdate { task_set, .. } => {
                self.task_set = task_set.clone();
                vec![]
            }
            Event::CycleComplete { meta, .. } => {
                self.task_set.progress = (self.task_set.progress + PROGRESS_STEP).min(1.0);
                vec![Event::TaskSetUpdate {
                    meta: *meta,
                    task_set: self.task_set.clone(),
                }]
            }
            Event::RequestState {
                meta,
                request: StateRequest::TaskSet,
                correlation_id,
            } => vec![Event::StateResponse {
                meta: *meta,
                response: StateResponse::TaskSet(self.task_set.clone()),
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
    use crate::runtime::types::{CycleId, CycleSummary};
    use std::time::Duration;

    fn meta() -> EventMeta {
        EventMeta {
            cycle_id: CycleId(1),

        }
    }

    #[tokio::test]
    async fn external_update_sets_task_set() {
        let bus = EventBus::new(16);
        let (_h, ready) = spawn_actor(bus.clone(), TaskSetActor::new());
        ready.await.unwrap();
        let mut rx = bus.subscribe();

        bus.publish(Event::TaskSetUpdate {
            meta: meta(),
            task_set: TaskSetState {
                goal: "answer the user".into(),
                priority: 1.0,
                progress: 0.0,
            },
        });
        bus.publish(Event::RequestState {
            meta: meta(),
            request: StateRequest::TaskSet,
            correlation_id: 1,
        });

        let task_set = tokio::time::timeout(Duration::from_secs(2), async {
            loop {
                match rx.recv().await.unwrap() {
                    Event::StateResponse {
                        response: StateResponse::TaskSet(task_set),
                        correlation_id: 1,
                        ..
                    } => return task_set,
                    _ => continue,
                }
            }
        })
        .await
        .unwrap();
        assert_eq!(task_set.goal, "answer the user");
    }

    #[tokio::test]
    async fn cycle_complete_advances_progress() {
        let bus = EventBus::new(16);
        let (_h, ready) = spawn_actor(bus.clone(), TaskSetActor::new());
        ready.await.unwrap();
        let mut rx = bus.subscribe();

        bus.publish(Event::TaskSetUpdate {
            meta: meta(),
            task_set: TaskSetState {
                goal: "answer the user".into(),
                priority: 1.0,
                progress: 0.0,
            },
        });
        bus.publish(Event::CycleComplete {
            meta: meta(),
            summary: CycleSummary::default(),
        });

        let task_set = tokio::time::timeout(Duration::from_secs(2), async {
            loop {
                match rx.recv().await.unwrap() {
                    Event::TaskSetUpdate { task_set, .. } if task_set.progress > 0.0 => {
                        return task_set;
                    }
                    _ => continue,
                }
            }
        })
        .await
        .unwrap();
        assert!(task_set.progress > 0.0);
    }
}
