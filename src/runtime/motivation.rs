use async_trait::async_trait;

use crate::runtime::actor::{ActorContext, CognitiveActor};
use crate::runtime::event::{Event, EventKind, StateRequest, StateResponse};
use crate::runtime::types::DriveState;

const SALIENCE_SMOOTH: f32 = 0.7;
const DRIVE_UPDATE_THRESHOLD: f32 = 0.05;

pub struct MotivationActor {
    drive: DriveState,
}

impl MotivationActor {
    pub fn new() -> Self {
        Self {
            drive: DriveState::default(),
        }
    }

    fn update(&mut self, event_meta: crate::runtime::event::EventMeta, drive: DriveState) -> Vec<Event> {
        let changed = (drive.homeostatic - self.drive.homeostatic).abs() >= DRIVE_UPDATE_THRESHOLD
            || (drive.curiosity - self.drive.curiosity).abs() >= DRIVE_UPDATE_THRESHOLD
            || (drive.salience - self.drive.salience).abs() >= DRIVE_UPDATE_THRESHOLD;
        self.drive = drive;
        if changed {
            vec![Event::DriveUpdate {
                meta: event_meta,
                drives: self.drive,
            }]
        } else {
            vec![]
        }
    }
}

impl Default for MotivationActor {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl CognitiveActor for MotivationActor {
    fn id(&self) -> &str {
        "motivation"
    }

    fn subscriptions(&self) -> Vec<EventKind> {
        vec![EventKind::Error, EventKind::TaskSet, EventKind::Modulation, EventKind::State]
    }

    async fn handle(&mut self, event: &Event, _ctx: &mut ActorContext) -> Vec<Event> {
        match event {
            Event::Rpe { meta, rpe, .. } => {
                let mut drive = self.drive;
                drive.salience = drive.salience * SALIENCE_SMOOTH
                    + rpe.0.abs() * (1.0 - SALIENCE_SMOOTH);
                self.update(*meta, drive)
            }
            Event::TaskSetUpdate { meta, task_set, .. } => {
                let mut drive = self.drive;
                drive.homeostatic = 1.0 - task_set.progress;
                self.update(*meta, drive)
            }
            Event::ModulatorUpdate { meta, state, .. } => {
                let mut drive = self.drive;
                drive.curiosity = state.acetylcholine;
                self.update(*meta, drive)
            }
            Event::RequestState {
                meta,
                request: StateRequest::Motivation,
                correlation_id,
            } => vec![Event::StateResponse {
                meta: *meta,
                response: StateResponse::Motivation(self.drive),
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
    use crate::runtime::types::{
        CognitiveMode, CycleId, ModulatorState, RpeSignal, TaskSetState,
    };
    use std::time::Duration;

    fn meta() -> EventMeta {
        EventMeta {
            cycle_id: CycleId(1),

        }
    }

    async fn query_drive(bus: &EventBus) -> DriveState {
        let mut rx = bus.subscribe();
        bus.publish(Event::RequestState {
            meta: meta(),
            request: StateRequest::Motivation,
            correlation_id: 7,
        });
        tokio::time::timeout(Duration::from_secs(2), async {
            loop {
                match rx.recv().await.unwrap() {
                    Event::StateResponse {
                        response: StateResponse::Motivation(drive),
                        correlation_id: 7,
                        ..
                    } => return drive,
                    _ => continue,
                }
            }
        })
        .await
        .unwrap()
    }

    #[tokio::test]
    async fn drives_follow_signals() {
        let bus = EventBus::new(16);
        let (_h, ready) = spawn_actor(bus.clone(), MotivationActor::new());
        ready.await.unwrap();

        bus.publish(Event::TaskSetUpdate {
            meta: meta(),
            task_set: TaskSetState {
                goal: "answer".into(),
                priority: 1.0,
                progress: 0.4,
            },
        });
        bus.publish(Event::Rpe {
            meta: meta(),
            rpe: RpeSignal(0.8),
        });
        bus.publish(Event::ModulatorUpdate {
            meta: meta(),
            state: ModulatorState {
                dopamine: 0.0,
                norepinephrine: 0.5,
                acetylcholine: 0.8,
                serotonin: 0.5,
            },
            mode: CognitiveMode::Automatic,
        });

        tokio::time::sleep(Duration::from_millis(100)).await;
        let drive = query_drive(&bus).await;
        assert!(drive.homeostatic > 0.5);
        assert!(drive.curiosity > 0.7);
        assert!(drive.salience > 0.0);
    }

    #[tokio::test]
    async fn drive_changes_broadcast_update() {
        let bus = EventBus::new(16);
        let (_h, ready) = spawn_actor(bus.clone(), MotivationActor::new());
        ready.await.unwrap();
        let mut rx = bus.subscribe();

        bus.publish(Event::TaskSetUpdate {
            meta: meta(),
            task_set: TaskSetState {
                goal: "answer".into(),
                priority: 1.0,
                progress: 0.4,
            },
        });

        let drive = tokio::time::timeout(Duration::from_secs(2), async {
            loop {
                match rx.recv().await.unwrap() {
                    Event::DriveUpdate { drives, .. } => return drives,
                    _ => continue,
                }
            }
        })
        .await
        .unwrap();
        assert!((drive.homeostatic - 0.6).abs() < 1e-3);
    }
}
