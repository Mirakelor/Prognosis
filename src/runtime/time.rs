use std::time::Duration;

use async_trait::async_trait;
use tokio::task::JoinHandle;

use crate::runtime::actor::{ActorContext, CognitiveActor};
use crate::runtime::event::{Event, EventKind, EventMeta};

pub struct TimeActor {
    interval: Duration,
    tick: u64,
    beat: Option<JoinHandle<()>>,
}

impl TimeActor {
    pub fn new(interval: Duration) -> Self {
        Self {
            interval,
            tick: 0,
            beat: None,
        }
    }
}

#[async_trait]
impl CognitiveActor for TimeActor {
    fn id(&self) -> &str {
        "time"
    }

    fn subscriptions(&self) -> Vec<EventKind> {
        vec![EventKind::Cycle]
    }

    async fn handle(&mut self, event: &Event, ctx: &mut ActorContext) -> Vec<Event> {
        if let Event::CycleStart { meta } = event {
            if let Some(beat) = &self.beat {
                beat.abort();
            }
            self.tick += 1;
            let bus = ctx.bus();
            let interval = self.interval;
            let cycle_id = meta.cycle_id;
            let mut tick = self.tick;
            self.beat = Some(tokio::spawn(async move {
                loop {
                    tokio::time::sleep(interval).await;
                    bus.publish(Event::Tick {
                        meta: EventMeta {
                            cycle_id,
                            timestamp: tick,
                        },
                    });
                    tick += 1;
                }
            }));
        }
        vec![]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::actor::spawn_actor;
    use crate::runtime::bus::EventBus;
    use crate::runtime::event::EventMeta;
    use crate::runtime::types::CycleId;
    use futures::StreamExt;

    fn meta() -> EventMeta {
        EventMeta {
            cycle_id: CycleId(1),
            timestamp: 0,
        }
    }

    #[tokio::test]
    async fn tick_events_published_after_cycle_start() {
        let bus = EventBus::new(16);
        let (_h, ready) = spawn_actor(bus.clone(), TimeActor::new(Duration::from_millis(10)));
        ready.await.unwrap();
        let mut ticks = Box::pin(bus.subscribe_kinds(&[EventKind::Time]));

        bus.publish(Event::CycleStart { meta: meta() });

        let first = tokio::time::timeout(Duration::from_secs(2), ticks.next())
            .await
            .unwrap()
            .unwrap();
        assert!(matches!(first, Event::Tick { .. }));
        assert_eq!(first.meta().unwrap().cycle_id, CycleId(1));
    }
}
