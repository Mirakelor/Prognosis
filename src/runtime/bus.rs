use std::collections::HashSet;

use futures::{stream, Stream};
use tokio::sync::broadcast;

use crate::runtime::event::{Event, EventKind};

#[derive(Clone)]
pub struct EventBus {
    tx: broadcast::Sender<Event>,
}

impl EventBus {
    pub fn new(capacity: usize) -> Self {
        let (tx, _) = broadcast::channel(capacity);
        Self { tx }
    }

    pub fn publish(&self, event: Event) {
        let _ = self.tx.send(event);
    }
    pub fn subscribe(&self) -> broadcast::Receiver<Event> {
        self.tx.subscribe()
    }

    pub fn subscribe_kinds(
        &self,
        kinds: &[EventKind],
    ) -> impl Stream<Item = Event> + Send + 'static {
        let allowed: std::sync::Arc<HashSet<EventKind>> =
            std::sync::Arc::new(kinds.iter().copied().collect());
        stream::unfold(self.tx.subscribe(), move |mut rx| {
            let allowed = allowed.clone();
            async move {
                loop {
                    match rx.recv().await {
                        Ok(event)
                            if allowed.contains(&event.kind())
                                || matches!(event, Event::Shutdown) =>
                        {
                            return Some((event, rx));
                        }
                        Ok(_) => continue,
                        Err(broadcast::error::RecvError::Lagged(_)) => continue,
                        Err(broadcast::error::RecvError::Closed) => return None,
                    }
                }
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::StreamExt;
    use crate::runtime::types::{CycleId, PerceptionPayload, PerceptionSource};

    fn meta() -> crate::runtime::event::EventMeta {
        crate::runtime::event::EventMeta {
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

    #[test]
    fn publish_reaches_all_subscribers() {
        let bus = EventBus::new(16);
        let mut rx1 = bus.subscribe();
        let mut rx2 = bus.subscribe();
        bus.publish(perception_event());
        assert_eq!(rx1.try_recv().unwrap().kind(), EventKind::Perception);
        assert_eq!(rx2.try_recv().unwrap().kind(), EventKind::Perception);
    }

    #[tokio::test]
    async fn subscribe_kinds_filters_events() {
        let bus = EventBus::new(16);
        let mut filtered = Box::pin(bus.subscribe_kinds(&[EventKind::Perception]));
        bus.publish(Event::CycleStart { meta: meta() });
        bus.publish(perception_event());
        let received = filtered.next().await.unwrap();
        assert_eq!(received.kind(), EventKind::Perception);
    }
}
