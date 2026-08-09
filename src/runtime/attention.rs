use async_trait::async_trait;

use crate::runtime::actor::{ActorContext, CognitiveActor};
use crate::runtime::event::{Event, EventKind, StateRequest, StateResponse};
use crate::runtime::types::{AttentionFocus, PerceptionPayload};

pub struct AttentionActor {
    goal: String,
    priority: f32,
    current: Option<AttentionFocus>,
}

impl AttentionActor {
    pub fn new() -> Self {
        Self {
            goal: String::new(),
            priority: 1.0,
            current: None,
        }
    }

    fn relevance(&self, payload: &PerceptionPayload) -> f32 {
        if self.goal.is_empty() {
            return 0.5;
        }
        let lower = payload.content.to_lowercase();
        let goal_terms: Vec<String> = self
            .goal
            .to_lowercase()
            .split_whitespace()
            .filter(|term| term.chars().count() > 2)
            .map(str::to_string)
            .collect();
        if goal_terms.is_empty() {
            return 0.5;
        }
        let hits = goal_terms
            .iter()
            .filter(|term| lower.contains(term.as_str()))
            .count();
        (hits as f32 / goal_terms.len() as f32).clamp(0.0, 1.0)
    }
}

impl Default for AttentionActor {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl CognitiveActor for AttentionActor {
    fn id(&self) -> &str {
        "attention"
    }

    fn subscriptions(&self) -> Vec<EventKind> {
        vec![EventKind::Sensed, EventKind::TaskSet, EventKind::Cycle]
    }

    async fn handle(&mut self, event: &Event, ctx: &mut ActorContext) -> Vec<Event> {
        match event {
            Event::Sensed {
                meta,
                payload,
                salience,
                ..
            } => {
                if let Some(StateResponse::TaskSet(task_set)) =
                    ctx.request_state(StateRequest::TaskSet).await
                {
                    self.goal = task_set.goal;
                    self.priority = task_set.priority;
                }
                let norepinephrine = match ctx.request_state(StateRequest::Modulator).await {
                    Some(StateResponse::Modulator(state)) => state.norepinephrine,
                    _ => 0.5,
                };
                let hold = 1.0 + norepinephrine * 0.5;
                let relevance = self.relevance(payload) * (0.5 + 0.5 * self.priority);
                let focus = AttentionFocus {
                    payload: payload.clone(),
                    salience: *salience,
                    relevance,
                };
                let wins = match &self.current {
                    Some(current) => {
                        focus.salience * focus.relevance
                            > current.salience * current.relevance * hold
                    }
                    None => true,
                };
                if wins {
                    self.current = Some(focus.clone());
                    vec![Event::Attention {
                        meta: *meta,
                        focus,
                    }]
                } else {
                    vec![]
                }
            }
            Event::TaskSetUpdate { task_set, .. } => {
                self.goal = task_set.goal.clone();
                self.priority = task_set.priority;
                vec![]
            }
            Event::CycleComplete { .. } => {
                self.current = None;
                vec![]
            }
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
    use crate::runtime::types::{CycleId, PerceptionFeatures, PerceptionSource, TaskSetState};
    use futures::StreamExt;
    use std::time::Duration;

    fn meta() -> EventMeta {
        EventMeta {
            cycle_id: CycleId(1),

        }
    }

    fn sensed_event(content: &str, salience: f32) -> Event {
        Event::Sensed {
            meta: meta(),
            payload: PerceptionPayload {
                source: PerceptionSource::User,
                content: content.into(),
                salience,
            },
            features: PerceptionFeatures::default(),
            salience,
        }
    }

    #[tokio::test]
    async fn attention_uses_goal_relevance() {
        let bus = EventBus::new(16);
        let (_h, ready) = spawn_actor(bus.clone(), AttentionActor::new());
        ready.await.unwrap();
        let mut rx = Box::pin(bus.subscribe_kinds(&[EventKind::Attention]));

        bus.publish(Event::TaskSetUpdate {
            meta: meta(),
            task_set: TaskSetState {
                goal: "find weather".into(),
                priority: 1.0,
                progress: 0.0,
            },
        });
        bus.publish(sensed_event("tell me about weather in beijing", 0.6));
        bus.publish(sensed_event("how are you doing", 0.9));

        let first = tokio::time::timeout(Duration::from_secs(2), rx.next())
            .await
            .unwrap()
            .unwrap();
        let focus = match first {
            Event::Attention { focus, .. } => focus,
            _ => panic!("expected attention event"),
        };
        assert!(focus.relevance > 0.0);
    }

    #[tokio::test]
    async fn attention_keeps_only_the_winner() {
        let bus = EventBus::new(16);
        let (_h, ready) = spawn_actor(bus.clone(), AttentionActor::new());
        ready.await.unwrap();
        let mut rx = Box::pin(bus.subscribe_kinds(&[EventKind::Attention]));

        bus.publish(Event::TaskSetUpdate {
            meta: meta(),
            task_set: TaskSetState {
                goal: "weather".into(),
                priority: 1.0,
                progress: 0.0,
            },
        });
        bus.publish(sensed_event("weather report please", 0.9));
        bus.publish(sensed_event("how are you", 0.2));

        let first = tokio::time::timeout(Duration::from_secs(2), rx.next())
            .await
            .unwrap()
            .unwrap();
        match first {
            Event::Attention { focus, .. } => {
                assert_eq!(focus.payload.content, "weather report please");
            }
            _ => panic!("expected attention event"),
        }
        let stray = tokio::time::timeout(Duration::from_millis(300), rx.next()).await;
        assert!(stray.is_err(), "losing candidate must not capture attention");
    }

    #[tokio::test]
    async fn new_cycle_releases_focus_for_next_input() {
        let bus = EventBus::new(16);
        let (_h, ready) = spawn_actor(bus.clone(), AttentionActor::new());
        ready.await.unwrap();
        let mut rx = Box::pin(bus.subscribe_kinds(&[EventKind::Attention]));

        bus.publish(sensed_event("first round question", 0.5));
        let first = tokio::time::timeout(Duration::from_secs(2), rx.next())
            .await
            .unwrap()
            .unwrap();
        assert!(matches!(first, Event::Attention { .. }));

        bus.publish(Event::CycleComplete {
            meta: meta(),
            summary: crate::runtime::types::CycleSummary {
                rpe: None,
                error: None,
                uncertainty: None,
                decision: None,
                modulation: None,
                user_input: None,
            },
        });
        bus.publish(sensed_event("second round tool request", 0.5));

        let second = tokio::time::timeout(Duration::from_secs(2), rx.next())
            .await
            .unwrap()
            .unwrap();
        match second {
            Event::Attention { focus, .. } => {
                assert_eq!(focus.payload.content, "second round tool request");
            }
            _ => panic!("expected attention for the next cycle"),
        }
    }
}
