use async_trait::async_trait;

use crate::adapter::types::{Message, ReasoningEffort, Temperature};
use crate::runtime::actor::{ActorContext, CognitiveActor};
use crate::runtime::event::{Event, EventKind, StateRequest, StateResponse};
use crate::runtime::types::{CognitiveMode, ModulationContext, ModulatorState};

const NE_ERROR_WEIGHT: f32 = 0.4;
const ACH_RPE_WEIGHT: f32 = 0.4;
const SEROTONIN_SMOOTH: f32 = 0.7;
const CHANGE_THRESHOLD: f32 = 0.05;
const CONTROLLED_NE: f32 = 0.8;
const AUTOMATIC_NE: f32 = 0.3;
const ACH_SELFCHECK: f32 = 0.5;
const TEMPERATURE_STEP: f32 = 0.1;

pub struct ModulatorActor {
    state: ModulatorState,
    mode: CognitiveMode,
    last_error: Option<f32>,
    last_emitted: ModulatorState,
    last_mode: CognitiveMode,
    last_effort: Option<ReasoningEffort>,
    progress: f32,
}

impl ModulatorActor {
    pub fn new() -> Self {
        Self {
            state: ModulatorState::default(),
            mode: CognitiveMode::Automatic,
            last_error: None,
            last_emitted: ModulatorState::default(),
            last_mode: CognitiveMode::Automatic,
            last_effort: None,
            progress: 0.0,
        }
    }

    fn update_serotonin(&mut self, frustration: f32) {
        let patience_need =
            (frustration + (1.0 - self.progress) * 0.5).clamp(0.0, 1.0);
        self.state.serotonin = self.state.serotonin * SEROTONIN_SMOOTH
            + patience_need * (1.0 - SEROTONIN_SMOOTH);
    }

    fn update_mode(&mut self) -> CognitiveMode {
        let mode = if self.state.norepinephrine >= CONTROLLED_NE {
            CognitiveMode::Controlled
        } else if self.state.norepinephrine <= AUTOMATIC_NE {
            CognitiveMode::Automatic
        } else {
            self.mode
        };
        self.mode = mode;
        mode
    }

    fn current_effort(&self) -> Option<ReasoningEffort> {
        if self.state.acetylcholine >= ACH_SELFCHECK {
            Some(ReasoningEffort::Max)
        } else {
            None
        }
    }

    fn emit_if_changed(&mut self, meta: &crate::runtime::event::EventMeta, events: &mut Vec<Event>) {
        let mode = self.update_mode();
        let effort = self.current_effort();
        let changed = (self.state.norepinephrine - self.last_emitted.norepinephrine).abs()
            >= CHANGE_THRESHOLD
            || (self.state.acetylcholine - self.last_emitted.acetylcholine).abs()
                >= CHANGE_THRESHOLD
            || (self.state.serotonin - self.last_emitted.serotonin).abs() >= CHANGE_THRESHOLD
            || mode != self.last_mode
            || effort != self.last_effort;
        if changed {
            self.last_emitted = self.state;
            self.last_mode = mode;
            self.last_effort = effort;
            events.push(Event::ModulatorUpdate {
                meta: *meta,
                state: self.state,
                mode,
            });
            events.push(Event::Modulate {
                meta: *meta,
                modulation: self.build_modulation(),
            });
        }
    }

    fn build_modulation(&self) -> ModulationContext {
        let mut modulation = ModulationContext::default();
        match self.mode {
            CognitiveMode::Controlled => {
                modulation.temperature = Temperature::new(
                    (0.7 - TEMPERATURE_STEP * self.state.norepinephrine).max(0.2),
                )
                .ok();
                modulation
                    .injected_messages
                    .push(Message::system(
                        "The agent's prediction of your turn was off, and the prediction error on this turn is high.\
\n\n# What This Means\
\nNorepinephrine-driven processing shifts toward the controlled end: instead of the usual fluent default, this turn gets more of the system's attention. Prediction errors are normal — the agent is not always right — so this is not an alarm; it only means the default assumption deserves less trust this time.\
\n\n# What To Do\
\n- Answer from what this turn actually says: take the actual message at face value, including anything that contradicts what the agent's expectation assumed.\
\n- If the turn is missing something the expectation assumed, or directly contradicts it, name the difference in one line rather than smoothing it over.\
\n- When you correct yourself or change direction, say so briefly so the user can follow.\
\n\n# Not Required\
\n- You do not need to re-verify claims already established in this conversation, re-read earlier turns, or audit the whole history. Only this turn's relationship to the expectation matters.",
                    ));
            }
            CognitiveMode::Automatic => {
                modulation.temperature =
                    Temperature::new(0.7 + TEMPERATURE_STEP * self.state.acetylcholine).ok();
                if self.state.acetylcholine >= ACH_SELFCHECK {
                    modulation
                        .injected_messages
                        .push(Message::system(
                            "Uncertainty about the agent's own output is high on this turn.\
\n\n# What This Means\
\nThe system's estimate of its own output reliability is lower than usual, which predicts a higher chance of internal contradictions in the answer. This is a continuous signal, not a failure: it only means the answer deserves one consistency pass before it goes out.\
\n\n# What To Do\
\n- Before sending, read your drafted answer once, looking only for claims that contradict each other.\
\n- If two claims conflict, keep the one backed by evidence and drop the other.\
\n- If a contradiction cannot be resolved, say so explicitly rather than hiding it.\
\n\n# Not Required\
\n- One pass is enough. Do not rehearse the answer repeatedly, do not re-derive earlier conclusions, and do not second-guess claims that are already consistent.",
                        ));
                }
            }
        }
        if let Some(effort) = self.current_effort() {
            modulation.reasoning_effort = Some(effort);
        }
        modulation
    }
}

impl Default for ModulatorActor {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl CognitiveActor for ModulatorActor {
    fn id(&self) -> &str {
        "modulator"
    }

    fn subscriptions(&self) -> Vec<EventKind> {
        vec![EventKind::Error, EventKind::TaskSet, EventKind::State]
    }

    async fn handle(&mut self, event: &Event, _ctx: &mut ActorContext) -> Vec<Event> {
        let mut events = Vec::new();
        match event {
            Event::Rpe { meta, rpe } => {
                self.state.dopamine = rpe.0;
                self.state.acetylcholine = self.state.acetylcholine * (1.0 - ACH_RPE_WEIGHT)
                    + rpe.0.abs() * ACH_RPE_WEIGHT;
                self.update_serotonin((-rpe.0).max(0.0));
                self.emit_if_changed(meta, &mut events);
            }
            Event::TaskSetUpdate { meta, task_set } => {
                self.progress = task_set.progress;
                self.update_serotonin(0.0);
                self.emit_if_changed(meta, &mut events);
            }
            Event::ErrorComputed { meta, error } => {
                let weighted = error.weighted();
                self.last_error = Some(weighted);
                self.state.norepinephrine = self.state.norepinephrine * (1.0 - NE_ERROR_WEIGHT)
                    + weighted * NE_ERROR_WEIGHT;
                self.emit_if_changed(meta, &mut events);
            }
            Event::RequestState {
                meta,
                request: StateRequest::Modulator,
                correlation_id,
            } => events.push(Event::StateResponse {
                meta: *meta,
                response: StateResponse::Modulator(self.state),
                correlation_id: *correlation_id,
            }),
            _ => {}
        }
        events
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::actor::spawn_actor;
    use crate::runtime::bus::EventBus;
    use crate::runtime::event::EventMeta;
    use crate::runtime::types::{CycleId, PredictionError, RpeSignal, TaskSetState};
    use futures::StreamExt;
    use std::time::Duration;

    fn meta() -> EventMeta {
        EventMeta {
            cycle_id: CycleId(1),

        }
    }

    fn rpe_event(value: f32) -> Event {
        Event::Rpe {
            meta: meta(),
            rpe: RpeSignal(value),
        }
    }

    fn error_event(weighted: f32, confidence: f32) -> Event {
        Event::ErrorComputed {
            meta: meta(),
            error: PredictionError {
                semantic: weighted,
                confidence,
                precision: 1.0,
            },
        }
    }

    #[tokio::test]
    async fn surprise_raises_ne_and_switches_mode() {
        let bus = EventBus::new(64);
        let (_h, ready) = spawn_actor(bus.clone(), ModulatorActor::new());
        ready.await.unwrap();
        let mut rx = Box::pin(bus.subscribe_kinds(&[EventKind::Modulation]));

        bus.publish(rpe_event(0.5));
        for _ in 0..3 {
            bus.publish(error_event(1.0, 1.0));
        }

        let mut controlled_seen = false;
        for _ in 0..3 {
            match tokio::time::timeout(Duration::from_secs(2), rx.next())
                .await
                .unwrap()
                .unwrap()
            {
                Event::ModulatorUpdate { mode, .. } => {
                    if mode == CognitiveMode::Controlled {
                        controlled_seen = true;
                    }
                }
                _ => continue,
            }
            if controlled_seen {
                break;
            }
        }
        assert!(controlled_seen, "large error delta should switch to controlled mode");
    }

    #[tokio::test]
    async fn high_uncertainty_emits_modulation_with_correction() {
        let bus = EventBus::new(64);
        let (_h, ready) = spawn_actor(bus.clone(), ModulatorActor::new());
        ready.await.unwrap();
        let mut rx = Box::pin(bus.subscribe_kinds(&[EventKind::Modulation]));
        bus.publish(rpe_event(-0.9));
        for _ in 0..4 {
            bus.publish(error_event(0.2, 0.9));
        }

        let modulation = tokio::time::timeout(Duration::from_secs(2), async {
            loop {
                match rx.next().await.unwrap() {
                    Event::Modulate { modulation, .. }
                        if !modulation.injected_messages.is_empty()
                            || modulation.reasoning_effort.is_some() =>
                    {
                        return modulation;
                    }
                    _ => continue,
                }
            }
        })
        .await
        .unwrap();
        assert!(
            !modulation.injected_messages.is_empty() || modulation.reasoning_effort.is_some()
        );
    }

    #[test]
    fn controlled_mode_temperature_drops_with_ne() {
        let mut actor = ModulatorActor::new();
        actor.mode = CognitiveMode::Controlled;
        actor.state.norepinephrine = 0.8;
        let temp = actor
            .build_modulation()
            .temperature
            .map(|t| t.get())
            .unwrap_or(0.7);
        assert!(temp < 0.7, "controlled mode must lower temperature, got {temp}");
        assert!((temp - 0.62).abs() < 1e-3, "0.7 - 0.1*0.8 = 0.62, got {temp}");

        actor.state.norepinephrine = 1.0;
        let temp = actor
            .build_modulation()
            .temperature
            .map(|t| t.get())
            .unwrap_or(0.7);
        assert!((temp - 0.6).abs() < 1e-3, "expected 0.6, got {temp}");

        actor.state.norepinephrine = 6.0;
        let temp = actor
            .build_modulation()
            .temperature
            .map(|t| t.get())
            .unwrap_or(0.7);
        assert_eq!(temp, 0.2, "temperature must clamp at the 0.2 floor");
    }

    #[tokio::test]
    async fn frustration_and_low_progress_raise_serotonin() {
        let bus = EventBus::new(64);
        let (_h, ready) = spawn_actor(bus.clone(), ModulatorActor::new());
        ready.await.unwrap();
        let mut rx = bus.subscribe();

        bus.publish(Event::TaskSetUpdate {
            meta: meta(),
            task_set: TaskSetState {
                goal: "finish the task".into(),
                priority: 1.0,
                progress: 0.0,
            },
        });
        for _ in 0..5 {
            bus.publish(rpe_event(-0.8));
        }

        bus.publish(Event::RequestState {
            meta: meta(),
            request: StateRequest::Modulator,
            correlation_id: 3,
        });
        let state = tokio::time::timeout(Duration::from_secs(2), async {
            loop {
                match rx.recv().await.unwrap() {
                    Event::StateResponse {
                        response: StateResponse::Modulator(state),
                        correlation_id: 3,
                        ..
                    } => return state,
                    _ => continue,
                }
            }
        })
        .await
        .unwrap();
        assert!(
            state.serotonin > 0.6,
            "frustration plus low progress should raise serotonin, got {}",
            state.serotonin
        );
    }


    #[test]
    fn slow_ach_decline_crossing_threshold_still_emits() {
        let bus = EventBus::new(16);
        let mut actor = ModulatorActor::new();
        let mut ctx = ActorContext::new(bus, meta());

        actor.state.acetylcholine = 0.6;
        let mut events = Vec::new();
        actor.emit_if_changed(&meta(), &mut events);
        let first_effort = events
            .iter()
            .find_map(|event| match event {
                Event::Modulate { modulation, .. } => modulation.reasoning_effort,
                _ => None,
            });
        assert_eq!(first_effort, Some(ReasoningEffort::Max));

        actor.state.acetylcholine = 0.52;
        let mut events = Vec::new();
        actor.emit_if_changed(&meta(), &mut events);
        assert!(!events.is_empty(), "0.08 change must emit");

        actor.state.acetylcholine = 0.48;
        let mut events = Vec::new();
        actor.emit_if_changed(&meta(), &mut events);
        let effort = events
            .iter()
            .find_map(|event| match event {
                Event::Modulate { modulation, .. } => modulation.reasoning_effort,
                _ => None,
            });
        assert_eq!(
            effort,
            None,
            "slow decline crossing 0.5 (delta 0.04 < threshold) must still re-emit with effort cleared"
        );
        let _ = &mut ctx;
    }
}
