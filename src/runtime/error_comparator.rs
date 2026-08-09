use std::sync::Arc;

use async_trait::async_trait;
use futures::StreamExt;
use strsim::{normalized_levenshtein, sorensen_dice};
use tokio_util::sync::CancellationToken;

use crate::adapter::types::{Message, ReasoningEffort, Temperature};
use crate::runtime::actor::{ActorContext, CognitiveActor};
use crate::runtime::event::{Event, EventKind};
use crate::runtime::ports::LlmPort;
use crate::runtime::types::{
    GenerateRequest, ModulationContext, PredictionError, PredictionTrajectory, RpeSignal,
};

const BASELINE_ERROR: f32 = 0.5;
const RPE_RATE_WEIGHT: f32 = 0.7;
const RPE_BASELINE_WEIGHT: f32 = 0.3;

pub(crate) fn similarity(a: &str, b: &str) -> f32 {
    sorensen_dice(a, b).max(normalized_levenshtein(a, b)) as f32
}

pub struct ErrorComparatorActor {
    port: Arc<dyn LlmPort>,
    prediction: Option<PredictionTrajectory>,
    last_cycle_error: Option<f32>,
}

impl ErrorComparatorActor {
    pub fn new(port: Arc<dyn LlmPort>) -> Self {
        Self {
            port,
            prediction: None,
            last_cycle_error: None,
        }
    }

    async fn judge_error(&self, prediction: &PredictionTrajectory, actual: &str) -> f32 {
        let system = "You are an error judge for a cognitive agent. The agent predicted several possible intents and topics for the user's next turn, reflecting genuine uncertainty. Your score becomes the agent's prediction error, which drives its learning signals (surprise, dopamine, memory consolidation). Judge semantic alignment only — never exact wording.\
\n\n# Output\
\nReply with JSON only, no other text:\
\n{\"error\": <float in [0, 1]>}\
\n\n# Rules\
\n- error = 0 when the actual message's intent or topic matches ANY predicted candidate well.\
\n- error = 1 when the actual message matches NONE of the predicted candidates.\
\n- Judge only semantic alignment of intent and topic; partial overlap with a candidate is a mid-range error (e.g. same topic, different intent ≈ 0.5-0.7; same intent, different topic ≈ 0.3-0.5).\
\n- Score against the BEST matching candidate, not the average: one close candidate means the prediction captured the turn even if other candidates missed.\
\n- Do not invent requirements; compare exactly the two inputs given.\
\n- Very short or purely social messages (\"ok\", \"thanks\", \"继续\") are near-zero error when the predicted reaction covers that kind of reply; they are not evidence of a wrong prediction.\
\n- If the prediction was for the assistant's tool round and the user instead asks an unrelated question, that is a high error — the prediction missed the turn entirely.";
        let candidates = if prediction.intent_candidates.is_empty() {
            vec![prediction.intent]
        } else {
            prediction.intent_candidates.clone()
        };
        let user = format!(
            "Predicted possible intents: {:?}\nPredicted possible topics: {:?}\nPredicted direction: {:.2}\nActual user message: {}",
            candidates, prediction.topics, prediction.direction, actual
        );
        let modulation = ModulationContext {
            reasoning_effort: Some(ReasoningEffort::None),
            temperature: Temperature::new(0.0).ok(),
            ..Default::default()
        };
        let request = GenerateRequest {
            messages: vec![Message::system(system), Message::user(user)],
            modulation,
            tools: None,
        };
        let cancel = CancellationToken::new();
        let mut stream = match self.port.generate(&request, &cancel).await {
            Ok(stream) => stream,
            Err(_) => return BASELINE_ERROR,
        };
        let mut content = String::new();
        while let Some(item) = stream.next().await {
            if let Ok(chunk) = item
                && let Some(text) = chunk.content() {
                    content.push_str(text);
                }
        }
        parse_error(&content)
    }

    fn compute_rpe(&mut self, error: f32) -> f32 {
        let rate = match self.last_cycle_error {
            Some(last) => error - last,
            None => 0.0,
        };
        let baseline_term = BASELINE_ERROR - error;
        let rpe = -RPE_RATE_WEIGHT * rate + RPE_BASELINE_WEIGHT * baseline_term;
        self.last_cycle_error = Some(error);
        rpe.clamp(-1.0, 1.0)
    }
}

fn parse_error(content: &str) -> f32 {
    let Some(json) = crate::util::extract_json_object(content) else {
        return BASELINE_ERROR;
    };
    let value: serde_json::Value = match serde_json::from_str(json) {
        Ok(json) => json,
        Err(_) => return BASELINE_ERROR,
    };
    value
        .get("error")
        .and_then(|value| value.as_f64())
        .map(|value| (value as f32).clamp(0.0, 1.0))
        .unwrap_or(BASELINE_ERROR)
}

#[async_trait]
impl CognitiveActor for ErrorComparatorActor {
    fn id(&self) -> &str {
        "error_comparator"
    }

    fn subscriptions(&self) -> Vec<EventKind> {
        vec![EventKind::Prediction, EventKind::Perception]
    }

    async fn handle(&mut self, event: &Event, _ctx: &mut ActorContext) -> Vec<Event> {
        match event {
            Event::Prediction { trajectory, .. } => {
                self.prediction = Some(trajectory.clone());
                vec![]
            }
            Event::Perception { meta, payload } => {
                if payload.source != crate::runtime::types::PerceptionSource::User {
                    return vec![];
                }
                let semantic = match &self.prediction {
                    Some(prediction) => self.judge_error(prediction, &payload.content).await,
                    None => BASELINE_ERROR,
                };
                let error = PredictionError {
                    semantic,
                    confidence: 0.0,
                    precision: 1.0,
                };
                let weighted = error.weighted();
                let rpe = self.compute_rpe(weighted);
                self.prediction = None;
                vec![
                    Event::ErrorComputed { meta: *meta, error },
                    Event::Rpe {
                        meta: *meta,
                        rpe: RpeSignal(rpe),
                    },
                ]
            }
            _ => vec![],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adapter::error::AdapterError;
    use crate::adapter::types::{ChunkDelta, CompletionChunk, FinishReason};
    use crate::runtime::actor::spawn_actor;
    use crate::runtime::bus::EventBus;
    use crate::runtime::event::EventMeta;
    use crate::runtime::types::{
        CycleId, IntentKind, PerceptionPayload, PerceptionSource,
    };
    use futures::Stream;
    use std::collections::VecDeque;
    use std::pin::Pin;
    use std::sync::Mutex;
    use std::time::Duration;

    struct JudgePort {
        errors: Mutex<VecDeque<f32>>,
        default: f32,
    }

    #[async_trait]
    impl LlmPort for JudgePort {
        async fn generate<'a>(
            &'a self,
            _request: &'a GenerateRequest,
            _cancel: &'a CancellationToken,
        ) -> Result<
            Pin<Box<dyn Stream<Item = Result<CompletionChunk, AdapterError>> + Send + 'a>>,
            AdapterError,
        > {
            let error = self
                .errors
                .lock()
                .unwrap()
                .pop_front()
                .unwrap_or(self.default);
            let chunk = CompletionChunk {
                model: "judge".into(),
                index: 0,
                delta: ChunkDelta {
                    role: None,
                    content: Some(format!(r#"{{"error": {error}}}"#)),
                    tool_calls: vec![],
                    logprobs: None,
                    reasoning: None,
                },
                finish_reason: Some(FinishReason::Stop),
                usage: None,
                request_id: None,
            };
            Ok(Box::pin(futures::stream::iter(vec![Ok(chunk)])))
        }
    }

    fn port(errors: Vec<f32>, default: f32) -> Arc<dyn LlmPort> {
        Arc::new(JudgePort {
            errors: Mutex::new(errors.into()),
            default,
        })
    }

    fn meta() -> EventMeta {
        EventMeta {
            cycle_id: CycleId(1),

        }
    }

    fn prediction_event() -> Event {
        Event::Prediction {
            meta: meta(),
            trajectory: PredictionTrajectory {
                topics: vec!["weather".into()],
                key_elements: vec!["sunny".into()],
                direction: 0.5,
                intent: IntentKind::Question,
                intent_candidates: vec![IntentKind::Question, IntentKind::Statement],
                reaction: "the user asks about the weather".into(),
                reaction_sentiment: 0.2,
            },
        }
    }

    fn user_input_event(content: &str) -> Event {
        Event::Perception {
            meta: meta(),
            payload: PerceptionPayload {
                source: PerceptionSource::User,
                content: content.into(),
                salience: 0.5,
            },
        }
    }

    #[test]
    fn external_similarity_measures() {
        assert!(similarity("sunny", "sunny") > 0.9);
        assert!(similarity("sunny", "rainy") < similarity("sunny", "sunny"));
    }

    #[test]
    fn error_json_parsing() {
        assert_eq!(parse_error(r#"{"error": 0.2}"#), 0.2);
        assert!((parse_error(r#"prefix {"error": 0.75} suffix"#) - 0.75).abs() < 1e-6);
        assert_eq!(parse_error("not json"), BASELINE_ERROR);
        assert_eq!(parse_error(r#"{"error": 3.0}"#), 1.0);
    }

    #[tokio::test]
    async fn low_judged_error_yields_low_weighted() {
        let bus = EventBus::new(32);
        let (_h, ready) = spawn_actor(bus.clone(), ErrorComparatorActor::new(port(vec![0.1], 0.5)));
        ready.await.unwrap();
        let mut rx = Box::pin(bus.subscribe_kinds(&[EventKind::Error]));

        bus.publish(prediction_event());
        bus.publish(user_input_event("is it sunny today?"));

        match tokio::time::timeout(Duration::from_secs(2), rx.next())
            .await
            .unwrap()
            .unwrap()
        {
            Event::ErrorComputed { error, .. } => {
                assert!(error.weighted() < 0.2, "low judged error expected");
            }
            other => panic!("expected error computed, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn high_judged_error_yields_high_weighted() {
        let bus = EventBus::new(32);
        let (_h, ready) = spawn_actor(bus.clone(), ErrorComparatorActor::new(port(vec![0.9], 0.5)));
        ready.await.unwrap();
        let mut rx = Box::pin(bus.subscribe_kinds(&[EventKind::Error]));

        bus.publish(prediction_event());
        bus.publish(user_input_event("write a rust web server"));

        match tokio::time::timeout(Duration::from_secs(2), rx.next())
            .await
            .unwrap()
            .unwrap()
        {
            Event::ErrorComputed { error, .. } => {
                assert!(error.weighted() > 0.8, "high judged error expected");
            }
            other => panic!("expected error computed, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn non_user_perception_does_not_trigger_error() {
        let bus = EventBus::new(32);
        let (_h, ready) = spawn_actor(bus.clone(), ErrorComparatorActor::new(port(vec![], 0.5)));
        ready.await.unwrap();
        let mut rx = Box::pin(bus.subscribe_kinds(&[EventKind::Error]));

        bus.publish(prediction_event());
        bus.publish(Event::Perception {
            meta: meta(),
            payload: PerceptionPayload {
                source: PerceptionSource::ToolResult,
                content: "(Tool read_file result)\nfn main() {}".into(),
                salience: 0.8,
            },
        });
        assert!(
            tokio::time::timeout(Duration::from_millis(200), rx.next()).await.is_err(),
            "tool results must not trigger user-feedback error"
        );
    }

    #[tokio::test]
    async fn rpe_uses_change_rate_across_turns() {
        let bus = EventBus::new(32);
        let (_h, ready) = spawn_actor(
            bus.clone(),
            ErrorComparatorActor::new(port(vec![0.1, 0.9], 0.5)),
        );
        ready.await.unwrap();
        let mut rx = Box::pin(bus.subscribe_kinds(&[EventKind::Error]));
        let mut rpes = Vec::new();

        bus.publish(prediction_event());
        bus.publish(user_input_event("is it sunny today?"));
        for _ in 0..2 {
            if let Event::Rpe { rpe, .. } = tokio::time::timeout(Duration::from_secs(2), rx.next())
                .await
                .unwrap()
                .unwrap() { rpes.push(rpe.0) }
        }

        bus.publish(prediction_event());
        bus.publish(user_input_event("write a rust web server"));
        for _ in 0..2 {
            if let Event::Rpe { rpe, .. } = tokio::time::timeout(Duration::from_secs(2), rx.next())
                .await
                .unwrap()
                .unwrap() { rpes.push(rpe.0) }
        }

        assert_eq!(rpes.len(), 2);
        assert!(
            rpes[1] < rpes[0],
            "rising judged error should produce more negative rpe: {rpes:?}"
        );
    }
}
