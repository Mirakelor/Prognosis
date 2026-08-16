use crate::frontend::state::{ToolStatus, UiState};
use crate::runtime::event::Event;
use crate::runtime::types::PerceptionSource;

pub fn handle_event(ui: &mut UiState, event: Event) -> Option<Event> {
    match event {
        Event::Perception { payload, .. } => match payload.source {
            PerceptionSource::User => {
                ui.reset_turn();
                ui.push_user(&payload.content);
                None
            }
            PerceptionSource::System | PerceptionSource::Internal | PerceptionSource::Scheduled => {
                if !payload.content.trim().is_empty() {
                    ui.push_system(&payload.content);
                }
                None
            }
            PerceptionSource::ToolResult => None,
        },
        Event::Chunk { chunk, .. } => {
            if ui.cancelled {
                return None;
            }
            let content = chunk.content().unwrap_or("");
            let reasoning = chunk.delta.reasoning.as_deref().unwrap_or("");
            ui.append_stream(content, reasoning);
            None
        }
        Event::StreamEnd { usage, .. } => {
            if let Some(usage) = usage {
                ui.context_tokens = usage.prompt_tokens as usize;
            }
            None
        }
        Event::ActionSelected { meta, decision } => {
            if ui.cancelled {
                return None;
            }
            if let crate::runtime::types::ActionCandidate::CallTool {
                name,
                arguments,
                tool_call_id,
                ..
            } = &decision.candidate
            {
                ui.finish_stream();
                ui.mark_tool_started(
                    tool_call_id.clone().unwrap_or_default(),
                    name.clone(),
                    arguments.to_string(),
                );
                Some(Event::ActionSelected { meta, decision })
            } else {
                ui.finish_stream();
                ui.finish_turn();
                None
            }
        }
        Event::ToolResult { result, verdict, .. } => {
            let status = if result.output.contains("user denied the tool call") {
                ToolStatus::Denied
            } else if matches!(verdict.as_deref(), Some("blocked")) {
                ToolStatus::Errored
            } else {
                ToolStatus::Done
            };
            let cleaned = crate::util::strip_ansi(&result.output)
                .replace("\r\n", "\n")
                .replace('\r', "\n");
            ui.mark_tool_finished(
                &result.tool_call_id.clone().unwrap_or_default(),
                &truncate(&cleaned, 300),
                &cleaned,
                status,
                verdict,
            );
            None
        }
        Event::GenerationError { error, .. } => {
            ui.push_system(&format!("(Generation error) {error}"));
            None
        }
        Event::CompactContext { summary, .. } => {
            ui.clear_conversation(&format!("(Conversation compacted)\n{summary}"));
            None
        }
        Event::ConversationCleared { .. } => {
            ui.clear_conversation("(Conversation cleared)");
            None
        }
        Event::ModulatorUpdate { state, mode, .. } => {
            crate::frontend::state::apply_modulator(ui, &state, mode);
            None
        }
        Event::EmotionUpdate { emotion, .. } => {
            crate::frontend::state::apply_emotion(ui, &emotion);
            None
        }
        Event::DriveUpdate { drives, .. } => {
            crate::frontend::state::apply_drive(ui, &drives);
            None
        }
        Event::MetaUpdate { meta_state, .. } => {
            crate::frontend::state::apply_meta(ui, &meta_state);
            None
        }
        Event::WorkingMemoryUpdate { snapshot, .. } => {
            crate::frontend::state::apply_wm(ui, &snapshot);
            None
        }
        Event::Rpe { rpe, .. } => {
            ui.cognitive.rpe = Some(rpe.0);
            None
        }
        Event::ErrorComputed { error, .. } => {
            ui.cognitive.last_error = Some(error.weighted());
            None
        }
        Event::Prediction { trajectory, .. } => {
            ui.cognitive.prediction_direction = Some(trajectory.direction);
            None
        }
        _ => None,
    }
}

fn truncate(text: &str, limit: usize) -> String {
    let first_line = text.lines().next().unwrap_or("").to_string();
    if first_line.chars().count() <= limit {
        first_line
    } else {
        let cut: String = first_line.chars().take(limit).collect();
        format!("{cut}…")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frontend::state::UiMessage;
    use crate::runtime::event::EventMeta;
    use crate::runtime::types::{
        ActionCandidate, ActionDecision, CycleId, PerceptionPayload, RpeSignal, ToolResult,
    };

    fn meta() -> EventMeta {
        EventMeta {
            cycle_id: CycleId(1),

        }
    }

    #[test]
    fn user_perception_pushes_user_message() {
        let mut ui = UiState::new();
        handle_event(&mut ui, Event::Perception {
            meta: meta(),
            payload: PerceptionPayload {
                source: PerceptionSource::User,
                content: "hello".into(),
                salience: 0.5,
            },
        });
        assert!(matches!(ui.messages[0], UiMessage::User { .. }));
    }

    #[test]
    fn chunks_stream_into_assistant_message() {
        let mut ui = UiState::new();
        let chunk = |content: Option<&str>, reasoning: Option<&str>| Event::Chunk {
            meta: meta(),
            chunk: crate::adapter::types::CompletionChunk {
                model: "m".into(),
                index: 0,
                delta: crate::adapter::types::ChunkDelta {
                    role: None,
                    content: content.map(str::to_string),
                    tool_calls: vec![],
                    logprobs: None,
                    reasoning: reasoning.map(str::to_string),
                },
                finish_reason: None,
                usage: None,
                request_id: None,
            },
        };
        handle_event(&mut ui, chunk(Some("hel"), Some("think")));
        handle_event(&mut ui, chunk(Some("lo"), None));
        match &ui.messages[0] {
            UiMessage::Assistant { content, reasoning } => {
                assert_eq!(content, "hello");
                assert_eq!(reasoning, "think");
            }
            _ => panic!("expected assistant message"),
        }
    }

    #[test]
    fn tool_call_lifecycle_updates_status() {
        let mut ui = UiState::new();
        let action = handle_event(&mut ui, Event::ActionSelected {
            meta: meta(),
            decision: ActionDecision {
                candidate: ActionCandidate::CallTool {
                    name: "read_file".into(),
                    arguments: serde_json::json!({"filepath": "a.rs"}),
                    tool_call_id: Some("call_1".into()),
                    reasoning: None,
                },
                confidence: 0.9,
                go: true,
            },
        });
        assert!(action.is_some(), "tool calls must be forwarded for execution");
        assert!(matches!(ui.messages[0], UiMessage::ToolCall(_)));
        handle_event(&mut ui, Event::ToolResult {
            meta: meta(),
            result: ToolResult {
                name: "read_file".into(),
                output: "fn main() {}".into(),
                tool_call_id: Some("call_1".into()),
            },
            verdict: Some("allowed".to_string()),
        });
        match &ui.messages[0] {
            UiMessage::ToolCall(call) => {
                assert_eq!(call.status, ToolStatus::Done);
                assert_eq!(call.summary, "fn main() {}");
            }
            _ => panic!("expected tool call message"),
        }
    }

    #[test]
    fn compact_clears_conversation() {
        let mut ui = UiState::new();
        ui.push_user("hi");
        handle_event(&mut ui, Event::CompactContext {
            meta: meta(),
            summary: "summary text".into(),
        });
        assert_eq!(ui.messages.len(), 1);
        match &ui.messages[0] {
            UiMessage::System { content } => assert!(content.contains("summary text")),
            _ => panic!("expected system message"),
        }
    }

    #[test]
    fn tool_call_action_ends_streaming() {
        let mut ui = UiState::new();
        ui.append_stream("some text", "");
        assert!(ui.is_generating());
        let _ = handle_event(&mut ui, Event::ActionSelected {
            meta: meta(),
            decision: ActionDecision {
                candidate: ActionCandidate::CallTool {
                    name: "ls".into(),
                    arguments: serde_json::json!({}),
                    tool_call_id: Some("call_x".into()),
                    reasoning: None,
                },
                confidence: 0.9,
                go: true,
            },
        });
        assert!(!ui.is_generating(), "tool execution must clear streaming");
        assert!(ui.has_running_tool());
    }

    #[test]
    fn conversation_cleared_clears_messages() {
        let mut ui = UiState::new();
        ui.push_user("hi");
        handle_event(&mut ui, Event::ConversationCleared { meta: meta() });
        assert_eq!(ui.messages.len(), 1);
        match &ui.messages[0] {
            UiMessage::System { content } => assert!(content.contains("Conversation cleared")),
            _ => panic!("expected system message"),
        }
    }

    #[test]
    fn drive_update_populates_cognitive_snapshot() {
        let mut ui = UiState::new();
        handle_event(&mut ui, Event::DriveUpdate {
            meta: meta(),
            drives: crate::runtime::types::DriveState {
                homeostatic: 0.7,
                curiosity: 0.4,
                salience: 0.9,
            },
        });
        assert_eq!(ui.cognitive.drive_homeostatic, 0.7);
        assert_eq!(ui.cognitive.drive_curiosity, 0.4);
        assert_eq!(ui.cognitive.drive_salience, 0.9);
    }

    #[test]
    fn stream_end_tracks_latest_context_prompt_tokens() {
        let mut ui = UiState::new();
        handle_event(&mut ui, Event::StreamEnd {
            meta: meta(),
            usage: Some(crate::adapter::types::TokenUsage {
                prompt_tokens: 8000,
                completion_tokens: 2000,
                total_tokens: 10000,
            }),
        });
        handle_event(&mut ui, Event::StreamEnd {
            meta: meta(),
            usage: Some(crate::adapter::types::TokenUsage {
                prompt_tokens: 12000,
                completion_tokens: 3000,
                total_tokens: 15000,
            }),
        });
        assert_eq!(
            ui.context_tokens, 12000,
            "context length is the latest prompt_tokens, not accumulated spend"
        );
    }

    #[test]
    fn rpe_updates_cognitive_snapshot() {
        let mut ui = UiState::new();
        handle_event(&mut ui, Event::Rpe {
            meta: meta(),
            rpe: RpeSignal(0.6),
        });
        assert_eq!(ui.cognitive.rpe, Some(0.6));
    }

    #[test]
    fn tool_result_denied_marks_denied() {
        let mut ui = UiState::new();
        let _ = handle_event(&mut ui, Event::ActionSelected {
            meta: meta(),
            decision: ActionDecision {
                candidate: ActionCandidate::CallTool {
                    name: "run_terminal_command".into(),
                    arguments: serde_json::json!({"command": "rm -rf /"}),
                    tool_call_id: Some("call_2".into()),
                    reasoning: None,
                },
                confidence: 0.9,
                go: true,
            },
        });
        handle_event(&mut ui, Event::ToolResult {
            meta: meta(),
            result: ToolResult {
                name: "run_terminal_command".into(),
                output: "user denied the tool call".into(),
                tool_call_id: Some("call_2".into()),
            },
            verdict: None,
        });
        match &ui.messages[0] {
            UiMessage::ToolCall(call) => assert_eq!(call.status, ToolStatus::Denied),
            _ => panic!("expected tool call message"),
        }
    }

    #[test]
    fn corrected_result_marks_done_not_errored() {
        let mut ui = UiState::new();
        let _ = handle_event(&mut ui, Event::ActionSelected {
            meta: meta(),
            decision: ActionDecision {
                candidate: ActionCandidate::CallTool {
                    name: "grep_search".into(),
                    arguments: serde_json::json!({"query": "useEffect"}),
                    tool_call_id: Some("call_3".into()),
                    reasoning: None,
                },
                confidence: 0.9,
                go: true,
            },
        });
        handle_event(&mut ui, Event::ToolResult {
            meta: meta(),
            result: ToolResult {
                name: "grep_search".into(),
                output: "(Tool call corrected by supervisor: the original grep_search call was repaired and executed instead as grep_search({\"query\": \"useEffect\", \"path\": \"src\"}) — review this result with the corrected call in mind)\n./src/a.tsx:  useEffect".into(),
                tool_call_id: Some("call_3".into()),
            },
            verdict: Some("corrected".into()),
        });
        match &ui.messages[0] {
            UiMessage::ToolCall(call) => {
                assert_eq!(
                    call.status,
                    ToolStatus::Done,
                    "a corrected call executed successfully and must not render as errored"
                );
                assert_eq!(call.verdict.as_deref(), Some("corrected"));
            }
            _ => panic!("expected tool call message"),
        }
    }
}
