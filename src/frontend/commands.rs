use crate::app::App;
use crate::frontend::state::{Mode, SelectorKind, UiState};
use crate::runtime::types::{ActionCandidate, TraceRecord};

pub const COMMANDS: &[(&str, &str)] = &[
    ("models", "List, switch, add, or remove models"),
    ("compact", "Compress the conversation into a summary"),
    ("approvals", "Manage remembered tool approvals"),
    ("status", "Show live cognitive signals"),
    ("task", "List or cancel scheduled tasks"),
    ("rules", "List rules (project + ~/.prognosis)"),
    ("skills", "List skills (.agents/skills + global)"),
    ("history", "List past sessions and load one fully"),
    ("continue", "Resume the most recent session"),
    ("remember", "Inject an archived session summary"),
    ("resume", "Resume a session by id (/resume <id>)"),
    ("trace", "Show recent cognitive trace records"),
    ("supervisor", "Toggle supervisor on / off"),
    ("clear", "Clear the conversation"),
    ("help", "Show key bindings and commands"),
];

pub fn all_commands() -> Vec<(String, String)> {
    COMMANDS
        .iter()
        .map(|(name, description)| (name.to_string(), description.to_string()))
        .collect()
}

pub fn filtered(filter: &str) -> Vec<(String, String)> {
    let lower = filter.to_lowercase();
    all_commands()
        .into_iter()
        .filter(|(name, _)| name.starts_with(&lower))
        .collect()
}

pub enum Action {
    Compact,
    OpenSelector(SelectorKind),
    ShowStatus,
    ShowHelp,
    ShowTrace,
    ClearConversation,
    ContinueSession,
    ResumeSession(String),
    ToggleSupervisor,
}

pub fn execute(name: &str) -> Option<Action> {
    match name {
        "models" => Some(Action::OpenSelector(SelectorKind::Models)),
        "compact" => Some(Action::Compact),
        "approvals" => Some(Action::OpenSelector(SelectorKind::Approvals)),
        "status" => Some(Action::ShowStatus),
        "task" => Some(Action::OpenSelector(SelectorKind::Tasks)),
        "rules" => Some(Action::OpenSelector(SelectorKind::Rules)),
        "skills" => Some(Action::OpenSelector(SelectorKind::Skills)),
        "history" => Some(Action::OpenSelector(SelectorKind::History)),
        "continue" => Some(Action::ContinueSession),
        "remember" => Some(Action::OpenSelector(SelectorKind::Remember)),
        "resume" => Some(Action::ResumeSession(String::new())),
        "trace" => Some(Action::ShowTrace),
        "supervisor" => Some(Action::ToggleSupervisor),
        "clear" => Some(Action::ClearConversation),
        "help" => Some(Action::ShowHelp),
        _ => None,
    }
}

pub fn resume_arg(input: &str) -> Option<String> {
    input
        .strip_prefix("resume ")
        .map(str::trim)
        .filter(|id| !id.is_empty())
        .map(str::to_string)
}

pub fn apply_action(app: &mut App, ui: &mut UiState, action: Action) -> Option<Action> {
    match action {
        Action::OpenSelector(kind) => {
            let selector = match kind {
                SelectorKind::Models => crate::frontend::select::models_selector(app),
                SelectorKind::Tasks => crate::frontend::select::tasks_selector(app),
                SelectorKind::Rules => crate::frontend::select::rules_selector(app),
                SelectorKind::Skills => crate::frontend::select::skills_selector(app),
                SelectorKind::Approvals => crate::frontend::select::approvals_selector(app),
                SelectorKind::Remember => crate::frontend::select::remember_selector(app),
                SelectorKind::History => crate::frontend::select::history_selector(app),
                SelectorKind::RemoveModel => crate::frontend::select::remove_models_selector(app),
            };
            ui.selector = Some(selector);
            ui.mode = Mode::Selector;
            None
        }
        Action::Compact => Some(Action::Compact),
        Action::ShowStatus => {
            ui.mode = Mode::Status;
            None
        }
        Action::ShowHelp => {
            ui.mode = Mode::Help;
            None
        }
        Action::ShowTrace => {
            let tools = format_tool_trace(&app.tool_trace());
            let cognitive = format_traces(&app.traces());
            ui.push_system(&format!("{tools}\n\n{cognitive}"));
            ui.mode = Mode::Chat;
            None
        }
        Action::ToggleSupervisor => {
            let enabled = app.supervisor.is_enabled();
            app.supervisor.set_enabled(!enabled);
            ui.push_system(&format!(
                "supervisor {}",
                if enabled { "off" } else { "on" }
            ));
            ui.mode = Mode::Chat;
            None
        }
        Action::ClearConversation => {
            app.clear();
            ui.mode = Mode::Chat;
            None
        }
        Action::ContinueSession => {
            let result = app.continue_session();
            ui.push_system(&match result {
                Ok(message) => format!("(resumed) {message}"),
                Err(error) => format!("(continue failed) {error}"),
            });
            ui.mode = Mode::Chat;
            None
        }
        Action::ResumeSession(id) => {
            if id.is_empty() {
                ui.push_system("usage: /resume <session id> — e.g. /resume s0001");
                ui.mode = Mode::Chat;
                return None;
            }
            let result = app.resume_session(&id);
            ui.push_system(&match result {
                Ok(message) => format!("(resumed) {message}"),
                Err(error) => format!("(resume failed) {error}"),
            });
            ui.mode = Mode::Chat;
            None
        }
    }
}

pub fn format_tool_trace(traces: &[crate::app::supervisor::ToolCallRecord]) -> String {
    if traces.is_empty() {
        return "(no tool calls yet)".to_string();
    }
    let lines: Vec<String> = traces
        .iter()
        .rev()
        .take(10)
        .rev()
        .map(|record| {
            let args: String = record.arguments.chars().take(40).collect();
            let output: String = record
                .output
                .chars()
                .take(40)
                .collect::<String>()
                .replace('\n', " ");
            format!("{} {args} -> {output}", record.name)
        })
        .collect();
    lines.join("\n")
}

pub fn format_traces(traces: &[TraceRecord]) -> String {
    let recent: Vec<&TraceRecord> = traces.iter().rev().take(15).collect();
    if recent.is_empty() {
        return "(no trace records yet)".to_string();
    }
    let lines: Vec<String> = recent
        .iter()
        .rev()
        .map(|trace| {
            let error = match (trace.error_before, trace.error_after) {
                (Some(before), Some(after)) => format!("rpe {before:.2}→{after:.2}"),
                _ => "rpe --".to_string(),
            };
            let prediction = match trace.prediction_direction {
                Some(direction) => format!("pred {direction:+.2}"),
                None => "pred --".to_string(),
            };
            let sentiment = trace
                .prediction_sentiment
                .map(|value| format!(" sent {value:+.2}"))
                .unwrap_or_default();
            let reaction = trace
                .prediction_reaction
                .as_ref()
                .map(|value| format!(" ({value})"))
                .unwrap_or_default();
            let decision = match &trace.decision {
                Some(decision) => match &decision.candidate {
                    ActionCandidate::CallTool { name, .. } => format!("tool {name}"),
                    ActionCandidate::Respond { .. } => "respond".to_string(),
                    ActionCandidate::AskClarification { .. } => "clarify".to_string(),
                },
                None => "no decision".to_string(),
            };
            let modulation = trace
                .modulation
                .as_ref()
                .map(|m| {
                    let effort = m
                        .reasoning_effort
                        .map(|e| format!("{e:?}"))
                        .unwrap_or_else(|| "default".to_string());
                    format!(" mod {effort}")
                })
                .unwrap_or_default();
            let retrieval = trace
                .retrieval
                .as_ref()
                .map(|r| format!(" · mem: {r}"))
                .unwrap_or_default();
            let writes = if trace.memory_writes.is_empty() {
                String::new()
            } else {
                format!(" · wrote: {}", trace.memory_writes.join(","))
            };
            format!(
                "#{} {error} · {prediction}{sentiment}{reaction}{modulation} · {decision}{retrieval}{writes}",
                trace.cycle_id.0
            )
        })
        .collect();
    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn command_filter_matches_prefix() {
        let all = filtered("");
        assert_eq!(all.len(), COMMANDS.len());
        assert!(all.iter().any(|(name, _)| name == "history"));
        assert!(all.iter().any(|(name, _)| name == "resume"));
        assert!(all.iter().any(|(name, _)| name == "trace"));
        assert!(all.iter().any(|(name, _)| name == "clear"));
        let models = filtered("mo");
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].0, "models");
        let none = filtered("zz");
        assert!(none.is_empty());
    }

    #[test]
    fn execute_returns_actions() {
        assert!(matches!(
            execute("status"),
            Some(Action::ShowStatus)
        ));
        assert!(matches!(execute("compact"), Some(Action::Compact)));
        assert!(matches!(
            execute("history"),
            Some(Action::OpenSelector(SelectorKind::History))
        ));
        assert!(matches!(execute("continue"), Some(Action::ContinueSession)));
        assert!(matches!(execute("trace"), Some(Action::ShowTrace)));
        assert!(matches!(
            execute("supervisor"),
            Some(Action::ToggleSupervisor)
        ));
        assert!(matches!(execute("clear"), Some(Action::ClearConversation)));
        assert!(matches!(
            execute("resume"),
            Some(Action::ResumeSession(_))
        ));
        assert!(execute("nope").is_none());
    }

    #[test]
    fn resume_arg_extracts_id() {
        assert_eq!(resume_arg("resume s0001"), Some("s0001".to_string()));
        assert_eq!(resume_arg("resume  12 "), Some("12".to_string()));
        assert_eq!(resume_arg("resume"), None);
        assert_eq!(resume_arg("resume "), None);
        assert_eq!(resume_arg("models"), None);
    }

    #[test]
    fn format_traces_renders_recent_lines() {
        use crate::adapter::types::ReasoningEffort;
        use crate::runtime::types::{ActionDecision, CycleId};
        let traces = vec![TraceRecord {
            cycle_id: CycleId(3),
            modulation: Some(crate::runtime::types::ModulationContext {
                reasoning_effort: Some(ReasoningEffort::High),
                ..Default::default()
            }),
            error_before: Some(0.4),
            error_after: Some(0.1),
            decision: Some(ActionDecision {
                candidate: ActionCandidate::CallTool {
                    name: "read_file".into(),
                    arguments: serde_json::json!({"filepath": "a.rs"}),
                    tool_call_id: None,
                    reasoning: None,
                },
                confidence: 0.8,
                go: true,
            }),
            retrieval: Some("episodic: asked about weather".into()),
            memory_writes: vec!["episodic s0.52".into()],
            prediction_direction: Some(0.3),
            prediction_sentiment: Some(0.2),
            prediction_reaction: Some("surprised".into()),
        }];
        let text = format_traces(&traces);
        assert!(text.contains("#3"), "{text}");
        assert!(text.contains("rpe 0.40→0.10"), "{text}");
        assert!(text.contains("pred +0.30"), "{text}");
        assert!(text.contains("sent +0.20"), "{text}");
        assert!(text.contains("(surprised)"), "{text}");
        assert!(text.contains("tool read_file"), "{text}");
        assert!(text.contains("mod High"), "{text}");
        assert!(text.contains("mem: episodic: asked about weather"), "{text}");
        assert!(text.contains("wrote: episodic s0.52"), "{text}");
    }

    #[test]
    fn format_traces_empty() {
        assert_eq!(format_traces(&[]), "(no trace records yet)");
    }
}
