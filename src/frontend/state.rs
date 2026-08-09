use crate::frontend::select::Selector;
use crate::runtime::types::{EmotionState, MetaState, ModulatorState, WorkingMemorySnapshot};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolStatus {
    Running,
    Done,
    Errored,
    Denied,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DiffLine {
    pub kind: char,
    pub line_no: Option<u32>,
    pub text: String,
}

#[derive(Debug, Clone)]
pub struct ToolCallMsg {
    pub tool_call_id: String,
    pub name: String,
    pub arguments: String,
    pub status: ToolStatus,
    pub summary: String,
    pub output: String,
    pub diff: Vec<DiffLine>,
    pub elapsed: Option<f64>,
    pub started: std::time::Instant,
    pub verdict: Option<String>,
}

#[derive(Debug, Clone)]
pub enum UiMessage {
    User { content: String, time: String },
    Assistant { content: String, reasoning: String },
    System { content: String },
    Summary(String),
    ToolCall(ToolCallMsg),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Chat,
    Command,
    Approve,
    Selector,
    Status,
    Help,
    Setup,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SelectorKind {
    Models,
    Tasks,
    Rules,
    Skills,
    Approvals,
    Remember,
    History,
    RemoveModel,
}

#[derive(Debug, Clone, Default)]
pub struct CognitiveSnapshot {
    pub dopamine: f32,
    pub norepinephrine: f32,
    pub acetylcholine: f32,
    pub serotonin: f32,
    pub valence: f32,
    pub arousal: f32,
    pub uncertainty: f32,
    pub conflict: f32,
    pub confidence: f32,
    pub wm_slots: Vec<String>,
    pub rpe: Option<f32>,
    pub last_error: Option<f32>,
    pub prediction_direction: Option<f32>,
    pub mode: String,
    pub drive_homeostatic: f32,
    pub drive_curiosity: f32,
    pub drive_salience: f32,
}

pub struct InputState {
    pub buffer: String,
    pub cursor: usize,
    pub history: Vec<String>,
    pub history_index: Option<usize>,
    pub command_selection: usize,
    pub pending_submit: Vec<String>,
}

impl Default for InputState {
    fn default() -> Self {
        Self::new()
    }
}

impl InputState {
    pub fn new() -> Self {
        Self {
            buffer: String::new(),
            cursor: 0,
            history: Vec::new(),
            history_index: None,
            command_selection: 0,
            pending_submit: Vec::new(),
        }
    }
}

pub struct SetupState {
    pub fields: Vec<(String, String)>,
    pub active: usize,
    pub cursor: usize,
    pub error: Option<String>,
}

pub struct UiState {
    pub messages: Vec<UiMessage>,
    pub streaming: Option<usize>,
    pub mode: Mode,
    pub input: InputState,
    pub cognitive: CognitiveSnapshot,
    pub total_tokens: usize,
    pub selector: Option<Selector>,
    pub setup: Option<SetupState>,
    pub scroll_offset: usize,
    pub last_tool_index: Option<usize>,
    pub tip: &'static str,
    pub kitty_supported: bool,
    pub spinner_index: usize,
    pub turn_start: std::time::Instant,
    pub turn_files: std::collections::HashSet<String>,
    pub turn_added: usize,
    pub turn_removed: usize,
    pub turn_tool_calls: usize,
    pub pending_tool_calls: std::collections::VecDeque<(String, serde_json::Value, Option<String>)>,
    pub last_recorded_assistant_index: Option<usize>,
    pub hitokoto: Option<crate::frontend::hitokoto::Hitokoto>,
    pub fold_expanded: bool,
    pub panel_scroll: usize,
    pub cancelled: bool,
}

impl Default for UiState {
    fn default() -> Self {
        Self::new()
    }
}

impl UiState {
    pub fn new() -> Self {
        Self {
            messages: Vec::new(),
            streaming: None,
            mode: Mode::Chat,
            input: InputState::new(),
            cognitive: CognitiveSnapshot::default(),
            total_tokens: 0,
            selector: None,
            setup: None,
            scroll_offset: 0,
            last_tool_index: None,
            tip: crate::frontend::logo::pick_tip(),
            kitty_supported: true,
            spinner_index: 0,
            turn_start: std::time::Instant::now(),
            turn_files: std::collections::HashSet::new(),
            turn_added: 0,
            turn_removed: 0,
            turn_tool_calls: 0,
            pending_tool_calls: std::collections::VecDeque::new(),
            last_recorded_assistant_index: None,
            hitokoto: None,
            fold_expanded: false,
            panel_scroll: 0,
            cancelled: false,
        }
    }

    pub fn reset_turn(&mut self) {
        self.turn_start = std::time::Instant::now();
        self.turn_files.clear();
        self.turn_added = 0;
        self.turn_removed = 0;
        self.turn_tool_calls = 0;
    }

    pub fn tick_spinner(&mut self) {
        self.spinner_index = (self.spinner_index + 1) % crate::frontend::theme::SPINNER.len();
    }

    pub fn spinner(&self) -> char {
        crate::frontend::theme::SPINNER[self.spinner_index % crate::frontend::theme::SPINNER.len()]
    }

    pub fn note_tool_started(&mut self, arguments: &str) {
        self.turn_tool_calls += 1;
        if let Some(path) = extract_filepath(arguments) {
            self.turn_files.insert(path);
        }
    }

    pub fn note_tool_finished(&mut self, output: &str) {
        let (added, removed) = count_diff_lines(output);
        self.turn_added += added;
        self.turn_removed += removed;
    }

    pub fn turn_elapsed(&self) -> std::time::Duration {
        self.turn_start.elapsed()
    }

    pub fn finish_turn(&mut self) {
        let elapsed = self.turn_elapsed();
        if self.turn_tool_calls == 0 {
            return;
        }
        let files = self.turn_files.len();
        let summary = format!(
            "{} changed · +{} −{} · {:.1}s",
            if files == 1 {
                "1 file".to_string()
            } else {
                format!("{files} files")
            },
            self.turn_added,
            self.turn_removed,
            elapsed.as_secs_f64()
        );
        self.messages.push(UiMessage::Summary(summary));
        self.reset_turn();
    }

    pub fn is_generating(&self) -> bool {
        self.streaming.is_some()
    }

    pub fn has_running_tool(&self) -> bool {
        self.messages.iter().any(|message| {
            matches!(
                message,
                UiMessage::ToolCall(call) if call.status == ToolStatus::Running
            )
        })
    }

    pub fn push_user(&mut self, content: &str) {
        let time = chrono::Local::now().format("%H:%M").to_string();
        self.messages.push(UiMessage::User {
            content: content.to_string(),
            time,
        });
    }

    pub fn push_system(&mut self, content: &str) {
        self.messages.push(UiMessage::System {
            content: content.to_string(),
        });
    }

    pub fn append_stream(&mut self, content: &str, reasoning: &str) {
        let index = match self.streaming {
            Some(index) => index,
            None => {
                self.messages.push(UiMessage::Assistant {
                    content: String::new(),
                    reasoning: String::new(),
                });
                self.streaming = Some(self.messages.len() - 1);
                self.messages.len() - 1
            }
        };
        if let Some(UiMessage::Assistant { content: c, reasoning: r }) =
            self.messages.get_mut(index)
        {
            c.push_str(content);
            r.push_str(reasoning);
        }
    }

    pub fn finish_stream(&mut self) {
        self.streaming = None;
    }

    pub fn mark_tool_started(
        &mut self,
        tool_call_id: String,
        name: String,
        arguments: String,
    ) -> usize {
        self.note_tool_started(&arguments);
        self.messages.push(UiMessage::ToolCall(ToolCallMsg {
            tool_call_id,
            name,
            arguments,
            status: ToolStatus::Running,
            summary: String::new(),
            output: String::new(),
            diff: Vec::new(),
            elapsed: None,
            started: std::time::Instant::now(),
            verdict: None,
        }));
        let index = self.messages.len() - 1;
        self.last_tool_index = Some(index);
        index
    }

    pub fn mark_tool_finished(
        &mut self,
        tool_call_id: &str,
        summary: &str,
        full_output: &str,
        status: ToolStatus,
        verdict: Option<String>,
    ) {
        self.note_tool_finished(full_output);
        let diff = parse_diff(full_output);
        for message in &mut self.messages {
            if let UiMessage::ToolCall(call) = message
                && call.tool_call_id == tool_call_id {
                    call.status = status;
                    call.summary = summary.to_string();
                    call.output = full_output.to_string();
                    call.diff = diff.clone();
                    call.elapsed = Some(call.started.elapsed().as_secs_f64());
                    call.verdict = verdict;
                    return;
                }
        }
        if let Some(index) = self.last_tool_index
            && let Some(UiMessage::ToolCall(call)) = self.messages.get_mut(index) {
                call.status = status;
                call.summary = summary.to_string();
                call.output = full_output.to_string();
                call.diff = diff;
                call.elapsed = Some(call.started.elapsed().as_secs_f64());
                call.verdict = verdict;
            }
    }

    pub fn clear_conversation(&mut self, summary: &str) {
        self.messages.clear();
        self.streaming = None;
        self.total_tokens = 0;
        self.push_system(summary);
    }
}

fn extract_filepath(arguments: &str) -> Option<String> {
    let value: serde_json::Value = serde_json::from_str(arguments).ok()?;
    value
        .get("filepath")
        .or_else(|| value.get("path"))
        .and_then(serde_json::Value::as_str)
        .map(str::to_string)
}

fn looks_like_diff(output: &str) -> bool {
    output.contains("diff --git") || output.contains("@@ -")
}

fn count_diff_lines(output: &str) -> (usize, usize) {
    if !looks_like_diff(output) {
        return (0, 0);
    }
    let mut added = 0;
    let mut removed = 0;
    for line in output.lines() {
        if line.starts_with("+++") || line.starts_with("---") || line.starts_with("@@") {
            continue;
        }
        if let Some(rest) = line.strip_prefix('+') {
            if !rest.is_empty() {
                added += 1;
            }
        } else if let Some(rest) = line.strip_prefix('-')
            && !rest.is_empty()
        {
            removed += 1;
        }
    }
    (added, removed)
}

fn parse_hunk(header: &str) -> Option<(u32, u32)> {
    let mut old = None;
    let mut new = None;
    for part in header.split_whitespace() {
        if let Some(rest) = part.strip_prefix('-') {
            old = rest.split(',').next().and_then(|s| s.parse().ok());
        } else if let Some(rest) = part.strip_prefix('+') {
            new = rest.split(',').next().and_then(|s| s.parse().ok());
        }
    }
    match (old, new) {
        (Some(old), Some(new)) => Some((old, new)),
        _ => None,
    }
}

fn parse_diff(output: &str) -> Vec<DiffLine> {
    if !looks_like_diff(output) {
        return vec![];
    }
    let mut lines = Vec::new();
    let mut old_no: u32 = 0;
    let mut new_no: u32 = 0;
    let mut in_hunk = false;
    for line in output.lines() {
        if line.starts_with("+++") || line.starts_with("---") || line.starts_with("diff --git") {
            continue;
        }
        if let Some(header) = line.strip_prefix("@@") {
            if let Some((old, new)) = parse_hunk(header) {
                old_no = old;
                new_no = new;
                in_hunk = true;
            }
            continue;
        }
        if in_hunk {
            if let Some(rest) = line.strip_prefix('+') {
                if !rest.is_empty() {
                    lines.push(DiffLine {
                        kind: '+',
                        line_no: Some(new_no),
                        text: rest.to_string(),
                    });
                    new_no += 1;
                }
            } else if let Some(rest) = line.strip_prefix('-') {
                if !rest.is_empty() {
                    lines.push(DiffLine {
                        kind: '-',
                        line_no: Some(old_no),
                        text: rest.to_string(),
                    });
                    old_no += 1;
                }
            } else {
                let text = line.strip_prefix(' ').unwrap_or(line);
                lines.push(DiffLine {
                    kind: ' ',
                    line_no: Some(new_no),
                    text: text.to_string(),
                });
                old_no += 1;
                new_no += 1;
            }
        } else if let Some(rest) = line.strip_prefix('+') {
            if !rest.is_empty() {
                lines.push(DiffLine {
                    kind: '+',
                    line_no: None,
                    text: rest.to_string(),
                });
            }
        } else if let Some(rest) = line.strip_prefix('-')
            && !rest.is_empty()
        {
            lines.push(DiffLine {
                kind: '-',
                line_no: None,
                text: rest.to_string(),
            });
        }
        if lines.len() >= 40 {
            break;
        }
    }
    lines
}

pub fn apply_modulator(
    state: &mut UiState,
    modulator: &ModulatorState,
    mode: crate::runtime::types::CognitiveMode,
) {
    state.cognitive.dopamine = modulator.dopamine;
    state.cognitive.norepinephrine = modulator.norepinephrine;
    state.cognitive.acetylcholine = modulator.acetylcholine;
    state.cognitive.serotonin = modulator.serotonin;
    state.cognitive.mode = format!("{mode:?}");
}

pub fn apply_emotion(state: &mut UiState, emotion: &EmotionState) {
    state.cognitive.valence = emotion.valence;
    state.cognitive.arousal = emotion.arousal;
}

pub fn apply_drive(state: &mut UiState, drive: &crate::runtime::types::DriveState) {
    state.cognitive.drive_homeostatic = drive.homeostatic;
    state.cognitive.drive_curiosity = drive.curiosity;
    state.cognitive.drive_salience = drive.salience;
}

pub fn apply_meta(state: &mut UiState, meta: &MetaState) {
    state.cognitive.uncertainty = meta.uncertainty;
    state.cognitive.conflict = meta.conflict;
    state.cognitive.confidence = meta.confidence;
}

pub fn apply_wm(state: &mut UiState, snapshot: &WorkingMemorySnapshot) {
    state.cognitive.wm_slots = snapshot
        .slots
        .iter()
        .map(|slot| slot.content.clone())
        .collect();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_diff_extracts_lines_with_numbers() {
        let output = "diff --git a/src/a.rs b/src/a.rs\n\
--- a/src/a.rs\n\
+++ b/src/a.rs\n\
@@ -1,3 +1,3 @@\n\
 fn main() {\n\
-    old();\n\
+    new();\n\
 }\n";
        let diff = parse_diff(output);
        assert_eq!(diff.len(), 4);
        assert_eq!(diff[0].kind, ' ');
        assert_eq!(diff[0].line_no, Some(1));
        assert_eq!(diff[0].text, "fn main() {");
        assert_eq!(diff[1].kind, '-');
        assert_eq!(diff[1].line_no, Some(2));
        assert_eq!(diff[1].text, "    old();");
        assert_eq!(diff[2].kind, '+');
        assert_eq!(diff[2].line_no, Some(2));
        assert_eq!(diff[2].text, "    new();");
        assert_eq!(diff[3].kind, ' ');
        assert_eq!(diff[3].line_no, Some(3));
        let (added, removed) = count_diff_lines(output);
        assert_eq!(added, 1);
        assert_eq!(removed, 1);
    }

    #[test]
    fn parse_hunk_header_extracts_start_lines() {
        assert_eq!(parse_hunk(" -21,4 +21,5 @@"), Some((21, 21)));
        assert_eq!(parse_hunk(" -1 +2 @@"), Some((1, 2)));
        assert_eq!(parse_hunk(" @@"), None);
    }

    #[test]
    fn parse_diff_without_hunk_keeps_added_removed() {
        let output = "diff --git a/x b/x\n+line a\n-line b\n context\n";
        let diff = parse_diff(output);
        assert_eq!(diff.len(), 2);
        assert_eq!(diff[0].kind, '+');
        assert_eq!(diff[0].line_no, None);
        assert_eq!(diff[1].kind, '-');
        assert_eq!(diff[1].line_no, None);
    }

    #[test]
    fn plain_output_is_not_parsed_as_diff() {
        let output = "- first\n- second\n-- flag\n+ plus\n";
        let diff = parse_diff(output);
        assert!(diff.is_empty(), "non-diff output must not render diff rows");
        let (added, removed) = count_diff_lines(output);
        assert_eq!((added, removed), (0, 0));
    }

    #[test]
    fn count_diff_lines_ignores_headers() {
        let output = "+++ b/x\n--- a/x\n@@ -1 +1 @@\n+line\n";
        let (added, removed) = count_diff_lines(output);
        assert_eq!(added, 1);
        assert_eq!(removed, 0);
    }

    #[test]
    fn parse_diff_limits_rows() {
        let mut output = "diff --git a/x b/x\n@@ -1 +1 @@\n".to_string();
        for i in 0..60 {
            output.push_str(&format!("+line {i}\n"));
        }
        let diff = parse_diff(&output);
        assert!(diff.len() <= 40);
    }

    #[test]
    fn extract_filepath_from_arguments() {
        assert_eq!(
            extract_filepath(r#"{"filepath": "src/a.rs"}"#),
            Some("src/a.rs".to_string())
        );
        assert_eq!(
            extract_filepath(r#"{"path": "src/b.rs"}"#),
            Some("src/b.rs".to_string())
        );
        assert_eq!(extract_filepath(r#"{"command": "ls"}"#), None);
        assert_eq!(extract_filepath("not json"), None);
    }

    #[test]
    fn turn_summary_counts_files() {
        let mut ui = UiState::new();
        ui.reset_turn();
        ui.note_tool_started(r#"{"filepath": "a.rs"}"#);
        ui.note_tool_finished("diff --git a/a.rs b/a.rs\n@@ -1 +1 @@\n+x\n-y\n");
        ui.note_tool_started(r#"{"filepath": "a.rs"}"#);
        ui.note_tool_finished("diff --git a/a.rs b/a.rs\n@@ -1 +1 @@\n+z\n");
        ui.note_tool_started(r#"{"command": "cargo test"}"#);
        ui.note_tool_finished("ok");
        assert_eq!(ui.turn_files.len(), 1);
        assert_eq!(ui.turn_added, 2);
        assert_eq!(ui.turn_removed, 1);
        assert_eq!(ui.turn_tool_calls, 3);
        ui.finish_turn();
        let summary = match ui.messages.last() {
            Some(UiMessage::Summary(s)) => s.clone(),
            _ => panic!("expected summary message"),
        };
        assert!(summary.contains("1 file changed"), "{summary}");
        assert!(summary.contains("+2"), "{summary}");
        assert!(summary.contains("−1"), "{summary}");
    }

    #[test]
    fn turn_without_tools_has_no_summary() {
        let mut ui = UiState::new();
        ui.reset_turn();
        ui.finish_turn();
        assert!(ui.messages.is_empty());
    }
}
