use std::sync::Arc;

use futures::StreamExt;
use tokio_util::sync::CancellationToken;

use crate::adapter::types::{Message, ReasoningEffort, Temperature};
use crate::runtime::ports::LlmPort;
use crate::runtime::types::{GenerateRequest, ModulationContext};

pub enum Verdict {
    Allow,
    Regenerate { reason: String },
    Corrected { calls: Vec<ToolCallRecord> },
}

#[derive(Debug, Clone)]
pub struct ToolCallRecord {
    pub name: String,
    pub arguments: String,
    pub output: String,
}

pub struct Supervisor {
    port: Arc<dyn LlmPort>,
    enabled: std::sync::atomic::AtomicBool,
    failure_patterns: std::sync::Mutex<Vec<String>>,
}

impl Supervisor {
    pub fn new(port: Arc<dyn LlmPort>) -> Self {
        Self {
            port,
            enabled: std::sync::atomic::AtomicBool::new(false),
            failure_patterns: std::sync::Mutex::new(Vec::new()),
        }
    }

    pub fn set_enabled(&self, enabled: bool) {
        self.enabled.store(enabled, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled.load(std::sync::atomic::Ordering::Relaxed)
    }

    pub fn failure_patterns(&self) -> Vec<String> {
        self.failure_patterns.lock().unwrap().clone()
    }

    fn judge_prompt(&self) -> String {
        let mut prompt = "You are a careful reviewer that gates every tool call an AI assistant makes.\n\n# Task\nDecide whether the pending tool call is acceptable, based on the user's request, the tools already used, and the pending call.\n\n# Checklist\nJudge each dimension independently. The user's request is the highest authority: when dimensions conflict, the user's request wins.\n1. Tool match: the right tool for the job — file reads (read_file), file creation (create_new_file), file edits (edit_existing_file, single_find_and_replace), content search (grep_search), name search (file_glob_search), directory listing (ls), git diff (view_diff), web search (search_web), URL fetch (fetch_url_content), rules (create_rule_block, request_rule), skills (read_skill), scheduling (schedule_task, cancel_task) — never a shell command (run_terminal_command) when a dedicated tool exists AND can do the job. The dedicated file tools only work inside the project directory; when the target path is outside the project (e.g. ~/Desktop), a shell command is the correct choice, not a violation. Read-only shell inspection (sed/awk/cat on specific line ranges, cat -A for invisible characters) is a legitimate verification use even inside the project: read_file has NO line-range parameters and returns the whole file, so do not correct a line-range inspection into read_file; the hard rule remains: never use shell commands to edit files.\n2. Arguments: well-formed and sufficient — required fields present, values plausible and relevant (e.g. the path names the file the task concerns). Do not invent arguments the tool does not support (read_file accepts only filepath).\n3. Order: the call follows the natural order of the task — gather needed information before acting on it; do not act on information not yet obtained; do not skip a needed preliminary step (e.g. investigate before answering, read before edit, search before read). Do NOT block because a tool output in the trace looks short or truncated — outputs may be truncated and the assistant may have seen the full output. But DO block when the trace shows the required prerequisite never happened: editing a file with no prior read of that file in the trace is a violation even when the user named the file — the assistant must read the file before editing it, UNLESS the user explicitly instructed the direct action (e.g. 'edit directly', 'do not read first', 'just run it'): an explicit user instruction to perform the action directly overrides the order requirement. Acting on information that was never gathered is also a violation. Verification calls that re-check current state are legitimate. Block only when a genuinely required prerequisite is missing.\n4. Goal alignment: stays on the requested task, no scope drift\n5. Safety: no destructive or risky actions beyond what the user asked for\n\n# Output\nReply with JSON only, no other text:\n{\"pass\": true|false, \"reason\": \"...\", \"action\": \"regenerate\"|\"correct\"}\n\n# Rules\n- pass is true unless at least one dimension has a concrete deficiency backed by evidence from the tool log\n- reason names the deficient dimension(s) and cites what you saw; when pass is true, write \"satisfactory\"\n- action is how to fix a failed call: \"correct\" if the call itself can be repaired in place (wrong tool, wrong arguments, missing preceding call); \"regenerate\" if the call should not happen at all (no tool needed, wrong approach) or the plan needs rethinking\n- For calls whose arguments embed large content (full file contents, long code blocks), prefer \"regenerate\" over \"correct\": correcting requires reproducing the whole call verbatim, which is unreliable at scale.\n- Do not invent flaws: if the call is plausible, allow it."
            .to_string();
        let failure_patterns = self.failure_patterns.lock().unwrap().clone();
        if !failure_patterns.is_empty() {
            prompt.push_str("\n\nKnown failure patterns from history: ");
            prompt.push_str(&failure_patterns.join("; "));
        }
        prompt
    }

    fn correction_prompt(available_tools: &[String]) -> String {
        format!(
            "You are an editor that repairs an AI assistant's tool call before it runs, so the user's request is served correctly. The reviewer found a defect in the pending call; your job is to fix the call itself, not the overall plan.\n\n# Task\nFix the pending tool call, given the user's request, the tools already used, the pending call, and the review finding.\n\n# Rules\n- Change only what is wrong: the wrong tool choice, wrong arguments, or a missing preceding call (e.g. read before edit). Everything else stays byte-identical — reproducing correct parts verbatim is what keeps the plan intact.\n- Large content arguments (full file contents, long code blocks in content/old/new) must be copied through unchanged. Do not summarize, truncate, or regenerate them; altering them breaks the edit the user asked for. If the defect is inside such an argument, fix only the small fields around it.\n- Output only the calls that must now run, in order. Do not repeat calls that already ran.\n- Use only the available tools: {}.\n- If the pending call should not run at all (no tool needed, wrong approach), output {{\"calls\": []}} so the assistant re-plans instead.\n- Do not explain your changes.\n\n# Output\nReply with JSON only, no other text:\n{{\"calls\": [{{\"tool\": \"<tool name>\", \"arguments\": {{<JSON arguments>}}}}]}}",
            available_tools.join(", ")
        )
    }

    async fn call(&self, messages: Vec<Message>) -> Result<String, String> {
        let request = GenerateRequest {
            messages,
            modulation: ModulationContext {
                reasoning_effort: Some(ReasoningEffort::None),
                temperature: Temperature::new(0.0).ok(),
                ..Default::default()
            },
            tools: None,
        };
        let cancel = CancellationToken::new();
        let stream = self.port.generate(&request, &cancel).await.map_err(|e| e.to_string())?;
        let mut stream = stream;
        let mut content = String::new();
        while let Some(item) = stream.next().await {
            if let Ok(chunk) = item
                && let Some(text) = chunk.content() {
                    content.push_str(text);
                }
        }
        Ok(content)
    }

    fn build_user_msg(
        user_input: &str,
        tool_trace: &[ToolCallRecord],
        pending: &ToolCallRecord,
    ) -> String {
        let mut msg = format!("User: {user_input}\n");
        if !tool_trace.is_empty() {
            msg.push_str("Tools used so far:");
            for (i, record) in tool_trace.iter().enumerate() {
                msg.push_str(&format!(
                    "\n{}. {}({}) -> {}",
                    i + 1,
                    record.name,
                    record.arguments,
                    record.output
                ));
            }
        }
        msg.push_str(&format!(
            "\nPending tool call: {}({})",
            pending.name, pending.arguments
        ));
        msg
    }

    pub async fn judge(
        &self,
        user_input: &str,
        tool_trace: &[ToolCallRecord],
        pending: &ToolCallRecord,
        available_tools: &[String],
    ) -> Verdict {
        if !self.is_enabled() {
            return Verdict::Allow;
        }
        let messages = vec![
            Message::system(self.judge_prompt()),
            Message::user(Self::build_user_msg(user_input, tool_trace, pending)),
        ];
        let output = match self.call(messages).await {
            Ok(output) => output,
            Err(_) => return Self::local_guard_verdict(pending),
        };
        match parse_verdict(&output) {
            Some((true, _, _)) => Verdict::Allow,
            Some((false, reason, action)) => {
                {
                    let mut patterns = self.failure_patterns.lock().unwrap();
                    patterns.push(reason.clone());
                    if patterns.len() > 20 {
                        patterns.remove(0);
                    }
                }
                match action.as_deref() {
                    Some("correct") => match self
                        .correct(user_input, tool_trace, pending, available_tools, &reason)
                        .await
                    {
                        Some(calls) => Verdict::Corrected { calls },
                        None => Verdict::Regenerate { reason },
                    },
                    _ => Verdict::Regenerate { reason },
                }
            }
            None => Self::local_guard_verdict(pending),
        }
    }

    fn local_guard_verdict(pending: &ToolCallRecord) -> Verdict {
        if let Some(reason) = local_guard_reason(pending) {
            Verdict::Regenerate { reason }
        } else {
            Verdict::Allow
        }
    }

    async fn correct(
        &self,
        user_input: &str,
        tool_trace: &[ToolCallRecord],
        pending: &ToolCallRecord,
        available_tools: &[String],
        reason: &str,
    ) -> Option<Vec<ToolCallRecord>> {
        let messages = vec![
            Message::system(Self::correction_prompt(available_tools)),
            Message::user(format!(
                "{}\nReview finding: {reason}",
                Self::build_user_msg(user_input, tool_trace, pending)
            )),
        ];
        let output = self.call(messages).await.ok()?;
        parse_calls(&output)
    }
}

fn local_guard_reason(pending: &ToolCallRecord) -> Option<String> {
    if pending.name != "run_terminal_command" {
        return None;
    }
    let command = pending.arguments.to_lowercase();
    let broad_target = ["rm -rf /", "rm -rf ~", "rm -rf $home", "rm -rf .", "rm -rf *"]
        .iter()
        .any(|pat| command.contains(pat));
    if broad_target {
        return Some(
            "the command targets a broad directory with rm -rf; resolve the exact target and confirm it is what the user asked to delete"
                .to_string(),
        );
    }
    let edits_file = ["sed -i", "perl -i", "awk -i", "> ", ">> ", "mv ", "rm ", "dd if=", "mkfs", "git reset --hard", "git checkout --"]
        .iter()
        .any(|pat| command.contains(pat));
    if edits_file {
        return Some(
            "the command modifies or deletes files or git state; use edit_existing_file / single_find_and_replace for file edits, and never run destructive git commands without explicit user approval"
                .to_string(),
        );
    }
    None
}

fn parse_verdict(output: &str) -> Option<(bool, String, Option<String>)> {
    let json: serde_json::Value =
        serde_json::from_str(crate::util::extract_json_object(output)?).ok()?;
    let pass = json.get("pass")?.as_bool()?;
    let reason = json
        .get("reason")
        .and_then(|r| r.as_str())
        .unwrap_or("")
        .to_string();
    let action = json
        .get("action")
        .and_then(|a| a.as_str())
        .map(str::to_string);
    Some((pass, reason, action))
}

fn parse_calls(output: &str) -> Option<Vec<ToolCallRecord>> {
    let json: serde_json::Value =
        serde_json::from_str(crate::util::extract_json_object(output)?).ok()?;
    let calls = json.get("calls")?.as_array()?;
    let mut records = Vec::new();
    for call in calls {
        let name = call.get("tool")?.as_str()?;
        let arguments = call.get("arguments").cloned().unwrap_or(serde_json::json!({}));
        records.push(ToolCallRecord {
            name: name.to_string(),
            arguments: arguments.to_string(),
            output: String::new(),
        });
    }
    Some(records)
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use crate::adapter::error::AdapterError;
    use crate::adapter::types::{ChunkDelta, CompletionChunk, FinishReason};
    use std::pin::Pin;

    #[test]
    fn local_guard_blocks_destructive_and_editing_commands() {
        let blocked = [
            r#"{"command":"rm -rf ~"}"#,
            r#"{"command":"rm -rf /tmp"}"#,
            r#"{"command":"sed -i 's/a/b/' x.py"}"#,
            r#"{"command":"perl -i -pe 's/a/b/' x.py"}"#,
            r#"{"command":"git reset --hard"}"#,
            r#"{"command":"git checkout -- src/a.rs"}"#,
            r#"{"command":"echo hi > x.py"}"#,
            r#"{"command":"mv a b"}"#,
        ];
        for args in blocked {
            let pending = ToolCallRecord {
                name: "run_terminal_command".into(),
                arguments: args.into(),
                output: String::new(),
            };
            assert!(
                local_guard_reason(&pending).is_some(),
                "must block: {args}"
            );
        }
        let allowed = [
            r#"{"command":"ls -la"}"#,
            r#"{"command":"sed -n '1,20p' x.py"}"#,
            r#"{"command":"cat -A x.py"}"#,
            r#"{"command":"python3 -m pytest"}"#,
        ];
        for args in allowed {
            let pending = ToolCallRecord {
                name: "run_terminal_command".into(),
                arguments: args.into(),
                output: String::new(),
            };
            assert!(
                local_guard_reason(&pending).is_none(),
                "must allow: {args}"
            );
        }
        let non_shell = ToolCallRecord {
            name: "read_file".into(),
            arguments: r#"{"filepath":"a.rs"}"#.into(),
            output: String::new(),
        };
        assert!(local_guard_reason(&non_shell).is_none());
    }

    #[test]
    fn judge_prompt_warns_about_truncated_traces_and_allows_verification() {
        let supervisor = Supervisor::new(Arc::new(JudgePort {
            outputs: vec![],
            calls: std::sync::Mutex::new(0),
        }));
        let prompt = supervisor.judge_prompt();
        assert!(
            prompt.contains("outputs may be truncated and the assistant may have seen the full output"),
            "judge must know traces can be truncated: {prompt}"
        );
        assert!(
            prompt.contains("editing a file with no prior read of that file in the trace is a violation"),
            "judge must block unread edits: {prompt}"
        );
        assert!(
            prompt.contains("Read-only shell inspection"),
            "judge must allow read-only shell inspection: {prompt}"
        );
    }

    struct JudgePort {
        outputs: Vec<String>,
        calls: std::sync::Mutex<usize>,
    }

    struct RecordingPort {
        outputs: Vec<String>,
        seen: std::sync::Mutex<Vec<String>>,
        modulations: std::sync::Mutex<Vec<ModulationContext>>,
    }

    #[async_trait]
    impl LlmPort for JudgePort {
        async fn generate<'a>(
            &'a self,
            _request: &'a GenerateRequest,
            _cancel: &'a CancellationToken,
        ) -> Result<
            Pin<Box<dyn futures::Stream<Item = Result<CompletionChunk, AdapterError>> + Send + 'a>>,
            AdapterError,
        > {
            let mut calls = self.calls.lock().unwrap();
            let output = self.outputs[*calls % self.outputs.len()].clone();
            *calls += 1;
            let chunk = CompletionChunk {
                model: "judge".into(),
                index: 0,
                delta: ChunkDelta {
                    role: None,
                    content: Some(output),
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

    #[async_trait]
    impl LlmPort for RecordingPort {
        async fn generate<'a>(
            &'a self,
            request: &'a GenerateRequest,
            _cancel: &'a CancellationToken,
        ) -> Result<
            Pin<Box<dyn futures::Stream<Item = Result<CompletionChunk, AdapterError>> + Send + 'a>>,
            AdapterError,
        > {
            for message in &request.messages {
                if message.role == crate::adapter::types::Role::User {
                    self.seen.lock().unwrap().push(message.content.to_plain_text());
                }
            }
            self.modulations.lock().unwrap().push(request.modulation.clone());
            let output = self.outputs[0].clone();
            let chunk = CompletionChunk {
                model: "judge".into(),
                index: 0,
                delta: ChunkDelta {
                    role: None,
                    content: Some(output),
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

    fn pending_call() -> ToolCallRecord {
        ToolCallRecord {
            name: "edit_file".into(),
            arguments: "src/main.rs".into(),
            output: String::new(),
        }
    }

    #[tokio::test]
    async fn plausible_call_is_allowed() {
        let port = JudgePort {
            outputs: vec![r#"{"pass":true,"reason":"satisfactory","action":"regenerate"}"#.into()],
            calls: std::sync::Mutex::new(0),
        };
        let supervisor = Supervisor::new(Arc::new(port));
        supervisor.set_enabled(true);
        let verdict = supervisor.judge("fix a bug", &[], &pending_call(), &[]).await;
        assert!(matches!(verdict, Verdict::Allow));
    }

    #[tokio::test]
    async fn flawed_call_regenerates_and_accumulates_pattern() {
        let port = JudgePort {
            outputs: vec![
                r#"{"pass":false,"reason":"no read before edit","action":"regenerate"}"#.into(),
            ],
            calls: std::sync::Mutex::new(0),
        };
        let supervisor = Supervisor::new(Arc::new(port));
        supervisor.set_enabled(true);
        let verdict = supervisor.judge("fix a bug", &[], &pending_call(), &[]).await;
        match verdict {
            Verdict::Regenerate { reason } => {
                assert_eq!(reason, "no read before edit");
            }
            _ => panic!("expected regenerate verdict"),
        }
        assert_eq!(supervisor.failure_patterns(), vec!["no read before edit"]);
    }

    #[tokio::test]
    async fn disabled_supervisor_allows_without_call() {
        let port = Arc::new(JudgePort {
            outputs: vec![],
            calls: std::sync::Mutex::new(0),
        });
        let supervisor = Supervisor::new(port.clone());
        let verdict = supervisor.judge("fix a bug", &[], &pending_call(), &[]).await;
        assert!(matches!(verdict, Verdict::Allow));
        assert_eq!(*port.calls.lock().unwrap(), 0);
    }

    #[tokio::test]
    async fn correct_action_returns_fixed_call_sequence() {
        let port = JudgePort {
            outputs: vec![
                r#"{"pass":false,"reason":"wrong tool","action":"correct"}"#.into(),
                r#"{"calls":[{"tool":"read_file","arguments":{"path":"src/main.rs"}}]}"#.into(),
            ],
            calls: std::sync::Mutex::new(0),
        };
        let supervisor = Supervisor::new(Arc::new(port));
        supervisor.set_enabled(true);
        let pending = ToolCallRecord {
            name: "run_command".into(),
            arguments: "cat src/main.rs".into(),
            output: String::new(),
        };
        let verdict = supervisor.judge("fix a bug", &[], &pending, &["read_file".into()]).await;
        match verdict {
            Verdict::Corrected { calls } => {
                assert_eq!(calls.len(), 1);
                assert_eq!(calls[0].name, "read_file");
                assert_eq!(calls[0].arguments, r#"{"path":"src/main.rs"}"#);
            }
            _ => panic!("expected corrected verdict"),
        }
    }

    #[tokio::test]
    async fn correct_failure_falls_back_to_regenerate() {
        let port = JudgePort {
            outputs: vec![
                r#"{"pass":false,"reason":"wrong tool","action":"correct"}"#.into(),
                "not json".into(),
            ],
            calls: std::sync::Mutex::new(0),
        };
        let supervisor = Supervisor::new(Arc::new(port));
        supervisor.set_enabled(true);
        let verdict = supervisor.judge("fix a bug", &[], &pending_call(), &[]).await;
        assert!(matches!(verdict, Verdict::Regenerate { .. }));
    }

    #[tokio::test]
    async fn tool_trace_and_pending_call_are_included_in_judge_input() {
        let port = Arc::new(RecordingPort {
            outputs: vec![r#"{"pass":true,"reason":"ok"}"#.into()],
            seen: std::sync::Mutex::new(Vec::new()),
            modulations: std::sync::Mutex::new(Vec::new()),
        });
        let supervisor = Supervisor::new(port.clone());
        supervisor.set_enabled(true);
        let trace = vec![ToolCallRecord {
            name: "read_file".into(),
            arguments: "src/main.rs".into(),
            output: "fn main() {}".into(),
        }];
        let pending = ToolCallRecord {
            name: "edit_file".into(),
            arguments: "src/main.rs".into(),
            output: String::new(),
        };
        let _ = supervisor.judge("fix the bug", &trace, &pending, &[]).await;
        let seen = port.seen.lock().unwrap();
        assert!(seen[0].contains("Tools used so far"));
        assert!(seen[0].contains("1. read_file(src/main.rs) -> fn main() {}"));
        assert!(seen[0].contains("Pending tool call: edit_file(src/main.rs)"));
    }

    #[tokio::test]
    async fn judge_and_correct_calls_disable_thinking() {
        let port = Arc::new(RecordingPort {
            outputs: vec![r#"{"pass":false,"reason":"wrong tool","action":"correct"}"#.into()],
            seen: std::sync::Mutex::new(Vec::new()),
            modulations: std::sync::Mutex::new(Vec::new()),
        });
        let supervisor = Supervisor::new(port.clone());
        supervisor.set_enabled(true);
        let pending = ToolCallRecord {
            name: "run_terminal_command".into(),
            arguments: "echo hi".into(),
            output: String::new(),
        };
        let _ = supervisor
            .judge(
                "task",
                &[],
                &pending,
                &["ls".to_string(), "run_terminal_command".to_string()],
            )
            .await;
        let mods = port.modulations.lock().unwrap();
        assert!(!mods.is_empty(), "judge must issue a light call");
        for modulation in mods.iter() {
            assert_eq!(
                modulation.reasoning_effort,
                Some(ReasoningEffort::None),
                "light calls must disable thinking"
            );
            assert!(
                modulation.temperature.is_some(),
                "light calls should be deterministic"
            );
        }
    }

    #[test]
    fn verdict_json_parsing() {
        assert_eq!(
            parse_verdict(r#"{"pass":true,"reason":"ok"}"#),
            Some((true, "ok".to_string(), None))
        );
        assert_eq!(
            parse_verdict(r#"{"pass":false,"reason":"bad","action":"regenerate"}"#),
            Some((false, "bad".to_string(), Some("regenerate".to_string())))
        );
        assert_eq!(parse_verdict("not json"), None);
    }

    #[test]
    fn calls_json_parsing() {
        let calls = parse_calls(
            r#"{"calls":[{"tool":"read_file","arguments":{"path":"a.rs"}},{"tool":"edit_file","arguments":{"path":"a.rs"}}]}"#,
        )
        .unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "read_file");
        assert_eq!(calls[1].name, "edit_file");
        assert!(parse_calls("not json").is_none());
        assert_eq!(parse_calls(r#"{"calls":[]}"#).map(|c| c.len()), Some(0));
    }

}
