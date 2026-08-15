use std::path::{Path, PathBuf};
use std::sync::Arc;

use futures::StreamExt;
use serde::{Deserialize, Serialize};
use tokio_util::sync::CancellationToken;

use crate::adapter::types::{Message, ReasoningEffort, Temperature};
use crate::runtime::ports::LlmPort;
use crate::runtime::types::{GenerateRequest, ModulationContext};

pub const MAX_FULL_SESSIONS: usize = 15;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationTurn {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionMeta {
    pub id: String,
    pub created: String,
    pub summary: String,
    pub turns: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchiveEntry {
    pub id: String,
    pub summary: String,
    pub highlights: Vec<String>,
    pub archived_at: String,
}

#[derive(Debug, Default, Serialize, Deserialize)]
struct Index {
    sessions: Vec<SessionMeta>,
    archive: Vec<ArchiveEntry>,
}

pub struct Remember {
    port: Arc<dyn LlmPort>,
    dir: PathBuf,
    index: Index,
    current_session: Option<SessionMeta>,
}

impl Remember {
    pub fn new(port: Arc<dyn LlmPort>, project_dir: &Path) -> Self {
        let dir = project_dir.join(".prognosis").join("history");
        let index = Self::load_index(&dir);
        Self {
            port,
            dir,
            index,
            current_session: None,
        }
    }

    fn load_index(dir: &Path) -> Index {
        std::fs::read_to_string(dir.join("index.json"))
            .ok()
            .and_then(|text| serde_json::from_str(&text).ok())
            .unwrap_or_default()
    }

    fn save_index(&self) {
        if let Err(err) = std::fs::create_dir_all(&self.dir) {
            eprintln!("[remember] cannot create history dir: {err}");
            return;
        }
        if let Err(err) = std::fs::write(
            self.dir.join("index.json"),
            serde_json::to_string_pretty(&self.index).unwrap_or_default(),
        ) {
            eprintln!("[remember] cannot save index: {err}");
        }
    }

    fn next_id(&self) -> String {
        let count = self.index.sessions.len() + self.index.archive.len() + 1;
        format!("s{count:04}")
    }

    pub fn current_session(&self) -> Option<&str> {
        self.current_session.as_ref().map(|meta| meta.id.as_str())
    }

    pub async fn start_session(&mut self) -> String {
        let id = self.next_id();
        if self.index.sessions.len() >= MAX_FULL_SESSIONS {
            let oldest = self.index.sessions.remove(0);
            let turns = self.load_session(&oldest.id);
            let summary = self.summarize_session(&oldest, &turns).await;
            self.index.archive.push(summary);
            let _ = std::fs::remove_file(self.dir.join(format!("sessions/{}.json", oldest.id)));
        }
        self.current_session = Some(SessionMeta {
            id: id.clone(),
            created: now_string(),
            summary: String::new(),
            turns: 0,
        });
        self.save_index();
        id
    }

    pub fn append_turn(&mut self, role: &str, content: &str) {
        let Some(meta) = self.current_session.clone() else {
            return;
        };
        let id = meta.id.clone();
        let mut turns = self.load_session(&id);
        turns.push(ConversationTurn {
            role: role.to_string(),
            content: content.to_string(),
        });
        if let Err(err) = std::fs::create_dir_all(self.dir.join("sessions")) {
            eprintln!("[remember] cannot create sessions dir: {err}");
            return;
        }
        if let Err(err) = std::fs::write(
            self.dir.join(format!("sessions/{id}.json")),
            serde_json::to_string_pretty(&turns).unwrap_or_default(),
        ) {
            eprintln!("[remember] cannot save session: {err}");
        }
        if !self.index.sessions.iter().any(|meta| meta.id == id) {
            self.index.sessions.push(meta);
        }
        if let Some(meta) = self.index.sessions.iter_mut().find(|meta| meta.id == id) {
            meta.turns = turns.len();
            if role == "user" && meta.summary.is_empty() && !content.is_empty() {
                meta.summary = content.chars().take(60).collect();
            }
        }
        self.save_index();
    }

    pub fn load_session(&self, id: &str) -> Vec<ConversationTurn> {
        std::fs::read_to_string(self.dir.join(format!("sessions/{id}.json")))
            .ok()
            .and_then(|text| serde_json::from_str(&text).ok())
            .unwrap_or_default()
    }

    pub fn list_sessions(&self) -> &[SessionMeta] {
        &self.index.sessions
    }

    pub fn archive(&self) -> &[ArchiveEntry] {
        &self.index.archive
    }

    pub fn set_current(&mut self, id: &str) -> bool {
        if let Some(meta) = self
            .index
            .sessions
            .iter()
            .find(|meta| meta.id == id)
            .cloned()
        {
            self.current_session = Some(meta);
            true
        } else {
            false
        }
    }

    async fn summarize_session(
        &self,
        meta: &SessionMeta,
        turns: &[ConversationTurn],
    ) -> ArchiveEntry {
        let transcript = turns
            .iter()
            .map(|turn| format!("{}: {}", turn.role, turn.content))
            .collect::<Vec<_>>()
            .join("\n");
        let output = if transcript.trim().is_empty() {
            String::new()
        } else {
            self.call_summarize(&transcript).await
        };
        let (summary, highlights) = match parse_summary(&output) {
            Some(parsed) => parsed,
            None => (
                meta.summary.clone(),
                turns
                    .iter()
                    .filter(|turn| turn.role == "user")
                    .map(|turn| turn.content.chars().take(60).collect())
                    .collect(),
            ),
        };
        ArchiveEntry {
            id: meta.id.clone(),
            summary,
            highlights,
            archived_at: now_string(),
        }
    }

    async fn call_summarize(&self, transcript: &str) -> String {
        let request = GenerateRequest {
            messages: vec![
                Message::system(
                    "You are an archivist. When a session is archived, it must leave a durable memory: a summary plus highlights that future sessions can recall. The summary is read weeks later by an agent that never saw this conversation; it must stand on its own.\
\n\n# Task\
\nSummarize the conversation and extract highlights worth remembering.\
\n\n# Output\
\nReply with JSON only, no other text:\
\n{\"summary\": \"<concise summary>\", \"highlights\": [\"<key fact>\"]}\
\n\n# Rules\
\n- summary: the whole session in a few sentences — what happened, what was decided, what was built, and where the work stands.\
\n- highlights: concrete facts, decisions, preferences, and open threads worth recalling later; 2-5 items, each specific enough to act on (\"user prefers Chinese replies\" over \"user preferences\").\
\n- Do not invent details not present in the transcript.\
\n- Do not include greetings or chit-chat.",
                ),
                Message::user(transcript),
            ],
            modulation: ModulationContext {
                reasoning_effort: Some(ReasoningEffort::None),
                temperature: Temperature::new(0.0).ok(),
                ..Default::default()
            },
            tools: None,
        };
        let cancel = CancellationToken::new();
        let stream = match self.port.generate(&request, &cancel).await {
            Ok(stream) => stream,
            Err(_) => return String::new(),
        };
        let mut stream = stream;
        let mut content = String::new();
        while let Some(item) = stream.next().await {
            if let Ok(chunk) = item
                && let Some(text) = chunk.content() {
                    content.push_str(text);
                }
        }
        content
    }
}

fn parse_summary(output: &str) -> Option<(String, Vec<String>)> {
    let json: serde_json::Value =
        serde_json::from_str(crate::util::extract_json_object(output)?).ok()?;
    let summary = json.get("summary")?.as_str()?.to_string();
    let highlights = json
        .get("highlights")
        .and_then(|h| h.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|item| item.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default();
    Some((summary, highlights))
}

fn now_string() -> String {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs().to_string())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use crate::adapter::error::AdapterError;
    use crate::adapter::types::{ChunkDelta, CompletionChunk, FinishReason};
    use std::pin::Pin;

    struct SummarizePort;

    #[async_trait]
    impl LlmPort for SummarizePort {
        async fn generate<'a>(
            &'a self,
            _request: &'a GenerateRequest,
            _cancel: &'a CancellationToken,
        ) -> Result<
            Pin<Box<dyn futures::Stream<Item = Result<CompletionChunk, AdapterError>> + Send + 'a>>,
            AdapterError,
        > {
            let chunk = CompletionChunk {
                model: "summarizer".into(),
                index: 0,
                delta: ChunkDelta {
                    role: None,
                    content: Some(
                        r#"{"summary":"user asked about weather","highlights":["weather","forecast"]}"#
                            .into(),
                    ),
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

    fn temp_dir() -> tempfile::TempDir {
        tempfile::TempDir::new().unwrap()
    }


    #[tokio::test]
    async fn session_rollover_archives_oldest() {
        let dir = temp_dir();
        let mut remember = Remember::new(Arc::new(SummarizePort), dir.path());
        for _ in 0..MAX_FULL_SESSIONS {
            let _id = remember.start_session().await;
            remember.append_turn("user", "hello");
            remember.append_turn("assistant", "hi there");
        }
        assert_eq!(remember.list_sessions().len(), MAX_FULL_SESSIONS);
        assert!(remember.archive().is_empty());

        remember.start_session().await;
        assert_eq!(
            remember.list_sessions().len(),
            MAX_FULL_SESSIONS - 1,
            "an empty session is not registered until its first turn"
        );
        assert_eq!(remember.archive().len(), 1);
        remember.append_turn("user", "after rollover");
        assert_eq!(remember.list_sessions().len(), MAX_FULL_SESSIONS);
        assert_eq!(remember.archive()[0].summary, "user asked about weather");
        assert_eq!(remember.archive()[0].highlights, vec!["weather", "forecast"]);
    }

    #[tokio::test]
    async fn turns_are_persisted_and_loaded() {
        let dir = temp_dir();
        let mut remember = Remember::new(Arc::new(SummarizePort), dir.path());
        let id = remember.start_session().await;
        remember.append_turn("user", "hello");
        remember.append_turn("assistant", "hi");

        let loaded = Remember::new(Arc::new(SummarizePort), dir.path());
        let turns = loaded.load_session(&id);
        assert_eq!(turns.len(), 2);
        assert_eq!(turns[0].role, "user");
        assert_eq!(turns[1].content, "hi");
    }

    #[tokio::test]
    async fn tool_turns_persist_but_do_not_set_summary() {
        let dir = temp_dir();
        let mut remember = Remember::new(Arc::new(SummarizePort), dir.path());
        let id = remember.start_session().await;
        remember.append_turn("user", "hello");
        remember.append_turn("tool", "ls({\"dirPath\":\".\"}) -> src/ [allowed]");

        let loaded = Remember::new(Arc::new(SummarizePort), dir.path());
        let turns = loaded.load_session(&id);
        assert_eq!(turns.len(), 2);
        assert_eq!(turns[1].role, "tool");
        assert!(turns[1].content.contains("ls("), "{}", turns[1].content);
        let meta = loaded.list_sessions().iter().find(|m| m.id == id).unwrap();
        assert_eq!(meta.summary, "hello", "summary must come from the user turn only");
        assert_eq!(meta.turns, 2);
    }

    #[tokio::test]
    async fn resume_selects_existing_session() {
        let dir = temp_dir();
        let mut remember = Remember::new(Arc::new(SummarizePort), dir.path());
        let id = remember.start_session().await;
        remember.append_turn("user", "keep this");

        let mut loaded = Remember::new(Arc::new(SummarizePort), dir.path());
        assert!(loaded.set_current(&id));
        assert_eq!(loaded.current_session(), Some(id.as_str()));
        assert_eq!(loaded.load_session(&id)[0].content, "keep this");
    }
}
