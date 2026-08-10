use std::collections::HashSet;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use futures::StreamExt;
use tokio_util::sync::CancellationToken;

use crate::adapter::types::{Message, ReasoningEffort, Temperature, ToolCall, ToolDefinition};
use crate::runtime::actor::{ActorContext, CognitiveActor};
use crate::runtime::bus::EventBus;
use crate::runtime::event::{Event, EventKind, EventMeta, StateRequest, StateResponse};
use crate::runtime::ports::LlmPort;
use crate::runtime::types::{
    GenerateRequest, ModulationContext, PredictionTrajectory, RuleContext, SkillContext,
    WorkingMemorySnapshot,
};

#[derive(serde::Deserialize)]
struct TrajectoryJson {
    #[serde(default)]
    topics: Vec<String>,
    #[serde(default)]
    key_elements: Vec<String>,
    #[serde(default)]
    direction: f32,
    #[serde(default)]
    intents: Vec<String>,
    #[serde(default)]
    reaction: String,
    #[serde(default)]
    reaction_sentiment: f32,
}

#[derive(Clone)]
struct ToolRound {
    id: String,
    name: String,
    arguments: serde_json::Value,
    reasoning: Option<String>,
    content: String,
    output: Option<String>,
}

pub struct PredictionActor {
    port: Arc<dyn LlmPort>,
    inhibited: HashSet<String>,
    tools: Vec<ToolDefinition>,
    rules: Vec<RuleContext>,
    skills: Vec<SkillContext>,
    last_trajectory: PredictionTrajectory,
    wm_snapshot: WorkingMemorySnapshot,
    session_summary: String,
    last_messages: Vec<Message>,
    tool_rounds: Vec<ToolRound>,
    stray_results: Vec<(String, String)>,
    executed_tools: Vec<(String, String)>,
    current_task: String,
    stream_cycle: Option<crate::runtime::types::CycleId>,
    stream_text: String,
    batch_generated: bool,
    meta_state: Option<crate::runtime::types::MetaState>,
}

impl PredictionActor {
    pub fn new(port: Arc<dyn LlmPort>) -> Self {
        Self {
            port,
            inhibited: HashSet::new(),
            tools: Vec::new(),
            rules: Vec::new(),
            skills: Vec::new(),
            last_trajectory: PredictionTrajectory::default(),
            wm_snapshot: WorkingMemorySnapshot::default(),
            session_summary: String::new(),
            last_messages: Vec::new(),
            tool_rounds: Vec::new(),
            stray_results: Vec::new(),
            executed_tools: Vec::new(),
            current_task: String::new(),
            stream_cycle: None,
            stream_text: String::new(),
            batch_generated: false,
            meta_state: None,
        }
    }

    pub fn with_tools(mut self, tools: Vec<ToolDefinition>) -> Self {
        self.tools = tools;
        self
    }

    pub fn with_rules(mut self, rules: Vec<RuleContext>) -> Self {
        self.rules = rules;
        self
    }

    fn spawn_prediction(&self, meta: EventMeta, input: String, bus: EventBus) {
        let port = self.port.clone();
        let rules = self.rules.clone();
        let inhibited = self.inhibited.iter().cloned().collect::<Vec<_>>();
        tokio::spawn(async move {
            let predictor = PredictionActor::new(port).with_rules(rules);
            if let Ok(Some(mut trajectory)) = predictor.predict(&input, &bus, meta).await {
                trajectory
                    .topics
                    .retain(|topic| !inhibited.contains(topic));
                trajectory
                    .key_elements
                    .retain(|element| !inhibited.contains(element));
                bus.publish(Event::Prediction { meta, trajectory });
            }
        });
    }

    async fn predict(
        &self,
        input: &str,
        bus: &EventBus,
        meta: EventMeta,
    ) -> Result<Option<PredictionTrajectory>, String> {
        let correlation_id = meta.cycle_id.0;
        bus.publish(Event::RequestState {
            meta,
            request: StateRequest::MemoryRetrieval {
                query: input.to_string(),
            },
            correlation_id,
        });
        let prior = {
            let mut responses = Box::pin(bus.subscribe_kinds(&[EventKind::State]));
            tokio::time::timeout(Duration::from_millis(500), async {
                loop {
                    match responses.next().await? {
                        Event::StateResponse {
                            correlation_id: id,
                            response: StateResponse::MemoryRetrieval(retrieval),
                            ..
                        } if id == correlation_id => return Some(retrieval),
                        _ => continue,
                    }
                }
            })
            .await
            .ok()
            .flatten()
            .map(|retrieval| {
                retrieval
                    .semantic
                    .iter()
                    .map(|entry| entry.content.clone())
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default()
        };
        let mut prior = prior;
        let rule_text = self
            .rules
            .iter()
            .filter(|rule| rule.always_apply == Some(true))
            .map(|rule| rule.rule.clone())
            .collect::<Vec<_>>();
        prior.extend(rule_text);
        let base_prompt = "You are a predictor for a cognitive agent. Before the assistant answers, you predict how the user will respond to the ongoing conversation. Your prediction is later compared against the user's actual next message, and the mismatch drives the agent's learning signals (prediction error, surprise, dopamine). Accurate, honest predictions matter more than confident ones: expressing uncertainty as candidate lists is correct behavior, not weakness.\
\n\n# Task\
\nGiven the conversation context, predict the user's next message.\
\n\n# Output\
\nReply with JSON only, no other text:\
\n{\"topics\": [\"<subject area>\"], \"key_elements\": [\"<concrete word or fact likely to appear>\"], \"direction\": <float in [-1, 1]>, \"intents\": [\"question\"|\"command\"|\"statement\"|\"smalltalk\"], \"reaction\": \"<predicted user message, one sentence>\", \"reaction_sentiment\": <float in [-1, 1]>}\
\n\nRules:\
\n- topics: the subject areas the user's next message may cover; 2-5 candidates, ordered by likelihood. If the conversation gives no signal, predict the most plausible continuation of the current topic rather than leaving this empty.\
\n- key_elements: concrete words, names, or facts likely to appear in the user's next message. Prefer specific items over generic ones (\"Athena\" over \"the database\").\
\n- direction: how the user positions itself — > 0 means agreement or continuation; < 0 means disagreement or redirection.\
\n- intents: the possible communicative intents of the user's next message, ranked most-likely first; give 1-3 candidates to reflect genuine uncertainty.\
\n- reaction: the most likely next user message, one sentence, written as the user would say it.\
\n- reaction_sentiment: the user's expected sentiment — > 0 satisfied or positive, < 0 dissatisfied or negative.\
\n- Predict what the user is likely to say, not what the assistant should say: you are predicting the user's side of the conversation, not drafting the answer.\
\n- Do not invent knowledge; predict from the context given. When the context is ambiguous, prefer a broader candidate set over a single guess.\
\n- When the last turn was a tool round (the assistant was executing tools), the user's next message usually reacts to what the tools revealed — predict that reaction.\
\n- When the conversation is a greeting or pure chit-chat, predict the natural reply to the greeting rather than forcing a work-related topic.";
        let prompt = if prior.is_empty() {
            base_prompt.to_string()
        } else {
            format!("{base_prompt}\n\nKnown knowledge: {}", prior.join("; "))
        };
        let modulation = ModulationContext {
            reasoning_effort: Some(ReasoningEffort::None),
            temperature: Temperature::new(0.0).ok(),
            ..Default::default()
        };
        let request = GenerateRequest {
            messages: vec![Message::system(prompt), Message::user(input)],
            modulation,
            tools: None,
        };
        let cancel = CancellationToken::new();
        let stream = match self.port.generate(&request, &cancel).await {
            Ok(stream) => stream,
            Err(err) => return Err(err.to_string()),
        };
        let mut stream = stream;
        let mut content = String::new();
        while let Some(item) = stream.next().await {
            if let Ok(chunk) = item
                && let Some(text) = chunk.content() {
                    content.push_str(text);
                }
        }
        Ok(parse_trajectory(&content))
    }

    fn active_rules(&self, context_text: &str) -> Vec<&RuleContext> {
        self.rules
            .iter()
            .filter(|rule| {
                if rule.always_apply == Some(true) {
                    return true;
                }
                let bare = rule.globs.is_empty()
                    && rule.regex.is_empty()
                    && rule.description.is_empty()
                    && rule.always_apply != Some(false);
                if bare {
                    return true;
                }
                if !rule.regex.is_empty()
                    && let Ok(re) = regex::Regex::new(&rule.regex)
                    && re.is_match(context_text)
                {
                    return true;
                }
                false
            })
            .collect()
    }

    fn build_generate(&self, input: &str) -> GenerateRequest {
        let background = self
            .wm_snapshot
            .slots
            .iter()
            .map(|slot| slot.content.clone())
            .collect::<Vec<_>>();
        let tool_result_text = self
            .wm_snapshot
            .slots
            .iter()
            .filter(|slot| slot.source == "ToolResult")
            .map(|slot| slot.content.clone())
            .collect::<Vec<_>>()
            .join("\n");
        let mut system = "You are a cognitive coding agent running inside the user's project directory. You act like a careful senior engineer: you read before you write, you verify before you claim, and you keep the user's goal in front of you at all times. A prediction-coding control system monitors your own uncertainty and shapes how you answer: when the situation is surprising or uncertain, slow down — gather evidence and verify before claiming anything; when you are confident, you may answer more fluently.\
\n\n# Ground rules\
\n- Ground every claim in evidence. Never fabricate file contents, search results, computed values, command outputs, or external facts. If you do not know something, say so explicitly.\
\n- Gather before you answer: if the answer depends on information you were not given, use the available tools to obtain it instead of guessing.\
\n- The user's request is the highest authority: when rules or evidence seem to conflict with it, resolve the conflict by verifying or by asking, never by silently picking a side.\
\n- Reply in the same language as the user's request, concisely. Match the user's level of detail: a one-line question gets a one-line answer unless more detail is genuinely needed.\
\n- Do not fabricate a response when you are blocked: if a tool is unavailable, a file is missing, or an action is not possible, say what is blocked and what you need to proceed.\
\n\n# Tool usage\
\n- Prefer the dedicated tool for each job: read_file for file contents, grep_search for content search, file_glob_search for finding files by name, ls for directory listings, create_new_file for new files, edit_existing_file / single_find_and_replace for modifications, view_diff for uncommitted changes, run_terminal_command only when no dedicated tool exists (builds, tests, git operations, servers) — or when the target lies outside the project directory, which the dedicated file tools cannot reach. Never use shell commands (sed, awk, etc.) to edit files.\
\n- Before calling a tool, know its required arguments: check the tool description and provide every required field. A call missing a required argument is rejected before it runs, and a rejected call costs a round trip.\
\n- Call each tool at most once per task unless new information genuinely requires a repeat. If a call failed, read the error and fix the arguments — a failed call usually means wrong arguments, not a broken tool.\
\n- When a command is run in the background, always suggest stopping it with shell commands, never Ctrl+C.\
\n\n# Reading and editing files\
\n- Read the relevant file (or the relevant part of it) before editing it, and re-read it after a previous edit changed the file, so your edits always apply to current content.\
\n- When editing, change only what the task requires. Preserve everything else byte-for-byte, including whitespace, comments, and unrelated code.\
\n- After an edit, verify the result when practical: re-read the edited region or run the relevant check (build, test, diff).\
\n\n# Failure recovery\
\n- A failed tool call usually means wrong arguments, not a broken tool: read the error message, fix the call, and retry.\
\n- If a tool result contradicts an earlier assumption, update your understanding instead of insisting on it.\
\n- If your call was rejected (denied, blocked, corrected), read the rejection reason and adjust: a denied call usually means the approach was wrong, a corrected call means the arguments were repaired for you — review the corrected result.\
\n- If the user's request is impossible or unsafe, say so instead of attempting it.\
\n- Do not loop: if the same call fails twice for the same reason, stop and change approach or ask the user, rather than retrying a third time.\
\n\n# Working style\
\n- When continuing after tool results, start with a NEW brief sentence about what you learned or what you will do next. Never repeat text you already wrote in this conversation.\
\n- If you are unsure what the user wants, ask one brief clarifying question rather than guessing.\
\n- If the current task involves code standards or preferences, request the relevant rule with the request_rule tool before answering.\
\n- When a task is complete, say so and summarize what changed in one or two lines — do not keep working."
            .to_string();
        let active_rules = self.active_rules(&tool_result_text);
        if !active_rules.is_empty() {
            let listed = active_rules
                .iter()
                .map(|rule| format!("- {}: {}", rule.name, rule.rule))
                .collect::<Vec<_>>()
                .join("\n");
            system.push_str(&format!("\n\n# Rules\n{listed}"));
        }
        if !self.skills.is_empty() {
            let listed = self
                .skills
                .iter()
                .map(|skill| format!("- {}: {}", skill.name, skill.description))
                .collect::<Vec<_>>()
                .join("\n");
            system.push_str(&format!(
                "\n\n# Available skills\n{listed}\nLoad the full instructions of any skill relevant to the current task with the read_skill tool before following it."
            ));
        }
        let mut messages = vec![Message::system(system)];
        if !self.session_summary.is_empty() {
            messages.push(Message::system(format!(
                "(Session summary — recalled context from a previous session; this is your memory of earlier work, not the current user input. Use it to continue the earlier work: pick up unfinished threads, respect decisions already made, and avoid re-litigating settled points. Treat facts here as background that may be stale: if they conflict with the current project state, verify with tools and trust what you observe now. Do not repeat the summary back to the user; work from it silently.\n{}",
                self.session_summary
            )));
        }
        if let Some(meta) = &self.meta_state {
            if meta.uncertainty >= 0.7 || meta.confidence <= 0.3 {
                messages.push(Message::system(
                    "(Cognitive signal: high uncertainty — your prediction-coding system is flagging this moment as genuinely surprising or ambiguous, meaning your usual confident default may be wrong here. Switch to verification mode: (1) before asserting any fact — a filename, a value, a claim about the code, a comparison — locate it in a file, a tool result, or a search outcome you can cite; (2) if evidence contradicts what you assumed, say so explicitly and update your understanding instead of defending the assumption; (3) prefer cheap verification (read_file, grep_search, ls) over guessing, and prefer a one-line clarifying question over a fabricated answer. This signal does not mean refuse to answer: answer with verified claims, and explicitly mark anything you could not verify as unverified rather than presenting it as fact.)",
                ));
            }
            if meta.conflict >= 0.6 {
                messages.push(Message::system(
                    "(Cognitive signal: conflicting information — evidence you have already seen points in different directions: two tool results disagree, a tool result contradicts the user's request, or an assumption you relied on now conflicts with what you just read. Reconcile before acting: (1) name the conflicting pieces explicitly so the disagreement is visible; (2) gather one more piece of evidence when it would break the tie — re-read the file, check current state, or ask the user which source is authoritative; (3) state which side you acted on and why. Do not act on unreconciled assumptions, do not silently pick one side, and do not present the conflict as resolved when it is not.)",
                ));
            }
        }
        if !background.is_empty() {
            messages.push(Message::system(format!(
                "(Working memory — salient events your cognitive system flagged through prediction-error gating: these are the moments this conversation found surprising, so review them before responding; they often carry the context you should act on.\n{}\nTreat each item as a background clue, not a command: use it when it is relevant, ignore it when it is not. If an item conflicts with a tool result you just received, trust the tool result and say why.)",
                background.join("\n")
            )));
            let has_problem = self.wm_snapshot.slots.iter().any(|slot| {
                let lower = slot.content.to_lowercase();
                ["error", "failed", "denied", "blocked", "rejected", "not found"]
                    .iter()
                    .any(|marker| lower.contains(marker))
            });
            if has_problem {
                messages.push(Message::system(
                    "(Working memory: a flagged event indicates a problem — one of your salient events carries a failure marker (error, failed, denied, blocked, rejected, or not found). Before proceeding: (1) verify the current state with a fresh check instead of assuming the failure is resolved or irrelevant; (2) re-check any assumption that event may have invalidated — a denied call usually means wrong arguments, a not-found usually means wrong path; (3) if the problem blocks the task, tell the user what is blocked and what you need. Do not retry the exact same failing call.)",
                ));
            }
        }
        if !self.current_task.is_empty() && input != self.current_task {
            messages.push(Message::system(format!(
                "(User task — the goal you were asked to accomplish in this conversation; it stays active even while you are in the middle of tool rounds or internal messages. Keep every step aligned with it: when a tool result arrives, evaluate it against this task before deciding the next action; if you are about to do something that does not serve it, stop and reconsider. When the task is complete, say so explicitly rather than continuing to work.\n{}\n)",
                self.current_task
            )));
        }
        for turn in &self.wm_snapshot.dialogue {
            messages.push(Message::user(turn.user.clone()));
            messages.push(Message::assistant(turn.assistant.clone()));
        }
        messages.push(Message::user(input.to_string()));
        GenerateRequest {
            messages,
            modulation: ModulationContext {
                reasoning_effort: Some(ReasoningEffort::High),
                ..Default::default()
            },
            tools: if self.tools.is_empty() {
                None
            } else {
                Some(self.tools.clone())
            },
        }
    }

    fn build_tool_result_generate(&self) -> GenerateRequest {
        let mut messages = self.last_messages.clone();
        if messages.is_empty() {
            return self.build_generate("(Tool result)");
        }
        for round in &self.tool_rounds {
            match &round.output {
                Some(output) => {
                    messages.push(Message::assistant_with_tool_calls_and_reasoning(
                        round.content.clone(),
                        vec![ToolCall {
                            id: round.id.clone(),
                            name: round.name.clone(),
                            arguments: round.arguments.clone(),
                        }],
                        round.reasoning.clone(),
                    ));
                    messages.push(Message::tool(round.id.clone(), output.clone()));
                }
                None => {
                    messages.push(Message::user(format!(
                        "(Tool call {} {} is still executing and its result has not arrived yet. Do NOT call it again and do NOT proceed as if it succeeded or failed — a duplicate call would be rejected, and acting on a missing result would be guessing. Wait for the result: it will arrive as a tool message right after this one. If you need to plan in the meantime, plan only conditionally — \"when the result arrives, then …\".)",
                        round.name, round.arguments
                    )));
                }
            }
        }
        for (_, output) in &self.stray_results {
            messages.push(Message::user(format!("(Tool result)\n{output}")));
        }
        if !self.executed_tools.is_empty() {
            let listed = self
                .executed_tools
                .iter()
                .map(|(name, args)| format!("- {name} {args}"))
                .collect::<Vec<_>>()
                .join("\n");
            let mut note = format!(
                "(Tools already executed in this task — each tool should be called at most once per task unless new information genuinely requires a repeat; calling an executed tool again is rejected by the runtime and wastes the user's time.\n{listed}\nUse these results as your evidence base: if the answer can be composed from what is already here, do not call more tools.)"
            );
            if let Some(last) = self.tool_rounds.last()
                && !last.content.trim().is_empty()
            {
                let snippet: String = last.content.trim().chars().take(80).collect();
                note.push_str(&format!(
                    "\n\n(Your previous round started with: \"{snippet}\" — write something new, do not repeat it. Models tend to reuse their opening sentence when resuming after tool results; the conversation already contains that text, and repeating it confuses the history. Open instead with one fresh sentence about what the latest result means for the task, then continue.)"
                ));
            }
            messages.push(Message::user(note));
        }
        GenerateRequest {
            messages,
            modulation: ModulationContext {
                reasoning_effort: Some(ReasoningEffort::High),
                ..Default::default()
            },
            tools: if self.tools.is_empty() {
                None
            } else {
                Some(self.tools.clone())
            },
        }
    }
}

fn parse_intent(raw: &str) -> crate::runtime::types::IntentKind {
    match raw {
        "question" => crate::runtime::types::IntentKind::Question,
        "command" => crate::runtime::types::IntentKind::Command,
        "smalltalk" => crate::runtime::types::IntentKind::Smalltalk,
        _ => crate::runtime::types::IntentKind::Statement,
    }
}

fn parse_trajectory(content: &str) -> Option<PredictionTrajectory> {
    let json: TrajectoryJson = serde_json::from_str(crate::util::extract_json_object(content)?).ok()?;
    let intent_candidates: Vec<crate::runtime::types::IntentKind> =
        json.intents.iter().map(|raw| parse_intent(raw)).collect();
    let intent = intent_candidates
        .first()
        .copied()
        .unwrap_or(crate::runtime::types::IntentKind::Statement);
    Some(PredictionTrajectory {
        topics: json.topics,
        key_elements: json.key_elements,
        direction: json.direction.clamp(-1.0, 1.0),
        intent,
        intent_candidates,
        reaction: json.reaction,
        reaction_sentiment: json.reaction_sentiment.clamp(-1.0, 1.0),
    })
}

#[async_trait]
impl CognitiveActor for PredictionActor {
    fn id(&self) -> &str {
        "prediction"
    }

    fn subscriptions(&self) -> Vec<EventKind> {
        vec![
            EventKind::Attention,
            EventKind::Inhibition,
            EventKind::State,
            EventKind::WorkingMemory,
            EventKind::Context,
            EventKind::Action,
            EventKind::Generation,
            EventKind::Modulation,
            EventKind::Prediction,
        ]
    }

    async fn handle(&mut self, event: &Event, ctx: &mut ActorContext) -> Vec<Event> {
        match event {
            Event::MetaUpdate { meta_state, .. } => {
                self.meta_state = Some(*meta_state);
                vec![]
            }
            Event::Chunk { meta, chunk } => {
                if self.stream_cycle != Some(meta.cycle_id) {
                    self.stream_cycle = Some(meta.cycle_id);
                    self.stream_text.clear();
                }
                if let Some(text) = chunk.content() {
                    self.stream_text.push_str(text);
                }
                vec![]
            }
            Event::ActionSelected { decision, .. } => {
                match &decision.candidate {
                    crate::runtime::types::ActionCandidate::CallTool {
                        name,
                        arguments,
                        tool_call_id,
                        reasoning,
                    } => {
                        self.tool_rounds.push(ToolRound {
                            id: tool_call_id.clone().unwrap_or_default(),
                            name: name.clone(),
                            arguments: arguments.clone(),
                            reasoning: reasoning.clone(),
                            content: self.stream_text.clone(),
                            output: None,
                        });
                        self.batch_generated = false;
                    }
                    _ => {
                        self.tool_rounds.clear();
                        self.stray_results.clear();
                    }
                }
                vec![]
            }
            Event::ToolResult { meta, result, .. } => {
                let id = result.tool_call_id.clone().unwrap_or_default();
                if id.is_empty() {
                    self.stray_results.push((id, result.output.clone()));
                } else if let Some(round) = self
                    .tool_rounds
                    .iter_mut()
                    .find(|round| !round.id.is_empty() && round.id == id)
                {
                    if round.output.is_none() {
                        round.output = Some(result.output.clone());
                        self.executed_tools
                            .push((round.name.clone(), round.arguments.to_string()));
                    }
                } else if id.is_empty()
                    && let Some(round) = self
                        .tool_rounds
                        .iter_mut()
                        .find(|round| round.id.is_empty() && round.output.is_none())
                {
                    round.output = Some(result.output.clone());
                    self.executed_tools
                        .push((round.name.clone(), round.arguments.to_string()));
                } else {
                    self.stray_results.push((id, result.output.clone()));
                }
                if !self.batch_generated
                    && !self.tool_rounds.is_empty()
                    && self.tool_rounds.iter().all(|r| r.output.is_some())
                {
                    self.batch_generated = true;
                    let generate = self.build_tool_result_generate();
                    return vec![Event::Generate {
                        meta: *meta,
                        request: generate,
                    }];
                }
                vec![]
            }
            Event::Attention { meta, focus } => {
                let input = focus.payload.content.clone();
                if self.wm_snapshot.slots.is_empty()
                    && let Some(StateResponse::WorkingMemory(snapshot)) = ctx
                        .request_state(StateRequest::WorkingMemory)
                        .await
                    {
                        self.wm_snapshot = snapshot;
                    }
                if focus.payload.source
                    == crate::runtime::types::PerceptionSource::ToolResult
                {
                    return vec![];
                }
                let generate = self.build_generate(&input);
                self.last_messages = generate.messages.clone();
                if focus.payload.source == crate::runtime::types::PerceptionSource::User {
                    self.current_task = input.clone();
                    self.tool_rounds.clear();
                    self.stray_results.clear();
                    self.executed_tools.clear();
                    self.batch_generated = false;
                }
                self.spawn_prediction(*meta, input, ctx.bus());
                vec![Event::Generate {
                    meta: *meta,
                    request: generate,
                }]
            }
            Event::Inhibition { signal, .. } => {
                self.inhibited.extend(signal.targets.iter().cloned());
                vec![]
            }
            Event::Prediction { trajectory, .. } => {
                self.last_trajectory = trajectory.clone();
                vec![]
            }
            Event::CompactContext { summary, .. } => {
                self.session_summary = summary.clone();
                self.wm_snapshot.dialogue.clear();
                self.tool_rounds.clear();
                self.stray_results.clear();
                self.last_messages.clear();
                vec![]
            }
            Event::ContextUpdate { rules, skills, .. } => {
                self.rules = rules.clone();
                self.skills = skills.clone();
                vec![]
            }
            Event::RequestState {
                meta,
                request: StateRequest::Prediction,
                correlation_id,
            } => vec![Event::StateResponse {
                meta: *meta,
                response: StateResponse::Prediction(self.last_trajectory.clone()),
                correlation_id: *correlation_id,
            }],
            Event::WorkingMemoryUpdate { snapshot, .. } => {
                self.wm_snapshot = snapshot.clone();
                vec![]
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
        AttentionFocus, CycleId, InhibitionSignal, PerceptionPayload, PerceptionSource,
    };
    use std::pin::Pin;
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    fn meta() -> EventMeta {
        EventMeta {
            cycle_id: CycleId(1),

        }
    }

    struct TrajectoryPort {
        requests: Arc<Mutex<Vec<GenerateRequest>>>,
    }

    #[async_trait]
    impl LlmPort for TrajectoryPort {
        async fn generate<'a>(
            &'a self,
            request: &'a GenerateRequest,
            _cancel: &'a CancellationToken,
        ) -> Result<
            Pin<Box<dyn futures::Stream<Item = Result<CompletionChunk, AdapterError>> + Send + 'a>>,
            AdapterError,
        > {
            self.requests.lock().unwrap().push(request.clone());
            let chunk = CompletionChunk {
                model: "predictor".into(),
                index: 0,
                delta: ChunkDelta {
                    role: None,
                    content: Some(
                        r#"{"topics":["weather"],"key_elements":["sunny","beijing"],"direction":0.8}"#
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

    struct MemoryProvider;

    #[async_trait]
    impl CognitiveActor for MemoryProvider {
        fn id(&self) -> &str {
            "memory_provider"
        }

        fn subscriptions(&self) -> Vec<EventKind> {
            vec![EventKind::State]
        }

        async fn handle(&mut self, event: &Event, _ctx: &mut ActorContext) -> Vec<Event> {
            match event {
                Event::RequestState {
                    meta,
                    request: StateRequest::MemoryRetrieval { .. },
                    correlation_id,
                } => vec![Event::StateResponse {
                    meta: *meta,
                    response: StateResponse::MemoryRetrieval(
                        crate::runtime::types::MemoryRetrieval {
                            episodic: vec![],
                            semantic: vec![crate::runtime::types::SemanticMemory {
                                id: 1,
                                content: "beijing is sunny in winter".into(),
                                strength: 1.0,
                                belief: 0.9,
                            }],
                        },
                    ),
                    correlation_id: *correlation_id,
                }],
                _ => vec![],
            }
        }
    }

    struct FailingPort;

    #[async_trait]
    impl LlmPort for FailingPort {
        async fn generate<'a>(
            &'a self,
            _request: &'a GenerateRequest,
            _cancel: &'a CancellationToken,
        ) -> Result<
            Pin<Box<dyn futures::Stream<Item = Result<CompletionChunk, AdapterError>> + Send + 'a>>,
            AdapterError,
        > {
            Err(AdapterError::invalid_request("predictor", "boom"))
        }
    }

    fn attention_event(content: &str) -> Event {
        Event::Attention {
            meta: meta(),
            focus: AttentionFocus {
                payload: PerceptionPayload {
                    source: PerceptionSource::User,
                    content: content.into(),
                    salience: 0.8,
                },
                salience: 0.8,
                relevance: 0.9,
            },
        }
    }

    #[tokio::test]
    async fn llm_prediction_publishes_trajectory() {
        let bus = EventBus::new(16);
        let port = TrajectoryPort {
            requests: Arc::new(Mutex::new(Vec::new())),
        };
        let (_h, ready) = spawn_actor(bus.clone(), PredictionActor::new(Arc::new(port)));
        ready.await.unwrap();
        let mut rx = Box::pin(bus.subscribe_kinds(&[EventKind::Prediction]));

        bus.publish(attention_event("what is the weather in beijing?"));
        let prediction = tokio::time::timeout(Duration::from_secs(2), rx.next())
            .await
            .unwrap()
            .unwrap();
        match prediction {
            Event::Prediction { trajectory, .. } => {
                assert!(trajectory.topics.contains(&"weather".to_string()));
                assert!(trajectory.key_elements.contains(&"sunny".to_string()));
                assert_eq!(trajectory.direction, 0.8);
            }
            _ => panic!("expected prediction event"),
        }
    }

    #[tokio::test]
    async fn llm_failure_is_silent_and_generation_still_runs() {
        let bus = EventBus::new(16);
        let (_h, ready) = spawn_actor(bus.clone(), PredictionActor::new(Arc::new(FailingPort)));
        ready.await.unwrap();
        let mut generations = Box::pin(bus.subscribe_kinds(&[EventKind::Generation]));
        let mut predictions = Box::pin(bus.subscribe_kinds(&[EventKind::Prediction]));

        bus.publish(attention_event("please help me with the weather report"));
        let event = tokio::time::timeout(Duration::from_secs(2), generations.next())
            .await
            .unwrap()
            .unwrap();
        assert!(
            matches!(event, Event::Generate { .. }),
            "main generation must start even when prediction fails"
        );
        let stray = tokio::time::timeout(Duration::from_millis(300), predictions.next()).await;
        assert!(stray.is_err(), "no prediction expected when the port fails");
    }

    #[tokio::test]
    async fn inhibited_elements_are_removed_from_prediction() {
        let bus = EventBus::new(16);
        let port = TrajectoryPort {
            requests: Arc::new(Mutex::new(Vec::new())),
        };
        let (_h, ready) = spawn_actor(bus.clone(), PredictionActor::new(Arc::new(port)));
        ready.await.unwrap();
        let mut rx = Box::pin(bus.subscribe_kinds(&[EventKind::Prediction]));

        bus.publish(Event::Inhibition {
            meta: meta(),
            signal: InhibitionSignal {
                targets: vec!["sunny".into()],
                strength: 0.9,
            },
        });
        bus.publish(attention_event("what is the weather in beijing?"));
        let prediction = tokio::time::timeout(Duration::from_secs(2), rx.next())
            .await
            .unwrap()
            .unwrap();
        match prediction {
            Event::Prediction { trajectory, .. } => {
                assert!(!trajectory.key_elements.contains(&"sunny".to_string()));
                assert!(trajectory.key_elements.contains(&"beijing".to_string()));
            }
            _ => panic!("expected prediction event"),
        }
    }

    #[tokio::test]
    async fn semantic_beliefs_are_injected_into_prediction() {
        let bus = EventBus::new(16);
        let requests = Arc::new(Mutex::new(Vec::new()));
        let port = TrajectoryPort {
            requests: requests.clone(),
        };
        let (_mem, mem_ready) = spawn_actor(bus.clone(), MemoryProvider);
        let (_h, ready) = spawn_actor(bus.clone(), PredictionActor::new(Arc::new(port)));
        mem_ready.await.unwrap();
        ready.await.unwrap();
        let mut rx = Box::pin(bus.subscribe_kinds(&[EventKind::Prediction]));

        bus.publish(attention_event("what is the weather in beijing?"));
        let _ = tokio::time::timeout(Duration::from_secs(2), rx.next())
            .await
            .unwrap()
            .unwrap();

        let recorded = requests.lock().unwrap();
        let request = recorded
            .iter()
            .find(|req| {
                req.messages[0]
                    .content
                    .to_plain_text()
                    .contains("predictor for a cognitive agent")
            })
            .expect("prediction request must be recorded");
        let prompt = request.messages[0].content.to_plain_text();
        assert!(
            prompt.contains("Known knowledge"),
            "semantic beliefs should be injected into the prediction prompt"
        );
        assert!(prompt.contains("beijing is sunny in winter"));
    }

    #[tokio::test]
    async fn dialogue_history_is_injected_into_generation() {
        let bus = EventBus::new(32);
        let requests = Arc::new(Mutex::new(Vec::new()));
        let port: Arc<dyn LlmPort> = Arc::new(TrajectoryPort {
            requests: requests.clone(),
        });
        let (_wm, wm_ready) = spawn_actor(
            bus.clone(),
            crate::runtime::working_memory::WorkingMemoryActor::new(),
        );
        let (_llm, llm_ready) = spawn_actor(bus.clone(), crate::runtime::llm_actor::LlmActor::new(port.clone()));
        let (_h, ready) = spawn_actor(bus.clone(), PredictionActor::new(port));
        wm_ready.await.unwrap();
        llm_ready.await.unwrap();
        ready.await.unwrap();
        let mut rx = Box::pin(bus.subscribe_kinds(&[EventKind::Prediction]));

        let meta1 = EventMeta {
            cycle_id: CycleId(1),

        };
        bus.publish(Event::Attention {
            meta: meta1,
            focus: AttentionFocus {
                payload: PerceptionPayload {
                    source: PerceptionSource::User,
                    content: "what is the weather?".into(),
                    salience: 0.8,
                },
                salience: 0.8,
                relevance: 0.8,
            },
        });
        bus.publish(Event::Rpe {
            meta: meta1,
            rpe: crate::runtime::types::RpeSignal(0.6),
        });
        let _ = tokio::time::timeout(Duration::from_secs(2), rx.next())
            .await
            .unwrap()
            .unwrap();
        bus.publish(Event::CycleComplete {
            meta: meta1,
            summary: crate::runtime::types::CycleSummary {
                decision: Some(crate::runtime::types::ActionDecision {
                    candidate: crate::runtime::types::ActionCandidate::Respond {
                        content: "the weather is sunny".into(),
                    },
                    confidence: 1.0,
                    go: true,
                }),
                ..Default::default()
            },
        });
        tokio::time::sleep(Duration::from_millis(100)).await;

        let meta2 = EventMeta {
            cycle_id: CycleId(2),

        };
        bus.publish(Event::Attention {
            meta: meta2,
            focus: AttentionFocus {
                payload: PerceptionPayload {
                    source: PerceptionSource::User,
                    content: "and tomorrow?".into(),
                    salience: 0.8,
                },
                salience: 0.8,
                relevance: 0.8,
            },
        });
        let _ = tokio::time::timeout(Duration::from_secs(2), rx.next())
            .await
            .unwrap()
            .unwrap();
        tokio::time::sleep(Duration::from_millis(100)).await;

        let recorded = requests.lock().unwrap();
        let generation = recorded
            .iter()
            .rev()
            .find(|req| {
                !req.messages[0]
                    .content
                    .to_plain_text()
                    .contains("predictor for a cognitive agent")
            })
            .expect("a main generation request should exist");
        let joined = generation
            .messages
            .iter()
            .map(|m| m.content.to_plain_text())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(
            joined.contains("what is the weather?"),
            "earlier user turn should be in the generation context"
        );
        assert!(
            joined.contains("the weather is sunny"),
            "earlier assistant answer should be in the generation context"
        );
        assert!(joined.contains("and tomorrow?"));
    }

    #[test]
    fn trajectory_json_parsing() {
        let parsed = parse_trajectory(r#"Here you go: {"topics":["a"],"key_elements":["b"],"direction":0.5}"#);
        assert!(parsed.is_some());
        let trajectory = parsed.unwrap();
        assert_eq!(trajectory.topics, vec!["a"]);
        assert_eq!(trajectory.direction, 0.5);
    }

    fn rule(name: &str, rule: &str, description: &str, globs: &str, regex: &str, always_apply: Option<bool>) -> RuleContext {
        RuleContext {
            name: name.into(),
            rule: rule.into(),
            description: description.into(),
            globs: globs.into(),
            regex: regex.into(),
            always_apply,
        }
    }

    #[test]
    fn active_rules_classifies_by_type() {
        let actor = PredictionActor::new(Arc::new(FailingPort));
        let rules = vec![
            rule("always", "always rule", "", "", "", Some(true)),
            rule("bare", "bare rule", "", "", "", None),
            rule("auto-hit", "auto rule", "", "", "useEffect", None),
            rule("auto-miss", "miss rule", "", "", "never_matches", None),
            rule("agent", "agent rule", "for react files", "", "", None),
            rule("manual", "manual rule", "", "", "", Some(false)),
        ];
        let actor = PredictionActor {
            rules,
            ..actor
        };
        let active = actor.active_rules("import { useEffect } from 'react'");
        let names: Vec<&str> = active.iter().map(|r| r.name.as_str()).collect();
        assert!(names.contains(&"always"), "{names:?}");
        assert!(names.contains(&"bare"), "{names:?}");
        assert!(names.contains(&"auto-hit"), "{names:?}");
        assert!(!names.contains(&"auto-miss"), "{names:?}");
        assert!(!names.contains(&"agent"), "{names:?}");
        assert!(!names.contains(&"manual"), "{names:?}");
        let miss_active = actor.active_rules("no match here");
        let miss_names: Vec<&str> = miss_active.iter().map(|r| r.name.as_str()).collect();
        assert!(!miss_names.contains(&"auto-hit"), "{miss_names:?}");
    }

    #[tokio::test]
    async fn rules_and_skills_injected_into_generation() {
        let bus = EventBus::new(32);
        let requests = Arc::new(Mutex::new(Vec::new()));
        let port: Arc<dyn LlmPort> = Arc::new(TrajectoryPort {
            requests: requests.clone(),
        });
        let (_llm, llm_ready) =
            spawn_actor(bus.clone(), crate::runtime::llm_actor::LlmActor::new(port.clone()));
        let (_h, ready) = spawn_actor(bus.clone(), PredictionActor::new(port));
        llm_ready.await.unwrap();
        ready.await.unwrap();
        let mut rx = Box::pin(bus.subscribe_kinds(&[EventKind::Prediction]));

        bus.publish(Event::ContextUpdate {
            meta: meta(),
            rules: vec![rule("always", "Always use PropTypes", "", "", "", Some(true))],
            skills: vec![SkillContext {
                name: "refactor".into(),
                description: "Refactor with confidence".into(),
            }],
        });
        bus.publish(attention_event("fix my component"));
        let _ = tokio::time::timeout(Duration::from_secs(2), rx.next())
            .await
            .unwrap()
            .unwrap();
        tokio::time::sleep(Duration::from_millis(100)).await;

        let recorded = requests.lock().unwrap();
        let generation = recorded
            .iter()
            .rev()
            .find(|req| {
                !req.messages[0]
                    .content
                    .to_plain_text()
                    .contains("predictor for a cognitive agent")
            })
            .expect("a main generation request should exist");
        let system = generation.messages[0].content.to_plain_text();
        assert!(system.contains("Always use PropTypes"), "{system}");
        assert!(system.contains("# Available skills"), "{system}");
        assert_eq!(
            generation.modulation.reasoning_effort,
            Some(ReasoningEffort::High),
            "main generation defaults to explicit High effort"
        );
        assert!(system.contains("refactor: Refactor with confidence"), "{system}");
    }

    #[tokio::test]
    async fn auto_attach_rule_matches_tool_result_only() {
        let bus = EventBus::new(32);
        let requests = Arc::new(Mutex::new(Vec::new()));
        let port: Arc<dyn LlmPort> = Arc::new(TrajectoryPort {
            requests: requests.clone(),
        });
        let (_llm, llm_ready) =
            spawn_actor(bus.clone(), crate::runtime::llm_actor::LlmActor::new(port.clone()));
        let (_h, ready) = spawn_actor(bus.clone(), PredictionActor::new(port));
        llm_ready.await.unwrap();
        ready.await.unwrap();
        let mut rx = Box::pin(bus.subscribe_kinds(&[EventKind::Prediction]));

        bus.publish(Event::ContextUpdate {
            meta: meta(),
            rules: vec![rule(
                "hooks",
                "Use named exports for hooks",
                "",
                "",
                "useEffect",
                None,
            )],
            skills: vec![],
        });
        bus.publish(Event::WorkingMemoryUpdate {
            meta: meta(),
            snapshot: WorkingMemorySnapshot {
                slots: vec![crate::runtime::types::WorkingMemorySlot {
                    id: 1,
                    content: "(Tool read_file result)\nimport { useEffect } from 'react'".into(),
                    source: "ToolResult".into(),
                    activation: 1.0,
                }],
                dialogue: vec![],
            },
        });
        bus.publish(attention_event("the user typed useEffect in chat"));
        let _ = tokio::time::timeout(Duration::from_secs(2), rx.next())
            .await
            .unwrap()
            .unwrap();
        tokio::time::sleep(Duration::from_millis(100)).await;

        let recorded = requests.lock().unwrap();
        let generation = recorded
            .iter()
            .rev()
            .find(|req| {
                !req.messages[0]
                    .content
                    .to_plain_text()
                    .contains("predictor for a cognitive agent")
            })
            .expect("a main generation request should exist");
        let system = generation.messages[0].content.to_plain_text();
        assert!(
            system.contains("Use named exports for hooks"),
            "rule must activate from tool result content: {system}"
        );
    }

    #[test]
    fn multiple_tool_results_pair_in_generation_order() {
        let mut actor = PredictionActor::new(Arc::new(FailingPort));
        actor.last_messages = vec![
            Message::system("system"),
            Message::user("user input"),
        ];
        actor.tool_rounds = vec![
            ToolRound {
                id: "call_a".into(),
                name: "ls".into(),
                arguments: serde_json::json!({}),
                reasoning: None,
                content: "listing dir".into(),
                output: Some("out_a".into()),
            },
            ToolRound {
                id: "call_b".into(),
                name: "read_file".into(),
                arguments: serde_json::json!({}),
                reasoning: None,
                content: "reading file".into(),
                output: Some("out_b".into()),
            },
        ];

        let request = actor.build_tool_result_generate();
        let mut tool_calls = 0;
        let mut tool_msgs = 0;
        for message in &request.messages {
            match message.role {
                crate::adapter::types::Role::Assistant => {
                    if message.tool_calls.is_some() {
                        tool_calls += 1;
                    }
                }
                crate::adapter::types::Role::Tool => tool_msgs += 1,
                _ => {}
            }
        }
        assert_eq!(tool_calls, 2, "both tool calls must be paired");
        assert_eq!(tool_msgs, 2, "both tool results must be attached");
        assert_eq!(
            request.messages[3].content.to_plain_text(),
            "out_a",
            "first tool result must follow its call"
        );
        assert_eq!(
            request.messages[5].content.to_plain_text(),
            "out_b",
            "second tool result must follow its call"
        );
    }

    #[test]
    fn pending_round_is_marked_as_still_running() {
        let mut actor = PredictionActor::new(Arc::new(FailingPort));
        actor.last_messages = vec![Message::system("system")];
        actor.tool_rounds = vec![ToolRound {
            id: "call_p".into(),
            name: "schedule_task".into(),
            arguments: serde_json::json!({}),
            reasoning: None,
            content: "scheduling".into(),
            output: None,
        }];

        let request = actor.build_tool_result_generate();
        assert_eq!(request.messages.len(), 2);
        assert!(
            request.messages[1]
                .content
                .to_plain_text()
                .contains("is still executing and its result has not arrived yet"),
            "pending tool must be described as still running"
        );
    }

    #[test]
    fn current_task_is_injected_into_non_user_generation() {
        let mut actor = PredictionActor::new(Arc::new(FailingPort));
        actor.current_task = "test all tools".into();
        let request = actor.build_generate("(Tool result)");
        let injected = request
            .messages
            .iter()
            .any(|message| {
                message
                    .content
                    .to_plain_text()
                    .contains("(User task — the goal you were asked to accomplish")
                    && message.content.to_plain_text().contains("test all tools")
            });
        assert!(injected, "user task must be injected for non-user input");
    }

    #[test]
    fn current_task_is_not_duplicated_for_user_input() {
        let mut actor = PredictionActor::new(Arc::new(FailingPort));
        actor.current_task = "test all tools".into();
        let request = actor.build_generate("test all tools");
        let duplicated = request
            .messages
            .iter()
            .filter(|message| message.content.to_plain_text().contains("(User task)"))
            .count();
        assert_eq!(duplicated, 0, "user input equal to task must not be re-injected");
    }

    #[tokio::test]
    async fn tool_round_carries_stream_content_and_no_stale_pending() {
        use crate::adapter::types::ChunkDelta;
        use crate::runtime::actor::ActorContext;

        let bus = EventBus::new(64);
        let mut actor = PredictionActor::new(Arc::new(FailingPort));
        let mut ctx = ActorContext::new(bus.clone(), meta());
        actor.last_messages = vec![Message::system("system"), Message::user("test tools")];

        let chunk_event = |cycle: u64, text: &str| Event::Chunk {
            meta: crate::runtime::event::EventMeta {
                cycle_id: CycleId(cycle),

            },
            chunk: CompletionChunk {
                model: "fake".into(),
                index: 0,
                delta: ChunkDelta {
                    role: None,
                    content: Some(text.into()),
                    tool_calls: vec![],
                    logprobs: None,
                    reasoning: None,
                },
                finish_reason: None,
                usage: None,
                request_id: None,
            },
        };
        let call_action = |name: &str, id: &str| Event::ActionSelected {
            meta: meta(),
            decision: crate::runtime::types::ActionDecision {
                candidate: crate::runtime::types::ActionCandidate::CallTool {
                    name: name.into(),
                    arguments: serde_json::json!({}),
                    tool_call_id: Some(id.into()),
                    reasoning: None,
                },
                confidence: 0.9,
                go: true,
            },
        };
        let tool_result_event = |id: &str, output: &str| Event::ToolResult {
            meta: meta(),
            result: crate::runtime::types::ToolResult {
                name: "x".into(),
                output: output.into(),
                tool_call_id: Some(id.into()),
            },
            verdict: None,
        };

        actor.handle(&chunk_event(1, "listing dir"), &mut ctx).await;
        actor.handle(&call_action("ls", "call_1"), &mut ctx).await;
        actor.handle(&chunk_event(2, "reading file"), &mut ctx).await;
        actor.handle(&call_action("read_file", "call_2"), &mut ctx).await;
        actor.handle(&tool_result_event("call_1", "out_a"), &mut ctx).await;
        let generate = actor.build_tool_result_generate();

        let assistant_calls: Vec<&Message> = generate
            .messages
            .iter()
            .filter(|m| m.role == crate::adapter::types::Role::Assistant && m.tool_calls.is_some())
            .collect();
        assert_eq!(assistant_calls.len(), 1, "only the finished round pairs a tool call");
        assert_eq!(
            assistant_calls[0].content.to_plain_text(),
            "listing dir",
            "round 1 assistant text must be carried into the tool round"
        );
        let pending_count = generate
            .messages
            .iter()
            .filter(|m| m.content.to_plain_text().contains("is still executing and its result has not arrived yet"))
            .count();
        assert_eq!(pending_count, 1, "only the unfinished round may show still running");
        assert!(
            generate
                .messages
                .iter()
                .any(|m| m.content.to_plain_text().contains("read_file")),
            "pending round must name the tool call"
        );
        assert!(
            generate.messages.iter().any(|m| m.content.to_plain_text().contains("out_a")),
            "finished round result must be attached"
        );

        actor.handle(&tool_result_event("call_2", "out_b"), &mut ctx).await;
        let generate = actor.build_tool_result_generate();
        let pending_count = generate
            .messages
            .iter()
            .filter(|m| m.content.to_plain_text().contains("still running"))
            .count();
        assert_eq!(pending_count, 0, "no stale pending after all results arrive");
        let text: String = generate
            .messages
            .iter()
            .map(|m| m.content.to_plain_text())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(text.contains("out_a") && text.contains("out_b"), "{text}");
    }


    #[test]
    fn system_prompt_forbids_repeating_own_text() {
        let actor = PredictionActor::new(Arc::new(FailingPort));
        let request = actor.build_generate("test all tools");
        let system = request.messages[0].content.to_plain_text();
        assert!(
            system.contains("Never repeat text you already wrote in this conversation"),
            "anti-repeat rule must be in the system prompt"
        );
    }

    #[tokio::test]
    async fn waits_for_all_tool_results_before_generating() {
        use crate::runtime::actor::ActorContext;

        let bus = EventBus::new(64);
        let mut actor = PredictionActor::new(Arc::new(FailingPort));
        let mut ctx = ActorContext::new(bus.clone(), meta());
        actor.last_messages = vec![Message::system("system"), Message::user("task")];
        actor.tool_rounds = vec![
            ToolRound {
                id: "call_a".into(),
                name: "ls".into(),
                arguments: serde_json::json!({}),
                reasoning: None,
                content: "listing".into(),
                output: None,
            },
            ToolRound {
                id: "call_b".into(),
                name: "read_file".into(),
                arguments: serde_json::json!({}),
                reasoning: None,
                content: "reading".into(),
                output: None,
            },
        ];

        let tool_result_event = |id: &str, output: &str| Event::ToolResult {
            meta: meta(),
            result: crate::runtime::types::ToolResult {
                name: "x".into(),
                output: output.into(),
                tool_call_id: Some(id.into()),
            },
            verdict: None,
        };

        let emitted = actor.handle(&tool_result_event("call_a", "out_a"), &mut ctx).await;
        assert!(
            emitted.is_empty(),
            "must not generate while a batch tool is still pending"
        );

        let emitted = actor.handle(&tool_result_event("call_b", "out_b"), &mut ctx).await;
        assert_eq!(emitted.len(), 1, "must generate once all results arrive");
        assert!(matches!(emitted[0], Event::Generate { .. }));
        let Event::Generate { request, .. } = &emitted[0] else {
            unreachable!()
        };
        let text: String = request
            .messages
            .iter()
            .map(|m| m.content.to_plain_text())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(text.contains("out_a") && text.contains("out_b"), "{text}");
    }
    #[test]
    fn meta_threshold_injects_semantic_reminder_not_numbers() {
        let mut actor = PredictionActor::new(Arc::new(FailingPort));
        actor.meta_state = Some(crate::runtime::types::MetaState {
            uncertainty: 0.85,
            conflict: 0.7,
            confidence: 0.2,
        });
        let request = actor.build_generate("test");
        let text: String = request
            .messages
            .iter()
            .map(|m| m.content.to_plain_text())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(text.contains("high uncertainty"), "{text}");
        assert!(text.contains("conflicting information"), "{text}");
        assert!(!text.contains("0.85"), "raw numbers must not be injected: {text}");
        assert!(!text.contains("0.7"), "raw numbers must not be injected: {text}");
    }

    #[test]
    fn meta_below_threshold_injects_nothing() {
        let actor = PredictionActor::new(Arc::new(FailingPort));
        let request = actor.build_generate("test");
        let text: String = request
            .messages
            .iter()
            .map(|m| m.content.to_plain_text())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(!text.contains("Cognitive state"), "{text}");
    }
    #[test]
    fn working_memory_injected_as_labeled_message_with_problem_reminder() {
        let mut actor = PredictionActor::new(Arc::new(FailingPort));
        actor.wm_snapshot = WorkingMemorySnapshot {
            slots: vec![
                crate::runtime::types::WorkingMemorySlot {
                    id: 1,
                    content: "(Tool read_file result)\nfn main() {}".into(),
                    source: "ToolResult".into(),
                    activation: 1.0,
                },
                crate::runtime::types::WorkingMemorySlot {
                    id: 2,
                    content: "(Tool grep result)\ncommand failed: pattern error".into(),
                    source: "ToolResult".into(),
                    activation: 0.9,
                },
            ],
            dialogue: vec![],
        };
        let request = actor.build_generate("test");
        let text: String = request
            .messages
            .iter()
            .map(|m| m.content.to_plain_text())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(
            text.contains("Working memory — salient events your cognitive system flagged"),
            "slots must be injected with the labeled header: {text}"
        );
        assert!(text.contains("fn main() {}"), "slot content must be kept: {text}");
        assert!(
            text.contains("a flagged event indicates a problem"),
            "problem reminder must be injected: {text}"
        );
    }
}

