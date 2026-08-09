use std::error::Error;
use std::io::{self, Write};
use std::sync::Arc;
use std::time::Duration;

use crossterm::event::{
    Event as TermEvent, KeyCode, KeyEvent, KeyModifiers, KeyboardEnhancementFlags,
    PopKeyboardEnhancementFlags, PushKeyboardEnhancementFlags,
};
use ratatui::backend::CrosstermBackend;
use ratatui::Terminal;
use tokio::sync::Mutex;

use crate::app::App;
use crate::frontend::commands::{self, Action};
use crate::frontend::messages;
use crate::frontend::render::{self, GitInfo};
use crate::frontend::state::{Mode, SelectorKind, SetupState, UiMessage, UiState};
use crate::runtime::event::Event;
use crate::runtime::types::ActionCandidate;

const TICK_MS: u64 = 100;

fn supports_kitty_keyboard() -> bool {
    let _ = io::stdout().write_all(b"\x1b[?u");
    let _ = io::stdout().flush();
    let deadline = std::time::Instant::now() + Duration::from_millis(120);
    while std::time::Instant::now() < deadline {
        if let Ok(true) = crossterm::event::poll(Duration::from_millis(15)) {
            let _ = crossterm::event::read();
            return true;
        }
    }
    false
}

pub async fn run(app: App) -> Result<(), Box<dyn Error>> {
    std::panic::set_hook(Box::new(|info| {
        let _ = crossterm::execute!(
            io::stdout(),
            crossterm::terminal::LeaveAlternateScreen
        );
        let _ = crossterm::terminal::disable_raw_mode();
        eprintln!("prognosis panicked: {info}");
        eprintln!("(terminal restored; please report this crash)");
    }));
    let git = git_info(&app);
    let mut ui = UiState::new();
    if app.needs_setup() {
        ui.mode = Mode::Setup;
        ui.setup = Some(SetupState {
            fields: vec![
                ("name".to_string(), "deepseek-v4-flash".to_string()),
                ("provider".to_string(), "deepseek".to_string()),
                ("base_url".to_string(), "https://api.deepseek.com".to_string()),
                ("api_key".to_string(), String::new()),
            ],
            active: 0,
            cursor: 0,
            error: None,
        });
    }
    let app = Arc::new(Mutex::new(app));

    let mut terminal = Terminal::new(CrosstermBackend::new(io::stdout()))?;
    crossterm::terminal::enable_raw_mode()?;
    let _ = crossterm::execute!(
        io::stdout(),
        crossterm::terminal::EnterAlternateScreen,
        crossterm::cursor::SetCursorStyle::SteadyBlock
    );
    let _ = terminal.clear();
    let _ = io::stdout().flush();
    let _ = crossterm::execute!(
        io::stdout(),
        PushKeyboardEnhancementFlags(KeyboardEnhancementFlags::DISAMBIGUATE_ESCAPE_CODES)
    );
    ui.kitty_supported = supports_kitty_keyboard();
    let result = match run_loop(&mut terminal, app.clone(), &mut ui, &git).await {
        Err(e) if e.to_string() == "quit" => Ok(()),
        other => other,
    };
    tokio::time::sleep(Duration::from_millis(200)).await;
    let _ = io::stdout().write_all(b"\x1b]0;\x07");
    let _ = io::stdout().flush();
    let _ = crossterm::execute!(
        io::stdout(),
        PopKeyboardEnhancementFlags,
        crossterm::cursor::SetCursorStyle::DefaultUserShape
    );
    let _ = crossterm::execute!(io::stdout(), crossterm::terminal::LeaveAlternateScreen);
    crossterm::terminal::disable_raw_mode()?;
    let _ = io::stdout().flush();
    result
}

fn git_info(app: &App) -> GitInfo {
    let project_name = app
        .project_dir()
        .canonicalize()
        .ok()
        .and_then(|path| path.file_name().map(|name| name.to_string_lossy().to_string()))
        .filter(|name| !name.is_empty())
        .unwrap_or_else(|| ".".to_string());
    let branch = std::process::Command::new("git")
        .args(["branch", "--show-current"])
        .current_dir(app.project_dir())
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_string())
        .filter(|name| !name.is_empty());
    GitInfo {
        project_name,
        branch,
    }
}

fn format_title(display: &str, model: &str) -> String {
    if model.is_empty() {
        format!("✦ {display}")
    } else {
        format!("✦ {display} · {model}")
    }
}

async fn run_loop(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    app: Arc<Mutex<App>>,
    ui: &mut UiState,
    git: &GitInfo,
) -> Result<(), Box<dyn Error>> {
    let (key_tx, mut key_rx) = tokio::sync::mpsc::unbounded_channel();
    std::thread::spawn(move || {
        while let Ok(event) = crossterm::event::read() {
            if key_tx.send(event).is_err() {
                break;
            }
        }
    });
    let (approve_tx, mut approve_rx) = tokio::sync::mpsc::unbounded_channel::<()>();
    let (hitokoto_tx, mut hitokoto_rx) =
        tokio::sync::mpsc::unbounded_channel::<Option<crate::frontend::hitokoto::Hitokoto>>();
    let spawned = Arc::new(std::sync::Mutex::new(Vec::<
        tokio::task::JoinHandle<()>,
    >::new()));
    spawned.lock().unwrap().push(tokio::spawn(async move {
        let _ = hitokoto_tx.send(crate::frontend::hitokoto::fetch_hitokoto().await);
    }));
    let mut bus = app.lock().await.runtime.bus().subscribe();
    let mut ticker = tokio::time::interval(Duration::from_millis(TICK_MS));
    ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    let mut redraw = true;
    let mut cached_ctx: Option<render::RenderCtx> = None;
    let mut last_title: Option<String> = None;

    let result: Result<(), Box<dyn Error>> = async {
        loop {
            if redraw {
                let ctx = match app.try_lock() {
                    Ok(guard) => {
                        let ctx = render::RenderCtx::capture(&guard).await;
                        cached_ctx = Some(ctx.clone());
                        Some(ctx)
                    }
                    Err(_) => cached_ctx.clone(),
                };
                if let Some(ctx) = ctx {
                    terminal.draw(|frame| render::draw(frame, &ctx, ui, git))?;
                    let title = format_title(&git.display(), &ctx.model);
                    if last_title.as_deref() != Some(title.as_str()) {
                        let _ = io::stdout()
                            .write_all(format!("\x1b]0;{title}\x07").as_bytes());
                        let _ = io::stdout().flush();
                        last_title = Some(title);
                    }
                }
            }
            tokio::select! {
                _ = ticker.tick() => {
                    if let Ok(mut app) = app.try_lock() {
                        app.scheduler_tick();
                    }
                    ui.tick_spinner();
                    redraw = ui.is_generating() || ui.has_running_tool();
                }
                event = bus.recv() => {
                    match event {
                        Ok(event) => {
                            if matches!(event, Event::StreamEnd { .. })
                                && let Some(text) = assistant_text(ui)
                                && let Ok(mut app) = app.try_lock()
                            {
                                app.note_assistant_turn(&text);
                            }
                            let tool_call = match messages::handle_event(ui, event) {
                                    Some(Event::ActionSelected { decision, .. }) => {
                                        match &decision.candidate {
                                            ActionCandidate::CallTool {
                                                name,
                                                arguments,
                                                tool_call_id,
                                                ..
                                            } => Some((
                                                name.clone(),
                                                arguments.clone(),
                                                tool_call_id.clone(),
                                            )),
                                            _ => None,
                                        }
                                    }
                                    _ => None,
                                };
                                if let Some((name, arguments, tool_call_id)) = tool_call {
                                    ui.pending_tool_calls
                                        .push_back((name, arguments, tool_call_id));
                            } else if !ui.is_generating() {
                                flush_pending(&app, ui).await;
                            }
                        }
                        Err(_) => break,
                    }
                    redraw = true;
                }
                _ = approve_rx.recv() => {
                    ui.mode = Mode::Approve;
                    redraw = true;
                }
                hitokoto = hitokoto_rx.recv(), if ui.hitokoto.is_none() => {
                    ui.hitokoto = Some(match hitokoto {
                        Some(fetched) => {
                            fetched.unwrap_or_else(crate::frontend::hitokoto::fallback_hitokoto)
                        }
                        None => crate::frontend::hitokoto::fallback_hitokoto(),
                    });
                    redraw = true;
                }
                _ = drain_pending_tools(&app, ui, &approve_tx, &spawned),
                    if ui.mode == Mode::Chat
                        && !ui.is_generating()
                        && !ui.pending_tool_calls.is_empty() =>
                {
                    redraw = true;
                }
                event = key_rx.recv() => {
                    match event {
                        Some(event) => handle_key(&app, ui, event, &spawned).await?,
                        None => break,
                    }
                    redraw = true;
                }
            }
        }
        Ok(())
    }
    .await;
    for handle in spawned.lock().unwrap().iter() {
        handle.abort();
    }
    result
}

fn assistant_text(ui: &UiState) -> Option<String> {
    ui.messages
        .iter()
        .rev()
        .find_map(|message| match message {
            crate::frontend::state::UiMessage::Assistant { content, .. }
                if !content.is_empty() =>
            {
                Some(content.clone())
            }
            _ => None,
        })
}

async fn cancel_running_tools(app: &Arc<Mutex<App>>, ui: &mut UiState) {
    let calls: Vec<(String, String)> = ui
        .messages
        .iter()
        .filter_map(|message| {
            if let UiMessage::ToolCall(call) = message
                && call.status == crate::frontend::state::ToolStatus::Running
            {
                Some((call.tool_call_id.clone(), call.name.clone()))
            } else {
                None
            }
        })
        .collect();
    if calls.is_empty() {
        return;
    }
    {
        let mut app = app.lock().await;
        app.cancel_tools(&calls);
    }
    for (id, _) in &calls {
        ui.mark_tool_finished(
            id,
            "(cancelled)",
            "(tool call cancelled by user)",
            crate::frontend::state::ToolStatus::Denied,
            Some("cancelled".to_string()),
        );
    }
    ui.pending_tool_calls.clear();
    ui.push_system("(tool calls cancelled)");
}

async fn drain_pending_tools(
    app: &Arc<Mutex<App>>,
    ui: &mut UiState,
    approve_tx: &tokio::sync::mpsc::UnboundedSender<()>,
    spawned: &Arc<std::sync::Mutex<Vec<tokio::task::JoinHandle<()>>>>,
) {
    if ui.mode != Mode::Chat || ui.is_generating() {
        return;
    }
    let Some((name, arguments, tool_call_id)) = ui.pending_tool_calls.pop_front() else {
        return;
    };
    let app = app.clone();
    let approve_tx = approve_tx.clone();
    spawned.lock().unwrap().push(tokio::spawn(async move {
        let check = {
            let mut app = app.lock().await;
            app.check_tool_call(name.clone(), arguments.clone(), tool_call_id.clone())
                .await
        };
        let verdict = match &check {
            crate::app::ToolCheck::Blocked => None,
            crate::app::ToolCheck::Ready => Some(crate::app::Verdict::Allow),
            crate::app::ToolCheck::Judge {
                input,
                trace,
                pending,
                tools,
            } => {
                let supervisor = app.lock().await.supervisor.clone();
                Some(supervisor.judge(input, trace, pending, tools).await)
            }
        };
        let plan = match (check, verdict) {
            (crate::app::ToolCheck::Blocked, _) => crate::app::ToolPlan::Blocked,
            (_, Some(verdict)) => {
                let mut app = app.lock().await;
                app.apply_verdict(verdict, name, arguments, tool_call_id)
                    .await
            }
            _ => crate::app::ToolPlan::Blocked,
        };
        match plan {
            crate::app::ToolPlan::Blocked => {}
            crate::app::ToolPlan::NeedConfirm { .. } => {
                let _ = approve_tx.send(());
            }
            crate::app::ToolPlan::Execute(calls) => {
                let mut handles = Vec::new();
                for call in calls {
                    let app = app.clone();
                    handles.push(tokio::spawn(async move {
                        let output =
                            crate::app::run_tool_handler(call.handler, call.name.clone(), call.arguments.clone())
                                .await;
                        app.lock()
                            .await
                            .finish_tool_call(
                                call.name,
                                call.arguments,
                                call.tool_call_id,
                                output,
                                &call.verdict,
                            )
                            .await;
                    }));
                }
                for handle in handles {
                    let _ = handle.await;
                }
            }
        }
    }));
}

async fn flush_pending(app: &Arc<Mutex<App>>, ui: &mut UiState) {
    if let Some(next) = ui.input.pending_submit.first().cloned()
        && let Ok(mut app) = app.try_lock()
    {
        ui.input.pending_submit.remove(0);
        app.submit(&next);
    }
}

async fn handle_key(
    app: &Arc<Mutex<App>>,
    ui: &mut UiState,
    event: TermEvent,
    spawned: &Arc<std::sync::Mutex<Vec<tokio::task::JoinHandle<()>>>>,
) -> Result<(), Box<dyn Error>> {
    let TermEvent::Key(key) = event else {
        return Ok(());
    };
    if key.code == KeyCode::Char('d') && key.modifiers.contains(KeyModifiers::CONTROL) {
        return Err("quit".into());
    }
    if key.code == KeyCode::Char('c') && key.modifiers.contains(KeyModifiers::CONTROL) {
        if ui.is_generating() {
            ui.cancelled = true;
            ui.pending_tool_calls.clear();
            let app = app.clone();
            spawned.lock().unwrap().push(tokio::spawn(async move {
                app.lock().await.cancel_generation();
            }));
            ui.finish_stream();
            ui.push_system("(generation cancelled)");
            return Ok(());
        }
        if ui.has_running_tool() {
            cancel_running_tools(app, ui).await;
            return Ok(());
        }
        return Err("quit".into());
    }
    match ui.mode {
        Mode::Approve => handle_approve_key(app, ui, key, spawned).await,
        Mode::Selector => handle_selector_key(app, ui, key).await,
        Mode::Status | Mode::Help => {
            match key.code {
                KeyCode::Esc => {
                    ui.mode = Mode::Chat;
                    ui.input.buffer.clear();
                    ui.input.cursor = 0;
                    ui.panel_scroll = 0;
                }
                KeyCode::Up => {
                    ui.panel_scroll = ui.panel_scroll.saturating_sub(1);
                }
                KeyCode::Down => {
                    ui.panel_scroll = ui.panel_scroll.saturating_add(1);
                }
                KeyCode::PageUp => {
                    ui.panel_scroll = ui.panel_scroll.saturating_add(10);
                }
                KeyCode::PageDown => {
                    ui.panel_scroll = ui.panel_scroll.saturating_sub(10);
                }
                _ => {}
            }
            Ok(())
        }
        Mode::Setup => handle_setup_key(app, ui, key).await,
        Mode::Chat | Mode::Command => handle_chat_key(app, ui, key, spawned).await,
    }
}

async fn handle_approve_key(
    app: &Arc<Mutex<App>>,
    ui: &mut UiState,
    key: KeyEvent,
    spawned: &Arc<std::sync::Mutex<Vec<tokio::task::JoinHandle<()>>>>,
) -> Result<(), Box<dyn Error>> {
    let approved = match key.code {
        KeyCode::Enter if key.modifiers.contains(KeyModifiers::SHIFT) => {
            let mut app = app.lock().await;
            if let Some(name) = app.pending_tool_name() {
                app.remember_approval(&name);
            }
            true
        }
        KeyCode::BackTab => {
            let mut app = app.lock().await;
            if let Some(name) = app.pending_tool_name() {
                app.remember_approval(&name);
            }
            true
        }
        KeyCode::Char('y') => {
            let mut app = app.lock().await;
            if let Some(name) = app.pending_tool_name() {
                app.remember_approval(&name);
            }
            true
        }
        KeyCode::Enter => true,
        KeyCode::Esc => false,
        _ => return Ok(()),
    };
    let plan = {
        let mut app = app.lock().await;
        app.take_pending_execution(approved)
    };
    if let Some(call) = plan {
        let app = app.clone();
        spawned.lock().unwrap().push(tokio::spawn(async move {
            let output = crate::app::run_tool_handler(
                call.handler,
                call.name.clone(),
                call.arguments.clone(),
            )
            .await;
            app.lock()
                .await
                .finish_tool_call(
                    call.name,
                    call.arguments,
                    call.tool_call_id,
                    output,
                    &call.verdict,
                )
                .await;
        }));
    }
    ui.mode = Mode::Chat;
    Ok(())
}

async fn handle_selector_key(
    app: &Arc<Mutex<App>>,
    ui: &mut UiState,
    key: KeyEvent,
) -> Result<(), Box<dyn Error>> {
    let Some(kind) = ui.selector.as_ref().map(|s| s.kind) else {
        ui.mode = Mode::Chat;
        return Ok(());
    };
    match key.code {
        KeyCode::Up => {
            if let Some(selector) = &mut ui.selector {
                selector.move_up();
            }
        }
        KeyCode::Down => {
            if let Some(selector) = &mut ui.selector {
                selector.move_down();
            }
        }
        KeyCode::Tab => {
            if let Some(kind) = ui.selector.as_ref().map(|s| s.kind)
                && matches!(kind, SelectorKind::Rules | SelectorKind::Skills)
                && let Some(selected) = ui
                    .selector
                    .as_ref()
                    .and_then(|selector| selector.current())
                    .map(selector_item_name)
                    .map(|item| {
                        item.split(" — ").next().unwrap_or("").to_string()
                    })
            {
                let mut app = app.lock().await;
                let now_enabled = match kind {
                    SelectorKind::Rules => app.toggle_rule(&selected),
                    SelectorKind::Skills => app.toggle_skill(&selected),
                    _ => false,
                };
                ui.push_system(&format!(
                    "{} {}",
                    if now_enabled { "enabled" } else { "disabled" },
                    selected
                ));
                ui.selector = Some(match kind {
                    SelectorKind::Rules => crate::frontend::select::rules_selector(&app),
                    SelectorKind::Skills => crate::frontend::select::skills_selector(&app),
                    _ => unreachable!(),
                });
            }
        }
        KeyCode::Enter => {
            let selected = ui
                .selector
                .as_ref()
                .and_then(|selector| selector.current())
                .map(str::to_string);
            ui.selector = None;
            if let Some(selected) = selected {
                select_enter(app, ui, kind, &selected).await;
            } else {
                ui.mode = Mode::Chat;
            }
        }
        KeyCode::Esc => {
            ui.selector = None;
            ui.mode = Mode::Chat;
        }
        _ => {}
    }
    Ok(())
}

fn selector_item_name(item: &str) -> String {
    item.strip_prefix("[on] ")
        .or_else(|| item.strip_prefix("[off] "))
        .unwrap_or(item)
        .to_string()
}

async fn select_enter(
    app: &Arc<Mutex<App>>,
    ui: &mut UiState,
    kind: SelectorKind,
    selected: &str,
) {
    match kind {
        SelectorKind::Models => {
            if selected == "+ Add model" {
                ui.mode = Mode::Setup;
                ui.setup = Some(SetupState {
                    fields: vec![
                        ("name".to_string(), "deepseek-v4-flash".to_string()),
                        ("provider".to_string(), "deepseek".to_string()),
                        (
                            "base_url".to_string(),
                            "https://api.deepseek.com".to_string(),
                        ),
                        ("api_key".to_string(), String::new()),
                    ],
                    active: 0,
                    cursor: 0,
                    error: None,
                });
                return;
            }
            if selected == "− Remove model" {
                let app = app.lock().await;
                ui.selector = Some(crate::frontend::select::remove_models_selector(&app));
                ui.mode = Mode::Selector;
                return;
            }
            let mut app = app.lock().await;
            match app.switch_model(selected) {
                Ok(message) => {
                    ui.push_system(&message);
                    ui.mode = Mode::Chat;
                }
                Err(error) => {
                    ui.push_system(&format!("(switch failed) {error}"));
                    ui.mode = Mode::Chat;
                }
            }
        }
        SelectorKind::RemoveModel => {
            let mut app = app.lock().await;
            match app.remove_model(selected) {
                Ok(message) => {
                    ui.push_system(&message);
                    ui.selector = Some(crate::frontend::select::models_selector(&app));
                    ui.mode = Mode::Selector;
                }
                Err(error) => {
                    ui.push_system(&format!("(remove failed) {error}"));
                    ui.selector = Some(crate::frontend::select::remove_models_selector(&app));
                    ui.mode = Mode::Selector;
                }
            }
        }
        SelectorKind::Remember => {
            let id = selected
                .trim_start_matches('#')
                .split_whitespace()
                .next()
                .unwrap_or(selected);
            let mut app = app.lock().await;
            match app.inject_archive_summary(id) {
                Ok(message) => {
                    ui.push_system(&message);
                    ui.mode = Mode::Chat;
                }
                Err(error) => {
                    ui.push_system(&format!("(remember failed) {error}"));
                    ui.mode = Mode::Chat;
                }
            }
        }
        SelectorKind::History => {
            let id = selected
                .trim_start_matches('#')
                .split_whitespace()
                .next()
                .unwrap_or(selected);
            let mut app = app.lock().await;
            match app.resume_session(id) {
                Ok(message) => {
                    ui.push_system(&message);
                    ui.mode = Mode::Chat;
                }
                Err(error) => {
                    ui.push_system(&format!("(resume failed) {error}"));
                    ui.mode = Mode::Chat;
                }
            }
        }
        SelectorKind::Tasks => {
            let id: u64 = selected
                .trim_start_matches('#')
                .split_whitespace()
                .next()
                .and_then(|part| part.parse().ok())
                .unwrap_or(0);
            let cancelled = app.lock().await.scheduler.lock().unwrap().cancel(id);
            if cancelled {
                ui.push_system(&format!("cancelled task #{id}"));
            } else {
                ui.push_system(&format!("task #{id} not found"));
            }
            let app = app.lock().await;
            ui.selector = Some(crate::frontend::select::tasks_selector(&app));
            ui.mode = Mode::Selector;
        }
        SelectorKind::Rules => {
            let name = selector_item_name(selected)
                .split(" — ")
                .next()
                .unwrap_or(selected)
                .to_string();
            let project_dir = app.lock().await.project_dir().to_path_buf();
            let rules = crate::app::tools::load_rules_all(&project_dir);
            if let Some(rule) = rules.iter().find(|rule| rule.name == name) {
                ui.push_system(&format!("Rule: {}\n\n{}", rule.name, rule.rule));
            }
            ui.mode = Mode::Chat;
        }
        SelectorKind::Skills => {
            let name = selector_item_name(selected)
                .split(" — ")
                .next()
                .unwrap_or(selected)
                .to_string();
            let project_dir = app.lock().await.project_dir().to_path_buf();
            let skills = crate::app::tools::load_skills_all(&project_dir);
            if let Some(skill) = skills.iter().find(|skill| skill.name == name) {
                ui.push_system(&format!(
                    "Skill: {}\n\n{}",
                    skill.name, skill.description
                ));
            }
            ui.mode = Mode::Chat;
        }
        SelectorKind::Approvals => {
            let mut app = app.lock().await;
            app.clear_approval(selected);
            ui.push_system(&format!("forgot approval for {selected}"));
            ui.selector = Some(crate::frontend::select::approvals_selector(&app));
            ui.mode = Mode::Selector;
        }
    }
}

async fn handle_setup_key(
    app: &Arc<Mutex<App>>,
    ui: &mut UiState,
    key: KeyEvent,
) -> Result<(), Box<dyn Error>> {
    let Some(setup) = &mut ui.setup else {
        ui.mode = Mode::Chat;
        return Ok(());
    };
    match key.code {
        KeyCode::Char(_c) if key.modifiers.contains(KeyModifiers::CONTROL) => {}
        KeyCode::Char(c) => {
            let value = &mut setup.fields[setup.active].1;
            let cursor = setup.cursor.min(value.chars().count());
            let byte = value
                .char_indices()
                .nth(cursor)
                .map(|(byte, _)| byte)
                .unwrap_or(value.len());
            value.insert(byte, c);
            setup.cursor = cursor + 1;
            setup.error = None;
        }
        KeyCode::Left => {
            setup.cursor = setup.cursor.saturating_sub(1);
        }
        KeyCode::Right => {
            let len = setup.fields[setup.active].1.chars().count();
            if setup.cursor < len {
                setup.cursor += 1;
            }
        }
        KeyCode::Home => {
            setup.cursor = 0;
        }
        KeyCode::End => {
            setup.cursor = setup.fields[setup.active].1.chars().count();
        }
        KeyCode::Backspace => {
            let value = &mut setup.fields[setup.active].1;
            if setup.cursor > 0 {
                let byte = value
                    .char_indices()
                    .nth(setup.cursor - 1)
                    .map(|(byte, _)| byte)
                    .unwrap_or(0);
                value.remove(byte);
                setup.cursor -= 1;
            }
        }
        KeyCode::Delete => {
            let value = &mut setup.fields[setup.active].1;
            let len = value.chars().count();
            if setup.cursor < len {
                let byte = value
                    .char_indices()
                    .nth(setup.cursor)
                    .map(|(byte, _)| byte)
                    .unwrap_or(0);
                value.remove(byte);
            }
        }
        KeyCode::Up => {
            if setup.active > 0 {
                setup.active -= 1;
                setup.cursor = setup.fields[setup.active].1.chars().count();
            }
        }
        KeyCode::Down => {
            if setup.active + 1 < setup.fields.len() {
                setup.active += 1;
                setup.cursor = setup.fields[setup.active].1.chars().count();
            }
        }
        KeyCode::Enter => {
            if setup.active + 1 < setup.fields.len() {
                setup.active += 1;
                setup.cursor = setup.fields[setup.active].1.chars().count();
                return Ok(());
            }
            let name = setup.fields[0].1.clone();
            let kind = setup.fields[1].1.clone();
            let base_url = setup.fields[2].1.clone();
            let api_key = setup.fields[3].1.clone();
            if api_key.is_empty() {
                setup.error = Some("api key must not be empty".to_string());
                return Ok(());
            }
            match app.lock().await.add_model(&name, &kind, &base_url, &api_key) {
                Ok(message) => {
                    ui.push_system(&message);
                    ui.setup = None;
                    ui.mode = Mode::Chat;
                }
                Err(error) => {
                    setup.error = Some(error);
                }
            }
        }
        KeyCode::Esc => {
            ui.setup = None;
            ui.mode = Mode::Chat;
        }
        _ => {}
    }
    Ok(())
}

async fn handle_chat_key(
    app: &Arc<Mutex<App>>,
    ui: &mut UiState,
    key: KeyEvent,
    spawned: &Arc<std::sync::Mutex<Vec<tokio::task::JoinHandle<()>>>>,
) -> Result<(), Box<dyn Error>> {
    let command_layer = ui.input.buffer.starts_with('/');
    match key.code {
        KeyCode::Char(c) => {
            crate::frontend::input::insert_char(&mut ui.input, c);
            if ui.input.buffer.starts_with('/') {
                ui.mode = Mode::Command;
            } else {
                ui.mode = Mode::Chat;
            }
        }
        KeyCode::Enter
            if key.modifiers.contains(KeyModifiers::SHIFT)
                || key.modifiers.contains(KeyModifiers::ALT) =>
        {
            crate::frontend::input::newline(&mut ui.input);
        }
        KeyCode::Enter => {
            if command_layer {
                let filter = ui.input.buffer.trim_start_matches('/');
                if let Some(id) = commands::resume_arg(filter) {
                    ui.input.buffer.clear();
                    ui.input.cursor = 0;
                    ui.mode = Mode::Chat;
                    let result = app.lock().await.resume_session(&id);
                    ui.push_system(&match result {
                        Ok(message) => format!("(resumed) {message}"),
                        Err(error) => format!("(resume failed) {error}"),
                    });
                    return Ok(());
                }
                let commands = commands::filtered(filter);
                if !commands.is_empty() {
                    let index = ui.input.command_selection % commands.len();
                    let name = commands[index].0.clone();
                    run_command(app, ui, &name, spawned).await?;
                    return Ok(());
                }
            }
            submit_input(app, ui, spawned).await;
        }
        KeyCode::Tab if command_layer => {
            let filter = ui.input.buffer.trim_start_matches('/');
            let commands = commands::filtered(filter);
            if let Some((name, _)) = commands.first() {
                ui.input.buffer = format!("/{name}");
                ui.input.cursor = ui.input.buffer.chars().count();
            }
        }
        KeyCode::Tab => {
            ui.fold_expanded = !ui.fold_expanded;
        }
        KeyCode::Up if command_layer => {
            let filter = ui.input.buffer.trim_start_matches('/');
            let count = commands::filtered(filter).len();
            if count > 0 {
                ui.input.command_selection = (ui.input.command_selection + count - 1) % count;
            }
        }
        KeyCode::Down if command_layer => {
            let filter = ui.input.buffer.trim_start_matches('/');
            let count = commands::filtered(filter).len();
            if count > 0 {
                ui.input.command_selection = (ui.input.command_selection + 1) % count;
            }
        }
        KeyCode::Up => {
            crate::frontend::input::history_previous(&mut ui.input);
        }
        KeyCode::Down => {
            crate::frontend::input::history_next(&mut ui.input);
        }
        KeyCode::Left => crate::frontend::input::move_left(&mut ui.input),
        KeyCode::Right => crate::frontend::input::move_right(&mut ui.input),
        KeyCode::Home => crate::frontend::input::move_home(&mut ui.input),
        KeyCode::End => crate::frontend::input::move_end(&mut ui.input),
        KeyCode::Backspace => {
            crate::frontend::input::backspace(&mut ui.input);
            ui.mode = if ui.input.buffer.starts_with('/') {
                Mode::Command
            } else {
                Mode::Chat
            };
        }
        KeyCode::Delete => {
            crate::frontend::input::delete_char(&mut ui.input);
            ui.mode = if ui.input.buffer.starts_with('/') {
                Mode::Command
            } else {
                Mode::Chat
            };
        }
        KeyCode::PageUp => ui.scroll_offset = ui.scroll_offset.saturating_add(10),
        KeyCode::PageDown => ui.scroll_offset = ui.scroll_offset.saturating_sub(10),
        KeyCode::Esc => {
            if ui.input.buffer.starts_with('/') {
                ui.input.buffer.clear();
                ui.input.cursor = 0;
                ui.mode = Mode::Chat;
            } else if ui.is_generating() {
                ui.cancelled = true;
                ui.pending_tool_calls.clear();
                let app = app.clone();
                let spawned = spawned.clone();
                spawned.lock().unwrap().push(tokio::spawn(async move {
                    app.lock().await.cancel_generation();
                }));
                ui.finish_stream();
                ui.push_system("(generation cancelled)");
            } else if ui.has_running_tool() {
                cancel_running_tools(app, ui).await;
            }
        }
        _ => {}
    }
    Ok(())
}

async fn submit_input(
    app: &Arc<Mutex<App>>,
    ui: &mut UiState,
    spawned: &Arc<std::sync::Mutex<Vec<tokio::task::JoinHandle<()>>>>,
) {
    let Some(text) = crate::frontend::input::commit(&mut ui.input) else {
        return;
    };
    if text.starts_with('/') {
        return;
    }
    if ui.is_generating() {
        ui.input.pending_submit.push(text);
        return;
    }
    ui.cancelled = false;
    let app = app.clone();
    spawned.lock().unwrap().push(tokio::spawn(async move {
        let mut app = app.lock().await;
        app.submit(&text);
    }));
}

async fn run_command(
    app: &Arc<Mutex<App>>,
    ui: &mut UiState,
    name: &str,
    spawned: &Arc<std::sync::Mutex<Vec<tokio::task::JoinHandle<()>>>>,
) -> Result<(), Box<dyn Error>> {
    ui.input.buffer.clear();
    ui.input.cursor = 0;
    match commands::execute(name) {
        Some(Action::Compact) => {
            ui.mode = Mode::Chat;
            let app = app.clone();
            spawned.lock().unwrap().push(tokio::spawn(async move {
                let mut app = app.lock().await;
                let _ = app.compact().await;
            }));
        }
        Some(action) => {
            match app.try_lock() {
                Ok(mut app) => {
                    commands::apply_action(&mut app, ui, action);
                }
                Err(_) => {
                    ui.push_system("(busy: a tool is running, retry the command)");
                }
            }
        }
        None => {
            ui.push_system(&format!("unknown command: /{name}"));
        }
    }
    Ok(())
}
