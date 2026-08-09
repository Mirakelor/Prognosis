use crate::app::App;
use crate::frontend::state::SelectorKind;

pub struct Selector {
    pub kind: SelectorKind,
    pub title: String,
    pub items: Vec<String>,
    pub selected: usize,
    pub hint: String,
}

impl Selector {
    pub fn new(kind: SelectorKind, title: String, items: Vec<String>, hint: String) -> Self {
        Self {
            kind,
            title,
            items,
            selected: 0,
            hint,
        }
    }

    pub fn move_up(&mut self) {
        if !self.items.is_empty() {
            self.selected = (self.selected + self.items.len() - 1) % self.items.len();
        }
    }

    pub fn move_down(&mut self) {
        if !self.items.is_empty() {
            self.selected = (self.selected + 1) % self.items.len();
        }
    }

    pub fn current(&self) -> Option<&str> {
        self.items.get(self.selected).map(String::as_str)
    }
}

pub fn models_selector(app: &App) -> Selector {
    let mut items = app.list_models();
    items.push("+ Add model".to_string());
    items.push("− Remove model".to_string());
    Selector::new(
        SelectorKind::Models,
        "Models".to_string(),
        items,
        "Enter: switch / add / remove · ↑↓ : move · Esc: close".to_string(),
    )
}

pub fn remove_models_selector(app: &App) -> Selector {
    Selector::new(
        SelectorKind::RemoveModel,
        "Remove Model".to_string(),
        app.list_models(),
        "Enter: remove · ↑↓ : move · Esc: cancel".to_string(),
    )
}

pub fn remember_selector(app: &App) -> Selector {
    let items = app
        .remember
        .archive()
        .iter()
        .map(|entry| {
            let preview: String = entry.summary.chars().take(60).collect();
            format!("#{} · {preview}", entry.id)
        })
        .collect();
    Selector::new(
        SelectorKind::Remember,
        "Archived Memories".to_string(),
        items,
        "Enter: inject summary · ↑↓ : move · Esc: close".to_string(),
    )
}

pub fn history_selector(app: &App) -> Selector {
    let items = app
        .remember
        .list_sessions()
        .iter()
        .map(|meta| {
            let preview: String = meta.summary.chars().take(40).collect();
            format!(
                "#{} · {} · {} turns · {preview}",
                meta.id, meta.created, meta.turns
            )
        })
        .collect();
    Selector::new(
        SelectorKind::History,
        "Session History".to_string(),
        items,
        "Enter: load full session · ↑↓ : move · Esc: close".to_string(),
    )
}

pub fn tasks_selector(app: &App) -> Selector {
    let items = app
        .scheduler
        .lock()
        .unwrap()
        .tasks()
        .iter()
        .map(|task| format!("#{} {}", task.id, task.describe()))
        .collect();
    Selector::new(
        SelectorKind::Tasks,
        "Scheduled Tasks".to_string(),
        items,
        "Enter: cancel · ↑↓ : move · Esc: close".to_string(),
    )
}

pub fn rules_selector(app: &App) -> Selector {
    let project_dir = app.project_dir().to_path_buf();
    let items = crate::app::tools::load_rules_all(&project_dir)
        .into_iter()
        .map(|rule| {
            let flag = if crate::app::tools::is_rule_enabled(&project_dir, &rule.name) {
                "[on] "
            } else {
                "[off] "
            };
            if rule.description.is_empty() {
                format!("{flag}{}", rule.name)
            } else {
                format!("{flag}{} — {}", rule.name, rule.description)
            }
        })
        .collect();
    Selector::new(
        SelectorKind::Rules,
        "Rules".to_string(),
        items,
        "Enter: view · Tab: toggle · ↑↓ : move · Esc: close".to_string(),
    )
}

pub fn skills_selector(app: &App) -> Selector {
    let project_dir = app.project_dir().to_path_buf();
    let items = crate::app::tools::load_skills_all(&project_dir)
        .into_iter()
        .map(|skill| {
            let flag = if crate::app::tools::is_skill_enabled(&project_dir, &skill.name) {
                "[on] "
            } else {
                "[off] "
            };
            if skill.description.is_empty() {
                format!("{flag}{}", skill.name)
            } else {
                format!("{flag}{} — {}", skill.name, skill.description)
            }
        })
        .collect();
    Selector::new(
        SelectorKind::Skills,
        "Skills".to_string(),
        items,
        "Enter: view · Tab: toggle · ↑↓ : move · Esc: close".to_string(),
    )
}

pub fn approvals_selector(app: &App) -> Selector {
    let items = app.approvals();
    Selector::new(
        SelectorKind::Approvals,
        "Remembered Approvals".to_string(),
        items,
        "Enter: forget · ↑↓ : move · Esc: close".to_string(),
    )
}

pub struct StatusRow {
    pub title: bool,
    pub color: ratatui::prelude::Color,
    pub text: String,
}

fn bar(value: f32, width: usize) -> String {
    let value = value.clamp(0.0, 1.0);
    let filled = (value * width as f32).round() as usize;
    let mut out = String::new();
    out.push_str(&"▓".repeat(filled));
    if filled < width {
        out.push('▒');
        out.push_str(&"░".repeat(width.saturating_sub(filled + 1)));
    }
    out
}

fn signal_row(label: &str, value: f32, color: ratatui::prelude::Color) -> StatusRow {
    StatusRow {
        title: false,
        color,
        text: format!(
            "{label:<12} {}  {:.0}%",
            bar(value, 10),
            value.clamp(0.0, 1.0) * 100.0
        ),
    }
}

pub fn status_lines(
    ctx: &crate::frontend::render::RenderCtx,
    ui: &crate::frontend::state::UiState,
) -> Vec<StatusRow> {
    use crate::frontend::theme;
    let cognitive = &ui.cognitive;
    let mut rows = vec![StatusRow {
        title: true,
        color: theme::PINK,
        text: "◆ Modulators".to_string(),
    }];
    rows.push(signal_row("DA", cognitive.dopamine, theme::PINK));
    rows.push(signal_row("NE", cognitive.norepinephrine, theme::PURPLE));
    rows.push(signal_row("ACh", cognitive.acetylcholine, theme::GOLD));
    rows.push(signal_row("5-HT", cognitive.serotonin, theme::GREEN));
    rows.push(StatusRow {
        title: true,
        color: theme::PINK,
        text: "◆ Emotion & Meta".to_string(),
    });
    rows.push(signal_row("valence", cognitive.valence, theme::TEXT));
    rows.push(signal_row("arousal", cognitive.arousal, theme::TEXT));
    rows.push(signal_row("uncertainty", cognitive.uncertainty, theme::TEXT));
    rows.push(signal_row("conflict", cognitive.conflict, theme::TEXT));
    rows.push(signal_row("confidence", cognitive.confidence, theme::TEXT));
    rows.push(StatusRow {
        title: true,
        color: theme::PINK,
        text: "◆ Drives".to_string(),
    });
    rows.push(signal_row("homeostatic", cognitive.drive_homeostatic, theme::GREEN));
    rows.push(signal_row("curiosity", cognitive.drive_curiosity, theme::PURPLE));
    rows.push(signal_row("salience", cognitive.drive_salience, theme::GOLD));
    rows.push(StatusRow {
        title: true,
        color: theme::PINK,
        text: "◆ Prediction & Error".to_string(),
    });
    rows.push(signal_row(
        "RPE",
        cognitive.rpe.unwrap_or(0.0),
        theme::PINK,
    ));
    rows.push(signal_row(
        "last error",
        cognitive.last_error.unwrap_or(0.0),
        theme::RED,
    ));
    let direction = cognitive.prediction_direction.unwrap_or(0.0);
    rows.push(StatusRow {
        title: false,
        color: theme::PURPLE,
        text: format!(
            "direction   {}  {direction:+.2}",
            bar((direction + 1.0) / 2.0, 10)
        ),
    });
    rows.push(StatusRow {
        title: true,
        color: theme::PINK,
        text: format!("◆ Working Memory ({})", cognitive.wm_slots.len()),
    });
    for slot in &cognitive.wm_slots {
        let preview: String = slot.chars().take(60).collect();
        rows.push(StatusRow {
            title: false,
            color: theme::TEXT,
            text: format!("  - {preview}"),
        });
    }
    let percent = if ctx.context_limit > 0 {
        ui.total_tokens.checked_mul(100).map_or(0.0, |tokens| {
            tokens as f32 / ctx.context_limit as f32
        })
    } else {
        0.0
    };
    rows.push(StatusRow {
        title: false,
        color: theme::META,
        text: format!(
            "context     {}  {percent:.0}% · {}k/{}k",
            bar(percent / 100.0, 10),
            ui.total_tokens / 1000,
            ctx.context_limit / 1000
        ),
    });
    rows.push(StatusRow {
        title: false,
        color: theme::META,
        text: format!(
            "trace records: {} · /trace shows the latest 15",
            ctx.traces_len
        ),
    });
    rows
}
