use ratatui::prelude::*;
use ratatui::widgets::{Clear, Paragraph, Wrap};

use crate::frontend::commands;
use crate::frontend::markdown::{self, SpanKind};
use crate::frontend::select;
use crate::frontend::state::{Mode, ToolStatus, UiMessage, UiState};
use crate::frontend::theme;

const REASONING_FOLD_LINES: usize = 6;
const TOOL_FOLD_LINES: usize = 7;
const TOOL_MAX_LINES: usize = 200;

pub struct GitInfo {
    pub project_name: String,
    pub branch: Option<String>,
}

#[derive(Clone)]
pub struct RenderCtx {
    pub model: String,
    pub context_limit: usize,
    pub traces_len: usize,
    pub supervisor_enabled: bool,
    pub project_dir: String,
}

impl RenderCtx {
    pub async fn capture(app: &crate::app::App) -> Self {
        let project_dir = std::fs::canonicalize(app.project_dir())
            .unwrap_or_else(|_| app.project_dir().to_path_buf());
        Self {
            model: app.current_model_name().to_string(),
            context_limit: app.context_limit(),
            traces_len: app.traces().len(),
            supervisor_enabled: app.supervisor.is_enabled(),
            project_dir: project_dir.to_string_lossy().to_string(),
        }
    }
}

impl GitInfo {
    pub fn display(&self) -> String {
        match &self.branch {
            Some(branch) => format!("{} ⊦ {branch}", self.project_name),
            None => self.project_name.clone(),
        }
    }
}

pub fn draw(frame: &mut Frame, ctx: &RenderCtx, ui: &mut UiState, git: &GitInfo) {
    let height = frame.area().height;
    let status_height = 1u16;
    let input_height = input_area_height(ui, frame.area().width);
    let top_height = status_height + 1;
    let bottom_height = status_height;
    let divider_height = 1u16;
    let message_area = Rect {
        x: 0,
        y: top_height,
        width: frame.area().width,
        height: height.saturating_sub(
            top_height + bottom_height + input_height + divider_height,
        ),
    };
    draw_top_bar(frame, ctx, ui, git, Rect {
        x: 0,
        y: 0,
        width: frame.area().width,
        height: top_height,
    });
    match ui.mode {
        Mode::Chat => {
            draw_messages(frame, ui, message_area);
            draw_divider(frame, message_area);
            draw_input(frame, ui, Rect {
                x: 0,
                y: message_area.y + message_area.height + divider_height,
                width: frame.area().width,
                height: input_height,
            });
        }
        Mode::Command => {
            draw_command_layer(frame, ui, message_area);
            draw_divider(frame, message_area);
            draw_input(frame, ui, Rect {
                x: 0,
                y: message_area.y + message_area.height + divider_height,
                width: frame.area().width,
                height: input_height,
            });
        }
        Mode::Approve => {
            let approve_area = message_area;
            let messages_area = Rect {
                x: message_area.x,
                y: message_area.y,
                width: message_area.width,
                height: message_area.height.saturating_sub(1),
            };
            draw_messages(frame, ui, messages_area);
            draw_approve(frame, ui, approve_area);
            draw_divider(frame, message_area);
        }
        Mode::Selector => {
            draw_selector(frame, ui, message_area);
        }
        Mode::Status => draw_status(frame, ui, ctx, frame.area()),
        Mode::Help => draw_help(frame, ui, frame.area()),
        Mode::Setup => draw_setup(frame, ui, frame.area()),
    }
    draw_bottom_bar(frame, ui, ctx, git, Rect {
        x: 0,
        y: height.saturating_sub(1),
        width: frame.area().width,
        height: 1,
    });
}

fn draw_top_bar(frame: &mut Frame, ctx: &RenderCtx, ui: &UiState, git: &GitInfo, area: Rect) {
    let left = Line::from(vec![
        Span::styled("✦ ", theme::PINK),
        Span::styled(git.display(), theme::text_style()),
        Span::styled(" · ", theme::divider_style()),
        Span::styled(ctx.model.clone(), theme::text_style()),
    ]);
    frame.render_widget(Paragraph::new(left), Rect {
        x: 1,
        y: area.y,
        width: area.width.saturating_sub(2),
        height: 1,
    });
    let working = ui.is_generating();
    let (color, label) = if working {
        (theme::PINK, format!("◐ {}", ui.spinner()))
    } else {
        (theme::GREEN, "●".to_string())
    };
    let sup_color = if ctx.supervisor_enabled {
        theme::GOLD
    } else {
        theme::META
    };
    let sup_text = if ctx.supervisor_enabled { "ON" } else { "OFF" };
    let mode = if ui.cognitive.mode.is_empty() {
        "Auto".to_string()
    } else {
        ui.cognitive.mode.clone()
    };
    let mode_color = if mode == "Controlled" {
        theme::PINK
    } else {
        theme::PURPLE
    };
    let right = Line::from(vec![
        Span::styled(format!(" {label}{}", if working { " Working" } else { " Ready" }), color),
        Span::styled(" · ", theme::divider_style()),
        Span::styled("🛡 ", theme::GOLD),
        Span::styled("Supervisor ", theme::meta_style()),
        Span::styled(sup_text, sup_color),
        Span::styled(" · ", theme::divider_style()),
        Span::styled("◉ ", theme::PURPLE),
        Span::styled("Mode ", theme::meta_style()),
        Span::styled(mode, mode_color),
    ]);
    let right_width = right.width() as u16;
    frame.render_widget(
        Paragraph::new(right),
        Rect {
            x: area.width.saturating_sub(right_width + 1),
            y: area.y,
            width: right_width,
            height: 1,
        },
    );
    frame.render_widget(Paragraph::new(divider_line(area.width)), Rect {
        x: 0,
        y: area.y + 1,
        width: area.width,
        height: 1,
    });
}

fn input_area_height(ui: &UiState, width: u16) -> u16 {
    if ui.mode == Mode::Approve {
        return 0;
    }
    let inner_width = width.saturating_sub(2).max(4) as usize;
    let mut lines = 1usize;
    let mut current = 0usize;
    for c in ui.input.buffer.chars() {
        if c == '\n' {
            lines += 1;
            current = 0;
            continue;
        }
        let w = unicode_width::UnicodeWidthChar::width(c).unwrap_or(0);
        if current + w > inner_width {
            lines += 1;
            current = 0;
        }
        current += w;
    }
    lines.min(5) as u16 + 1
}

fn draw_messages(frame: &mut Frame, ui: &mut UiState, area: Rect) {
    if ui.messages.is_empty() && ui.mode != Mode::Approve {
        draw_intro(frame, ui, area);
        return;
    }
    let width = area.width.saturating_sub(2).max(10) as usize;
    let rows = render_rows(ui, width);
    let total = rows.len() as u16;
    let viewport = area.height.saturating_sub(1);
    let max_offset = total.saturating_sub(viewport) as usize;
    let back = ui.scroll_offset.min(max_offset);
    ui.scroll_offset = back;
    let scroll = total.saturating_sub(viewport + back as u16);
    let paragraph = Paragraph::new(rows).scroll((scroll, 0));
    frame.render_widget(paragraph, area);
}

fn render_rows(ui: &UiState, width: usize) -> Vec<Line<'static>> {
    let mut rows = Vec::new();
    for (index, message) in ui.messages.iter().enumerate() {
        match message {
            UiMessage::User { content, time } => {
                let time_width = time.chars().count();
                let pad = width.saturating_sub(2 + 3 + time_width);
                rows.push(Line::from(vec![
                    Span::styled("❯ ", theme::PINK),
                    Span::styled("You", theme::text_style().bold()),
                    Span::raw(" ".repeat(pad)),
                    Span::styled(time.clone(), theme::meta_style()),
                ]));
                for line in wrap_text(content, width) {
                    rows.push(Line::from(vec![
                        Span::raw("  "),
                        Span::styled(line, theme::text_style()),
                    ]));
                }
            }
            UiMessage::Assistant { content, reasoning } => {
                let generating = ui.streaming == Some(index);
                let point = "●".to_string();
                let label = if generating {
                    format!(" Agent · Thinking {}", ui.spinner())
                } else {
                    " Agent".to_string()
                };
                rows.push(Line::from(vec![
                    Span::styled(point, theme::PINK),
                    Span::styled(label, theme::text_style().bold()),
                ]));
                if !reasoning.is_empty() {
                    let wrapped = wrap_text(reasoning, width);
                    if !ui.fold_expanded && wrapped.len() > REASONING_FOLD_LINES {
                        for line in &wrapped[..REASONING_FOLD_LINES] {
                            rows.push(Line::from(Span::styled(
                                format!("  {line}"),
                                theme::meta_style().italic(),
                            )));
                        }
                        let remaining = wrapped.len() - REASONING_FOLD_LINES;
                        rows.push(Line::from(Span::styled(
                            format!("  ▸ … {remaining} more lines · Tab to expand"),
                            theme::meta_style().italic(),
                        )));
                    } else {
                        for line in wrapped {
                            rows.push(Line::from(Span::styled(
                                format!("  {line}"),
                                theme::meta_style().italic(),
                            )));
                        }
                    }
                }
                for markdown_line in markdown::parse(content) {
                    if markdown_line.code_block {
                        for code_line in markdown_line.spans[0].text.split('\n') {
                            rows.push(Line::from(Span::styled(
                                format!("    {code_line}"),
                                Style::default().fg(theme::PURPLE),
                            )));
                        }
                        continue;
                    }
                    if let Some(table) = &markdown_line.table {
                        for row in render_table(table, width) {
                            rows.push(row);
                        }
                        continue;
                    }
                    if markdown_line.heading {
                        let text = markdown_line.spans[0].text.clone();
                        let wrapped = wrap_text(&text, width.saturating_sub(4));
                        for (index, line) in wrapped.iter().enumerate() {
                            if index == 0 {
                                rows.push(Line::from(vec![
                                    Span::styled("  ▍", theme::GOLD),
                                    Span::styled(
                                        line.clone(),
                                        theme::text_style().bold().fg(theme::GOLD),
                                    ),
                                ]));
                            } else {
                                rows.push(Line::from(Span::styled(
                                    format!("    {line}"),
                                    theme::text_style().bold().fg(theme::GOLD),
                                )));
                            }
                        }
                        continue;
                    }
                    let spans = wrap_spans(&markdown_line.spans, width.saturating_sub(2));
                    for wrapped in spans {
                        let mut styled = Vec::new();
                        styled.push(Span::raw("  "));
                        if markdown_line.list_item {
                            styled.push(Span::styled("• ", theme::meta_style()));
                        }
                        styled.extend(wrapped.into_iter().map(|span| style_span(span)));
                        rows.push(Line::from(styled));
                    }
                }
            }
            UiMessage::System { content } => {
                for line in wrap_text(content, width) {
                    rows.push(Line::from(Span::styled(
                        line,
                        theme::meta_style().italic(),
                    )));
                }
            }
            UiMessage::Summary(summary) => {
                rows.push(Line::from(vec![
                    Span::styled("✦ ", theme::PINK),
                    Span::styled(summary.clone(), theme::text_style().bold()),
                ]));
            }
            UiMessage::ToolCall(call) => {
                let (color, marker, icon) = match call.status {
                    ToolStatus::Running => (theme::PINK, "◐", tool_icon(&call.name)),
                    ToolStatus::Done => (theme::GREEN, "✓", tool_icon(&call.name)),
                    ToolStatus::Errored | ToolStatus::Denied => {
                        (theme::RED, "✗", tool_icon(&call.name))
                    }
                };
                let path = extract_filepath(&call.arguments);
                let mut spans = vec![
                    Span::styled(format!("  {marker} "), color),
                    Span::styled(
                        format!("{icon} {:<24}", call.name),
                        theme::text_style().bold(),
                    ),
                ];
                if let Some(path) = &path {
                    spans.push(Span::styled(path.clone(), theme::PURPLE));
                }
                let stats = format_stats(call);
                if !stats.is_empty() {
                    let used: usize = spans
                        .iter()
                        .map(|span| unicode_width::UnicodeWidthStr::width(span.content.as_ref()))
                        .sum();
                    let stats_width = stats.chars().count();
                    let pad = width.saturating_sub(used + stats_width);
                    if pad > 0 {
                        spans.push(Span::raw(" ".repeat(pad)));
                    }
                    spans.push(Span::styled(stats, theme::meta_style()));
                }
                rows.push(Line::from(spans));
                if matches!(call.status, ToolStatus::Done | ToolStatus::Errored | ToolStatus::Denied)
                    && !call.output.trim().is_empty()
                {
                    let is_diff = !call.diff.is_empty();
                    let raw_lines: Vec<&str> = call.output.lines().collect();
                    let lines: Vec<&str> = if is_diff {
                        raw_lines
                            .iter()
                            .filter(|line| !is_diff_source_line(line))
                            .copied()
                            .collect()
                    } else {
                        raw_lines
                    };
                    let mut shown: Vec<&str> = lines.clone();
                    let mut folded = false;
                    if !ui.fold_expanded && shown.len() > TOOL_FOLD_LINES {
                        shown = shown[..TOOL_FOLD_LINES].to_vec();
                        folded = true;
                    }
                    if shown.len() > TOOL_MAX_LINES {
                        shown.truncate(TOOL_MAX_LINES);
                    }
                    for line in shown {
                        if line.trim().is_empty() {
                            continue;
                        }
                        rows.push(Line::from(Span::styled(
                            format!("  │ {line}"),
                            theme::meta_style(),
                        )));
                    }
                    if folded {
                        let remaining = lines.len().saturating_sub(TOOL_FOLD_LINES);
                        rows.push(Line::from(Span::styled(
                            format!("  ▸ … {remaining} more lines · Tab to expand"),
                            theme::meta_style().italic(),
                        )));
                    } else if lines.len() > TOOL_MAX_LINES {
                        rows.push(Line::from(Span::styled(
                            format!("  ▸ … {} more lines (truncated)",
                                lines.len().saturating_sub(TOOL_MAX_LINES)),
                            theme::meta_style().italic(),
                        )));
                    }
                }
                if !call.diff.is_empty() {
                    let mut shown: &[crate::frontend::state::DiffLine] = &call.diff;
                    let mut folded = false;
                    if !ui.fold_expanded && shown.len() > TOOL_FOLD_LINES {
                        shown = &shown[..TOOL_FOLD_LINES];
                        folded = true;
                    }
                    for line in shown {
                        rows.push(render_diff_line(line, &call.diff));
                    }
                    if folded {
                        let remaining = call.diff.len().saturating_sub(TOOL_FOLD_LINES);
                        rows.push(Line::from(Span::styled(
                            format!("  ▸ … {remaining} more diff lines · Tab to expand"),
                            theme::meta_style().italic(),
                        )));
                    }
                }
            }
        }
    }
    rows
}

fn is_diff_source_line(line: &str) -> bool {
    line.starts_with("diff --git")
        || line.starts_with("---")
        || line.starts_with("+++")
        || line.starts_with("@@")
        || line.starts_with('+')
        || line.starts_with('-')
        || line.starts_with(' ')
}

fn cell_width(cell: &[crate::frontend::markdown::Span]) -> usize {
    cell.iter()
        .map(|span| unicode_width::UnicodeWidthStr::width(span.text.as_str()))
        .sum()
}

fn render_table(
    rows: &[crate::frontend::markdown::TableRow],
    width: usize,
) -> Vec<Line<'static>> {
    let columns = rows
        .iter()
        .filter(|row| !row.separator)
        .map(|row| row.cells.len())
        .max()
        .unwrap_or(0)
        .max(1);
    let mut col_widths = vec![0usize; columns];
    for row in rows.iter().filter(|row| !row.separator) {
        for (i, cell) in row.cells.iter().enumerate().take(columns) {
            col_widths[i] = col_widths[i].max(cell_width(cell));
        }
    }
    let max_inner = width.saturating_sub(4).max(10);
    let total: usize = col_widths.iter().sum::<usize>() + columns + 1;
    if total > max_inner {
        let mut overflow = total - max_inner;
        for col in col_widths.iter_mut().rev() {
            if overflow == 0 {
                break;
            }
            let cut = (*col).min(overflow);
            *col = col.saturating_sub(cut);
            overflow -= cut;
        }
    }
    let mut out = Vec::new();
    let divider = theme::divider_style();
    let mut top = vec![Span::raw("  ┌")];
    for (i, col) in col_widths.iter().enumerate() {
        top.push(Span::styled("─".repeat(col + 2), divider));
        top.push(Span::styled(
            if i + 1 < col_widths.len() { "┬" } else { "┐" },
            divider,
        ));
    }
    out.push(Line::from(top));
    for row in rows {
        if row.separator {
            let mut spans = vec![Span::raw("  ├")];
            for (i, col) in col_widths.iter().enumerate() {
                spans.push(Span::styled("─".repeat(col + 2), divider));
                spans.push(Span::styled(
                    if i + 1 < col_widths.len() { "┼" } else { "┤" },
                    divider,
                ));
            }
            out.push(Line::from(spans));
            continue;
        }
        let mut spans = vec![Span::raw("  │")];
        for (i, col) in col_widths.iter().enumerate() {
            let cell = row.cells.get(i).cloned().unwrap_or_default();
            let content = cell_width(&cell);
            let pad = col.saturating_sub(content);
            let (left_pad, right_pad) = match row.align.get(i).copied().unwrap_or(crate::frontend::markdown::Align::Left) {
                crate::frontend::markdown::Align::Center => (pad / 2, pad - pad / 2),
                crate::frontend::markdown::Align::Right => (pad, 0),
                crate::frontend::markdown::Align::Left => (0, pad),
            };
            spans.push(Span::raw(" ".repeat(left_pad + 1)));
            if row.header {
                for span in &cell {
                    spans.push(Span::styled(
                        span.text.clone(),
                        Style::default().fg(theme::GOLD).bold(),
                    ));
                }
            } else {
                for span in &cell {
                    spans.push(style_span(span.clone()));
                }
            }
            spans.push(Span::raw(" ".repeat(right_pad + 1)));
            spans.push(Span::styled("│", divider));
        }
        out.push(Line::from(spans));
    }
    let mut bottom = vec![Span::raw("  └")];
    for (i, col) in col_widths.iter().enumerate() {
        bottom.push(Span::styled("─".repeat(col + 2), divider));
        bottom.push(Span::styled(
            if i + 1 < col_widths.len() { "┴" } else { "┘" },
            divider,
        ));
    }
    out.push(Line::from(bottom));
    out
}

fn tool_icon(name: &str) -> &'static str {
    match name {
        "edit_existing_file" => "✎",
        "single_find_and_replace" => "⇋",
        "create_new_file" => "✚",
        "run_terminal_command" => "$",
        "read_file" => "▤",
        "ls" => "▦",
        "grep_search" => "⌗",
        "file_glob_search" => "✱",
        "view_diff" => "⇄",
        "search_web" => "⌕",
        "fetch_url_content" => "⇪",
        "create_rule_block" => "⚑",
        "request_rule" => "⚐",
        "read_skill" => "❖",
        "schedule_task" => "◷",
        "cancel_task" => "⊘",
        _ => "○",
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

fn format_stats(call: &crate::frontend::state::ToolCallMsg) -> String {
    let mut stats = Vec::new();
    if call.diff.iter().any(|line| line.kind == '+') {
        let added = call.diff.iter().filter(|line| line.kind == '+').count();
        let removed = call.diff.iter().filter(|line| line.kind == '-').count();
        stats.push(format!("+{added} −{removed}"));
    }
    let elapsed = match call.elapsed {
        Some(elapsed) => elapsed,
        None => call.started.elapsed().as_secs_f64(),
    };
    if call.status == crate::frontend::state::ToolStatus::Running
        || call.elapsed.is_some()
    {
        stats.push(format!("{elapsed:.1}s"));
    }
    if let Some(verdict) = &call.verdict {
        stats.push(format!("🛡 {verdict}"));
    }
    if call.name == "run_terminal_command" {
        if call.status == crate::frontend::state::ToolStatus::Done {
            stats.push("ok".to_string());
        } else if call.status == crate::frontend::state::ToolStatus::Errored {
            stats.push("failed".to_string());
        }
    } else if !call.summary.is_empty() && call.status != crate::frontend::state::ToolStatus::Running {
        let summary: String = call.summary.chars().take(60).collect();
        stats.push(summary);
    }
    stats.join("  ")
}

fn render_diff_line(
    line: &crate::frontend::state::DiffLine,
    all: &[crate::frontend::state::DiffLine],
) -> Line<'static> {
    let width = all
        .iter()
        .filter_map(|line| line.line_no)
        .max()
        .map(|max| max.to_string().len())
        .unwrap_or(2)
        .max(2);
    let number = line
        .line_no
        .map(|no| format!("{no:>width$}"))
        .unwrap_or_else(|| " ".repeat(width));
    match line.kind {
        '+' => Line::from(vec![
            Span::styled(format!("  +  {number}  "), Style::default().fg(theme::GREEN)),
            Span::styled(line.text.clone(), Style::default().fg(theme::GREEN)),
        ]),
        '-' => Line::from(vec![
            Span::styled(format!("  −  {number}  "), Style::default().fg(theme::RED)),
            Span::styled(line.text.clone(), Style::default().fg(theme::RED)),
        ]),
        _ => Line::from(vec![
            Span::styled(format!("     {number}  "), theme::meta_style()),
            Span::styled(line.text.clone(), theme::meta_style()),
        ]),
    }
}

fn style_span(span: markdown::Span) -> Span<'static> {
    let mut style = theme::text_style();
    style = match span.kind {
        SpanKind::Normal => style,
        SpanKind::Bold => style.bold(),
        SpanKind::Italic => style.italic(),
        SpanKind::Dim => theme::meta_style().italic(),
        SpanKind::Code => style.fg(theme::PURPLE),
        SpanKind::Strike => style.crossed_out(),
    };
    Span::styled(span.text, style)
}

fn wrap_text(text: &str, width: usize) -> Vec<String> {
    let mut lines = Vec::new();
    for raw in text.split('\n') {
        let mut current = String::new();
        let mut current_width = 0usize;
        for c in raw.chars() {
            let w = unicode_width::UnicodeWidthChar::width(c).unwrap_or(0);
            if current_width + w > width && !current.is_empty() {
                lines.push(std::mem::take(&mut current));
                current_width = 0;
            }
            current.push(c);
            current_width += w;
        }
        lines.push(current);
    }
    lines
}

fn wrap_spans(spans: &[markdown::Span], width: usize) -> Vec<Vec<markdown::Span>> {
    let mut result = Vec::new();
    let mut current: Vec<markdown::Span> = Vec::new();
    let mut current_width = 0usize;
    for span in spans {
        let mut text = span.text.as_str();
        while !text.is_empty() {
            let remaining = width.saturating_sub(current_width);
            if remaining == 0 {
                result.push(std::mem::take(&mut current));
                current_width = 0;
                continue;
            }
            let mut take = 0usize;
            let mut take_width = 0usize;
            for c in text.chars() {
                let w = unicode_width::UnicodeWidthChar::width(c).unwrap_or(0);
                if take_width + w > remaining {
                    break;
                }
                take += c.len_utf8();
                take_width += w;
            }
            if take == 0 {
                take = text.chars().next().map(|c| c.len_utf8()).unwrap_or(0);
            }
            let (head, rest) = text.split_at(take);
            current.push(markdown::Span {
                text: head.to_string(),
                kind: span.kind,
            });
            current_width += take_width;
            text = rest;
            if current_width >= width {
                result.push(std::mem::take(&mut current));
                current_width = 0;
            }
        }
    }
    if !current.is_empty() {
        result.push(current);
    }
    result
}

fn draw_intro(frame: &mut Frame, ui: &UiState, area: Rect) {
    let wrap_width = area.width.saturating_sub(4).max(10) as usize;
    let mut lines: Vec<Line<'static>> = Vec::new();
    if let Some(hitokoto) = &ui.hitokoto {
        for wrapped in wrap_text(&hitokoto.text, wrap_width) {
            lines.push(centered_line(
                Span::styled(wrapped, theme::text_style().italic()),
                area.width,
            ));
        }
        if let Some(source) = &hitokoto.source {
            lines.push(centered_line(
                Span::styled(format!("—— {source}"), theme::meta_style()),
                area.width,
            ));
        }
        lines.push(centered_line(Span::raw(""), area.width));
    }
    lines.push(centered_line(
        Line::from(vec![
            Span::styled("✦ ", theme::PINK),
            Span::styled(ui.tip, theme::meta_style().italic()),
        ]),
        area.width,
    ));
    let y = area.y + area.height.saturating_sub(lines.len() as u16) / 2;
    let height = lines.len() as u16;
    frame.render_widget(
        Paragraph::new(lines),
        Rect {
            x: 0,
            y,
            width: area.width,
            height,
        },
    );
}

fn centered_line(content: impl Into<Line<'static>>, area_width: u16) -> Line<'static> {
    let line = content.into();
    let text: String = line.spans.iter().map(|span| span.content.as_ref()).collect();
    let width = unicode_width::UnicodeWidthStr::width(text.as_str());
    let pad = area_width.saturating_sub(width as u16) / 2;
    let mut spans = vec![Span::raw(" ".repeat(pad as usize))];
    spans.extend(line.spans);
    Line::from(spans)
}

fn draw_command_layer(frame: &mut Frame, ui: &UiState, area: Rect) {
    let filter = ui.input.buffer.trim_start_matches('/');
    let commands = commands::filtered(filter);
    if commands.is_empty() {
        return;
    }
    let height = (commands.len() as u16).min(area.height);
    let width = area.width.min(60);
    let popup = Rect {
        x: area.x + (area.width.saturating_sub(width)) / 2,
        y: area.y + (area.height.saturating_sub(height)) / 2,
        width,
        height,
    };
    frame.render_widget(Clear, popup);
    let selected = ui.input.command_selection % commands.len();
    for (index, (name, description)) in commands.iter().enumerate().take(height as usize) {
        let marker = if index == selected { "➤ " } else { "  " };
        let style = if index == selected {
            Style::default().fg(theme::PINK).bold()
        } else {
            theme::text_style()
        };
        frame.render_widget(
            Paragraph::new(Line::from(vec![
                Span::styled(marker, style),
                Span::styled(format!("/{name}"), style),
                Span::styled(format!("  {description}"), theme::meta_style()),
            ])),
            Rect {
                x: popup.x,
                y: popup.y + index as u16,
                width: popup.width,
                height: 1,
            },
        );
    }
}

fn divider_line(width: u16) -> Line<'static> {
    let total = width.saturating_sub(5) as usize;
    let left = total / 2;
    let right = total - left;
    Line::from(vec![
        Span::raw(" "),
        Span::styled("─".repeat(left), theme::divider_style()),
        Span::styled(" ✧ ", theme::divider_style()),
        Span::styled("─".repeat(right), theme::divider_style()),
    ])
}

fn draw_divider(frame: &mut Frame, message_area: Rect) {
    frame.render_widget(
        Paragraph::new(divider_line(message_area.width)),
        Rect {
            x: message_area.x,
            y: message_area.y + message_area.height,
            width: message_area.width,
            height: 1,
        },
    );
}

fn draw_input(frame: &mut Frame, ui: &UiState, area: Rect) {
    if ui.input.buffer.is_empty() {
        let placeholder = if ui.input.pending_submit.is_empty() {
            "Ask anything, / for slash commands".to_string()
        } else {
            format!(
                "Ask anything, / for slash commands · {} queued — will send when the agent finishes",
                ui.input.pending_submit.len()
            )
        };
        frame.render_widget(
            Paragraph::new(Line::from(vec![
                Span::styled("❯ ", theme::PINK),
                Span::styled(placeholder, theme::meta_style().italic()),
            ]))
            .wrap(Wrap { trim: false }),
            area,
        );
        frame.set_cursor_position((area.x + 2, area.y));
        return;
    }
    let buffer = &ui.input.buffer;
    let cursor_byte = char_index_to_byte(buffer, ui.input.cursor);
    let lines: Vec<&str> = buffer.split('\n').collect();
    let cursor_line = buffer[..cursor_byte].matches('\n').count();
    let line_start = buffer[..cursor_byte].rfind('\n').map(|i| i + 1).unwrap_or(0);
    let cursor_in_line = buffer[line_start..cursor_byte].chars().count();
    let mut rendered = Vec::new();
    for (index, line) in lines.iter().enumerate() {
        if index == 0 {
            let mut spans = vec![Span::styled("❯ ", theme::PINK)];
            if !line.is_empty() {
                spans.push(Span::styled(line.to_string(), theme::text_style()));
            }
            rendered.push(Line::from(spans));
        } else if !line.is_empty() {
            rendered.push(Line::from(vec![Span::styled(
                line.to_string(),
                theme::text_style(),
            )]));
        }
    }
    frame.render_widget(
        Paragraph::new(rendered).wrap(Wrap { trim: false }),
        area,
    );
    let before: String = lines[cursor_line].chars().take(cursor_in_line).collect();
    let before_width = unicode_width::UnicodeWidthStr::width(before.as_str()) as u16;
    let x = area.x + if cursor_line == 0 { 2 } else { 0 } + before_width;
    let y = (area.y + cursor_line as u16).min(area.bottom().saturating_sub(1));
    frame.set_cursor_position((x.min(area.right().saturating_sub(1)), y));
}

fn draw_approve(frame: &mut Frame, ui: &UiState, area: Rect) {
    let name = ui
        .last_tool_index
        .and_then(|i| ui.messages.get(i))
        .map(|m| match m {
            UiMessage::ToolCall(call) => call.name.clone(),
            _ => String::new(),
        })
        .unwrap_or_default();
    let keys = if ui.kitty_supported {
        "Enter: Approve · Shift+Enter: Remember · Esc: Deny"
    } else {
        "Enter: Approve · Shift+Tab/y: Remember · Esc: Deny"
    };
    let line = centered_line(
        Line::from(vec![
            Span::styled(
                "◉ Tool Approval",
                Style::default().fg(theme::PINK).bold(),
            ),
            Span::styled(" · ", theme::divider_style()),
            Span::styled(name, theme::text_style().bold()),
            Span::styled(" · ", theme::divider_style()),
            Span::styled(keys, theme::meta_style()),
        ]),
        area.width,
    );
    let row = area.y + area.height.saturating_sub(1);
    frame.render_widget(
        Paragraph::new(line),
        Rect {
            x: 0,
            y: row,
            width: area.width,
            height: 1,
        },
    );
}

fn draw_selector(frame: &mut Frame, ui: &UiState, area: Rect) {
    let Some(selector) = &ui.selector else {
        return;
    };
    let width = area.width.min(72);
    let height = (selector.items.len() as u16).min(14) + 2;
    let popup = Rect {
        x: area.x + (area.width.saturating_sub(width)) / 2,
        y: area.y + (area.height.saturating_sub(height)) / 2,
        width,
        height,
    };
    frame.render_widget(Clear, popup);
    let title = Line::from(Span::styled(
        selector.title.clone(),
        Style::default().fg(theme::PINK).bold(),
    ));
    frame.render_widget(
        Paragraph::new(title),
        Rect {
            x: popup.x,
            y: popup.y,
            width: popup.width,
            height: 1,
        },
    );
    for (index, item) in selector
        .items
        .iter()
        .enumerate()
        .take(popup.height.saturating_sub(2) as usize)
    {
        let selected = index == selector.selected;
        let marker = if selected { "➤ " } else { "  " };
        let style = if selected {
            Style::default().fg(theme::PINK).bold()
        } else {
            theme::text_style()
        };
        frame.render_widget(
            Paragraph::new(Line::from(vec![
                Span::styled(marker, style),
                Span::styled(item.clone(), style),
            ])),
            Rect {
                x: popup.x,
                y: popup.y + 1 + index as u16,
                width: popup.width,
                height: 1,
            },
        );
    }
    frame.render_widget(
        Paragraph::new(Line::from(Span::styled(
            selector.hint.clone(),
            theme::meta_style(),
        ))),
        Rect {
            x: popup.x,
            y: popup.y + popup.height.saturating_sub(1),
            width: popup.width,
            height: 1,
        },
    );
}

fn draw_status(frame: &mut Frame, ui: &mut UiState, ctx: &RenderCtx, area: Rect) {
    let mut all = vec![Line::from(Span::styled(
        "Cognitive Status",
        Style::default().fg(theme::PINK).bold(),
    ))];
    for row in select::status_lines(ctx, ui) {
        let style = if row.title {
            Style::default().fg(row.color).bold()
        } else {
            Style::default().fg(row.color)
        };
        all.push(Line::from(Span::styled(row.text, style)));
    }
    let max_width = all
        .iter()
        .map(|line| {
            line.spans
                .iter()
                .map(|span| unicode_width::UnicodeWidthStr::width(span.content.as_ref()))
                .sum::<usize>()
        })
        .max()
        .unwrap_or(0) as u16;
    let panel_width = (max_width + 4).min(area.width);
    let all: Vec<Line<'static>> = all
        .into_iter()
        .map(|line| centered_line(line, panel_width))
        .collect();
    let panel_height = (area.height.saturating_sub(6)).min(all.len() as u16) as usize;
    let max_scroll = all.len().saturating_sub(panel_height);
    let scroll = ui.panel_scroll.min(max_scroll);
    ui.panel_scroll = scroll;
    let visible: Vec<Line<'static>> = all[scroll..scroll + panel_height].to_vec();
    let popup = Rect {
        x: area.x + (area.width.saturating_sub(panel_width)) / 2,
        y: area.y + 2 + (area.height.saturating_sub(6).saturating_sub(panel_height as u16)) / 2,
        width: panel_width,
        height: panel_height as u16,
    };
    frame.render_widget(Clear, popup);
    frame.render_widget(Paragraph::new(visible).scroll((0, 0)), popup);
}

fn draw_help(frame: &mut Frame, ui: &mut UiState, area: Rect) {
    let bindings: &[(&str, &str)] = &[
        ("Enter", "Submit / Approve"),
        ("Shift+Enter", "Newline / Approve & Remember"),
        ("Shift+Tab / y", "Approve & Remember (fallback)"),
        ("Esc", "Interrupt, Close Panels, Deny"),
        ("↑/↓", "Input History / Scroll Panels"),
        ("Tab", "Complete Command / Expand & Collapse Folds"),
        ("PageUp/Down", "Scroll Message History"),
        ("Paste", "Insert text (multi-line ok) — Enter sends"),
        ("Ctrl+C", "Interrupt Generation / Quit when idle"),
        ("Ctrl+D", "Quit"),
    ];
    let bindings_max = bindings
        .iter()
        .map(|(key, desc)| {
            unicode_width::UnicodeWidthStr::width(*key)
                + unicode_width::UnicodeWidthStr::width(*desc)
        })
        .max()
        .unwrap_or(0);
    let commands_max = commands::COMMANDS
        .iter()
        .map(|(name, desc)| {
            2 + unicode_width::UnicodeWidthStr::width(*name)
                + unicode_width::UnicodeWidthStr::width(*desc)
        })
        .max()
        .unwrap_or(0);
    let panel_width = (bindings_max.max(commands_max) as u16 + 4).min(area.width);
    let mut all = vec![
        centered_line(
            Span::styled("Help", Style::default().fg(theme::PINK).bold()),
            panel_width,
        ),
        centered_line(
            Span::styled(
                "Key Bindings",
                Style::default().fg(theme::PINK).bold(),
            ),
            panel_width,
        ),
    ];
    for (key, desc) in bindings {
        all.push(help_row(key, desc, panel_width));
    }
    all.push(centered_line(
        Span::styled("Commands", Style::default().fg(theme::PINK).bold()),
        panel_width,
    ));
    for (name, description) in commands::COMMANDS {
        let row = Line::from(vec![
            Span::styled(
                format!("/{name}"),
                Style::default().fg(theme::PURPLE),
            ),
            Span::styled(format!("  {description}"), theme::meta_style()),
        ]);
        all.push(centered_line(row, panel_width));
    }
    let panel_height = (area.height.saturating_sub(6)).min(all.len() as u16) as usize;
    let max_scroll = all.len().saturating_sub(panel_height);
    let scroll = ui.panel_scroll.min(max_scroll);
    ui.panel_scroll = scroll;
    let visible: Vec<Line<'static>> = all[scroll..scroll + panel_height].to_vec();
    let popup = Rect {
        x: area.x + (area.width.saturating_sub(panel_width)) / 2,
        y: area.y + 2 + (area.height.saturating_sub(6).saturating_sub(panel_height as u16)) / 2,
        width: panel_width,
        height: panel_height as u16,
    };
    frame.render_widget(Clear, popup);
    frame.render_widget(Paragraph::new(visible), popup);
}

fn help_row(key: &str, desc: &str, panel_width: u16) -> Line<'static> {
    let key_width = unicode_width::UnicodeWidthStr::width(key);
    let desc_width = unicode_width::UnicodeWidthStr::width(desc);
    let pad = panel_width.saturating_sub(key_width as u16 + desc_width as u16) as usize;
    Line::from(vec![
        Span::styled(key.to_string(), theme::text_style()),
        Span::raw(" ".repeat(pad)),
        Span::styled(desc.to_string(), theme::meta_style()),
    ])
}

fn draw_setup(frame: &mut Frame, ui: &UiState, area: Rect) {
    let Some(setup) = &ui.setup else {
        return;
    };
    let mut content: Vec<Line<'static>> = Vec::new();
    let mut cursor_pos = None;
    for (index, (label, value)) in setup.fields.iter().enumerate() {
        let active = index == setup.active;
        let prefix = if active { "➤ " } else { "  " };
        let label_style = if active {
            Style::default().fg(theme::PINK).bold()
        } else {
            theme::text_style()
        };
        if active {
            let before: String = value.chars().take(setup.cursor).collect();
            let column = 2 + label.chars().count() as u16 + 2
                + unicode_width::UnicodeWidthStr::width(before.as_str()) as u16;
            cursor_pos = Some((content.len() as u16 + 1, column));
        }
        content.push(Line::from(vec![
            Span::styled(prefix, label_style),
            Span::styled(format!("{label}: "), label_style),
            Span::styled(value.clone(), theme::text_style()),
        ]));
    }
    if let Some(error) = &setup.error {
        content.push(Line::from(Span::styled(
            error.clone(),
            Style::default().fg(theme::RED),
        )));
    }
    content.push(Line::from(Span::styled(
        "Enter: Next Field / Finish · ↑↓ : Move · Esc: Cancel",
        theme::meta_style(),
    )));
    let max_width = content
        .iter()
        .map(|line| {
            line.spans
                .iter()
                .map(|span| unicode_width::UnicodeWidthStr::width(span.content.as_ref()))
                .sum::<usize>()
        })
        .max()
        .unwrap_or(0) as u16;
    let width = (max_width + 6).min(area.width);
    let mut lines = vec![Line::from(Span::styled(
        "Model Setup",
        Style::default().fg(theme::PINK).bold(),
    ))];
    lines.extend(content);
    let height = lines.len() as u16;
    let popup = Rect {
        x: area.x + (area.width.saturating_sub(width)) / 2,
        y: area.y + (area.height.saturating_sub(height)) / 2,
        width,
        height,
    };
    frame.render_widget(Clear, popup);
    frame.render_widget(Paragraph::new(lines), popup);
    if let Some((row, column)) = cursor_pos {
        frame.set_cursor_position((
            (popup.x + column).min(popup.right().saturating_sub(1)),
            (popup.y + row).min(popup.bottom().saturating_sub(1)),
        ));
    }
}

fn draw_bottom_bar(frame: &mut Frame, ui: &UiState, ctx: &RenderCtx, _git: &GitInfo, area: Rect) {
    let percent = if ctx.context_limit > 0 {
        ui.total_tokens.checked_mul(100).map_or(0, |tokens| {
            tokens.checked_div(ctx.context_limit).unwrap_or(0)
        })
    } else {
        0
    };
    let bar_width = 14usize;
    let filled = (percent * bar_width) / 100;
    let mut bar = String::new();
    bar.push_str(&"▓".repeat(filled));
    if filled < bar_width {
        bar.push('▒');
        bar.push_str(&"░".repeat(bar_width - filled - 1));
    }
    let right_text = "↑ History · Esc Stop · Ctrl+D Quit";
    let right_width = right_text.chars().count() as u16;
    let left_max = area.width.saturating_sub(right_width + 2) as usize;
    let queued = ui.input.pending_submit.len();
    if queued > 0 {
        let mut spans = vec![
            Span::styled("◆ ", theme::PINK),
            Span::styled(format!("{queued} queued"), theme::meta_style()),
        ];
        let path = fit_path(&ctx.project_dir, left_max.saturating_sub(14));
        if !path.is_empty() {
            spans.push(Span::styled(" · ", theme::divider_style()));
            spans.push(Span::styled(path, theme::meta_style()));
        }
        let left = Line::from(spans);
        frame.render_widget(Paragraph::new(left), Rect {
            x: 1,
            y: area.y,
            width: area.width.saturating_sub(2),
            height: 1,
        });
        frame.render_widget(
            Paragraph::new(Line::from(Span::styled(right_text, theme::meta_style()))),
            Rect {
                x: area.width.saturating_sub(right_width + 1),
                y: area.y,
                width: right_width,
                height: 1,
            },
        );
        return;
    }
    let mut left = Line::from(vec![
        Span::styled("◐ ", theme::PINK),
        Span::styled("ctx ", theme::meta_style()),
        Span::styled(bar, theme::PINK),
        Span::styled(
            format!(" {percent}% · {}k/{}k", ui.total_tokens / 1000, ctx.context_limit / 1000),
            theme::meta_style(),
        ),
    ]);
    let path = fit_path(&ctx.project_dir, left_max.saturating_sub(left.width() as usize + 4));
    if !path.is_empty() {
        left.spans.push(Span::styled(" · ", theme::divider_style()));
        left.spans.push(Span::styled(path, theme::meta_style()));
    }
    frame.render_widget(Paragraph::new(left), Rect {
        x: 1,
        y: area.y,
        width: area.width.saturating_sub(2),
        height: 1,
    });
    frame.render_widget(
        Paragraph::new(Line::from(Span::styled(right_text, theme::meta_style()))),
        Rect {
            x: area.width.saturating_sub(right_width + 1),
            y: area.y,
            width: right_width,
            height: 1,
        },
    );
}

fn fit_path(path: &str, max: usize) -> String {
    let width = unicode_width::UnicodeWidthStr::width(path);
    if width <= max {
        return path.to_string();
    }
    if max <= 1 {
        return String::new();
    }
    let mut kept = String::new();
    let mut kept_width = 0usize;
    for c in path.chars().rev() {
        let cw = unicode_width::UnicodeWidthChar::width(c).unwrap_or(0);
        if kept_width + cw > max.saturating_sub(1) {
            break;
        }
        kept.push(c);
        kept_width += cw;
    }
    let tail = kept.chars().rev().collect::<String>();
    format!("…{tail}")
}

fn char_index_to_byte(text: &str, char_index: usize) -> usize {
    text.char_indices()
        .nth(char_index)
        .map(|(byte, _)| byte)
        .unwrap_or(text.len())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frontend::state::{DiffLine, ToolCallMsg, ToolStatus, UiState};

    fn row_text(row: &Line<'static>) -> String {
        row.spans.iter().map(|span| span.content.as_ref()).collect()
    }

    #[test]
    fn diff_line_renders_aligned_numbers() {
        let lines = vec![
            DiffLine { kind: ' ', line_no: Some(21), text: "def verify():".into() },
            DiffLine { kind: '+', line_no: Some(22), text: "if expired():".into() },
            DiffLine { kind: '-', line_no: Some(22), text: "if none:".into() },
        ];
        assert_eq!(
            row_text(&render_diff_line(&lines[0], &lines)),
            "     21  def verify():"
        );
        assert_eq!(
            row_text(&render_diff_line(&lines[1], &lines)),
            "  +  22  if expired():"
        );
        assert_eq!(
            row_text(&render_diff_line(&lines[2], &lines)),
            "  −  22  if none:"
        );
    }

    #[test]
    fn diff_folds_and_skips_source_lines_in_output() {
        assert!(is_diff_source_line("diff --git a/x b/x"));
        assert!(is_diff_source_line("@@ -1 +1 @@"));
        assert!(is_diff_source_line("+added"));
        assert!(is_diff_source_line("-removed"));
        assert!(is_diff_source_line(" context"));
        assert!(!is_diff_source_line("normal output"));
        assert!(!is_diff_source_line("2 spaces is context in diff"));

        let mut ui = UiState::new();
        let diff: Vec<DiffLine> = (0..10)
            .map(|i| DiffLine {
                kind: '+',
                line_no: Some(i),
                text: format!("line {i}"),
            })
            .collect();
        ui.messages.push(UiMessage::ToolCall(ToolCallMsg {
            tool_call_id: "1".into(),
            name: "view_diff".into(),
            arguments: "{}".into(),
            status: ToolStatus::Done,
            summary: String::new(),
            output: "diff --git a/x b/x\n+line 0\n+line 1\n".into(),
            diff,
            elapsed: Some(0.1),
            started: std::time::Instant::now(),
            verdict: None,
        }));
        let rows = render_rows(&ui, 80);
        let text: String = rows
            .iter()
            .map(|row| {
                row.spans
                    .iter()
                    .map(|span| span.content.as_ref())
                    .collect::<String>()
            })
            .collect::<Vec<_>>()
            .join("\n");
        assert!(
            text.contains("more diff lines · Tab to expand"),
            "diff block must fold: {text}"
        );
        assert!(
            !text.contains("│ diff --git"),
            "diff source lines must not repeat in the output fold: {text}"
        );
        ui.fold_expanded = true;
        let rows = render_rows(&ui, 80);
        let text: String = rows
            .iter()
            .map(|row| {
                row.spans
                    .iter()
                    .map(|span| span.content.as_ref())
                    .collect::<String>()
            })
            .collect::<Vec<_>>()
            .join("\n");
        assert!(
            !text.contains("more diff lines"),
            "expanded diff must show all lines: {text}"
        );
    }

    #[test]
    fn diff_line_without_number_keeps_column() {
        let lines = vec![DiffLine { kind: '+', line_no: None, text: "x".into() }];
        assert_eq!(row_text(&render_diff_line(&lines[0], &lines)), "  +      x");
    }

    #[test]
    fn fit_path_shows_full_when_fits() {
        assert_eq!(fit_path("/a/b.rs", 10), "/a/b.rs");
    }

    #[test]
    fn fit_path_truncates_tail_when_too_long() {
        assert_eq!(fit_path("/home/user/project/src/main.rs", 12), "…src/main.rs");
        assert_eq!(fit_path("/a/very/long/path/here", 4), "…ere");
    }

    #[test]
    fn fit_path_counts_display_width_not_chars() {
        let fitted = fit_path("/home/winterist/桌面/project", 16);
        assert_eq!(fitted, "…st/桌面/project");
        assert!(
            unicode_width::UnicodeWidthStr::width(fitted.as_str()) <= 16,
            "fitted path must not exceed the width budget"
        );
    }

    #[test]
    fn fit_path_returns_empty_when_no_room() {
        assert_eq!(fit_path("/a/b.rs", 1), "");
        assert_eq!(fit_path("/a/b.rs", 0), "");
    }

    #[test]
    fn command_stats_show_elapsed() {
        let call = ToolCallMsg {
            tool_call_id: "1".into(),
            name: "run_terminal_command".into(),
            arguments: "{}".into(),
            status: ToolStatus::Done,
            summary: "12 passed".into(),
            output: String::new(),
            diff: Vec::new(),
            elapsed: Some(1.8),
            started: std::time::Instant::now(),
            verdict: None,
        };
        assert_eq!(format_stats(&call), "1.8s  ok");
    }

    #[test]
    fn command_stats_show_verdict() {
        let call = ToolCallMsg {
            tool_call_id: "1".into(),
            name: "ls".into(),
            arguments: "{}".into(),
            status: ToolStatus::Done,
            summary: String::new(),
            output: String::new(),
            diff: Vec::new(),
            elapsed: Some(0.4),
            started: std::time::Instant::now(),
            verdict: Some("allowed".to_string()),
        };
        assert_eq!(format_stats(&call), "0.4s  🛡 allowed");
    }

    #[test]
    fn other_tools_stats_show_summary() {
        let call = ToolCallMsg {
            tool_call_id: "1".into(),
            name: "edit_existing_file".into(),
            arguments: r#"{"filepath": "a.rs"}"#.into(),
            status: ToolStatus::Done,
            summary: "applied".into(),
            output: String::new(),
            diff: Vec::new(),
            elapsed: Some(0.4),
            started: std::time::Instant::now(),
            verdict: None,
        };
        assert_eq!(format_stats(&call), "0.4s  applied");
    }

    #[test]
    fn user_header_right_aligns_time() {
        let mut ui = UiState::new();
        ui.push_user("hi");
        let rows = render_rows(&ui, 60);
        let text = row_text(&rows[0]);
        assert_eq!(text.chars().count(), 60);
        assert!(text.starts_with("❯ You"), "{text}");
        assert!(text.ends_with(":00") || text.contains(':'));

    }
    #[test]
    fn table_renders_aligned_columns() {
        let lines = crate::frontend::markdown::parse(
            "| 工具 | 结果 |\n|------|:----:|\n| ls | ✅ |\n| read_file | ✗ |",
        );
        let table = lines[0].table.as_ref().unwrap();
        let rows = render_table(table, 80);
        assert_eq!(rows.len(), 6, "top and bottom borders must render");
        let text: Vec<String> = rows
            .iter()
            .map(|row| {
                row.spans
                    .iter()
                    .map(|s| s.content.as_ref())
                    .collect::<String>()
            })
            .collect();
        assert!(text[0].contains("┌"), "top border: {:?}", text);
        assert!(text[1].contains("工具"), "{:?}", text);
        assert!(text[2].contains("├"), "header separator: {:?}", text);
        assert!(text[3].contains("✅"), "{:?}", text);
        assert!(text[4].contains("read_file"), "{:?}", text);
        assert!(text[5].contains("└"), "bottom border: {:?}", text);
        let col_start_3 = text[3].find("ls").unwrap();
        let col_start_4 = text[4].find("read_file").unwrap();
        assert_eq!(col_start_3, col_start_4, "columns must align: {:?}", text);
    }
}
