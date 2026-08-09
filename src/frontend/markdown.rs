#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpanKind {
    Normal,
    Bold,
    Italic,
    Dim,
    Code,
    Strike,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Span {
    pub text: String,
    pub kind: SpanKind,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TableRow {
    pub header: bool,
    pub separator: bool,
    pub align: Vec<Align>,
    pub cells: Vec<Vec<Span>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Align {
    Left,
    Center,
    Right,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Line {
    pub spans: Vec<Span>,
    pub code_block: bool,
    pub heading: bool,
    pub list_item: bool,
    pub table: Option<Vec<TableRow>>,
}

pub fn parse(text: &str) -> Vec<Line> {
    let mut lines = Vec::new();
    let mut in_code_block = false;
    let mut code_buf = Vec::new();
    let raw_lines: Vec<&str> = text.split('\n').collect();
    let mut i = 0;
    while i < raw_lines.len() {
        let raw = raw_lines[i];
        let trimmed = raw.trim();
        if let Some(lang) = trimmed.strip_prefix("```") {
            if in_code_block {
                if !code_buf.is_empty() {
                    lines.push(Line {
                        spans: vec![Span {
                            text: code_buf.join("\n"),
                            kind: SpanKind::Code,
                        }],
                        code_block: true,
                        heading: false,
                        list_item: false,
                        table: None,
                    });
                }
                code_buf.clear();
                in_code_block = false;
            } else {
                let _ = lang;
                in_code_block = true;
            }
            i += 1;
            continue;
        }
        if in_code_block {
            code_buf.push(raw.to_string());
            i += 1;
            continue;
        }
        if trimmed.starts_with('#') {
            let heading = trimmed.trim_start_matches('#').trim();
            if !heading.is_empty() {
                lines.push(Line {
                    spans: vec![Span {
                        text: heading.to_string(),
                        kind: SpanKind::Bold,
                    }],
                    code_block: false,
                    heading: true,
                    list_item: false,
                    table: None,
                });
                i += 1;
                continue;
            }
        }
        if is_table_row(trimmed) && i + 1 < raw_lines.len() && is_separator_row(raw_lines[i + 1].trim()) {
            let mut rows = Vec::new();
            let header_cells = split_table_row(trimmed);
            let align = parse_align(raw_lines[i + 1].trim(), header_cells.len());
            rows.push(TableRow {
                header: true,
                separator: false,
                align: align.clone(),
                cells: header_cells,
            });
            rows.push(TableRow {
                header: false,
                separator: true,
                align: align.clone(),
                cells: Vec::new(),
            });
            i += 2;
            while i < raw_lines.len() && is_table_row(raw_lines[i].trim()) {
                rows.push(TableRow {
                    header: false,
                    separator: false,
                    align: align.clone(),
                    cells: split_table_row(raw_lines[i].trim()),
                });
                i += 1;
            }
            lines.push(Line {
                spans: Vec::new(),
                code_block: false,
                heading: false,
                list_item: false,
                table: Some(rows),
            });
            continue;
        }
        let list_item = trimmed.starts_with("- ") || trimmed.starts_with("* ");
        let body = if list_item { &trimmed[2..] } else { trimmed };
        lines.push(Line {
            spans: parse_inline(body),
            code_block: false,
            heading: false,
            list_item,
            table: None,
        });
        i += 1;
    }
    if in_code_block && !code_buf.is_empty() {
        lines.push(Line {
            spans: vec![Span {
                text: code_buf.join("\n"),
                kind: SpanKind::Code,
            }],
            code_block: true,
            heading: false,
            list_item: false,
            table: None,
        });
    }
    lines
}

fn is_table_row(trimmed: &str) -> bool {
    trimmed.starts_with('|') && trimmed.contains('|') && trimmed.len() > 2
}

fn is_separator_row(trimmed: &str) -> bool {
    if !trimmed.starts_with('|') {
        return false;
    }
    let inner = trimmed.trim_start_matches('|').trim_end_matches('|');
    !inner.is_empty()
        && inner
            .split('|')
            .all(|cell| {
                let cell = cell.trim();
                !cell.is_empty() && cell.chars().all(|c| c == '-' || c == ':' || c == ' ')
            })
}

fn split_table_row(trimmed: &str) -> Vec<Vec<Span>> {
    let inner = trimmed.trim_start_matches('|').trim_end_matches('|');
    inner
        .split('|')
        .map(|cell| parse_inline(cell.trim()))
        .collect()
}

fn parse_align(separator: &str, columns: usize) -> Vec<Align> {
    let inner = separator.trim_start_matches('|').trim_end_matches('|');
    let mut aligns: Vec<Align> = inner
        .split('|')
        .map(|cell| {
            let cell = cell.trim();
            let left = cell.starts_with(':');
            let right = cell.ends_with(':');
            if left && right {
                Align::Center
            } else if right {
                Align::Right
            } else {
                Align::Left
            }
        })
        .collect();
    while aligns.len() < columns {
        aligns.push(Align::Left);
    }
    aligns.truncate(columns);
    aligns
}

fn parse_inline(text: &str) -> Vec<Span> {
    let mut spans = Vec::new();
    let mut current = String::new();
    let mut kind = SpanKind::Normal;
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '`' => {
                flush(&mut spans, &mut current, kind);
                let mut code = String::new();
                while let Some(&next) = chars.peek() {
                    if next == '`' {
                        chars.next();
                        break;
                    }
                    code.push(chars.next().unwrap());
                }
                spans.push(Span {
                    text: code,
                    kind: SpanKind::Code,
                });
                kind = SpanKind::Normal;
            }
            '*' => {
                if chars.peek() == Some(&'*') {
                    chars.next();
                    flush(&mut spans, &mut current, kind);
                    kind = toggle(kind, SpanKind::Bold);
                } else {
                    flush(&mut spans, &mut current, kind);
                    kind = toggle(kind, SpanKind::Italic);
                }
            }
            '~' => {
                if chars.peek() == Some(&'~') {
                    chars.next();
                    flush(&mut spans, &mut current, kind);
                    kind = toggle(kind, SpanKind::Strike);
                } else {
                    current.push('~');
                }
            }
            _ => current.push(c),
        }
    }
    flush(&mut spans, &mut current, kind);
    spans
}

fn toggle(current: SpanKind, target: SpanKind) -> SpanKind {
    if current == target {
        SpanKind::Normal
    } else {
        target
    }
}

fn flush(spans: &mut Vec<Span>, current: &mut String, kind: SpanKind) {
    if !current.is_empty() {
        spans.push(Span {
            text: std::mem::take(current),
            kind,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_text_is_normal() {
        let lines = parse("hello world");
        assert_eq!(lines.len(), 1);
        assert_eq!(lines[0].spans[0].kind, SpanKind::Normal);
        assert_eq!(lines[0].spans[0].text, "hello world");
    }

    #[test]
    fn bold_and_inline_code() {
        let lines = parse("use **rust** and `cargo`");
        assert_eq!(lines.len(), 1);
        let kinds: Vec<SpanKind> = lines[0].spans.iter().map(|s| s.kind).collect();
        assert!(kinds.contains(&SpanKind::Bold));
        assert!(kinds.contains(&SpanKind::Code));
    }

    #[test]
    fn code_block_is_single_line() {
        let lines = parse("before\n```rust\nfn main() {}\n```\nafter");
        assert_eq!(lines.len(), 3);
        assert!(lines[1].code_block);
        assert_eq!(lines[1].spans[0].text, "fn main() {}");
        assert_eq!(lines[1].spans[0].kind, SpanKind::Code);
    }

    #[test]
    fn heading_and_list() {
        let lines = parse("# Title\n- item one\n- item two");
        assert_eq!(lines.len(), 3);
        assert!(lines[0].heading);
        assert!(lines[1].list_item);
        assert!(lines[2].list_item);
        assert_eq!(lines[1].spans[0].text, "item one");
    }

    #[test]
    fn strike_through() {
        let lines = parse("~~removed~~ kept");
        assert_eq!(lines[0].spans[0].kind, SpanKind::Strike);
        assert_eq!(lines[0].spans[0].text, "removed");
        assert_eq!(lines[0].spans[1].text, " kept");
    }

    #[test]
    fn empty_code_block_then_normal() {
        let lines = parse("```\n```\ntext");
        assert_eq!(lines.len(), 1);
        assert!(!lines[0].code_block);
        assert_eq!(lines[0].spans[0].text, "text");
    }
    #[test]
    fn table_parses_header_separator_and_rows() {
        let lines = parse("| 工具 | 结果 |\n|------|------|\n| `ls` | ✅ |\n| read | ✗ |");
        assert_eq!(lines.len(), 1, "table must collapse into one line");
        let table = lines[0].table.as_ref().expect("table block");
        assert_eq!(table.len(), 4);
        assert!(table[0].header);
        assert!(table[1].separator);
        assert!(!table[2].header);
        assert_eq!(table[2].cells.len(), 2);
        assert_eq!(table[2].cells[0][0].kind, SpanKind::Code);
        assert_eq!(table[3].cells[1][0].text, "✗");
    }

    #[test]
    fn table_align_detected_from_separator() {
        let lines = parse("| a | b | c |\n|:--|:-:|--:|\n| 1 | 2 | 3 |");
        let table = lines[0].table.as_ref().unwrap();
        assert_eq!(table[0].align, vec![Align::Left, Align::Center, Align::Right]);
    }

    #[test]
    fn plain_pipe_lines_are_not_tables() {
        let lines = parse("no | pipe here");
        assert!(lines[0].table.is_none());
        let lines = parse("| single");
        assert!(lines[0].table.is_none());
    }
}
