use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::time::{Duration, Instant};

use regex::{Regex, RegexBuilder};
use serde_json::Value;

const BROWSER_UA: &str = "Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0";
const GLOB_MAX_RESULTS: usize = 100;
const LS_MAX_LINES: usize = 200;
const GREP_MAX_RESULTS: usize = 100;
const GREP_MAX_CHARS: usize = 7500;
const DIFF_MAX_LINES: usize = 5000;
const FETCH_MAX_CHARS: usize = 20_000;
const SEARCH_MAX_CHARS: usize = 8000;
const COMMAND_TIMEOUT: Duration = Duration::from_secs(120);

const IGNORED_DIRS: &[&str] = &[
    ".git", ".svn", "node_modules", "dist", "build", "Build", "target", "out", "bin",
    ".pytest_cache", ".vscode-test", "__pycache__", "site-packages", ".gradle", ".mvn",
    ".cache", "gems", "vendor", ".venv", "venv", ".vscode", ".idea", ".vs", ".prognosis",
];

const IGNORED_FILETYPES: &[&str] = &[
    "*.png", "*.jpg", "*.jpeg", "*.gif", "*.mp4", "*.svg", "*.ico", "*.pdf", "*.zip", "*.gz",
    "*.tar", "*.dmg", "*.tgz", "*.rar", "*.7z", "*.exe", "*.dll", "*.obj", "*.o", "*.o.d", "*.a",
    "*.lib", "*.so", "*.dylib", "*.ncb", "*.sdf", "*.woff", "*.woff2", "*.eot", "*.cur", "*.avi",
    "*.mpg", "*.mpeg", "*.mov", "*.mp3", "*.mkv", "*.webm", "*.jar", "*.onnx", "*.parquet",
    "*.pqt", "*.wav", "*.webp", "*.wasm", "*.plist", "*.profraw", "*.gcda", "*.gcno", "go.sum",
    "*.gitignore", "*.gitkeep", "*.csv", "*.uasset", "*.pdb", "*.bin", "*.pag", "*.swp",
    "*.jsonl",
];

pub fn resolve_path(project_dir: &Path, requested: &str) -> Result<PathBuf, String> {
    let root = project_dir
        .canonicalize()
        .map_err(|err| format!("cannot resolve project dir: {err}"))?;
    let expanded = expand_requested(requested);
    let candidate = if Path::new(&expanded).is_absolute() {
        PathBuf::from(&expanded)
    } else {
        root.join(&expanded)
    };
    let file_name = candidate
        .file_name()
        .ok_or_else(|| format!("invalid path: {requested}"))?;
    let parent = match candidate.parent() {
        Some(parent) if !parent.as_os_str().is_empty() => parent.to_path_buf(),
        _ => root.clone(),
    };
    let canonical_parent = parent
        .canonicalize()
        .map_err(|err| format!("cannot access {requested}: {err}"))?;
    let resolved = canonical_parent.join(file_name);
    if resolved.starts_with(&root) {
        Ok(resolved)
    } else {
        Err(format!("path outside project directory: {requested}"))
    }
}

fn expand_requested(requested: &str) -> String {
    if let Some(rest) = requested.strip_prefix("file://") {
        let decoded = percent_decode(rest);
        if decoded.starts_with('/') {
            decoded
        } else {
            format!("/{decoded}")
        }
    } else if let Some(rest) = requested.strip_prefix("~/") {
        let home = std::env::var("HOME").unwrap_or_default();
        if home.is_empty() {
            requested.to_string()
        } else {
            format!("{home}/{rest}")
        }
    } else {
        requested.to_string()
    }
}

fn percent_decode(input: &str) -> String {
    let bytes = input.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len()
            && let (Some(hi), Some(lo)) = (hex_val(bytes[i + 1]), hex_val(bytes[i + 2])) {
                out.push(hi << 4 | lo);
                i += 3;
                continue;
            }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn hex_val(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

pub fn read_file(project_dir: &Path, filepath: &str) -> Result<String, String> {
    let file = resolve_path(project_dir, filepath)?;
    if !file.is_file() {
        return Err(format!(
            "File \"{filepath}\" does not exist or is not accessible. You might want to check the path and try again."
        ));
    }
    std::fs::read_to_string(&file).map_err(|_| {
        format!(
            "File \"{filepath}\" does not exist or is not accessible. You might want to check the path and try again."
        )
    })
}

fn unified_diff(original: &str, updated: &str) -> String {
    let a: Vec<&str> = original.lines().collect();
    let b: Vec<&str> = updated.lines().collect();
    let (n, m) = (a.len(), b.len());
    let mut dp = vec![vec![0usize; m + 1]; n + 1];
    for i in (0..n).rev() {
        for j in (0..m).rev() {
            dp[i][j] = if a[i] == b[j] {
                dp[i + 1][j + 1] + 1
            } else {
                dp[i + 1][j].max(dp[i][j + 1])
            };
        }
    }
    let mut lines = Vec::new();
    let (mut i, mut j) = (0, 0);
    while i < n && j < m {
        if a[i] == b[j] {
            lines.push(format!(" {}", a[i]));
            i += 1;
            j += 1;
        } else if dp[i + 1][j] >= dp[i][j + 1] {
            lines.push(format!("-{}", a[i]));
            i += 1;
        } else {
            lines.push(format!("+{}", b[j]));
            j += 1;
        }
    }
    while i < n {
        lines.push(format!("-{}", a[i]));
        i += 1;
    }
    while j < m {
        lines.push(format!("+{}", b[j]));
        j += 1;
    }
    format!("@@ -{n} +{m} @@\n{}", lines.join("\n"))
}

pub fn create_new_file(
    project_dir: &Path,
    filepath: &str,
    contents: &str,
) -> Result<String, String> {
    let file = resolve_path(project_dir, filepath)?;
    if file.exists() {
        return Err(format!(
            "File {filepath} already exists. Use the edit tool to edit this file"
        ));
    }
    std::fs::write(&file, contents).map_err(|err| format!("write failed: {err}"))?;
    Ok(format!(
        "File created successfully\n{}",
        unified_diff("", contents)
    ))
}

pub fn run_terminal_command(project_dir: &Path, command: &str, wait: bool) -> String {
    let mut child = match Command::new("sh")
        .arg("-c")
        .arg(command)
        .current_dir(project_dir)
        .env("FORCE_COLOR", "1")
        .env("TERM", "xterm-256color")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(child) => child,
        Err(err) => return format!("command failed: {err}"),
    };
    if !wait {
        return "Command is running in the background...".to_string();
    }
    let mut stdout = child.stdout.take().unwrap();
    let mut stderr = child.stderr.take().unwrap();
    let start = Instant::now();
    let (tx, rx) = mpsc::channel();
    let tx_out = tx.clone();
    let reader = std::thread::spawn(move || {
        let mut buf = Vec::new();
        let _ = stdout.read_to_end(&mut buf);
        let _ = tx_out.send((0, buf));
    });
    let tx_err = tx;
    let reader_err = std::thread::spawn(move || {
        let mut buf = Vec::new();
        let _ = stderr.read_to_end(&mut buf);
        let _ = tx_err.send((1, buf));
    });
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) => {}
            Err(_) => break child.wait().unwrap_or_default(),
        }
        if start.elapsed() >= COMMAND_TIMEOUT {
            let _ = child.kill();
            let _ = child.wait();
            let _ = reader.join();
            let _ = reader_err.join();
            let mut collected = String::new();
            while let Ok((kind, buf)) = rx.recv_timeout(Duration::from_millis(100)) {
                let text = String::from_utf8_lossy(&buf);
                if kind == 1 && !text.is_empty() {
                    collected.push_str(&format!("\n[stderr] {text}"));
                } else {
                    collected.push_str(&text);
                }
            }
            return format!("\n[Timeout: process killed after 2 minutes]\n{collected}");
        }
        std::thread::sleep(Duration::from_millis(50));
    };
    let mut text = String::new();
    while let Ok((kind, buf)) = rx.recv_timeout(Duration::from_secs(1)) {
        let chunk = String::from_utf8_lossy(&buf);
        if kind == 1 && !chunk.is_empty() {
            text.push_str(&format!("\n[stderr] {chunk}"));
        } else {
            text.push_str(&chunk);
        }
    }
    let _ = reader.join();
    let _ = reader_err.join();
    if text.trim().is_empty() {
        format!("(exit {})", status.code().unwrap_or(-1))
    } else {
        text
    }
}

pub fn file_glob_search(project_dir: &Path, pattern: &str) -> String {
    if pattern.trim().is_empty() {
        return "empty search pattern".to_string();
    }
    let matcher = match glob_matcher(pattern) {
        Ok(matcher) => matcher,
        Err(err) => return format!("invalid glob pattern: {err}"),
    };
    let mut results: Vec<String> = Vec::new();
    walk_files(project_dir, "", &mut |rel, _| {
        if matcher.matches(rel) {
            results.push(rel.to_string());
        }
        results.len() < GLOB_MAX_RESULTS
    });
    if results.is_empty() {
        return "The glob search returned no results.".to_string();
    }
    let mut out = results.join("\n");
    if results.len() == GLOB_MAX_RESULTS {
        out.push_str(&format!(
            "\n\nWarning: the results above were truncated to the first {GLOB_MAX_RESULTS} files. If the results are not satisfactory, refine your search pattern"
        ));
    }
    out
}

struct GlobMatcher {
    full: Regex,
    basename: Option<Regex>,
}

impl GlobMatcher {
    fn matches(&self, rel_path: &str) -> bool {
        match &self.basename {
            Some(base) => {
                let name = rel_path.rsplit('/').next().unwrap_or(rel_path);
                base.is_match(name)
            }
            None => self.full.is_match(rel_path),
        }
    }
}

fn glob_matcher(pattern: &str) -> Result<GlobMatcher, String> {
    let regex_source = glob_to_regex(pattern)?;
    let full = Regex::new(&format!("^{regex_source}$"))
        .map_err(|err| format!("invalid glob pattern: {err}"))?;
    if pattern.contains('/') {
        Ok(GlobMatcher {
            full,
            basename: None,
        })
    } else {
        Ok(GlobMatcher {
            basename: Some(full.clone()),
            full,
        })
    }
}

fn glob_to_regex(pattern: &str) -> Result<String, String> {
    let chars: Vec<char> = pattern.chars().collect();
    let mut re = String::new();
    let mut i = 0;
    while i < chars.len() {
        match chars[i] {
            '*' => {
                if i + 1 < chars.len() && chars[i + 1] == '*' {
                    if i + 2 < chars.len() && chars[i + 2] == '/' {
                        re.push_str("(?:[^/]*/)*");
                        i += 3;
                        continue;
                    }
                    re.push_str(".*");
                    i += 2;
                } else {
                    re.push_str("[^/]*");
                    i += 1;
                }
            }
            '?' => {
                re.push_str("[^/]");
                i += 1;
            }
            c => {
                re.push_str(&regex::escape(&c.to_string()));
                i += 1;
            }
        }
    }
    Ok(re)
}

fn walk_files(project_dir: &Path, prefix: &str, visit: &mut dyn FnMut(&str, &str) -> bool) {
    let dir = if prefix.is_empty() {
        project_dir.to_path_buf()
    } else {
        project_dir.join(prefix)
    };
    let Ok(entries) = std::fs::read_dir(&dir) else {
        return;
    };
    let mut entries: Vec<_> = entries.filter_map(|entry| entry.ok()).collect();
    entries.sort_by_key(|entry| entry.file_name());
    for entry in entries {
        let name = entry.file_name().to_string_lossy().to_string();
        if entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
            if IGNORED_DIRS.contains(&name.as_str()) {
                continue;
            }
            let rel = if prefix.is_empty() {
                name.clone()
            } else {
                format!("{prefix}/{name}")
            };
            if !visit(&rel, "") {
                return;
            }
            walk_files(project_dir, &rel, visit);
        } else if entry.file_type().map(|t| t.is_file()).unwrap_or(false) {
            if ignored_file_name(&name) {
                continue;
            }
            let rel = if prefix.is_empty() {
                name
            } else {
                format!("{prefix}/{name}")
            };
            let content = std::fs::read_to_string(entry.path()).unwrap_or_default();
            if !visit(&rel, &content) {
                return;
            }
        }
    }
}

fn ignored_file_name(name: &str) -> bool {
    IGNORED_FILETYPES.iter().any(|pat| {
        if let Some(ext) = pat.strip_prefix("*.") {
            name.ends_with(&format!(".{ext}"))
        } else {
            *pat == name
        }
    })
}

pub fn grep_search(project_dir: &Path, query: &str) -> String {
    let re = match RegexBuilder::new(query).case_insensitive(true).build() {
        Ok(re) => re,
        Err(err) => {
            return format!(
                "The search failed due to an invalid regex pattern.\n\nOriginal query: {query}\n\nError: {err}\n\nTip: If you're searching for literal text with special characters, the query was automatically escaped. If you need regex patterns, ensure they use proper regex syntax."
            );
        }
    };
    let mut out = String::new();
    let mut hits = 0usize;
    let mut hit_limit = false;
    walk_files(project_dir, "", &mut |rel, content| {
        let lines: Vec<&str> = content.lines().collect();
        let matched: Vec<usize> = lines
            .iter()
            .enumerate()
            .filter(|(_, line)| re.is_match(line))
            .map(|(idx, _)| idx)
            .collect();
        if matched.is_empty() {
            return true;
        }
        if !out.is_empty() {
            out.push_str("\n--\n");
        }
        out.push_str(&format!("./{rel}"));
        let mut printed = Vec::new();
        for idx in matched {
            let start = idx.saturating_sub(2);
            let end = (idx + 2).min(lines.len() - 1);
            for (offset, line) in lines[start..=end].iter().enumerate() {
                let ctx = start + offset;
                if !printed.contains(&ctx) {
                    printed.push(ctx);
                    out.push_str(&format!("\n  {line}"));
                }
            }
            hits += 1;
            if hits >= GREP_MAX_RESULTS || out.len() >= GREP_MAX_CHARS {
                hit_limit = true;
                return false;
            }
        }
        true
    });
    if hits == 0 {
        return "The search returned no results.".to_string();
    }
    if hit_limit {
        let reasons = if hits >= GREP_MAX_RESULTS {
            format!("the number of results exceeded {GREP_MAX_RESULTS}")
        } else {
            format!("the number of characters exceeded {GREP_MAX_CHARS}")
        };
        out.push_str(&format!(
            "\n\nThe above search results were truncated because {reasons}. If the results are not satisfactory, try refining your search query."
        ));
    }
    out
}

pub fn view_diff(project_dir: &Path) -> String {
    let mut parts = Vec::new();
    for args in [&["diff"][..], &["diff", "--cached"][..]] {
        match Command::new("git")
            .args(args)
            .current_dir(project_dir)
            .output()
        {
            Ok(output) if output.status.success() => {
                let text = String::from_utf8_lossy(&output.stdout);
                if !text.trim().is_empty() {
                    parts.push(text.trim_end().to_string());
                }
            }
            Ok(output) => {
                let stderr = String::from_utf8_lossy(&output.stderr);
                return format!(
                    "git diff failed: {}",
                    stderr.trim().lines().next().unwrap_or("unknown error")
                );
            }
            Err(err) => {
                if err.kind() == std::io::ErrorKind::NotFound {
                    return "git is not installed or not in PATH".to_string();
                }
                return format!("git diff failed: {err}");
            }
        }
    }
    let combined = parts.join("\n");
    if combined.trim().is_empty() {
        return "The current diff is empty".to_string();
    }
    let lines: Vec<&str> = combined.lines().collect();
    if lines.len() > DIFF_MAX_LINES {
        return format!(
            "{}\n\nThe git diff was truncated because it exceeded {DIFF_MAX_LINES} lines. Consider viewing specific files or focusing on smaller changes.",
            lines[..DIFF_MAX_LINES].join("\n")
        );
    }
    combined
}

pub fn ls_dir(project_dir: &Path, dir_path: Option<&str>, recursive: bool) -> Result<String, String> {
    let requested = dir_path.unwrap_or(".");
    let dir = resolve_path(project_dir, requested)?;
    if !dir.is_dir() {
        return Err(format!(
            "Directory {requested} not found or is not accessible. You can use absolute paths, relative paths, or paths starting with ~"
        ));
    }
    let mut entries: Vec<String> = Vec::new();
    if recursive {
        walk_all(project_dir, &dir, "", &mut entries);
    } else {
        let mut items: Vec<_> = std::fs::read_dir(&dir)
            .map_err(|err| format!("cannot read {requested}: {err}"))?
            .filter_map(|entry| entry.ok())
            .collect();
        items.sort_by_key(|entry| entry.file_name());
        for entry in items {
            let name = entry.file_name().to_string_lossy().to_string();
            let is_dir = entry.file_type().map(|t| t.is_dir()).unwrap_or(false);
            if is_dir {
                entries.push(format!("{name}/"));
            } else {
                entries.push(name);
            }
        }
    }
    let total = entries.len();
    entries.truncate(LS_MAX_LINES);
    if entries.is_empty() {
        return Ok(format!("No files/folders found in {}", dir.display()));
    }
    let mut out = entries.join("\n");
    if total > LS_MAX_LINES {
        let mut warning = format!("{total} ls entries were truncated");
        if recursive {
            warning.push_str(". Try using a non-recursive search");
        }
        out.push_str(&format!("\n\n{warning}"));
    }
    Ok(out)
}

fn walk_all(_project_dir: &Path, dir: &Path, prefix: &str, out: &mut Vec<String>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    let mut entries: Vec<_> = entries.filter_map(|entry| entry.ok()).collect();
    entries.sort_by_key(|entry| entry.file_name());
    for entry in entries {
        let name = entry.file_name().to_string_lossy().to_string();
        let rel = if prefix.is_empty() {
            name.clone()
        } else {
            format!("{prefix}/{name}")
        };
        if entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
            out.push(format!("{rel}/"));
            walk_all(_project_dir, &entry.path(), &rel, out);
        } else {
            out.push(rel);
        }
    }
}

pub struct RuleSpec<'a> {
    pub name: &'a str,
    pub rule: &'a str,
    pub description: Option<&'a str>,
    pub globs: Option<&'a str>,
    pub regex: Option<&'a str>,
    pub always_apply: Option<bool>,
    pub scope: Option<&'a str>,
}

pub fn create_rule_block(project_dir: &Path, spec: RuleSpec) -> Result<String, String> {
    let safe = sanitize_rule_name(spec.name);
    if safe.is_empty() {
        return Err("rule name must contain at least one alphanumeric character".to_string());
    }
    let mut frontmatter = Vec::new();
    if let Some(globs) = spec.globs
        && !globs.trim().is_empty() {
            frontmatter.push(format!("globs: {}", yaml_str(globs.trim())));
        }
    if let Some(regex) = spec.regex
        && !regex.trim().is_empty() {
            frontmatter.push(format!("regex: {}", yaml_str(regex.trim())));
        }
    if let Some(description) = spec.description
        && !description.trim().is_empty() {
            frontmatter.push(format!("description: {}", yaml_str(description.trim())));
        }
    if let Some(always_apply) = spec.always_apply {
        frontmatter.push(format!("alwaysApply: {always_apply}"));
    }
    let dir = if spec.scope == Some("global") {
        global_config_dir().join("rules")
    } else {
        project_dir.join(".prognosis").join("rules")
    };
    std::fs::create_dir_all(&dir).map_err(|err| format!("cannot create rules dir: {err}"))?;
    let content = format!("---\n{}\n---\n\n{}", frontmatter.join("\n"), spec.rule);
    std::fs::write(dir.join(format!("{safe}.md")), content)
        .map_err(|err| format!("write failed: {err}"))?;
    Ok(if spec.scope == Some("global") {
        "Rule created successfully (global)".to_string()
    } else {
        "Rule created successfully".to_string()
    })
}

pub fn global_config_dir() -> std::path::PathBuf {
    if let Some(dir) = std::env::var_os("PROGNOSIS_CONFIG_DIR") {
        return std::path::PathBuf::from(dir);
    }
    std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(|home| std::path::PathBuf::from(home).join(".prognosis"))
        .unwrap_or_default()
}

fn skills_global_dirs() -> Vec<std::path::PathBuf> {
    let home = std::env::var_os("PROGNOSIS_SKILLS_HOME")
        .or_else(|| std::env::var_os("HOME"))
        .or_else(|| std::env::var_os("USERPROFILE"));
    let Some(home) = home else {
        return Vec::new();
    };
    let home = std::path::PathBuf::from(home);
    vec![
        home.join(".agents").join("skills"),
        home.join(".config").join("agents").join("skills"),
    ]
}

fn sanitize_rule_name(name: &str) -> String {
    let mut out = String::new();
    let mut prev_dash = false;
    for c in name.to_lowercase().chars() {
        if c.is_ascii_alphanumeric() || c == ' ' || c == '-' {
            let ch = if c == ' ' { '-' } else { c };
            if ch == '-' && prev_dash {
                continue;
            }
            out.push(ch);
            prev_dash = ch == '-';
        } else {
            if !prev_dash {
                out.push('-');
                prev_dash = true;
            }
        }
    }
    out.trim_matches('-').to_string()
}

fn yaml_str(value: &str) -> String {
    let escaped = value.replace('\\', "\\\\").replace('"', "\\\"");
    format!("\"{escaped}\"")
}

pub fn request_rule(project_dir: &Path, name: &str) -> Result<String, String> {
    let file = find_rule_file(project_dir, name)?;
    std::fs::read_to_string(&file)
        .map_err(|err| format!("cannot read rule {name}: {err}"))
}

fn find_rule_file(project_dir: &Path, name: &str) -> Result<PathBuf, String> {
    let wanted = sanitize_rule_name(name);
    for dir in [
        project_dir.join(".prognosis").join("rules"),
        global_config_dir().join("rules"),
    ] {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let file_name = entry.file_name().to_string_lossy().to_string();
            let stem = file_name.strip_suffix(".md").unwrap_or(&file_name);
            if stem == name || stem == wanted {
                return Ok(entry.path());
            }
        }
    }
    Err(format!(
        "Rule with name \"{name}\" not found or has no file path"
    ))
}

pub fn load_rules(project_dir: &Path) -> Vec<crate::runtime::types::RuleContext> {
    let disabled = disabled_names(project_dir, "rules");
    load_rules_all(project_dir)
        .into_iter()
        .filter(|rule| !disabled.iter().any(|d| d == &rule.name))
        .collect()
}

pub fn load_rules_all(project_dir: &Path) -> Vec<crate::runtime::types::RuleContext> {
    let mut rules: Vec<crate::runtime::types::RuleContext> = Vec::new();
    for dir in [
        global_config_dir().join("rules"),
        project_dir.join(".prognosis").join("rules"),
    ] {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let file_name = entry.file_name().to_string_lossy().to_string();
            let Some(name) = file_name.strip_suffix(".md") else {
                continue;
            };
            let content = std::fs::read_to_string(entry.path()).unwrap_or_default();
            let rule = crate::runtime::types::RuleContext {
                name: name.to_string(),
                rule: rule_body(&content).unwrap_or_else(|| content.clone()),
                description: frontmatter_field(&content, "description").unwrap_or_default(),
                globs: frontmatter_field(&content, "globs").unwrap_or_default(),
                regex: frontmatter_field(&content, "regex").unwrap_or_default(),
                always_apply: frontmatter_bool(&content, "alwaysApply"),
            };
            if let Some(existing) = rules.iter_mut().find(|r| r.name == rule.name) {
                *existing = rule;
            } else {
                rules.push(rule);
            }
        }
    }
    rules.sort_by(|a, b| a.name.cmp(&b.name));
    rules
}

fn disabled_names(project_dir: &Path, subdir: &str) -> Vec<String> {
    migrate_state(project_dir);
    let path = project_dir.join(".prognosis").join("state.json");
    let Ok(text) = std::fs::read_to_string(&path) else {
        return Vec::new();
    };
    serde_json::from_str::<serde_json::Value>(&text)
        .ok()
        .and_then(|value| value.get(format!("{subdir}_disabled")).cloned())
        .and_then(|value| serde_json::from_value(value).ok())
        .unwrap_or_default()
}

fn migrate_state(project_dir: &Path) {
    let path = project_dir.join(".prognosis").join("state.json");
    if path.is_file() {
        return;
    }
    let mut merged = serde_json::Map::new();
    for subdir in ["rules", "skills"] {
        let legacy = project_dir.join(".prognosis").join(subdir).join("state.json");
        let Ok(text) = std::fs::read_to_string(&legacy) else {
            continue;
        };
        let disabled: Vec<String> = serde_json::from_str::<serde_json::Value>(&text)
            .ok()
            .and_then(|value| value.get("disabled").cloned())
            .and_then(|value| serde_json::from_value(value).ok())
            .unwrap_or_default();
        if !disabled.is_empty() {
            merged.insert(format!("{subdir}_disabled"), serde_json::json!(disabled));
        }
    }
    if merged.is_empty() {
        return;
    }
    let _ = std::fs::create_dir_all(path.parent().unwrap());
    let _ = std::fs::write(
        &path,
        serde_json::to_string_pretty(&serde_json::Value::Object(merged)).unwrap_or_default(),
    );
}

pub fn is_rule_enabled(project_dir: &Path, name: &str) -> bool {
    !disabled_names(project_dir, "rules").iter().any(|d| d == name)
}

pub fn is_skill_enabled(project_dir: &Path, name: &str) -> bool {
    !disabled_names(project_dir, "skills").iter().any(|d| d == name)
}

pub fn set_rule_enabled(project_dir: &Path, name: &str, enabled: bool) {
    set_disabled(project_dir, "rules", name, enabled);
}

pub fn set_skill_enabled(project_dir: &Path, name: &str, enabled: bool) {
    set_disabled(project_dir, "skills", name, enabled);
}

fn set_disabled(project_dir: &Path, subdir: &str, name: &str, enabled: bool) {
    migrate_state(project_dir);
    let path = project_dir.join(".prognosis").join("state.json");
    let mut disabled: Vec<String> = disabled_names(project_dir, subdir);
    if enabled {
        disabled.retain(|d| d != name);
    } else if !disabled.iter().any(|d| d == name) {
        disabled.push(name.to_string());
    }
    let mut value = serde_json::from_str::<serde_json::Value>(
        &std::fs::read_to_string(&path).unwrap_or_else(|_| "{}".to_string()),
    )
    .unwrap_or(serde_json::Value::Object(serde_json::Map::new()));
    value[format!("{subdir}_disabled")] = serde_json::json!(disabled);
    let _ = std::fs::create_dir_all(path.parent().unwrap());
    let _ = std::fs::write(&path, serde_json::to_string_pretty(&value).unwrap_or_default());
}

fn rule_body(content: &str) -> Option<String> {
    let body = content.strip_prefix("---")?;
    let end = body.find("\n---")?;
    let rest = &body[end + 4..];
    Some(rest.trim_start_matches('\n').to_string())
}

fn frontmatter_bool(content: &str, field: &str) -> Option<bool> {
    frontmatter_field(content, field).and_then(|value| value.parse::<bool>().ok())
}

pub fn read_skill(project_dir: &Path, skill_name: &str) -> Result<String, String> {
    for dir in std::iter::once(project_dir.join(".agents").join("skills"))
        .chain(skills_global_dirs())
    {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let file_name = entry.file_name().to_string_lossy().to_string();
            if entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                let skill_file = path.join("SKILL.md");
                if skill_file.is_file() {
                    let content = std::fs::read_to_string(&skill_file)
                        .map_err(|err| format!("cannot read skill {skill_name}: {err}"))?;
                    let frontmatter_name = frontmatter_field(&content, "name").unwrap_or_default();
                    if frontmatter_name == skill_name || file_name == skill_name {
                        let supporting = std::fs::read_dir(&path)
                            .map(|entries| {
                                entries
                                    .flatten()
                                    .map(|e| e.file_name().to_string_lossy().to_string())
                                    .filter(|name| name != "SKILL.md")
                                    .collect::<Vec<_>>()
                            })
                            .unwrap_or_default();
                        if supporting.is_empty() {
                            return Ok(content);
                        }
                        return Ok(format!(
                            "{content}\n\n## Supporting Files\nSkill directory:\n{}\n\nUse the read file tool to access these files as needed.",
                            supporting.join("\n")
                        ));
                    }
                }
            } else {
                let stem = file_name.strip_suffix(".md").unwrap_or(&file_name);
                if stem == skill_name {
                    let content = std::fs::read_to_string(&path)
                        .map_err(|err| format!("cannot read skill {skill_name}: {err}"))?;
                    return Ok(content);
                }
            }
        }
    }
    let available = load_skills(project_dir)
        .into_iter()
        .map(|skill| skill.name)
        .collect::<Vec<_>>()
        .join(", ");
    Err(format!(
        "Skill \"{skill_name}\" not found. Available skills: {available}"
    ))
}

pub fn load_skills(project_dir: &Path) -> Vec<crate::runtime::types::SkillContext> {
    let disabled = disabled_names(project_dir, "skills");
    load_skills_all(project_dir)
        .into_iter()
        .filter(|skill| !disabled.iter().any(|d| d == &skill.name))
        .collect()
}

pub fn load_skills_all(project_dir: &Path) -> Vec<crate::runtime::types::SkillContext> {
    let mut skills: Vec<crate::runtime::types::SkillContext> = Vec::new();
    for dir in std::iter::once(project_dir.join(".agents").join("skills"))
        .chain(skills_global_dirs())
    {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let file_name = entry.file_name().to_string_lossy().to_string();
            let skill = if entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                let skill_file = path.join("SKILL.md");
                if skill_file.is_file() {
                    let content = std::fs::read_to_string(&skill_file).unwrap_or_default();
                    Some(crate::runtime::types::SkillContext {
                        name: frontmatter_field(&content, "name").unwrap_or_else(|| file_name.clone()),
                        description: frontmatter_field(&content, "description").unwrap_or_default(),
                    })
                } else {
                    None
                }
            } else if let Some(name) = file_name.strip_suffix(".md") {
                let content = std::fs::read_to_string(&path).unwrap_or_default();
                Some(crate::runtime::types::SkillContext {
                    name: frontmatter_field(&content, "name").unwrap_or_else(|| name.to_string()),
                    description: frontmatter_field(&content, "description").unwrap_or_default(),
                })
            } else {
                None
            };
            if let Some(skill) = skill
                && !skills.iter().any(|s| s.name == skill.name)
            {
                skills.push(skill);
            }
        }
    }
    skills.sort_by(|a, b| a.name.cmp(&b.name));
    skills
}

fn frontmatter_field(content: &str, field: &str) -> Option<String> {
    let body = content.strip_prefix("---")?;
    let end = body.find("\n---")?;
    let frontmatter = &body[..end];
    for line in frontmatter.lines() {
        if let Some(value) = line.strip_prefix(&format!("{field}:")) {
            let value = value.trim();
            if value == "true" || value == "false" {
                return Some(value.to_string());
            }
            return Some(value.trim_matches('"').to_string());
        }
    }
    None
}

pub fn single_find_and_replace(
    project_dir: &Path,
    filepath: &str,
    old_string: &str,
    new_string: &str,
    replace_all: bool,
) -> Result<String, String> {
    if old_string.is_empty() {
        return Err("old_string must not be empty".to_string());
    }
    if old_string == new_string {
        return Err("new_string MUST be different from old_string".to_string());
    }
    let file = resolve_path(project_dir, filepath)?;
    if !file.is_file() {
        return Err(format!(
            "File \"{filepath}\" does not exist or is not accessible. You might want to check the path and try again."
        ));
    }
    let content = std::fs::read_to_string(&file)
        .map_err(|err| format!("cannot read {filepath}: {err}"))?;
    let count = content.match_indices(old_string).count();
    if count == 0 {
        return Err(format!(
            "old_string not found in {filepath}. Read the file and provide an exact match."
        ));
    }
    if count > 1 && !replace_all {
        return Err(format!(
            "old_string is not unique in the file (appears {count} times). Provide a larger string with more surrounding context or use replace_all."
        ));
    }
    let new_content = if replace_all {
        content.replace(old_string, new_string)
    } else {
        content.replacen(old_string, new_string, 1)
    };
    let diff = unified_diff(&content, &new_content);
    std::fs::write(&file, new_content).map_err(|err| format!("write failed: {err}"))?;
    Ok(format!("Edited {filepath}\n{diff}"))
}

pub fn edit_existing_file(
    project_dir: &Path,
    filepath: &str,
    changes: &str,
) -> Result<String, String> {
    let file = resolve_path(project_dir, filepath)?;
    if !file.is_file() {
        return Err(format!(
            "File \"{filepath}\" does not exist or is not accessible. You might want to check the path and try again."
        ));
    }
    let content = std::fs::read_to_string(&file)
        .map_err(|err| format!("cannot read {filepath}: {err}"))?;
    let cleaned = strip_codeblock(changes);
    let change_lines: Vec<&str> = cleaned.lines().collect();
    if change_lines.is_empty() {
        return Err("changes must contain code to apply".to_string());
    }
    let file_lines: Vec<&str> = content.lines().collect();
    let mut positions: Vec<Option<usize>> = vec![None; change_lines.len()];
    let mut search_from = 0usize;
    for (idx, line) in change_lines.iter().enumerate() {
        if is_placeholder(line) {
            continue;
        }
        if let Some(offset) = file_lines[search_from..]
            .iter()
            .position(|fl| fl.trim_end() == line.trim_end())
        {
            positions[idx] = Some(search_from + offset);
            search_from += offset + 1;
        }
    }
    if !positions.iter().any(|p| p.is_some()) {
        return Err(
            "Could not apply the changes: none of the provided code matched the file contents. Read the file again and provide changes that match the current contents."
                .to_string(),
        );
    }
    let first_anchor = change_lines
        .iter()
        .position(|line| !is_placeholder(line) && !line.trim().is_empty());
    let last_anchor = change_lines
        .iter()
        .rposition(|line| !is_placeholder(line) && !line.trim().is_empty());
    if let (Some(first), Some(last)) = (first_anchor, last_anchor) {
        if positions[first].is_none() {
            return Err(
                "Could not apply the changes: the first line of the changes does not match the file contents. Read the file again and provide changes that match its current contents."
                    .to_string(),
            );
        }
        if positions[last].is_none() {
            return Err(
                "Could not apply the changes: the last line of the changes does not match the file contents. Read the file again and provide changes that match its current contents."
                    .to_string(),
            );
        }
    }
    let mut replacements: Vec<Option<usize>> = vec![None; change_lines.len()];
    for (idx, line) in change_lines.iter().enumerate() {
        if positions[idx].is_some() || is_placeholder(line) {
            continue;
        }
        let start = positions[..idx]
            .iter()
            .rev()
            .find_map(|p| *p)
            .map(|p| p + 1)
            .unwrap_or(0);
        let end = positions[idx + 1..]
            .iter()
            .find_map(|p| *p)
            .unwrap_or(file_lines.len());
        let mut best: Option<(usize, f64)> = None;
        for (offset, file_line) in file_lines[start..end].iter().enumerate() {
            let j = start + offset;
            if positions.contains(&Some(j)) {
                continue;
            }
            let sim = strsim::sorensen_dice(line.trim_end(), file_line.trim_end());
            if best.is_none_or(|(_, best_sim)| sim > best_sim) {
                best = Some((j, sim));
            }
        }
        if let Some((j, sim)) = best
            && sim >= 0.9 {
                replacements[idx] = Some(j);
            }
    }
    let mut new_lines: Vec<String> = Vec::with_capacity(file_lines.len() + change_lines.len());
    let mut last_consumed = 0usize;
    for (idx, line) in change_lines.iter().enumerate() {
        if is_placeholder(line) {
            continue;
        }
        if let Some(pos) = positions[idx] {
            new_lines.extend(file_lines[last_consumed..pos].iter().map(|l| (*l).to_string()));
            new_lines.push((*line).to_string());
            last_consumed = pos + 1;
        } else if let Some(rep) = replacements[idx] {
            new_lines.extend(file_lines[last_consumed..rep].iter().map(|l| (*l).to_string()));
            new_lines.push((*line).to_string());
            last_consumed = rep + 1;
        } else if let Some(target) = unique_interval_target(&positions, idx) {
            new_lines.extend(file_lines[last_consumed..target].iter().map(|l| (*l).to_string()));
            new_lines.push((*line).to_string());
            last_consumed = target + 1;
        } else {
            let next = positions[idx + 1..]
                .iter()
                .chain(replacements[idx + 1..].iter())
                .find_map(|p| *p);
            let Some(next) = next else {
                return Err(format!(
                    "Could not apply the changes: line {} of the changes does not match the file contents and has no following anchor line. Read the file again and provide changes that match its current contents.",
                    idx + 1
                ));
            };
            new_lines.extend(file_lines[last_consumed..next].iter().map(|l| (*l).to_string()));
            new_lines.push((*line).to_string());
            last_consumed = next;
        }
    }
    new_lines.extend(file_lines[last_consumed..].iter().map(|l| (*l).to_string()));
    let mut joined = new_lines.join("\n");
    if content.ends_with('\n') {
        joined.push('\n');
    }
    if joined == content {
        return Err(
            "The changes did not modify the file: the provided code already matches the file contents."
                .to_string(),
        );
    }
    std::fs::write(&file, &joined).map_err(|err| format!("write failed: {err}"))?;
    Ok(format!(
        "Edited {filepath}\n{}",
        unified_diff(&content, &joined)
    ))
}

fn unique_interval_target(positions: &[Option<usize>], idx: usize) -> Option<usize> {
    let prev = positions[..idx].iter().rev().find_map(|p| *p)?;
    let next = positions[idx + 1..].iter().find_map(|p| *p)?;
    if next == prev + 2 {
        Some(prev + 1)
    } else {
        None
    }
}

fn strip_codeblock(changes: &str) -> &str {
    let trimmed = changes.trim();
    let Some(rest) = trimmed.strip_prefix("```") else {
        return trimmed;
    };
    let Some(newline) = rest.find('\n') else {
        return trimmed;
    };
    let body = &rest[newline + 1..];
    match body.rfind("```") {
        Some(end) => body[..end].trim_end_matches('\n'),
        None => trimmed,
    }
}

fn is_placeholder(line: &str) -> bool {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return false;
    }
    if is_dots(trimmed) && trimmed.contains("...") {
        return true;
    }
    let stripped = strip_comment_prefix(trimmed);
    let stripped = strip_comment_suffix(stripped).trim();
    if is_dots(stripped) && stripped.contains("...") {
        return true;
    }
    if stripped.starts_with("...") && stripped.ends_with("...") {
        return true;
    }
    false
}

fn strip_comment_suffix(line: &str) -> &str {
    for suffix in ["*/", "-->", "/"] {
        if let Some(rest) = line.strip_suffix(suffix) {
            return rest;
        }
    }
    line
}

fn strip_comment_prefix(line: &str) -> &str {
    for prefix in ["//", "#", "--", "/*", "*/", "<!--", "-->", ";", "'", "*"] {
        if let Some(rest) = line.strip_prefix(prefix) {
            return rest;
        }
    }
    line
}

fn is_dots(s: &str) -> bool {
    s.chars().all(|c| c == '.' || c.is_whitespace())
}

pub fn fetch_url_content(url: &str) -> String {
    let client = match reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(30))
        .user_agent(BROWSER_UA)
        .build()
    {
        Ok(client) => client,
        Err(err) => return format!("Failed to fetch URL: {err}"),
    };
    let response = match client.get(url).send() {
        Ok(response) => response,
        Err(err) => return format!("Failed to fetch URL: {err}"),
    };
    let html = match response.bytes() {
        Ok(bytes) => decode_response(&bytes),
        Err(err) => return format!("Failed to fetch URL: {err}"),
    };
    let plain = html_to_text(&html);
    if plain.chars().count() > FETCH_MAX_CHARS {
        let cut: String = plain.chars().take(FETCH_MAX_CHARS).collect();
        format!(
            "{cut}\n\nThe content from {url} was truncated because it exceeded the {FETCH_MAX_CHARS} character limit. If you need more content, consider fetching specific sections or using a more targeted approach."
        )
    } else {
        plain
    }
}

const BLOCK_TAGS: &[&str] = &[
    "p", "div", "li", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "ul", "ol", "table", "section",
    "article", "header", "footer", "pre", "blockquote",
];

fn html_to_text(html: &str) -> String {
    let script_re =
        Regex::new(r"(?is)<(script|style|noscript)[^>]*>.*?</(script|style|noscript)>").unwrap();
    let cleaned = script_re.replace_all(html, "");
    let mut out = String::with_capacity(cleaned.len());
    let mut in_tag = false;
    let mut tag = String::new();
    for c in cleaned.chars() {
        if c == '<' {
            in_tag = true;
            tag.clear();
            continue;
        }
        if c == '>' && in_tag {
            in_tag = false;
            let name = tag
                .trim_start_matches('/')
                .split_whitespace()
                .next()
                .unwrap_or("")
                .to_lowercase();
            if name == "br" || BLOCK_TAGS.contains(&name.as_str()) {
                out.push('\n');
            }
            continue;
        }
        if in_tag {
            tag.push(c);
            continue;
        }
        out.push(c);
    }
    let decoded = decode_entities(&out);
    let mut result = String::with_capacity(decoded.len());
    let mut prev_blank = false;
    for line in decoded.lines() {
        let line = line.trim();
        if line.is_empty() {
            if !prev_blank {
                result.push('\n');
            }
            prev_blank = true;
        } else {
            if !result.is_empty() && !result.ends_with('\n') {
                result.push('\n');
            }
            result.push_str(line);
            prev_blank = false;
        }
    }
    result.trim().to_string()
}

fn decode_entities(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(pos) = rest.find('&') {
        out.push_str(&rest[..pos]);
        rest = &rest[pos..];
        if let Some(end) = rest.find(';') {
            let entity = &rest[..end + 1];
            let decoded = match entity {
                "&amp;" => Some("&".to_string()),
                "&lt;" => Some("<".to_string()),
                "&gt;" => Some(">".to_string()),
                "&quot;" => Some("\"".to_string()),
                "&apos;" => Some("'".to_string()),
                "&#39;" => Some("'".to_string()),
                "&nbsp;" => Some(" ".to_string()),
                _ => {
                    if let Some(num) = entity
                        .strip_prefix("&#")
                        .and_then(|n| n.strip_suffix(';'))
                        .and_then(|n| n.parse::<u32>().ok())
                    {
                        char::from_u32(num).map(|c| c.to_string())
                    } else {
                        None
                    }
                }
            };
            if let Some(decoded) = decoded {
                out.push_str(&decoded);
                rest = &rest[end + 1..];
                continue;
            }
        }
        out.push('&');
        rest = &rest[1..];
    }
    out.push_str(rest);
    out
}

pub fn search_web(query: &str) -> String {
    if query.trim().is_empty() {
        return "empty search query".to_string();
    }
    let client = match reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(30))
        .user_agent(BROWSER_UA)
        .build()
    {
        Ok(client) => client,
        Err(err) => return format!("Failed to search the web: {err}"),
    };
    let url = format!("https://www.bing.com/search?q={}&count=10", url_encode(query));
    let response = match client
        .get(&url)
        .header("Accept-Language", "en-US,en;q=0.9")
        .send()
    {
        Ok(response) => response,
        Err(err) => return format!("Failed to search the web: {err}"),
    };
    let html = match response.bytes() {
        Ok(bytes) => decode_response(&bytes),
        Err(err) => return format!("Failed to search the web: {err}"),
    };
    let results = parse_bing_results(&html);
    if results.is_empty() {
        return "The web search returned no results.".to_string();
    }
    let mut out = String::new();
    for (title, url, snippet) in results {
        let entry = format!("{title}\n{url}\n{snippet}");
        if out.chars().count() + entry.chars().count() > SEARCH_MAX_CHARS {
            out.push_str(&format!(
                "\n\nThe content from the following search results was truncated because it exceeded the {SEARCH_MAX_CHARS} character limit: {title}. For more detailed information, consider refining your search query."
            ));
            return out;
        }
        if !out.is_empty() {
            out.push('\n');
        }
        out.push_str(&entry);
    }
    out
}

fn parse_bing_results(html: &str) -> Vec<(String, String, String)> {
    let mut results = Vec::new();
    let mut rest = html;
    while let Some(start) = rest.find("<li class=\"b_algo\"") {
        let block_start = start;
        let Some(end) = rest[block_start..].find("</li>") else {
            break;
        };
        let block = &rest[block_start..block_start + end];
        let title = extract_tag_text(block, "<h2", "</h2>")
            .and_then(|inner| extract_tag_text(&inner, "<a", "</a>"))
            .map(|t| decode_entities(&strip_tags(&t)).trim().to_string())
            .filter(|t| !t.is_empty());
        let href = extract_href(block);
        let url = href
            .as_deref()
            .and_then(decode_bing_url)
            .filter(|u| !u.is_empty());
        let snippet = extract_tag_text(block, "<p", "</p>")
            .map(|t| decode_entities(&strip_tags(&t)).trim().to_string())
            .filter(|s| !s.is_empty());
        if title.is_some() || url.is_some() {
            results.push((
                title.unwrap_or_default(),
                url.unwrap_or_default(),
                snippet.unwrap_or_default(),
            ));
        }
        rest = &rest[block_start + end + 5..];
    }
    results
}

fn extract_tag_text(haystack: &str, open: &str, close: &str) -> Option<String> {
    let start = haystack.find(open)?;
    let inner_start = haystack[start..].find('>')? + start + 1;
    let inner_end = haystack[inner_start..].find(close)? + inner_start;
    Some(haystack[inner_start..inner_end].to_string())
}

fn extract_href(block: &str) -> Option<String> {
    let start = block.find("<a")?;
    let end = block[start..].find('>')? + start;
    let tag = &block[start..end];
    let href_start = tag.find("href=\"")? + 6;
    let href_end = tag[href_start..].find('"')? + href_start;
    Some(tag[href_start..href_end].to_string())
}

fn decode_bing_url(href: &str) -> Option<String> {
    let marker = "u=a1";
    let idx = href.find(marker)?;
    let encoded = &href[idx + marker.len()..];
    let encoded = encoded.split('&').next()?;
    let bytes = base64_decode(encoded)?;
    String::from_utf8(bytes).ok()
}

fn base64_decode(input: &str) -> Option<Vec<u8>> {
    let mut out = Vec::new();
    let mut buf = 0u32;
    let mut bits = 0u32;
    for c in input.chars() {
        if c == '=' {
            break;
        }
        let value = match c {
            'A'..='Z' => c as u32 - 'A' as u32,
            'a'..='z' => c as u32 - 'a' as u32 + 26,
            '0'..='9' => c as u32 - '0' as u32 + 52,
            '+' => 62,
            '/' => 63,
            _ => continue,
        };
        buf = (buf << 6) | value;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((buf >> bits) as u8);
        }
    }
    Some(out)
}

fn strip_tags(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut in_tag = false;
    for c in text.chars() {
        if c == '<' {
            in_tag = true;
            continue;
        }
        if c == '>' {
            in_tag = false;
            continue;
        }
        if !in_tag {
            out.push(c);
        }
    }
    out
}

fn decode_response(bytes: &[u8]) -> String {
    match std::str::from_utf8(bytes) {
        Ok(text) => text.to_string(),
        Err(_) => {
            let (text, _, _) = encoding_rs::GBK.decode(bytes);
            text.into_owned()
        }
    }
}

fn url_encode(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    for byte in input.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(byte as char);
            }
            _ => out.push_str(&format!("%{byte:02X}")),
        }
    }
    out
}

pub fn string_arg(args: &Value, key: &str) -> Result<String, String> {
    args.get(key)
        .and_then(Value::as_str)
        .map(str::to_string)
        .ok_or_else(|| format!("missing string argument: {key}"))
}

pub fn optional_string_arg(args: &Value, key: &str) -> Option<String> {
    args.get(key).and_then(Value::as_str).map(str::to_string)
}

pub fn optional_bool_arg(args: &Value, key: &str) -> Option<bool> {
    args.get(key).and_then(Value::as_bool)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn with_global_dirs(f: impl FnOnce(&Path)) {
        let _guard = crate::test_util::ENV_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let global = tempfile::TempDir::new().unwrap();
        unsafe {
            std::env::set_var("PROGNOSIS_CONFIG_DIR", global.path());
            std::env::set_var("PROGNOSIS_SKILLS_HOME", global.path());
        }
        f(global.path());
        unsafe {
            std::env::remove_var("PROGNOSIS_CONFIG_DIR");
            std::env::remove_var("PROGNOSIS_SKILLS_HOME");
        }
    }

    #[test]
    fn decode_response_handles_utf8_and_gbk() {
        let utf8 = "Rust 程序设计语言";
        assert_eq!(decode_response(utf8.as_bytes()), utf8);
        let (gbk_bytes, _, _) = encoding_rs::GBK.encode("中文测试");
        assert_eq!(decode_response(&gbk_bytes), "中文测试");
    }


    #[test]
    fn rule_toggle_persists_and_filters() {
        with_global_dirs(|_| {
            let dir = tmp_project();
            let rules_dir = dir.path().join(".prognosis").join("rules");
            std::fs::create_dir_all(&rules_dir).unwrap();
            std::fs::write(rules_dir.join("alpha.md"), "---\ndescription: \"a\"\n---\nbody").unwrap();
            std::fs::write(rules_dir.join("beta.md"), "---\ndescription: \"b\"\n---\nbody").unwrap();

            assert_eq!(load_rules(dir.path()).len(), 2);
            assert!(is_rule_enabled(dir.path(), "alpha"));

            set_rule_enabled(dir.path(), "alpha", false);
            assert!(!is_rule_enabled(dir.path(), "alpha"));
            assert_eq!(load_rules(dir.path()).len(), 1, "disabled rule must be filtered");
            assert_eq!(load_rules_all(dir.path()).len(), 2, "selector must see all rules");

            set_rule_enabled(dir.path(), "alpha", true);
            assert!(is_rule_enabled(dir.path(), "alpha"));
            assert_eq!(load_rules(dir.path()).len(), 2);
        });
    }

    fn tmp_project() -> tempfile::TempDir {
        tempfile::TempDir::new().unwrap()
    }

    #[test]
    fn path_resolution_stays_in_project() {
        let dir = tmp_project();
        std::fs::create_dir(dir.path().join("sub")).unwrap();
        let new_file = resolve_path(dir.path(), "sub/new.txt");
        assert_eq!(new_file.as_deref().unwrap().file_name().unwrap(), "new.txt");
        assert!(resolve_path(dir.path(), "missing/new.txt").is_err());
        assert!(resolve_path(dir.path(), "/etc/passwd").is_err());
        assert!(resolve_path(dir.path(), "../outside").is_err());
    }

    #[test]
    fn tilde_and_file_uri_expansion() {
        assert!(expand_requested("~/x").starts_with('/'));
        assert!(expand_requested("file:///tmp/a%20b.txt").ends_with("a b.txt"));
        assert!(expand_requested("file:///tmp/x.txt").ends_with("x.txt"));
    }

    #[test]
    fn read_and_create_file() {
        let dir = tmp_project();
        let file = resolve_path(dir.path(), "a.txt").unwrap();
        std::fs::write(&file, "hello").unwrap();
        assert_eq!(read_file(dir.path(), "a.txt").unwrap(), "hello");
        assert!(read_file(dir.path(), "missing.txt")
            .unwrap_err()
            .contains("does not exist"));
        let created = create_new_file(dir.path(), "new.txt", "content").unwrap();
        assert!(created.contains("File created successfully"), "{created}");
        assert!(created.contains("@@"), "output must carry a diff: {created}");
        assert!(created.contains("+content"), "{created}");
        let err = create_new_file(dir.path(), "new.txt", "again").unwrap_err();
        assert!(err.contains("already exists"));
    }

    #[test]
    fn terminal_command_runs_and_captures() {
        let dir = tmp_project();
        let out = run_terminal_command(dir.path(), "echo hello", true);
        assert!(out.contains("hello"), "{out}");
        let bg = run_terminal_command(dir.path(), "sleep 0.1", false);
        assert!(bg.contains("background"), "{bg}");
        let err = run_terminal_command(dir.path(), "echo oops >&2", true);
        assert!(err.contains("oops"), "{err}");
        let exit = run_terminal_command(dir.path(), "exit 3", true);
        assert!(exit.contains("exit 3"), "{exit}");
    }

    #[test]
    fn glob_search_respects_ignores_and_wildcards() {
        let dir = tmp_project();
        std::fs::create_dir_all(dir.path().join("src")).unwrap();
        std::fs::create_dir_all(dir.path().join("target")).unwrap();
        std::fs::create_dir_all(dir.path().join("src/nested")).unwrap();
        std::fs::write(dir.path().join("src/main.rs"), "").unwrap();
        std::fs::write(dir.path().join("src/lib.rs"), "").unwrap();
        std::fs::write(dir.path().join("src/nested/mod.rs"), "").unwrap();
        std::fs::write(dir.path().join("target/debug.rs"), "").unwrap();
        std::fs::write(dir.path().join("src/main.txt"), "").unwrap();
        let all = file_glob_search(dir.path(), "**/*.rs");
        assert!(all.contains("src/main.rs"), "{all}");
        assert!(all.contains("src/nested/mod.rs"), "{all}");
        assert!(!all.contains("target/debug.rs"), "target must be ignored: {all}");
        let top = file_glob_search(dir.path(), "*.rs");
        assert!(top.contains("src/nested/mod.rs"), "basename globs match any depth: {top}");
        assert!(!top.contains("main.txt"), "{top}");
        let none = file_glob_search(dir.path(), "*.py");
        assert_eq!(none, "The glob search returned no results.");
    }

    #[test]
    fn grep_search_finds_with_context() {
        let dir = tmp_project();
        std::fs::create_dir_all(dir.path().join("src")).unwrap();
        std::fs::write(
            dir.path().join("src/a.rs"),
            "line one\nfn target() {}\nline three\nline four\n",
        )
        .unwrap();
        std::fs::write(dir.path().join("src/b.rs"), "fn other() {}").unwrap();
        let out = grep_search(dir.path(), "fn target");
        assert!(out.contains("./src/a.rs"), "{out}");
        assert!(out.contains("line one"), "context lines expected: {out}");
        assert!(!out.contains("b.rs"), "{out}");
        let none = grep_search(dir.path(), "zzz_nothing");
        assert!(none.contains("no results"), "{none}");
        let bad = grep_search(dir.path(), "[");
        assert!(bad.contains("invalid regex"), "{bad}");
    }

    #[test]
    fn unified_diff_marks_added_and_removed_lines() {
        let out = unified_diff("a\nb\nc\n", "a\nx\nc\n");
        assert!(out.contains("@@ -3 +3 @@"), "{out}");
        assert!(out.contains("-b"), "{out}");
        assert!(out.contains("+x"), "{out}");
        assert!(!out.contains("-a"), "{out}");
        let created = unified_diff("", "l1\nl2");
        assert!(created.starts_with("@@ -0 +2 @@"), "{created}");
        assert!(created.contains("+l1"), "{created}");
        assert!(created.contains("+l2"), "{created}");
    }

    #[test]
    fn find_replace_exact_match_rules() {
        let dir = tmp_project();
        let file = dir.path().join("a.txt");
        std::fs::write(&file, "hello world\nhello again").unwrap();
        let err = single_find_and_replace(dir.path(), "a.txt", "x", "y", false).unwrap_err();
        assert!(err.contains("not found"));
        let err = single_find_and_replace(dir.path(), "a.txt", "hello", "hi", false).unwrap_err();
        assert!(err.contains("not unique"));
        single_find_and_replace(dir.path(), "a.txt", "hello", "hi", true).unwrap();
        assert_eq!(
            std::fs::read_to_string(&file).unwrap(),
            "hi world\nhi again"
        );
        let err = single_find_and_replace(dir.path(), "a.txt", "same", "same", false).unwrap_err();
        assert!(err.contains("MUST be different"));
    }

    #[test]
    fn edit_existing_file_applies_placeholder_changes() {
        let dir = tmp_project();
        let file = dir.path().join("a.rs");
        std::fs::write(
            &file,
            "fn main() {\n    let x = 1;\n    let y = 2;\n    println!(\"{x}\");\n}\n",
        )
        .unwrap();
        let changes = "fn main() {\n    let x = 1;\n    let y = 2;\n    // ... existing code ...\n    println!(\"{x}\");\n}\n";
        let err = edit_existing_file(dir.path(), "a.rs", changes).unwrap_err();
        assert!(err.contains("already matches"), "{err}");
        let changes = "fn main() {\n    let x = 10;\n    let y = 2;\n    // ... existing code ...\n    println!(\"{x}\");\n}\n";
        edit_existing_file(dir.path(), "a.rs", changes).unwrap();
        assert_eq!(
            std::fs::read_to_string(&file).unwrap(),
            "fn main() {\n    let x = 10;\n    let y = 2;\n    println!(\"{x}\");\n}\n"
        );
        let inserted = "fn main() {\n    let x = 10;\n    let y = 2;\n    let z = 3;\n    // ... existing code ...\n    println!(\"{x}\");\n}\n";
        edit_existing_file(dir.path(), "a.rs", inserted).unwrap();
        let content = std::fs::read_to_string(&file).unwrap();
        assert!(content.contains("let z = 3"), "{content}");
        let broken = edit_existing_file(dir.path(), "a.rs", "completely different code").unwrap_err();
        assert!(broken.contains("none of the provided code"), "{broken}");
    }

    #[test]
    fn edit_existing_file_rejects_mismatched_anchor_lines() {
        let dir = tmp_project();
        let file = dir.path().join("c.rs");
        std::fs::write(&file, "fn a() {\n    old();\n}\n").unwrap();
        let bad_first = edit_existing_file(dir.path(), "c.rs", "fn missing() {\n    new();\n}\n").unwrap_err();
        assert!(bad_first.contains("first line"), "{bad_first}");
        let bad_last = edit_existing_file(dir.path(), "c.rs", "fn a() {\n    new();\n}\n}\n").unwrap_err();
        assert!(bad_last.contains("last line"), "{bad_last}");
        assert_eq!(
            std::fs::read_to_string(&file).unwrap(),
            "fn a() {\n    old();\n}\n",
            "rejected edits must not modify the file"
        );
    }

    #[test]
    fn edit_existing_file_inserts_between_anchors_only() {
        let dir = tmp_project();
        let file = dir.path().join("e.rs");
        std::fs::write(&file, "fn a() {\n    keep();\n}\n").unwrap();
        edit_existing_file(
            dir.path(),
            "e.rs",
            "fn a() {\n    keep();\n    extra();\n}\n",
        )
        .unwrap();
        assert_eq!(
            std::fs::read_to_string(&file).unwrap(),
            "fn a() {\n    keep();\n    extra();\n}\n"
        );
    }

    #[test]
    fn edit_existing_file_does_not_blur_replacement_into_wrong_row() {
        let dir = tmp_project();
        let file = dir.path().join("f.rs");
        std::fs::write(&file, "fn a() {\n    x1();\n    x2();\n}\n").unwrap();
        edit_existing_file(
            dir.path(),
            "f.rs",
            "fn a() {\n    x1();\n    brand();\n    x2();\n}\n",
        )
        .unwrap();
        assert_eq!(
            std::fs::read_to_string(&file).unwrap(),
            "fn a() {\n    x1();\n    brand();\n    x2();\n}\n",
            "a dissimilar line between two anchors must insert, not replace"
        );
    }

    #[test]
    fn edit_existing_file_strips_codeblock_fence() {
        let dir = tmp_project();
        let file = dir.path().join("b.rs");
        std::fs::write(&file, "fn a() {\n    old();\n}\n").unwrap();
        let changes = "```rust b.rs\nfn a() {\n    new();\n}\n```";
        edit_existing_file(dir.path(), "b.rs", changes).unwrap();
        assert_eq!(
            std::fs::read_to_string(&file).unwrap(),
            "fn a() {\n    new();\n}\n"
        );
    }

    #[test]
    fn placeholder_detection() {
        assert!(is_placeholder("// ... existing code ..."));
        assert!(is_placeholder("..."));
        assert!(is_placeholder("  # ... rest of file ..."));
        assert!(is_placeholder("/* ... */"));
        assert!(!is_placeholder("let x = 1;"));
        assert!(!is_placeholder("fn f(...args) {}"));
        assert!(!is_placeholder(""));
    }

    #[test]
    fn ls_lists_with_and_without_recursion() {
        let dir = tmp_project();
        std::fs::create_dir_all(dir.path().join("a/b")).unwrap();
        std::fs::write(dir.path().join("a/f1.txt"), "").unwrap();
        std::fs::write(dir.path().join("a/b/f2.txt"), "").unwrap();
        let flat = ls_dir(dir.path(), Some("a"), false).unwrap();
        assert!(flat.contains("b/"), "{flat}");
        assert!(!flat.contains("f2.txt"), "{flat}");
        let rec = ls_dir(dir.path(), Some("a"), true).unwrap();
        assert!(rec.contains("b/f2.txt"), "{rec}");
        let err = ls_dir(dir.path(), Some("missing"), false).unwrap_err();
        assert!(err.contains("not found or is not accessible"), "{err}");
    }

    #[test]
    fn view_diff_reports_working_changes() {
        let dir = tmp_project();
        run_terminal_command(dir.path(), "git init -q && git config user.email t@t && git config user.name t && echo v1 > f.txt && git add f.txt && git commit -qm init", true);
        let empty = view_diff(dir.path());
        assert!(empty.contains("empty"), "{empty}");
        std::fs::write(dir.path().join("f.txt"), "v2").unwrap();
        let diff = view_diff(dir.path());
        assert!(diff.contains("-v1"), "{diff}");
        assert!(diff.contains("+v2"), "{diff}");
    }

    #[test]
    fn rule_block_roundtrip() {
        with_global_dirs(|_| {
            let dir = tmp_project();
            create_rule_block(
                dir.path(),
                RuleSpec {
                    name: "Use Prop Types",
                    rule: "Always use PropTypes",
                    description: Some("For React components"),
                    globs: Some("**/*.js"),
                    regex: None,
                    always_apply: Some(false),
                    scope: None,
                },
            )
            .unwrap();
            let rules = load_rules(dir.path());
            assert_eq!(rules.len(), 1);
            assert_eq!(rules[0].name, "use-prop-types");
            assert_eq!(rules[0].description, "For React components");
            assert_eq!(rules[0].globs, "**/*.js");
            assert_eq!(rules[0].rule, "Always use PropTypes");
            assert_eq!(rules[0].always_apply, Some(false));
            let content = request_rule(dir.path(), "Use Prop Types").unwrap();
            assert!(content.contains("Always use PropTypes"), "{content}");
            assert!(content.contains("alwaysApply: false"), "{content}");
            let err = request_rule(dir.path(), "nope").unwrap_err();
            assert!(err.contains("not found"), "{err}");
        });
    }

    #[test]
    fn rules_merge_global_and_project_with_project_priority() {
        with_global_dirs(|global| {
            let dir = tmp_project();
            let global_rules = global.join("rules");
            let project_rules = dir.path().join(".prognosis").join("rules");
            std::fs::create_dir_all(&global_rules).unwrap();
            std::fs::create_dir_all(&project_rules).unwrap();
            std::fs::write(
                global_rules.join("shared.md"),
                "---\ndescription: \"global shared\"\n---\nglobal body",
            )
            .unwrap();
            std::fs::write(
                global_rules.join("conflict.md"),
                "---\ndescription: \"global version\"\n---\nglobal conflict",
            )
            .unwrap();
            std::fs::write(
                project_rules.join("conflict.md"),
                "---\ndescription: \"project version\"\n---\nproject conflict",
            )
            .unwrap();

            let rules = load_rules_all(dir.path());
            assert_eq!(rules.len(), 2);
            let shared = rules.iter().find(|r| r.name == "shared").unwrap();
            assert_eq!(shared.description, "global shared");
            let conflict = rules.iter().find(|r| r.name == "conflict").unwrap();
            assert_eq!(conflict.description, "project version", "project must override global");
            assert_eq!(conflict.rule, "project conflict");
            let content = request_rule(dir.path(), "conflict").unwrap();
            assert!(content.contains("project conflict"), "{content}");
            let content = request_rule(dir.path(), "shared").unwrap();
            assert!(content.contains("global body"), "{content}");
        });
    }

    #[test]
    fn create_rule_global_scope_writes_global_dir() {
        with_global_dirs(|global| {
            let dir = tmp_project();
            create_rule_block(
                dir.path(),
                RuleSpec {
                    name: "Global Style",
                    rule: "Use tabs",
                    description: Some("global rule"),
                    globs: None,
                    regex: None,
                    always_apply: None,
                    scope: Some("global"),
                },
            )
            .unwrap();
            assert!(!dir.path().join(".prognosis/rules").exists());
            let content =
                std::fs::read_to_string(global.join("rules").join("global-style.md")).unwrap();
            assert!(content.contains("Use tabs"), "{content}");
            assert_eq!(load_rules(dir.path()).len(), 1);
            let content = request_rule(dir.path(), "Global Style").unwrap();
            assert!(content.contains("Use tabs"), "{content}");
        });
    }

    #[test]
    fn skill_roundtrip_with_subdirectory() {
        with_global_dirs(|_| {
            let dir = tmp_project();
            let skills = dir.path().join(".agents/skills");
            std::fs::create_dir_all(&skills).unwrap();
            std::fs::write(
                skills.join("refactor.md"),
                "---\ndescription: Refactor with confidence\n---\n\nStep 1: read the file",
            )
            .unwrap();
            std::fs::create_dir_all(skills.join("deploy")).unwrap();
            std::fs::write(
                skills.join("deploy/SKILL.md"),
                "---\nname: deploy\n---\n\nStep 1: build",
            )
            .unwrap();
            let listed = load_skills(dir.path());
            assert_eq!(listed.len(), 2);
            let refactor = listed.iter().find(|s| s.name == "refactor").unwrap();
            assert_eq!(refactor.description, "Refactor with confidence");
            let deploy = listed.iter().find(|s| s.name == "deploy").unwrap();
            assert_eq!(deploy.description, "");
            let content = read_skill(dir.path(), "refactor").unwrap();
            assert!(content.contains("Step 1"), "{content}");
            let content = read_skill(dir.path(), "deploy").unwrap();
            assert!(content.contains("Step 1: build"), "{content}");
            let err = read_skill(dir.path(), "missing").unwrap_err();
            assert!(err.contains("Available skills: deploy, refactor"), "{err}");
        });
    }

    #[test]
    fn skills_merge_from_agents_and_global_dirs() {
        with_global_dirs(|global| {
            let dir = tmp_project();
            let project_skills = dir.path().join(".agents/skills");
            let global_agents = global.join(".agents/skills");
            let global_config = global.join(".config/agents/skills");
            std::fs::create_dir_all(&project_skills).unwrap();
            std::fs::create_dir_all(&global_agents).unwrap();
            std::fs::create_dir_all(&global_config).unwrap();
            std::fs::create_dir_all(global_agents.join("fmt")).unwrap();
            std::fs::create_dir_all(global_config.join("web")).unwrap();
            std::fs::create_dir_all(project_skills.join("fmt")).unwrap();
            std::fs::write(
                global_agents.join("fmt/SKILL.md"),
                "---\nname: fmt\ndescription: Formatting rules\n---\nfmt body",
            )
            .unwrap();
            std::fs::write(
                global_config.join("web/SKILL.md"),
                "---\nname: web\ndescription: Web checks\n---\nweb body",
            )
            .unwrap();
            std::fs::write(
                project_skills.join("fmt/SKILL.md"),
                "---\nname: fmt\ndescription: Project override\n---\nproject fmt",
            )
            .unwrap();

            let skills = load_skills_all(dir.path());
            assert_eq!(skills.len(), 2, "project override must dedupe global");
            let fmt = skills.iter().find(|s| s.name == "fmt").unwrap();
            assert_eq!(fmt.description, "Project override", "project must override global");
            assert!(skills.iter().any(|s| s.name == "web"));
            let content = read_skill(dir.path(), "fmt").unwrap();
            assert!(content.contains("project fmt"), "{content}");
            let content = read_skill(dir.path(), "web").unwrap();
            assert!(content.contains("web body"), "{content}");
        });
    }

    #[test]
    fn state_migrates_from_legacy_files() {
        with_global_dirs(|_| {
            let dir = tmp_project();
            let legacy_rules = dir.path().join(".prognosis/rules");
            let legacy_skills = dir.path().join(".prognosis/skills");
            std::fs::create_dir_all(&legacy_rules).unwrap();
            std::fs::create_dir_all(&legacy_skills).unwrap();
            std::fs::write(
                legacy_rules.join("state.json"),
                r#"{"disabled": ["alpha"]}"#,
            )
            .unwrap();
            std::fs::write(
                legacy_skills.join("state.json"),
                r#"{"disabled": ["deploy"]}"#,
            )
            .unwrap();

            assert!(!is_rule_enabled(dir.path(), "alpha"));
            assert!(is_rule_enabled(dir.path(), "beta"));
            assert!(!is_skill_enabled(dir.path(), "deploy"));
            let migrated = std::fs::read_to_string(dir.path().join(".prognosis/state.json")).unwrap();
            assert!(migrated.contains("alpha"), "{migrated}");
            assert!(migrated.contains("deploy"), "{migrated}");

            set_skill_enabled(dir.path(), "deploy", true);
            assert!(is_skill_enabled(dir.path(), "deploy"));
            assert!(!is_rule_enabled(dir.path(), "alpha"), "rules state must survive");
        });
    }

    #[test]
    fn html_to_text_converts_and_decodes() {
        let html = "<html><head><script>var x = 1;</script></head><body><h1>Title</h1><p>Hello &amp; goodbye<br>line2</p></body></html>";
        let text = html_to_text(html);
        assert!(text.contains("Title"), "{text}");
        assert!(text.contains("Hello & goodbye"), "{text}");
        assert!(!text.contains("var x"), "{text}");
        assert_eq!(decode_entities("a &lt;b&gt; &#39;c&#39;"), "a <b> 'c'");
    }

    #[test]
    fn base64_decode_works() {
        let decoded = base64_decode("aHR0cHM6Ly9ydXN0LWxhbmcub3JnLw==").unwrap();
        assert_eq!(decoded, b"https://rust-lang.org/");
        assert_eq!(url_encode("a b&c"), "a%20b%26c");
    }

    #[test]
    fn glob_matcher_handles_doublestar() {
        let matcher = glob_matcher("**/*.rs").unwrap();
        assert!(matcher.matches("src/main.rs"));
        assert!(matcher.matches("main.rs"));
        let matcher = glob_matcher("src/*.rs").unwrap();
        assert!(matcher.matches("src/main.rs"));
        assert!(!matcher.matches("src/nested/mod.rs"));
        let matcher = glob_matcher("*.rs").unwrap();
        assert!(matcher.matches("src/main.rs"));
        assert!(!matcher.matches("src/main.txt"));
    }

    #[test]
    fn sanitize_rule_names() {
        assert_eq!(sanitize_rule_name("Use Prop Types"), "use-prop-types");
        assert_eq!(sanitize_rule_name("React!! Standards??"), "react-standards");
    }

    #[test]
    fn fetch_url_fetches_webpage() {
        let out = fetch_url_content("https://example.com");
        assert!(
            out.contains("Example Domain") || out.contains("example"),
            "unexpected fetch result: {out}"
        );
    }

    #[test]
    fn search_web_chinese_decodes_correctly() {
        let out = search_web("Rust 程序设计语言");
        let first = out.lines().next().unwrap_or("");
        println!("SEARCH_FIRST: {first}");
        assert!(!first.contains('Ã'), "mojibake detected: {first}");
        assert!(!first.contains('ç'), "mojibake detected: {first}");
    }

    #[test]
    fn search_web_returns_bing_results() {
        let out = search_web("rust programming language");
        assert!(
            out.contains("rust") || out.contains("Rust"),
            "unexpected search result: {out}"
        );
        assert!(out.lines().count() > 3, "expected multiple result lines: {out}");
    }
}
