pub mod models;
pub mod remember;
pub mod scheduler;
pub mod supervisor;
pub mod tools;

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::adapter::error::AdapterError;
use crate::adapter::types::ToolDefinition;
use crate::runtime::event::{Event, EventMeta};
use crate::runtime::ports::{LlmAdapter, LlmPort};
use crate::runtime::types::{
    CycleId, DialogueTurn, GenerateRequest, ModulationContext, PerceptionPayload,
    PerceptionSource, TaskSetState, ToolResult,
};
use crate::runtime::Runtime;

use models::{context_window, ModelEntry, ModelsStore, SwitchableAdapter};
use scheduler::{Condition, Fired, Scheduler, TaskAction, TaskKind};
pub use supervisor::{Supervisor, ToolCallRecord, Verdict};

pub enum AdapterKind {
    OpenAi,
    DeepSeek,
}

pub struct AppConfig {
    pub adapter: AdapterKind,
    pub model: Option<String>,
    pub supervisor_enabled: bool,
    pub project_dir: PathBuf,
    pub tick_interval: Duration,
}

const NO_PARALLEL_TOOL_CALLING_INSTRUCTION: &str =
    "This tool CANNOT be called in parallel with any other tools, including itself";

const CHANGES_DESCRIPTION: &str =
    "Any modifications to the file, showing only needed changes. Do NOT wrap this in a codeblock or write anything besides the code changes. In larger files, use brief language-appropriate placeholders for large unmodified sections, e.g. '// ... existing code ...'";

const EDIT_CODE_INSTRUCTIONS: &str = r#"  When addressing code modification requests, present a concise code snippet that
  emphasizes only the necessary changes and uses abbreviated placeholders for
  unmodified sections. For example:

  ```language /path/to/file

  {{ modified code here }}


  {{ another modification }}

  ```

  In existing files, you should always restate the function or class that the snippet belongs to:

  ```language /path/to/file

  function exampleFunction() {

    {{ modified code here }}

  }

  ```

  Since users have access to their complete file, they prefer reading only the
  relevant modifications. It's perfectly acceptable to omit unmodified portions
  at the beginning, middle, or end of files using these "lazy" comments. Only
  provide the complete file when explicitly requested. Include a concise explanation
  of changes unless the user specifically asks for code only."#;

const SINGLE_FIND_AND_REPLACE_DESCRIPTION: &str = r#"Performs exact string replacements in a file.

IMPORTANT:
- ALWAYS use the `read_file` tool just before making edits, to understand the file's up-to-date contents and context. The user can also edit the file while you are working with it.
- This tool CANNOT be called in parallel with any other tools, including itself
- When editing text from `read_file` tool output, ensure you preserve exact whitespace/indentation.
- Only use emojis if the user explicitly requests it. Avoid adding emojis to files unless asked.
- Use `replace_all` for replacing and renaming strings across the file. This parameter is useful if you want to rename a variable, for instance.

WARNINGS:
- When not using `replace_all`, the edit will FAIL if `old_string` is not unique in the file. Either provide a larger string with more surrounding context to make it unique or use `replace_all` to change every instance of `old_string`.
- The edit will likely fail if you have not recently used the `read_file` tool to view up-to-date file contents."#;

const CREATE_RULE_BLOCK_DESCRIPTION: &str = r#"Creates a "rule" that can be referenced in future conversations. This should be used whenever you want to establish code standards / preferences that should be applied consistently, or when you want to avoid making a mistake again. To modify existing rules, use the edit tool instead.

Rule Types:
- Always: Include only "rule" (always included in model context)
- Auto Attached: Include "rule", "globs", and/or "regex" (included when files match patterns)
- Agent Requested: Include "rule" and "description" (AI decides when to apply based on description)
- Manual: Include only "rule" (only included when explicitly mentioned using @ruleName)

Scope: set "scope" to "global" to store the rule in ~/.prognosis/rules so it applies to all projects; omit it to store in the current project's .prognosis/rules."#;

const NAME_ARG_DESC: &str = "Short, descriptive name summarizing the rule's purpose (e.g. 'React Standards', 'Type Hints')";
const RULE_ARG_DESC: &str = "Clear, imperative instruction for future code generation (e.g. 'Use named exports', 'Add Python type hints'). Each rule should focus on one specific standard.";
const DESC_ARG_DESC: &str = "Description of when this rule should be applied. Required for Agent Requested rules (AI decides when to apply). Optional for other types.";
const GLOB_ARG_DESC: &str = "Optional file patterns to which this rule applies (e.g. ['**/*.{ts,tsx}'] or ['src/**/*.ts', 'tests/**/*.ts'])";
const REGEX_ARG_DESC: &str = "Optional regex patterns to match against file content. Rule applies only to files whose content matches the pattern (e.g. 'useEffect' for React hooks or '\\bclass\\b' for class definitions)";
const ALWAYS_APPLY_DESC: &str = "Whether this rule should always be applied. Set to false for Agent Requested and Manual rules. Omit or set to true for Always and Auto Attached rules.";

fn run_terminal_command_description() -> String {
    let platform_info = format!(
        "Choose terminal commands and scripts optimized for {} and {} and shell {}.",
        std::env::consts::OS,
        std::env::consts::ARCH,
        std::env::var("SHELL").unwrap_or_else(|_| "/bin/bash".into())
    );
    format!(
        "Run a terminal command in the project directory and return its output.\n\nGuidelines:\n- Each invocation starts a fresh shell with NO memory of previous commands; chain dependent commands in a single call (e.g. 'cd dir && make test').\n- Prefer dedicated tools over shell commands: use the file tools for reading and editing files; use the shell only for builds, tests, git operations, and other actions that have no dedicated tool.\n- NEVER use shell commands (sed, awk, perl, etc.) to edit files.\n- When a command runs in the background (waitForCompletion=false), ALWAYS suggest stopping it with a shell command (e.g. 'kill <pid>' or 'pkill -f <name>'); NEVER suggest Ctrl+C.\n- When you suggest follow-up shell commands, always format them as shell code blocks.\n- Do NOT run commands that require special/admin privileges.\n- Prefer '&&' chaining over separate calls when the second command depends on the first.\n{platform_info}"
    )
}

fn request_rule_description(project_dir: &Path) -> String {
    let prefix = "Retrieve the full content of a rule by name. Rules are project or global instructions that apply automatically when their file patterns match, or when explicitly mentioned. Before following a rule, load its full text with this tool. Available rules:\n";
    let rules = tools::load_rules(project_dir);
    if rules.is_empty() {
        format!("{prefix}No rules available.")
    } else {
        let listed = rules
            .iter()
            .map(|rule| format!("{}: {}", rule.name, rule.description))
            .collect::<Vec<_>>()
            .join("\n");
        format!("{prefix}{listed}")
    }
}

fn read_skill_description(project_dir: &Path) -> String {
    let mut description = "Read the full instructions of a skill by name. Skills package detailed, task-specific procedures (steps, checklists, reference material). Load a skill's content with this tool when the current task matches its description, then follow its instructions. Available skills:\n"
        .to_string();
    for skill in tools::load_skills(project_dir) {
        description.push_str(&format!(
            "\nname: {}\ndescription: {}\n",
            skill.name, skill.description
        ));
    }
    description
}

fn tool_spec(name: &str, project_dir: &Path) -> (Option<String>, serde_json::Value) {
    match name {
        "read_file" => (
            Some(
                "Read the contents of an existing file in the workspace and return them with line numbers. Use this before editing a file, before quoting from a file, and whenever you need to know what the file actually contains — never guess file contents from memory or from an earlier version. Re-read a file after a previous edit changed it, so your edits always apply to current content; the file may have changed since you last saw it.\n\nBehavior:\n- Output includes line numbers, so you can refer to specific lines (\"around line 42\") in follow-up calls.\n- Large files are read in full; if the output is very long, prefer reading the relevant part or using grep_search first to locate the section you need.\n- The path can be relative to the project root, absolute, a tilde path (~/...), or a file:// URI.\n- If the file does not exist or cannot be read, the tool returns an explicit error message instead of guessing content.".into(),
            ),
            serde_json::json!({
                "type": "object",
                "properties": {
                    "filepath": {"type": "string", "description": "The path of the file to read. Can be a relative path (from workspace root), absolute path, tilde path (~/...), or file:// URI"},
                },
                "required": ["filepath"],
            }),
        ),
        "create_new_file" => (
            Some(
                "Create a brand-new file with the given contents. Use this tool ONLY when the file does not exist yet — never to overwrite an existing file; if the file already exists, use edit_existing_file or single_find_and_replace instead.\n\nBehavior:\n- The parent directory is created automatically when needed.\n- The contents are written exactly as provided; do not include markdown fences or explanatory text in the contents argument.\n- If a file already exists at the path, the call is rejected — check with ls or read_file first when you are unsure whether the file exists.".into(),
            ),
            serde_json::json!({
                "type": "object",
                "properties": {
                    "filepath": {"type": "string", "description": "The path where the new file should be created. Can be a relative path (from workspace root), absolute path, tilde path (~/...), or file:// URI."},
                    "contents": {"type": "string", "description": "The contents to write to the new file"},
                },
                "required": ["filepath", "contents"],
            }),
        ),
        "run_terminal_command" => (
            Some(run_terminal_command_description()),
            serde_json::json!({
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The command to run. This will be passed directly into the IDE shell."},
                    "waitForCompletion": {"type": "boolean", "description": "Whether to wait for the command to complete before returning. Default is true. Set to false to run the command in the background. Set to true to run the command in the foreground and wait to collect the output."},
                },
                "required": ["command"],
            }),
        ),
        "file_glob_search" => (
            Some(
                "Search for files by name or path pattern recursively across the project, using glob syntax. Use this to locate a file when you know part of its name or path but not its exact location (e.g. find the test file for a module, locate a config file).\n\nBehavior:\n- '**' matches any number of directories (e.g. 'src/**/tests/*.rs').\n- Build, cache, and dependency directories (target, node_modules, .git, etc.) are excluded — use ls to inspect those.\n- Results may be truncated; prefer targeted patterns over broad ones like '*'. If a broad pattern returns nothing useful, narrow it with the directory you expect the file in.".into(),
            ),
            serde_json::json!({
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Glob pattern for file path matching"},
                },
                "required": ["pattern"],
            }),
        ),
        "view_diff" => (
            Some("Show the uncommitted working-tree changes (git diff) of the project — the exact lines you added and removed, with file names. Use this before summarizing what changed, before writing a commit message, and to review your own edits after a task so you can confirm only the intended lines were touched.\n\nBehavior:\n- Read-only: it never modifies files or the git index.\n- Diff blocks are folded in the UI for large changes; the full detail is available when expanded.\n- If the project is not a git repository, returns a friendly notice instead of an error.".into()),
            serde_json::json!({"type": "object"}),
        ),
        "ls" => (
            Some(
                "List the contents of a directory with file names and sizes. Use this to discover the project layout, confirm what a directory contains, or find where a file lives before reading or editing it. Call it on the project root first when you start a task in an unfamiliar workspace — the listing tells you which files exist and what to read next.\n\nBehavior:\n- Paths can be relative to the project root, absolute, or tilde paths.\n- With recursive=true the whole subtree is listed — use it sparingly on large trees (node_modules, build output) because the output can be very long.\n- If the directory does not exist or cannot be read, returns an explicit error message.".into(),
            ),
            serde_json::json!({
                "type": "object",
                "properties": {
                    "dirPath": {"type": "string", "description": "The directory path. Can be relative to project root, absolute path, tilde path (~/...), or file:// URI. Use forward slash paths"},
                    "recursive": {"type": "boolean", "description": "If true, lists files and folders recursively. To prevent unexpected large results, use this sparingly"},
                },
            }),
        ),
        "create_rule_block" => (
            Some(CREATE_RULE_BLOCK_DESCRIPTION.into()),
            serde_json::json!({
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": NAME_ARG_DESC},
                    "rule": {"type": "string", "description": RULE_ARG_DESC},
                    "description": {"type": "string", "description": DESC_ARG_DESC},
                    "globs": {"type": "string", "description": GLOB_ARG_DESC},
                    "regex": {"type": "string", "description": REGEX_ARG_DESC},
                    "alwaysApply": {"type": "boolean", "description": ALWAYS_APPLY_DESC},
                    "scope": {"type": "string", "description": "Where to store the rule: \"local\" (project .prognosis/rules, default) or \"global\" (~/.prognosis/rules, applies to all projects)"},
                },
                "required": ["name", "rule"],
            }),
        ),
        "fetch_url_content" => (
            Some(
                "Fetch and view the text content of a public web page by URL. Use this when the answer needs current, external information that is not in the project and not in your training data — for example a library's current version, an API change, or a public announcement.\n\nBehavior:\n- The page is fetched over the network and returned as plain text; interactive or heavily scripted pages may return little content.\n- Do NOT use it for local files — use read_file for those.\n- Only fetch public, non-sensitive URLs; the request is visible to the site owner.\n- Some sites block automated fetches; if a fetch fails, try search_web first to find an alternative source.".into(),
            ),
            serde_json::json!({
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The URL to read"},
                },
                "required": ["url"],
            }),
        ),
        "request_rule" => (
            Some(request_rule_description(project_dir)),
            serde_json::json!({
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Name of the rule"},
                },
                "required": ["name"],
            }),
        ),
        "read_skill" => (
            Some(read_skill_description(project_dir)),
            serde_json::json!({
                "type": "object",
                "properties": {
                    "skillName": {"type": "string", "description": "The name of the skill to read. This should match the name from the available skills."},
                },
                "required": ["skillName"],
            }),
        ),
        "search_web" => (
            Some(
                "Search the web and return top results with titles, URLs, and snippets. Use this only when the answer requires specialized, external, or up-to-date knowledge that is not in the project and not in your training data (e.g. current events, library versions, documentation changes). Common programming questions about the code in this project do NOT require a web search — read the code instead.\n\nBehavior:\n- Results come from a web search index; snippets are summaries, not full pages — if a result page itself is needed, follow up with fetch_url_content.\n- Prefer official sources (documentation, package registries, the project's own site) over forums and blogs when they are available.\n- If the search returns nothing relevant, rephrase the query with different terms rather than repeating it verbatim.".into(),
            ),
            serde_json::json!({
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The natural language search query"},
                },
                "required": ["query"],
            }),
        ),
        "edit_existing_file" => (
            Some(format!(
                "Use this tool to edit an existing file. If you don't know the contents of the file, read it first.\n{EDIT_CODE_INSTRUCTIONS}\n{NO_PARALLEL_TOOL_CALLING_INSTRUCTION}"
            )),
            serde_json::json!({
                "type": "object",
                "properties": {
                    "filepath": {"type": "string", "description": "The path of the file to edit, relative to the root of the workspace."},
                    "changes": {"type": "string", "description": CHANGES_DESCRIPTION},
                },
                "required": ["filepath", "changes"],
            }),
        ),
        "single_find_and_replace" => (
            Some(SINGLE_FIND_AND_REPLACE_DESCRIPTION.into()),
            serde_json::json!({
                "type": "object",
                "properties": {
                    "filepath": {"type": "string", "description": "The path to the file to modify, relative to the root of the workspace"},
                    "old_string": {"type": "string", "description": "The text to replace - must be exact including whitespace/indentation"},
                    "new_string": {"type": "string", "description": "The text to replace it with (MUST be different from old_string)"},
                    "replace_all": {"type": "boolean", "description": "Replace all occurrences of old_string (default false)"},
                },
                "required": ["filepath", "old_string", "new_string"],
            }),
        ),
        "grep_search" => (
            Some(
                "Search file contents across the project with a regular expression (ripgrep). Use this to find where a symbol is defined, used, or referenced, or to locate code matching a pattern — it is the fastest way to answer questions like \"where is this function called?\" or \"which files use this constant?\".\n\nBehavior:\n- The query is a regex; use alternation (e.g. 'word1|word2|word3') or character classes to find multiple potential spellings in a single search.\n- Build, cache, and dependency directories are excluded.\n- Results may be truncated — narrow the pattern (e.g. 'fn handle_call_tool' or 'trait Lang.*Adapter') instead of using a broad term.\n- The pattern is passed directly to ripgrep, not to a shell, so no quoting or escaping for the shell is needed.".into(),
            ),
            serde_json::json!({
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The regex pattern to search for within file contents. Use regex with alternation (e.g., 'word1|word2|word3') or character classes to find multiple potential words in a single search."},
                },
                "required": ["query"],
            }),
        ),
        "schedule_task" => (
            Some(
                "Schedule a tool call to run later. Use it when the user asks for something to happen after a delay, on a recurring interval, or when a condition becomes true (e.g. run a command when a file appears).\n\nTask types:\n- delay: run the action once after `seconds`, or at wall-clock time `at` (HH:MM, today or tomorrow)\n- schedule: run the action every `interval_seconds`\n- monitor: check `condition` every `check_every_seconds` until it holds or `timeout_seconds` elapses, then run the action once\n\nReturns a task id; use cancel_task to stop a scheduled or monitor task.".into(),
            ),
            serde_json::json!({
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": ["delay", "schedule", "monitor"], "description": "The kind of task to schedule: delay runs once, schedule repeats, monitor waits for a condition."},
                    "seconds": {"type": "integer", "description": "delay: run the action after this many seconds."},
                    "at": {"type": "string", "description": "delay: run the action at this wall-clock time HH:MM."},
                    "interval_seconds": {"type": "integer", "description": "schedule: repeat the action every this many seconds."},
                    "condition": {"type": "object", "description": "monitor: the condition to watch.", "properties": {
                        "type": {"type": "string", "enum": ["output_contains", "exit_code", "file_exists"], "description": "The kind of condition to check."},
                        "cmd": {"type": "string", "description": "The shell command to run for output_contains / exit_code conditions."},
                        "contains": {"type": "string", "description": "The text the command output must contain for output_contains."},
                        "path": {"type": "string", "description": "The file path to watch for file_exists."},
                    }, "required": ["type"]},
                    "check_every_seconds": {"type": "integer", "description": "monitor: check the condition every this many seconds (default 5)."},
                    "timeout_seconds": {"type": "integer", "description": "monitor: give up after this many seconds."},
                    "action": {"type": "object", "description": "The tool call to run when the task triggers.", "properties": {
                        "tool": {"type": "string", "description": "The tool to run."},
                        "arguments": {"type": "object", "description": "The arguments for the tool."},
                    }, "required": ["tool"]},
                },
                "required": ["type", "action"],
            }),
        ),
        "cancel_task" => (
            Some(
                "Cancel a previously scheduled task by its id. Use this when a scheduled or monitor task should no longer run — for example the user changed their mind, or the condition it was waiting for became irrelevant.\n\nBehavior:\n- The id is the number returned by schedule_task (shown in the task list as #id).\n- Only tasks that have not fired yet can be cancelled; a delay task that already triggered, or a monitor task that already matched, is no longer cancellable.\n- Returns an explicit error if no task with the given id exists — check the task list first if unsure.".into(),
            ),
            serde_json::json!({
                "type": "object",
                "properties": {
                    "id": {"type": "integer", "description": "The task id returned by schedule_task."},
                },
                "required": ["id"],
            }),
        ),
        _ => (None, serde_json::json!({"type": "object"})),
    }
}

fn register_default_tools(app: &mut App) {
    let dir = app.project_dir.clone();
    app.register_tool(
        "read_file",
        move |args| match tools::string_arg(args, "filepath") {
            Ok(filepath) => tools::read_file(&dir, &filepath).unwrap_or_else(|err| err),
            Err(err) => err,
        },
        false,
    );
    let dir = app.project_dir.clone();
    app.register_tool(
        "create_new_file",
        move |args| {
            let filepath = match tools::string_arg(args, "filepath") {
                Ok(filepath) => filepath,
                Err(err) => return err,
            };
            let contents = match tools::string_arg(args, "contents") {
                Ok(contents) => contents,
                Err(err) => return err,
            };
            tools::create_new_file(&dir, &filepath, &contents).unwrap_or_else(|err| err)
        },
        true,
    );
    let dir = app.project_dir.clone();
    app.register_tool(
        "run_terminal_command",
        move |args| {
            let command = match tools::string_arg(args, "command") {
                Ok(command) => command,
                Err(err) => return err,
            };
            let wait = tools::optional_bool_arg(args, "waitForCompletion").unwrap_or(true);
            tools::run_terminal_command(&dir, &command, wait)
        },
        true,
    );
    let dir = app.project_dir.clone();
    app.register_tool(
        "file_glob_search",
        move |args| match tools::string_arg(args, "pattern") {
            Ok(pattern) => tools::file_glob_search(&dir, &pattern),
            Err(err) => err,
        },
        false,
    );
    let dir = app.project_dir.clone();
    app.register_tool(
        "view_diff",
        move |_| tools::view_diff(&dir),
        false,
    );
    let dir = app.project_dir.clone();
    app.register_tool(
        "ls",
        move |args| {
            let dir_path = tools::optional_string_arg(args, "dirPath");
            let recursive = tools::optional_bool_arg(args, "recursive").unwrap_or(false);
            tools::ls_dir(&dir, dir_path.as_deref(), recursive).unwrap_or_else(|err| err)
        },
        false,
    );
    let dir = app.project_dir.clone();
    app.register_tool(
        "create_rule_block",
        move |args| {
            let name = match tools::string_arg(args, "name") {
                Ok(name) => name,
                Err(err) => return err,
            };
            let rule = match tools::string_arg(args, "rule") {
                Ok(rule) => rule,
                Err(err) => return err,
            };
            let description = tools::optional_string_arg(args, "description");
            let globs = tools::optional_string_arg(args, "globs");
            let regex = tools::optional_string_arg(args, "regex");
            let always_apply = tools::optional_bool_arg(args, "alwaysApply");
            let scope = tools::optional_string_arg(args, "scope");
            tools::create_rule_block(
                &dir,
                tools::RuleSpec {
                    name: &name,
                    rule: &rule,
                    description: description.as_deref(),
                    globs: globs.as_deref(),
                    regex: regex.as_deref(),
                    always_apply,
                    scope: scope.as_deref(),
                },
            )
            .unwrap_or_else(|err| err)
        },
        true,
    );
    app.register_tool(
        "fetch_url_content",
        move |args| match tools::string_arg(args, "url") {
            Ok(url) => tools::fetch_url_content(&url),
            Err(err) => err,
        },
        true,
    );
    let dir = app.project_dir.clone();
    app.register_tool(
        "request_rule",
        move |args| match tools::string_arg(args, "name") {
            Ok(name) => tools::request_rule(&dir, &name).unwrap_or_else(|err| err),
            Err(err) => err,
        },
        false,
    );
    let dir = app.project_dir.clone();
    app.register_tool(
        "read_skill",
        move |args| match tools::string_arg(args, "skillName") {
            Ok(skill_name) => tools::read_skill(&dir, &skill_name).unwrap_or_else(|err| err),
            Err(err) => err,
        },
        false,
    );
    app.register_tool(
        "search_web",
        move |args| match tools::string_arg(args, "query") {
            Ok(query) => tools::search_web(&query),
            Err(err) => err,
        },
        false,
    );
    let dir = app.project_dir.clone();
    app.register_tool(
        "edit_existing_file",
        move |args| {
            let filepath = match tools::string_arg(args, "filepath") {
                Ok(filepath) => filepath,
                Err(err) => return err,
            };
            let changes = match tools::string_arg(args, "changes") {
                Ok(changes) => changes,
                Err(err) => return err,
            };
            tools::edit_existing_file(&dir, &filepath, &changes).unwrap_or_else(|err| err)
        },
        true,
    );
    let dir = app.project_dir.clone();
    app.register_tool(
        "single_find_and_replace",
        move |args| {
            let filepath = match tools::string_arg(args, "filepath") {
                Ok(filepath) => filepath,
                Err(err) => return err,
            };
            let old_string = match tools::string_arg(args, "old_string") {
                Ok(old_string) => old_string,
                Err(err) => return err,
            };
            let new_string = match tools::string_arg(args, "new_string") {
                Ok(new_string) => new_string,
                Err(err) => return err,
            };
            let replace_all = tools::optional_bool_arg(args, "replace_all").unwrap_or(false);
            tools::single_find_and_replace(
                &dir,
                &filepath,
                &old_string,
                &new_string,
                replace_all,
            )
            .unwrap_or_else(|err| err)
        },
        true,
    );
    let dir = app.project_dir.clone();
    app.register_tool(
        "grep_search",
        move |args| match tools::string_arg(args, "query") {
            Ok(query) => tools::grep_search(&dir, &query),
            Err(err) => err,
        },
        false,
    );
    let scheduler = app.scheduler.clone();
    app.register_tool(
        "schedule_task",
        move |args| match schedule_task(&scheduler, args) {
            Ok(msg) => msg,
            Err(err) => err,
        },
        true,
    );
    let scheduler = app.scheduler.clone();
    app.register_tool(
        "cancel_task",
        move |args| match args.get("id").and_then(serde_json::Value::as_u64) {
            Some(id) => {
                if scheduler.lock().unwrap().cancel(id) {
                    format!("cancelled task #{id}")
                } else {
                    format!("task #{id} not found")
                }
            }
            None => "cancel_task: missing id".into(),
        },
        false,
    );
}

fn parse_action(args: &serde_json::Value) -> Result<TaskAction, String> {
    let action = args
        .get("action")
        .ok_or("task: missing action")?;
    let tool = action
        .get("tool")
        .and_then(serde_json::Value::as_str)
        .ok_or("task: action.tool missing")?
        .to_string();
    let arguments = action
        .get("arguments")
        .cloned()
        .unwrap_or(serde_json::Value::Object(Default::default()));
    Ok(TaskAction { tool, arguments })
}

fn parse_at(at: &str) -> Result<Instant, String> {
    let mut parts = at.split(':');
    let hour: u32 = parts
        .next()
        .and_then(|p| p.parse().ok())
        .ok_or("task delay: invalid at format, use HH:MM")?;
    let minute: u32 = parts
        .next()
        .and_then(|p| p.parse().ok())
        .ok_or("task delay: invalid at format, use HH:MM")?;
    let now = chrono::Local::now();
    let mut due = now
        .date_naive()
        .and_hms_opt(hour, minute, 0)
        .ok_or("task delay: invalid time")?
        .and_local_timezone(chrono::Local)
        .single()
        .ok_or("task delay: invalid time")?;
    if due <= now {
        due += chrono::Duration::days(1);
    }
    let delta = (due - now).to_std().map_err(|_| "task delay: invalid time")?;
    Ok(Instant::now() + delta)
}

fn parse_condition(args: &serde_json::Value) -> Result<Condition, String> {
    let cond = args
        .get("condition")
        .ok_or("task monitor: missing condition")?;
    let ctype = cond
        .get("type")
        .and_then(serde_json::Value::as_str)
        .ok_or("task monitor: condition.type missing")?;
    match ctype {
        "output_contains" => Ok(Condition::OutputContains {
            cmd: cond
                .get("cmd")
                .and_then(serde_json::Value::as_str)
                .ok_or("task monitor: output_contains needs cmd")?
                .into(),
            contains: cond
                .get("contains")
                .and_then(serde_json::Value::as_str)
                .ok_or("task monitor: output_contains needs contains")?
                .into(),
        }),
        "exit_code" => Ok(Condition::ExitZero {
            cmd: cond
                .get("cmd")
                .and_then(serde_json::Value::as_str)
                .ok_or("task monitor: exit_code needs cmd")?
                .into(),
        }),
        "file_exists" => Ok(Condition::FileExists {
            path: cond
                .get("path")
                .and_then(serde_json::Value::as_str)
                .ok_or("task monitor: file_exists needs path")?
                .into(),
        }),
        other => Err(format!("task monitor: unknown condition type {other}")),
    }
}

fn schedule_task(scheduler: &Arc<Mutex<Scheduler>>, args: &serde_json::Value) -> Result<String, String> {
    let kind = args
        .get("type")
        .and_then(serde_json::Value::as_str)
        .ok_or("task: missing type (delay|schedule|monitor)")?;
    let action = parse_action(args)?;
    let (task_kind, detail) = match kind {
        "delay" => {
            if let Some(seconds) = args.get("seconds").and_then(serde_json::Value::as_u64) {
                (
                    TaskKind::Delay { due: Instant::now() + Duration::from_secs(seconds) },
                    format!("{seconds}s"),
                )
            } else if let Some(at) = args.get("at").and_then(serde_json::Value::as_str) {
                let due = parse_at(at)?;
                (TaskKind::Delay { due }, at.to_string())
            } else {
                return Err("task delay: provide seconds or at".into());
            }
        }
        "schedule" => {
            let interval = args
                .get("interval_seconds")
                .and_then(serde_json::Value::as_u64)
                .ok_or("task schedule: provide interval_seconds")?;
            let interval = Duration::from_secs(interval);
            (
                TaskKind::Schedule { interval, next: Instant::now() + interval },
                format!("every {}s", interval.as_secs()),
            )
        }
        "monitor" => {
            let condition = parse_condition(args)?;
            let check_every = Duration::from_secs(
                args.get("check_every_seconds")
                    .and_then(serde_json::Value::as_u64)
                    .unwrap_or(5),
            );
            let deadline = args
                .get("timeout_seconds")
                .and_then(serde_json::Value::as_u64)
                .map(|s| Instant::now() + Duration::from_secs(s));
            (TaskKind::Monitor { condition, check_every, deadline, last_check: None }, "monitor".into())
        }
        other => return Err(format!("task: unknown type {other}")),
    };
    let id = scheduler.lock().unwrap().register(task_kind, action);
    Ok(format!("task #{id} scheduled ({kind}, {detail})"))
}

type ToolHandler = Arc<dyn Fn(&serde_json::Value) -> String + Send + Sync>;

pub struct PlannedCall {
    pub name: String,
    pub arguments: serde_json::Value,
    pub tool_call_id: Option<String>,
    pub handler: Option<ToolHandler>,
    pub verdict: String,
}

pub enum ToolPlan {
    Blocked,
    NeedConfirm {
        name: String,
        arguments: serde_json::Value,
        tool_call_id: Option<String>,
    },
    Execute(Vec<PlannedCall>),
}

pub enum ToolCheck {
    Blocked,
    Ready,
    Judge {
        input: String,
        trace: Vec<ToolCallRecord>,
        pending: ToolCallRecord,
        tools: Vec<String>,
    },
}

struct ToolEntry {
    handler: ToolHandler,
    confirm: bool,
}

pub struct App {
    pub runtime: Runtime,
    pub supervisor: Arc<Supervisor>,
    pub remember: remember::Remember,
    models: Arc<SwitchableAdapter>,
    pub scheduler: Arc<Mutex<Scheduler>>,
    port: Arc<dyn LlmPort>,
    models_store: ModelsStore,
    model_store_dir: PathBuf,
    current_model: String,
    approvals: std::collections::HashSet<String>,
    approvals_path: PathBuf,
    project_dir: PathBuf,
    tools: std::collections::HashMap<String, ToolEntry>,
    pending_tool: std::collections::VecDeque<(String, serde_json::Value, Option<String>)>,
    tool_lock: Arc<tokio::sync::Mutex<()>>,
    tool_trace: Vec<ToolCallRecord>,
    current_user_input: String,
    supervise_blocks: u32,
    tool_rounds_executed: u32,
    executed_calls: Vec<(String, String)>,
    cancelled_tool_ids: std::collections::HashSet<String>,
    cycle: u64,
}

const TRACE_ARG_LIMIT: usize = 200;
const TRACE_OUTPUT_LIMIT: usize = 400;
const MAX_SUPERVISE_BLOCKS: u32 = 5;
const MAX_TOOL_ROUNDS: u32 = 200;

pub(crate) async fn run_tool_handler(
    handler: Option<ToolHandler>,
    name: String,
    arguments: serde_json::Value,
) -> String {
    match handler {
        Some(handler) => {
            let display_name = name.clone();
            tokio::task::spawn_blocking(move || {
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || handler(&arguments)))
                    .unwrap_or_else(|_| format!("(tool {display_name} panicked)"))
            })
            .await
            .unwrap_or_else(|_| format!("(tool {name} execution failed)"))
        }
        None => format!("unknown tool: {name}"),
    }
}

fn truncate(text: &str, limit: usize) -> String {
    if text.chars().count() <= limit {
        text.to_string()
    } else {
        let mut cut: String = text.chars().take(limit).collect();
        cut.push('…');
        cut
    }
}

fn pair_dialogue(turns: &[crate::app::remember::ConversationTurn]) -> Vec<DialogueTurn> {
    let mut dialogue = Vec::new();
    let mut user: Option<String> = None;
    let mut assistant_parts: Vec<String> = Vec::new();
    for turn in turns {
        match turn.role.as_str() {
            "user" => {
                if let Some(user) = user.take()
                    && !assistant_parts.is_empty()
                {
                    dialogue.push(DialogueTurn {
                        user,
                        assistant: assistant_parts.join("\n"),
                    });
                }
                assistant_parts.clear();
                user = Some(turn.content.clone());
            }
            "assistant" if user.is_some() && !turn.content.trim().is_empty() => {
                assistant_parts.push(turn.content.clone());
            }
            _ => {}
        }
    }
    if let Some(user) = user
        && !assistant_parts.is_empty()
    {
        dialogue.push(DialogueTurn {
            user,
            assistant: assistant_parts.join("\n"),
        });
    }
    dialogue
}

fn missing_required_argument(
    name: &str,
    arguments: &serde_json::Value,
    project_dir: &Path,
) -> Option<String> {
    let (_, schema) = tool_spec(name, project_dir);
    let required = schema.get("required")?.as_array()?;
    for field in required {
        let key = field.as_str().unwrap_or("");
        if arguments.get(key).is_none() {
            return Some(key.to_string());
        }
    }
    None
}

fn is_correction_broken(output: &str) -> bool {
    output.starts_with("missing string argument")
        || output.starts_with("missing number argument")
        || output.starts_with("unknown tool")
}

impl App {
    pub fn new(config: AppConfig) -> Result<Self, AdapterError> {
        let kind = match config.adapter {
            AdapterKind::OpenAi => "openai",
            AdapterKind::DeepSeek => "deepseek",
        };
        let env_client = models::build_from_env(kind, config.model.clone());
        let switchable = Arc::new(SwitchableAdapter::new(match &env_client {
            Ok(client) => client.clone(),
            Err(_) => models::placeholder(),
        }));
        let port: Arc<dyn LlmPort> = Arc::new(LlmAdapter::new(switchable.clone()));
        let app = Self::from_port(port, switchable, &config)?;
        Ok(app)
    }

    fn from_port(
        port: Arc<dyn LlmPort>,
        models: Arc<SwitchableAdapter>,
        config: &AppConfig,
    ) -> Result<Self, AdapterError> {
        let runtime = Runtime::new(port.clone(), config.tick_interval, vec![]);
        let supervisor = Arc::new(Supervisor::new(port.clone()));
        let app_port = port.clone();
        supervisor.set_enabled(config.supervisor_enabled);
        let remember = remember::Remember::new(port.clone(), &config.project_dir);
        let store_dir = config.project_dir.join(".prognosis");
        models::migrate_from_project(&store_dir);
        let models_dir = tools::global_config_dir();
        let models_store = ModelsStore::load(&models_dir);
        let approvals_path = store_dir.join("approvals.json");
        let approvals: std::collections::HashSet<String> = std::fs::read_to_string(&approvals_path)
            .ok()
            .and_then(|text| serde_json::from_str(&text).ok())
            .unwrap_or_default();
        let current_model = {
            let name = models_store.current.clone();
            if name.is_empty() {
                let kind = match config.adapter {
                    AdapterKind::DeepSeek => "deepseek",
                    AdapterKind::OpenAi => "openai",
                };
                config
                    .model
                    .clone()
                    .unwrap_or_else(|| models::default_model_name(kind))
            } else {
                name
            }
        };
        if let Some(entry) = models_store.get(&current_model)
            && let Ok(client) = models::build_client(entry) {
                models.switch(client);
            }
        let mut app = Self {
            runtime,
            supervisor,
            remember,
            port: app_port,
            models,
            project_dir: config.project_dir.clone(),
            tools: std::collections::HashMap::new(),
            scheduler: Arc::new(Mutex::new(Scheduler::new())),
            models_store,
            model_store_dir: models_dir,
            current_model,
            approvals,
            approvals_path: approvals_path.clone(),
            pending_tool: std::collections::VecDeque::new(),
            tool_lock: Arc::new(tokio::sync::Mutex::new(())),
            tool_trace: Vec::new(),
            current_user_input: String::new(),
            supervise_blocks: 0,
            tool_rounds_executed: 0,
            executed_calls: Vec::new(),
            cancelled_tool_ids: std::collections::HashSet::new(),
            cycle: 0,
        };
        register_default_tools(&mut app);
        let tool_defs: Vec<ToolDefinition> = app
            .tools
            .keys()
            .map(|name| {
                let (description, schema) = tool_spec(name, &config.project_dir);
                ToolDefinition::function(name.clone(), description, schema)
            })
            .collect();
        app.runtime.shutdown();
        app.runtime = Runtime::new(port.clone(), config.tick_interval, tool_defs);
        Ok(app)
    }

    pub fn current_model_name(&self) -> &str {
        &self.current_model
    }

    pub fn project_dir(&self) -> &Path {
        &self.project_dir
    }

    pub fn needs_setup(&self) -> bool {
        std::env::var("DEEPSEEK_API_KEY").is_err()
            && std::env::var("OPENAI_API_KEY").is_err()
            && self.models_store.entries.is_empty()
    }

    pub fn traces(&self) -> Vec<crate::runtime::types::TraceRecord> {
        self.runtime.trace_records().lock().unwrap().clone()
    }

    pub fn tool_trace(&self) -> Vec<ToolCallRecord> {
        self.tool_trace.clone()
    }

    pub fn context_limit(&self) -> usize {
        context_window(&self.current_model)
    }

    pub fn list_models(&self) -> Vec<String> {
        let mut names: Vec<String> = self
            .models_store
            .entries
            .iter()
            .map(|e| e.name.clone())
            .collect();
        names.sort();
        names.dedup();
        names
    }

    pub fn switch_model(&mut self, name: &str) -> Result<String, String> {
        let entry = self
            .models_store
            .get(name)
            .ok_or_else(|| format!("model {name} not found"))?
            .clone();
        models::switch_adapter(&self.models, &mut self.models_store, &self.model_store_dir, &entry)?;
        self.current_model = entry.name;
        Ok(format!("switched to {name}"))
    }

    pub fn add_model(
        &mut self,
        name: &str,
        kind: &str,
        base_url: &str,
        api_key: &str,
    ) -> Result<String, String> {
        let entry = ModelEntry {
            name: name.to_string(),
            kind: kind.to_string(),
            base_url: base_url.to_string(),
            api_key: api_key.to_string(),
        };
        models::switch_adapter(&self.models, &mut self.models_store, &self.model_store_dir, &entry)?;
        self.current_model = name.to_string();
        Ok(format!("model {name} added and activated"))
    }

    pub fn remove_model(&mut self, name: &str) -> Result<String, String> {
        if !self.models_store.remove(name) {
            return Err(format!("model {name} not found"));
        }
        self.models_store.save(&self.model_store_dir);
        if self.current_model == name {
            match self.models_store.entries.first().cloned() {
                Some(entry) => {
                    models::switch_adapter(
                        &self.models,
                        &mut self.models_store,
                        &self.model_store_dir,
                        &entry,
                    )?;
                    self.current_model = entry.name;
                }
                None => {
                    self.models.switch(models::placeholder());
                    self.current_model = String::new();
                }
            }
        }
        Ok(format!("model {name} removed"))
    }

    pub fn resume_session(&mut self, id: &str) -> Result<String, String> {
        let turns = self.remember.load_session(id);
        if turns.is_empty() {
            return Err(format!("session #{id} not found"));
        }
        let dialogue = pair_dialogue(&turns);
        let meta = self.next_meta();
        self.runtime
            .publish(Event::RestoreDialogue { meta, turns: dialogue });
        self.inject_context(&format!(
            "(Session #{id} resumed — the full conversation history is loaded; continue the earlier work.)"
        ));
        self.remember.set_current(id);
        Ok(format!("resumed session #{id} ({} turns)", turns.len()))
    }

    pub fn continue_session(&mut self) -> Result<String, String> {
        let current = self.remember.current_session().map(str::to_string);
        let id = self
            .remember
            .list_sessions()
            .iter()
            .rev()
            .find(|meta| Some(&meta.id) != current.as_ref())
            .map(|meta| meta.id.clone())
            .ok_or_else(|| "no previous session".to_string())?;
        self.resume_session(&id)
    }

    pub fn inject_archive_summary(&mut self, id: &str) -> Result<String, String> {
        let entry = self
            .remember
            .archive()
            .iter()
            .find(|entry| entry.id == id)
            .ok_or_else(|| format!("no archived session #{id}"))?;
        let mut text = format!("(Memory: #{id})\n{}", entry.summary);
        if !entry.highlights.is_empty() {
            text.push_str(&format!(
                "\nhighlights: {}",
                entry.highlights.join(", ")
            ));
        }
        self.inject_context(&text);
        Ok(format!("remembered session #{id}"))
    }

    pub fn note_assistant_turn(&mut self, content: &str) {
        if !content.trim().is_empty() {
            self.remember.append_turn("assistant", content);
        }
    }

    pub fn clear(&mut self) {
        self.tool_trace.clear();
        self.supervise_blocks = 0;
        self.current_user_input = String::new();
        let meta = self.next_meta();
        self.runtime.publish(Event::ConversationCleared { meta });
    }

    fn save_approvals(&self) {
        if let Ok(text) = serde_json::to_string(&self.approvals.iter().collect::<Vec<_>>()) {
            let _ = std::fs::create_dir_all(self.approvals_path.parent().unwrap_or(Path::new(".")));
            let _ = std::fs::write(&self.approvals_path, text);
        }
    }

    pub fn is_approved(&self, name: &str) -> bool {
        self.approvals.contains(name)
    }

    pub fn remember_approval(&mut self, name: &str) {
        self.approvals.insert(name.to_string());
        self.save_approvals();
    }

    pub fn clear_approval(&mut self, name: &str) {
        self.approvals.remove(name);
        self.save_approvals();
    }

    pub fn approvals(&self) -> Vec<String> {
        let mut list: Vec<String> = self.approvals.iter().cloned().collect();
        list.sort();
        list
    }

    pub fn toggle_rule(&mut self, name: &str) -> bool {
        let enabled = tools::is_rule_enabled(&self.project_dir, name);
        tools::set_rule_enabled(&self.project_dir, name, !enabled);
        self.refresh_context();
        !enabled
    }

    pub fn toggle_skill(&mut self, name: &str) -> bool {
        let enabled = tools::is_skill_enabled(&self.project_dir, name);
        tools::set_skill_enabled(&self.project_dir, name, !enabled);
        self.refresh_context();
        !enabled
    }

    pub fn register_tool(
        &mut self,
        name: impl Into<String>,
        handler: impl Fn(&serde_json::Value) -> String + Send + Sync + 'static,
        confirm: bool,
    ) {
        let name = name.into();
        self.tools.insert(
            name,
            ToolEntry {
                handler: Arc::new(handler),
                confirm,
            },
        );
    }

    pub fn pending_tool_name(&self) -> Option<String> {
        self.pending_tool.front().map(|(name, _, _)| name.clone())
    }

    pub fn cancel_generation(&mut self) {
        let meta = self.next_meta();
        self.runtime.publish(Event::CancelGeneration { meta });
    }

    pub fn take_pending_execution(&mut self, approved: bool) -> Option<PlannedCall> {
        let (name, arguments, tool_call_id) = self.pending_tool.pop_front()?;
        if !approved {
            let output = "user denied the tool call".to_string();
            self.tool_trace.push(ToolCallRecord {
                name: name.clone(),
                arguments: truncate(&arguments.to_string(), TRACE_ARG_LIMIT),
                output: truncate(&output, TRACE_OUTPUT_LIMIT),
            });
            let meta = self.next_meta();
            self.runtime.publish(Event::ToolResult {
                meta,
                result: ToolResult {
                    name: name.clone(),
                    output: output.clone(),
                    tool_call_id,
                },
                verdict: None,
            });
            self.runtime.publish(Event::CycleStart { meta });
            self.runtime.publish(Event::Perception {
                meta,
                payload: PerceptionPayload {
                    source: PerceptionSource::ToolResult,
                    content: format!("(Tool {name} result)\n{output}"),
                    salience: 0.8,
                },
            });
            return None;
        }
        let handler = self.tools.get(&name).map(|entry| entry.handler.clone());
        Some(PlannedCall {
            name,
            arguments,
            tool_call_id,
            handler,
            verdict: "allowed".to_string(),
        })
    }

    fn trace_summary(&self) -> String {
        if self.tool_trace.is_empty() {
            return String::new();
        }
        let lines = self
            .tool_trace
            .iter()
            .map(|record| format!("- {} {}", record.name, record.arguments))
            .collect::<Vec<_>>()
            .join("\n");
        format!("\n\n[Tools already executed or supervised in this task]\n{lines}")
    }

    #[cfg(test)]
    async fn execute_tool(&self, name: &str, arguments: &serde_json::Value) -> String {
        let handler = self.tools.get(name).map(|entry| entry.handler.clone());
        run_tool_handler(handler, name.to_string(), arguments.clone()).await
    }

    pub async fn start(&mut self) {
        self.remember.start_session().await;
        self.runtime.wait_ready().await;
    }

    fn next_meta(&mut self) -> EventMeta {
        self.cycle += 1;
        EventMeta {
            cycle_id: CycleId(self.cycle),

        }
    }

    fn refresh_context(&mut self) {
        let rules = tools::load_rules(&self.project_dir);
        let skills = tools::load_skills(&self.project_dir);
        let meta = self.next_meta();
        self.runtime.publish(Event::ContextUpdate { meta, rules, skills });
    }

    pub fn submit(&mut self, text: &str) {
        self.tool_trace.clear();
        self.supervise_blocks = 0;
        self.tool_rounds_executed = 0;
        self.executed_calls.clear();
        self.current_user_input = text.to_string();
        self.remember.append_turn("user", text);
        self.refresh_context();
        let meta = self.next_meta();
        self.runtime.publish(Event::TaskSetUpdate {
            meta,
            task_set: TaskSetState {
                goal: text.to_string(),
                priority: 1.0,
                progress: 0.0,
            },
        });
        self.runtime.publish(Event::CycleStart { meta });
        self.runtime.publish(Event::Perception {
            meta,
            payload: PerceptionPayload {
                source: PerceptionSource::User,
                content: text.to_string(),
                salience: 0.5,
            },
        });
    }

    pub fn inject_context(&mut self, background: &str) {
        let meta = self.next_meta();
        self.runtime.publish(Event::CycleStart { meta });
        self.runtime.publish(Event::Perception {
            meta,
            payload: PerceptionPayload {
                source: PerceptionSource::System,
                content: format!("(Conversation background)\n{background}"),
                salience: 0.3,
            },
        });
    }

    pub async fn handle_call_tool(
        &mut self,
        name: String,
        arguments: serde_json::Value,
        tool_call_id: Option<String>,
    ) -> Option<String> {
        match self
            .prepare_tool_call(name.clone(), arguments.clone(), tool_call_id.clone())
            .await
        {
            ToolPlan::Blocked => None,
            ToolPlan::NeedConfirm { .. } => {
                Some(format!("Tool {name} requests execution, please confirm"))
            }
            ToolPlan::Execute(calls) => {
                for call in calls {
                    let output = run_tool_handler(
                        call.handler,
                        call.name.clone(),
                        call.arguments.clone(),
                    )
                    .await;
                    self.finish_tool_call(
                        call.name,
                        call.arguments,
                        call.tool_call_id,
                        output,
                        &call.verdict,
                    )
                    .await;
                }
                None
            }
        }
    }

    pub async fn prepare_tool_call(
        &mut self,
        name: String,
        arguments: serde_json::Value,
        tool_call_id: Option<String>,
    ) -> ToolPlan {
        match self
            .check_tool_call(name.clone(), arguments.clone(), tool_call_id.clone())
            .await
        {
            ToolCheck::Blocked => ToolPlan::Blocked,
            ToolCheck::Ready => {
                self.apply_verdict(Verdict::Allow, name, arguments, tool_call_id)
                    .await
            }
            ToolCheck::Judge {
                input,
                trace,
                pending,
                tools,
            } => {
                let verdict = self
                    .supervisor
                    .judge(&input, &trace, &pending, &tools)
                    .await;
                self.apply_verdict(verdict, name, arguments, tool_call_id)
                    .await
            }
        }
    }

    pub async fn check_tool_call(
        &mut self,
        name: String,
        arguments: serde_json::Value,
        tool_call_id: Option<String>,
    ) -> ToolCheck {
        let tool_lock = self.tool_lock.clone();
        let _guard = tool_lock.lock().await;
        let pending = ToolCallRecord {
            name: name.clone(),
            arguments: truncate(&arguments.to_string(), TRACE_ARG_LIMIT),
            output: String::new(),
        };
        if let Some(missing) = missing_required_argument(&name, &arguments, &self.project_dir)
            && self.supervise_blocks < MAX_SUPERVISE_BLOCKS {
                self.supervise_blocks += 1;
                self.tool_trace.push(ToolCallRecord {
                    name: name.clone(),
                    arguments: truncate(&arguments.to_string(), TRACE_ARG_LIMIT),
                    output: format!("(rejected: missing required argument \"{missing}\")"),
                });
                let meta2 = self.next_meta();
                self.runtime.publish(Event::CycleStart { meta: meta2 });
                self.runtime.publish(Event::ToolResult {
                    meta: meta2,
                    result: ToolResult {
                        name: name.clone(),
                        output: format!("(rejected: missing required argument \"{missing}\")"),
                        tool_call_id: tool_call_id.clone(),
                    },
                    verdict: Some("blocked".to_string()),
                });
                self.runtime.publish(Event::Perception {
                    meta: meta2,
                    payload: PerceptionPayload {
                        source: PerceptionSource::Internal,
                        content: format!(
                            "(Tool call rejected: missing required argument \"{missing}\" for {name}) Reissue the call with complete arguments.{}",
                            self.trace_summary()
                        ),
                        salience: 0.9,
                    },
                });
                return ToolCheck::Blocked;
            }
        if self.tool_rounds_executed >= MAX_TOOL_ROUNDS {
            let meta2 = self.next_meta();
            self.runtime.publish(Event::CycleStart { meta: meta2 });
            self.runtime.publish(Event::ToolResult {
                meta: meta2,
                result: ToolResult {
                    name: name.clone(),
                    output: "(tool round limit reached; call not executed) Stop calling tools and provide your final answer now."
                        .to_string(),
                    tool_call_id: tool_call_id.clone(),
                },
                verdict: Some("blocked".to_string()),
            });
            self.runtime.publish(Event::Perception {
                meta: meta2,
                payload: PerceptionPayload {
                    source: PerceptionSource::Internal,
                    content: format!(
                        "(Tool round limit reached ({MAX_TOOL_ROUNDS}). Stop calling tools and provide your final answer now.){}",
                        self.trace_summary()
                    ),
                    salience: 0.9,
                },
            });
            return ToolCheck::Blocked;
        }
        let args_json = arguments.to_string();
        let is_duplicate = self
            .executed_calls
            .iter()
            .any(|(exec_name, exec_args)| exec_name == &name && exec_args == &args_json);
        if is_duplicate {
            let meta2 = self.next_meta();
            self.runtime.publish(Event::CycleStart { meta: meta2 });
            self.runtime.publish(Event::ToolResult {
                meta: meta2,
                result: ToolResult {
                    name: name.clone(),
                    output: "(duplicate tool call with identical arguments; not executed) Use the result you already received or choose a different action."
                        .to_string(),
                    tool_call_id: tool_call_id.clone(),
                },
                verdict: Some("blocked".to_string()),
            });
            self.runtime.publish(Event::Perception {
                meta: meta2,
                payload: PerceptionPayload {
                    source: PerceptionSource::Internal,
                    content: format!(
                        "(Duplicate tool call: {name} with identical arguments was already executed. Do not repeat it; use its result or change your approach.){}",
                        self.trace_summary()
                    ),
                    salience: 0.9,
                },
            });
            return ToolCheck::Blocked;
        }
        if !self.supervisor.is_enabled() {
            return ToolCheck::Ready;
        }
        ToolCheck::Judge {
            input: self.current_user_input.clone(),
            trace: self.tool_trace.clone(),
            pending,
            tools: self.tools.keys().cloned().collect(),
        }
    }

    pub async fn apply_verdict(
        &mut self,
        verdict: Verdict,
        name: String,
        arguments: serde_json::Value,
        tool_call_id: Option<String>,
    ) -> ToolPlan {
        let tool_lock = self.tool_lock.clone();
        let _guard = tool_lock.lock().await;
        match verdict {
            Verdict::Regenerate { reason } => {
                if self.supervise_blocks < MAX_SUPERVISE_BLOCKS {
                    self.supervise_blocks += 1;
                    self.tool_trace.push(ToolCallRecord {
                        name: name.clone(),
                        arguments: truncate(&arguments.to_string(), TRACE_ARG_LIMIT),
                        output: format!("(blocked by supervisor: {reason})"),
                    });
                    let meta2 = self.next_meta();
                    self.runtime.publish(Event::CycleStart { meta: meta2 });
                    self.runtime.publish(Event::ToolResult {
                        meta: meta2,
                        result: ToolResult {
                            name: name.clone(),
                            output: format!("(blocked by supervisor: {reason})"),
                            tool_call_id: tool_call_id.clone(),
                        },
                        verdict: Some("blocked".to_string()),
                    });
                    self.runtime.publish(Event::Perception {
                        meta: meta2,
                        payload: PerceptionPayload {
                            source: PerceptionSource::Internal,
                            content: format!(
                                "(Tool call blocked by supervisor: {reason}) Reconsider your tool plan.{}",
                                self.trace_summary()
                            ),
                            salience: 0.9,
                        },
                    });
                    return ToolPlan::Blocked;
                }
            }
            Verdict::Corrected { calls } => {
                if self.supervise_blocks < MAX_SUPERVISE_BLOCKS {
                    self.supervise_blocks += 1;
                    if calls.is_empty() {
                        self.tool_trace.push(ToolCallRecord {
                            name: name.clone(),
                            arguments: truncate(&arguments.to_string(), TRACE_ARG_LIMIT),
                            output: "(corrected by supervisor to: empty plan)".to_string(),
                        });
                        let meta2 = self.next_meta();
                        self.runtime.publish(Event::CycleStart { meta: meta2 });
                        self.runtime.publish(Event::ToolResult {
                            meta: meta2,
                            result: ToolResult {
                                name: name.clone(),
                                output: "(corrected by supervisor to: empty plan)".to_string(),
                                tool_call_id: tool_call_id.clone(),
                            },
                            verdict: Some("blocked".to_string()),
                        });
                        self.runtime.publish(Event::Perception {
                            meta: meta2,
                            payload: PerceptionPayload {
                                source: PerceptionSource::Internal,
                                content: format!(
                                    "(Tool call correction declined by supervisor: the corrected plan was empty) Reconsider your tool plan.{}",
                                    self.trace_summary()
                                ),
                                salience: 0.9,
                            },
                        });
                        return ToolPlan::Blocked;
                    }
                    self.tool_trace.push(ToolCallRecord {
                        name: name.clone(),
                        arguments: truncate(&arguments.to_string(), TRACE_ARG_LIMIT),
                        output: format!(
                            "(corrected by supervisor to: {})",
                            calls
                                .iter()
                                .map(|call| format!("{} {}", call.name, call.arguments))
                                .collect::<Vec<_>>()
                                .join("; ")
                        ),
                    });
                    let planned = calls
                        .into_iter()
                        .filter(|call| self.tools.contains_key(&call.name))
                        .map(|call| {
                            let args = serde_json::from_str(&call.arguments)
                                .unwrap_or(serde_json::json!({}));
                            PlannedCall {
                                name: call.name.clone(),
                                arguments: args,
                                tool_call_id: tool_call_id.clone(),
                                handler: self.tools.get(&call.name).map(|e| e.handler.clone()),
                                verdict: "corrected".to_string(),
                            }
                        })
                        .collect();
                    return ToolPlan::Execute(planned);
                }
            }
            Verdict::Allow => {}
        }
        let needs_confirm = self
            .tools
            .get(&name)
            .map(|entry| entry.confirm)
            .unwrap_or(false)
            && !self.is_approved(&name);
        if needs_confirm {
            self.pending_tool
                .push_back((name.clone(), arguments.clone(), tool_call_id.clone()));
            return ToolPlan::NeedConfirm {
                name,
                arguments,
                tool_call_id,
            };
        }
        let handler = self.tools.get(&name).map(|entry| entry.handler.clone());
        ToolPlan::Execute(vec![PlannedCall {
            name,
            arguments,
            tool_call_id,
            handler,
            verdict: "allowed".to_string(),
        }])
    }

    pub fn cancel_tools(&mut self, calls: &[(String, String)]) {
        for (id, name) in calls {
            self.cancelled_tool_ids.insert(id.clone());
            let meta = self.next_meta();
            self.runtime.publish(Event::ToolResult {
                meta,
                result: ToolResult {
                    name: name.clone(),
                    output: "(tool call cancelled by user)".to_string(),
                    tool_call_id: Some(id.clone()),
                },
                verdict: Some("blocked".to_string()),
            });
            self.runtime.publish(Event::CycleStart { meta });
        }
    }

    pub async fn finish_tool_call(
        &mut self,
        name: String,
        arguments: serde_json::Value,
        tool_call_id: Option<String>,
        output: String,
        verdict: &str,
    ) {
        if let Some(id) = &tool_call_id
            && self.cancelled_tool_ids.contains(id)
        {
            self.cancelled_tool_ids.remove(id);
            return;
        }
        self.refresh_context();
        self.tool_rounds_executed += 1;
        self.executed_calls
            .push((name.clone(), arguments.to_string()));
        self.tool_trace.push(ToolCallRecord {
            name: name.clone(),
            arguments: truncate(&arguments.to_string(), TRACE_ARG_LIMIT),
            output: truncate(&output, TRACE_OUTPUT_LIMIT),
        });
        let meta2 = self.next_meta();
        self.runtime.publish(Event::ToolResult {
            meta: meta2,
            result: ToolResult {
                name: name.clone(),
                output: output.clone(),
                tool_call_id,
            },
            verdict: Some(verdict.to_string()),
        });
        self.runtime.publish(Event::CycleStart { meta: meta2 });
        let note = if verdict == "corrected" && is_correction_broken(&output) {
            format!(
                "(Tool call correction FAILED: the corrected call was malformed: {output}) Do not retry the same call; reconsider your tool plan and issue the call yourself with complete arguments."
            )
        } else if verdict == "corrected" {
            format!("(Tool call corrected by supervisor: {name} — the corrected call was executed instead; review its result)\n{output}")
        } else {
            format!("(Tool {name} result)\n{output}")
        };
        self.runtime.publish(Event::Perception {
            meta: meta2,
            payload: PerceptionPayload {
                source: PerceptionSource::ToolResult,
                content: note,
                salience: 0.8,
            },
        });
    }

    pub async fn compact(&mut self) -> String {
        let transcript = match self.remember.current_session() {
            Some(id) => self
                .remember
                .load_session(id)
                .iter()
                .map(|turn| format!("{}: {}", turn.role, turn.content))
                .collect::<Vec<_>>()
                .join("\n"),
            None => String::new(),
        };
        let prompt = "You are an archivist. The conversation context must be compressed so the agent can continue without losing important information. This summary is the only thing the agent will remember about the earlier conversation, so completeness of decisions matters more than brevity.\
\n\n# Task\
\nCompress the conversation into a single summary paragraph.\
\n\n# Rules\
\n- Keep: topic flow, resolved items, user preferences, open questions, current task state, decisions and their reasons, concrete facts (names, numbers, paths).\
\n- Drop: greetings, chit-chat, tool call details, repeated restatements.\
\n- Preserve the language of the conversation; if the conversation mixes languages, follow the dominant one.\
\n- Do not invent anything not present in the conversation; if something is unresolved, say it is unresolved rather than smoothing it over.\
\n- Aim for one dense paragraph (roughly 5-15 sentences depending on conversation size); the goal is that a reader who never saw the conversation can continue the work from this paragraph alone.\
\n- Reply with the summary text only, no preface, no JSON.";
        let summary = match self
            .call_llm(prompt, &transcript)
            .await
        {
            Some(text) if !text.trim().is_empty() => text.trim().to_string(),
            _ => {
                let fallback = format!("(compressed {} turns)", transcript.lines().count() / 2);
                fallback
            }
        };
        let meta = self.next_meta();
        self.runtime.publish(Event::CompactContext {
            meta,
            summary: summary.clone(),
        });
        summary
    }

    async fn call_llm(&self, system: &str, user: &str) -> Option<String> {
        let request = GenerateRequest {
            messages: vec![
                crate::adapter::types::Message::system(system),
                crate::adapter::types::Message::user(user),
            ],
            modulation: ModulationContext::default(),
            tools: None,
        };
        let cancel = tokio_util::sync::CancellationToken::new();
        let stream = self.port.generate(&request, &cancel).await.ok()?;
        let mut stream = stream;
        let mut content = String::new();
        while let Some(item) = futures::StreamExt::next(&mut stream).await {
            if let Ok(chunk) = item
                && let Some(text) = chunk.content() {
                    content.push_str(text);
                }
        }
        if content.trim().is_empty() {
            None
        } else {
            Some(content)
        }
    }

    pub fn scheduler_tick(&mut self) {
        let fired = self.scheduler.lock().unwrap().poll(Instant::now());
        for item in fired {
            match item {
                Fired::Execute { id, action, label } => {
                    let meta = self.next_meta();
                    self.refresh_context();
                    let handler = self
                        .tools
                        .get(&action.tool)
                        .map(|entry| entry.handler.clone());
                    let name = action.tool.clone();
                    let arguments = action.arguments.clone();
                    let bus = self.runtime.bus();
                    tokio::spawn(async move {
                        let output = run_tool_handler(handler, name.clone(), arguments).await;
                        bus.publish(Event::CycleStart { meta });
                        bus.publish(Event::Perception {
                            meta,
                            payload: PerceptionPayload {
                                source: PerceptionSource::Scheduled,
                                content: format!("(Scheduled task #{id} — {label})\n{output}"),
                                salience: 0.8,
                            },
                        });
                    });
                }
                Fired::MonitorTimeout { id } => {
                    let meta = self.next_meta();
                    self.runtime.publish(Event::CycleStart { meta });
                    self.runtime.publish(Event::Perception {
                        meta,
                        payload: PerceptionPayload {
                            source: PerceptionSource::Scheduled,
                            content: format!("(Monitor task #{id} timed out)"),
                            salience: 0.6,
                        },
                    });
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use futures::{Stream, StreamExt};
    use std::pin::Pin;

    use crate::adapter::error::AdapterError;
    use crate::adapter::traits::LanguageModelAdapter;
    use crate::adapter::types::CompletionRequest;
    use crate::runtime::event::EventKind;
    use crate::runtime::ports::LlmPort;
    use crate::runtime::types::GenerateRequest;

    struct MockPort;

    #[async_trait]
    impl LlmPort for MockPort {
        async fn generate<'a>(
            &'a self,
            _request: &'a GenerateRequest,
            _cancel: &'a tokio_util::sync::CancellationToken,
        ) -> Result<
            Pin<Box<dyn Stream<Item = Result<crate::adapter::types::CompletionChunk, AdapterError>> + Send + 'a>>,
            AdapterError,
        > {
            Ok(Box::pin(futures::stream::empty()))
        }
    }

    struct MockAdapter;

    #[async_trait]
    impl LanguageModelAdapter for MockAdapter {
        fn id(&self) -> &str {
            "mock"
        }
        fn capabilities(&self) -> crate::adapter::types::AdapterCapabilities {
            crate::adapter::types::AdapterCapabilities::default()
        }
        async fn stream<'a>(
            &'a self,
            _request: CompletionRequest,
            _cancel: &'a tokio_util::sync::CancellationToken,
        ) -> Result<
            Pin<Box<dyn Stream<Item = Result<crate::adapter::types::CompletionChunk, AdapterError>> + Send + 'a>>,
            AdapterError,
        > {
            Ok(Box::pin(futures::stream::empty()))
        }
    }

    fn test_app() -> App {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|duration| duration.as_nanos())
            .unwrap_or(0);
        let dir = std::env::temp_dir().join(format!(
            "prognosis_app_test_{}_{unique}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let config = AppConfig {
            adapter: AdapterKind::OpenAi,
            model: None,
            supervisor_enabled: false,
            project_dir: dir,
            tick_interval: Duration::from_millis(100),
        };
        let port: Arc<dyn LlmPort> = Arc::new(MockPort);
        let adapter: Arc<dyn LanguageModelAdapter> = Arc::new(MockAdapter);
        let switchable = Arc::new(SwitchableAdapter::new(adapter));
        let mut app = App::from_port(port, switchable, &config).unwrap();
        let models_dir = config.project_dir.join("test_models");
        app.models_store = models::ModelsStore::load(&models_dir);
        app.model_store_dir = models_dir;
        app.current_model = String::new();
        app
    }

    #[tokio::test]
    async fn task_tool_registers_delay_task() {
        let app = test_app();
        let output = app.execute_tool(
            "schedule_task",
            &serde_json::json!({
                "type": "delay",
                "seconds": 30,
                "action": {"tool": "ls", "arguments": {}}
            }),
        ).await;
        assert!(output.contains("task #1 scheduled (delay, 30s)"), "{output}");
        assert_eq!(app.scheduler.lock().unwrap().tasks().len(), 1);
    }

    #[tokio::test]
    async fn task_tool_registers_schedule_and_monitor() {
        let app = test_app();
        let out = app.execute_tool(
            "schedule_task",
            &serde_json::json!({
                "type": "schedule",
                "interval_seconds": 60,
                "action": {"tool": "ls", "arguments": {}}
            }),
        ).await;
        assert!(out.contains("task #1 scheduled (schedule, every 60s)"), "{out}");
        let out = app.execute_tool(
            "schedule_task",
            &serde_json::json!({
                "type": "monitor",
                "condition": {"type": "output_contains", "cmd": "echo hi", "contains": "hi"},
                "check_every_seconds": 2,
                "timeout_seconds": 60,
                "action": {"tool": "ls", "arguments": {}}
            }),
        ).await;
        assert!(out.contains("task #2 scheduled (monitor"), "{out}");
        assert_eq!(app.scheduler.lock().unwrap().tasks().len(), 2);
    }

    #[tokio::test]
    async fn task_tool_rejects_bad_input() {
        let app = test_app();
        let out = app.execute_tool(
            "schedule_task",
            &serde_json::json!({"type": "nope", "action": {"tool": "ls"}}),
        ).await;
        assert!(out.contains("unknown type"), "{out}");
        let out = app.execute_tool(
            "schedule_task",
            &serde_json::json!({"type": "delay", "action": {"tool": "ls"}}),
        ).await;
        assert!(out.contains("provide seconds or at"), "{out}");
    }

    #[tokio::test]
    async fn cancel_task_removes_scheduled_task() {
        let app = test_app();
        app.execute_tool(
            "schedule_task",
            &serde_json::json!({
                "type": "schedule",
                "interval_seconds": 60,
                "action": {"tool": "ls", "arguments": {}}
            }),
        ).await;
        let out = app.execute_tool("cancel_task", &serde_json::json!({"id": 1})).await;
        assert!(out.contains("cancelled task #1"), "{out}");
        assert!(app.scheduler.lock().unwrap().tasks().is_empty());
        let out = app.execute_tool("cancel_task", &serde_json::json!({"id": 1})).await;
        assert!(out.contains("not found"), "{out}");
    }

    #[tokio::test]
    async fn handle_call_tool_rejects_missing_required_argument_before_supervisor() {
        let mut app = test_app();
        let bus = app.runtime.bus();
        let mut rx = Box::pin(bus.subscribe_kinds(&[EventKind::Perception]));
        app.submit("write a file");
        let result = app
            .handle_call_tool("create_new_file".into(), serde_json::json!({"contents": "x"}), None)
            .await;
        assert!(result.is_none(), "must be rejected without passing through supervisor");
        assert_eq!(app.tool_trace.len(), 1, "rejection must be recorded");
        assert!(app.tool_trace[0].output.contains("missing required argument"));
        let rejected = tokio::time::timeout(Duration::from_secs(2), async {
            loop {
                match rx.next().await {
                    Some(Event::Perception { payload, .. }) => {
                        if payload.content.contains("missing required argument") {
                            break payload.content.clone();
                        }
                    }
                    _ => continue,
                }
            }
        })
        .await
        .expect("rejection perception never arrived");
        assert!(rejected.contains("missing required argument \"filepath\""));
    }

    #[tokio::test]
    async fn handle_call_tool_executes_non_confirm_tool_and_publishes_result() {
        let mut app = test_app();
        let bus = app.runtime.bus();
        let mut rx = Box::pin(bus.subscribe_kinds(&[EventKind::Action]));
        app.submit("what time is it");
        let result = app
            .handle_call_tool("ls".into(), serde_json::json!({}), None)
            .await;
        assert!(result.is_none());
        assert_eq!(app.tool_trace.len(), 1);
        match tokio::time::timeout(Duration::from_secs(2), rx.next()).await {
            Ok(Some(Event::ToolResult { result, .. })) => {
                assert_eq!(result.name, "ls");
            }
            other => panic!("expected tool result, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn handle_call_tool_requests_confirmation_for_confirm_tool() {
        let mut app = test_app();
        app.submit("write a file");
        let result = app
            .handle_call_tool(
                "create_new_file".into(),
                serde_json::json!({"filepath": "tmp.txt", "contents": "x"}),
                None,
            )
            .await;
        assert!(result.is_some());
        assert!(!app.pending_tool.is_empty());
    }

    #[tokio::test]
    async fn compact_publishes_context_event() {
        let mut app = test_app();
        app.submit("what is the weather");
        app.remember.append_turn("user", "what is the weather");
        app.remember.append_turn("assistant", "it is sunny");
        let bus = app.runtime.bus();
        let mut rx = Box::pin(bus.subscribe_kinds(&[EventKind::Context]));
        let summary = app.compact().await;
        assert!(!summary.trim().is_empty());
        match tokio::time::timeout(Duration::from_secs(2), rx.next()).await {
            Ok(Some(Event::CompactContext { summary: event_summary, .. })) => {
                assert_eq!(event_summary, summary);
            }
            other => panic!("expected compact context, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn approvals_persist_and_gate_confirm() {
        let mut app = test_app();
        assert!(!app.is_approved("run_terminal_command"));
        app.remember_approval("run_terminal_command");
        assert!(app.is_approved("run_terminal_command"));
        assert!(app.approvals().contains(&"run_terminal_command".to_string()));
        app.clear_approval("run_terminal_command");
        assert!(!app.is_approved("run_terminal_command"));
        let result = app
            .handle_call_tool("run_terminal_command".into(), serde_json::json!({"command": "echo hi"}), None)
            .await;
        assert!(result.is_some(), "unapproved confirm tool still asks");
        app.remember_approval("run_terminal_command");
        let result = app
            .handle_call_tool("run_terminal_command".into(), serde_json::json!({"command": "echo hi"}), None)
            .await;
        assert!(result.is_none(), "approved confirm tool executes directly");
    }

    #[tokio::test]
    async fn model_store_roundtrip_in_project_dir() {
        let mut app = test_app();
        let name = format!("test-model-{}", std::process::id());
        let msg = app
            .add_model(&name, "openai", "", "sk-test-0000")
            .unwrap_or_else(|err| panic!("add model failed: {err}"));
        assert!(msg.contains("added and activated"), "{msg}");
        assert_eq!(app.current_model_name(), name);
        assert!(app.list_models().contains(&name));
        let msg = app.switch_model(&name).unwrap_or_else(|err| panic!("switch failed: {err}"));
        assert!(msg.contains("switched"), "{msg}");
    }

    #[tokio::test]
    async fn submit_publishes_task_set() {
        let mut app = test_app();
        let bus = app.runtime.bus();
        let mut rx = Box::pin(bus.subscribe_kinds(&[EventKind::TaskSet]));
        app.submit("fix the bug");
        match tokio::time::timeout(Duration::from_secs(2), rx.next()).await {
            Ok(Some(Event::TaskSetUpdate { task_set, .. })) => {
                assert_eq!(task_set.goal, "fix the bug");
                assert_eq!(task_set.progress, 0.0);
            }
            other => panic!("expected task set update, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn scheduler_tick_fires_delay_and_publishes_perception() {
        let mut app = test_app();
        let bus = app.runtime.bus();
        let mut rx = Box::pin(bus.subscribe_kinds(&[EventKind::Perception]));
        app.scheduler.lock().unwrap().register(
            TaskKind::Delay { due: Instant::now() - Duration::from_secs(1) },
            TaskAction {
                tool: "ls".into(),
                arguments: serde_json::json!({}),
            },
        );
        app.scheduler_tick();
        assert!(app.scheduler.lock().unwrap().tasks().is_empty());
        match tokio::time::timeout(Duration::from_secs(2), rx.next()).await {
            Ok(Some(Event::Perception { payload, .. })) => {
                assert_eq!(payload.source, PerceptionSource::Scheduled);
                assert!(payload.content.contains("Scheduled task #1"));
            }
            other => panic!("expected scheduled perception, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn scheduler_tick_timeout_publishes_perception() {
        let mut app = test_app();
        let bus = app.runtime.bus();
        let mut rx = Box::pin(bus.subscribe_kinds(&[EventKind::Perception]));
        app.scheduler.lock().unwrap().register(
            TaskKind::Monitor {
                condition: Condition::ExitZero { cmd: "false".into() },
                check_every: Duration::from_secs(1),
                deadline: Some(Instant::now() - Duration::from_secs(1)),
                last_check: None,
            },
            TaskAction {
                tool: "ls".into(),
                arguments: serde_json::json!({}),
            },
        );
        app.scheduler_tick();
        assert!(app.scheduler.lock().unwrap().tasks().is_empty());
        match tokio::time::timeout(Duration::from_secs(2), rx.next()).await {
            Ok(Some(Event::Perception { payload, .. })) => {
                assert_eq!(payload.source, PerceptionSource::Scheduled);
                assert!(payload.content.contains("timed out"), "{}", payload.content);
            }
            other => panic!("expected timeout perception, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn resume_session_injects_and_sets_current() {
        let mut app = test_app();
        app.remember.start_session().await;
        app.remember.append_turn("user", "hello");
        app.remember.append_turn("assistant", "hi there");
        let id = app.remember.list_sessions()[0].id.clone();
        let message = app.resume_session(&id).expect("resume should succeed");
        assert!(message.contains("resumed session"), "{message}");
        assert!(message.contains(&id), "{message}");
        assert_eq!(app.remember.current_session(), Some(id.as_str()));
    }

    #[tokio::test]
    async fn resume_session_unknown_id_errors() {
        let mut app = test_app();
        assert!(app.resume_session("s9999").is_err());
    }

    #[tokio::test]
    async fn continue_session_resumes_most_recent() {
        let mut app = test_app();
        assert!(app.continue_session().is_err(), "no sessions yet");
        app.remember.start_session().await;
        app.remember.append_turn("user", "first turn");
        app.remember.start_session().await;
        app.remember.append_turn("user", "current turn");
        let message = app.continue_session().expect("continue should succeed");
        assert!(message.contains("resumed session #s0001"), "{message}");
        assert_eq!(
            app.remember.current_session(),
            Some("s0001"),
            "continue must switch to the previous session"
        );
    }

    #[tokio::test]
    async fn inject_archive_summary_finds_latest_archived() {
        let mut app = test_app();
        for _ in 0..(crate::app::remember::MAX_FULL_SESSIONS + 1) {
            app.remember.start_session().await;
            app.remember.append_turn("user", "archived turn");
        }
        assert!(!app.remember.archive().is_empty(), "archive must have entries");
        let id = app.remember.archive().last().unwrap().id.clone();
        let message = app.inject_archive_summary(&id).expect("inject should succeed");
        assert!(message.contains("remembered session"), "{message}");
        assert!(app.inject_archive_summary("s9999").is_err());
    }

    #[tokio::test]
    async fn remove_model_updates_current_and_clears_when_empty() {
        let mut app = test_app();
        app.add_model("alpha", "openai", "http://example.com", "k").unwrap();
        app.add_model("beta", "openai", "http://example.com", "k").unwrap();
        assert!(app.remove_model("alpha").is_ok());
        assert_eq!(app.current_model, "beta");
        assert!(app.remove_model("nope").is_err());
        assert!(app.remove_model("beta").is_ok());
        assert_eq!(app.current_model, "");
    }

    #[tokio::test]
    async fn clear_publishes_conversation_cleared() {
        let mut app = test_app();
        let mut rx = Box::pin(app.runtime.bus().subscribe_kinds(&[EventKind::Context]));
        app.submit("some task");
        app.clear();
        loop {
            match rx.next().await {
                Some(Event::ConversationCleared { .. }) => break,
                _ => continue,
            }
        }
    }

    #[tokio::test]
    async fn submit_records_user_turn_into_remember() {
        let mut app = test_app();
        app.remember.start_session().await;
        app.submit("record me");
        let turns = app.remember.load_session(app.remember.current_session().unwrap());
        assert_eq!(turns.len(), 1);
        assert_eq!(turns[0].role, "user");
        assert_eq!(turns[0].content, "record me");
    }
}
