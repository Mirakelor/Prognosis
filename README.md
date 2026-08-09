# ✦ Prognosis

> 🧠 A cognitive-neuroscience-inspired AI coding agent that lives in your terminal.
> A prediction-coding runtime (RPE, neuromodulation, attention, working memory), an LLM app layer with 16 tools and a supervisor judge, OpenAI/DeepSeek adapters, and a ratatui TUI.

---

## ✨ Features

| | Feature | What it means for you |
|---|---|---|
| 🧠 | **Cognitive runtime** | Prediction-error (RPE) learning signals, dopamine/norepinephrine/acetylcholine/serotonin modulation, attention salience, working memory, and metacognitive monitoring — all visible live in the TUI |
| 🛠️ | **16-tool ecosystem** | File read/edit/create, glob/grep search, terminal, web search & fetch, git diff, rules, skills, task scheduling — with an approval workflow and an LLM supervisor judge that catches bad calls before they run |
| 📜 | **Rules & Skills** | Project and global rules with auto-attachment (globs/regex), and skills following the open Agent Skills standard (`SKILL.md`), loaded from shared ecosystem locations so skills installed for Codex/Cursor also work here |
| ⚡ | **Batch tool execution** | Independent tool calls run in parallel; the model waits for the whole batch before generating again — no mid-batch guessing, no repeated calls |
| 💾 | **Session memory** | Full conversation history is always injected (never silently summarized); archive, resume, and re-inject past sessions with `/history`, `/continue`, `/remember` |
| 🖥️ | **Terminal-first UI** | Model switching, approval keys, live cognitive signal panel, Markdown rendering (headings, lists, code, **tables**), diff folding, mouse-wheel scrolling |

---

## 🚀 Install

### One-command install (recommended)

No Rust, no compiler, no `sudo` required on machines where a prebuilt release exists:

```bash
bash <(curl -fsSL https://raw.githubusercontent.com/Mirakelor/Prognosis/master/install.sh)
```

Or download and run it locally:

```bash
curl -fsSL -o install.sh https://raw.githubusercontent.com/Mirakelor/Prognosis/master/install.sh
bash install.sh
```

**What the installer does, step by step:**

1. 🔍 Detects your platform (`linux`/`macos` × `x86_64`/`aarch64`)
2. 📦 Checks GitHub Releases for a matching prebuilt binary (`prognosis-<os>-<arch>.tar.gz`) — if found, downloads, extracts, and installs it. **Zero dependencies, zero sudo, done.**
3. 🔧 No release for your platform? Falls back to building from source automatically:
   - Installs **Rust** via rustup (user-level, **no sudo**, silent `-y --profile minimal`)
   - Installs a **C compiler** if missing (`sudo apt-get install -y build-essential` on Debian/Ubuntu — the *only* sudo in the whole flow; on macOS it tells you to run `xcode-select --install`)
   - Clones the repository, runs `cargo build --release`, installs the binary
4. 📁 Installs to `~/.local/bin/prognosis` (override with `PROGNOSIS_INSTALL_DIR`)
5. 🛤️ Adds `~/.local/bin` to your `PATH` in `~/.bashrc`/`~/.zshrc`/`~/.profile` if it isn't already there
6. 🧹 Cleans up temporary files; prints the next step (`/models`)

The script is idempotent — re-run it anytime to update.

### Build from source (manual)

```bash
git clone https://github.com/Mirakelor/Prognosis.git
cd Prognosis
cargo build --release
./target/release/prognosis
```

Requirements: Rust (stable) + a C linker (`cc`/`gcc`/`clang`). The dependency tree is 100% pure Rust — no OpenSSL, no system C libraries.

### Installer environment variables

| Variable | Default | Purpose |
|---|---|---|
| `PROGNOSIS_INSTALL_DIR` | `~/.local/bin` | Where the binary is installed |
| `PROGNOSIS_REPO` | `Mirakelor/Prognosis` | Repository to fetch releases/source from |
| `PROGNOSIS_SKIP_RELEASE` | `0` | Set to `1` to skip the prebuilt download and always build from source |
| `PROGNOSIS_SKIP_BUILD_DEPS` | `0` | Set to `1` to refuse auto-installing a C compiler (fails with guidance instead) |

---

## 🏁 Quick start

```bash
prognosis
```

1. **Add a model**: type `/models`, choose *add*, and paste your API key. Keys are stored in `~/.prognosis/models.json` — **never in the project**, so they can't leak into git. Supported: DeepSeek and any OpenAI-compatible endpoint (custom `base_url`).
2. **Start working**: type your request in any language — the agent replies in the same language.
3. **Approvals**: some tool calls ask for confirmation before running:
   - `Enter` — approve this call
   - `Shift+Enter` — approve and remember the tool forever (no more prompts)
   - `Esc` — deny
4. **Interrupting**: `Esc` stops generation or cancels running tool calls; `Ctrl+C` does the same; `Ctrl+D` quits.

### ⌨️ Key bindings

| Key | Action |
|---|---|
| `Enter` | Send message / approve / select |
| `Shift+Enter` / `Alt+Enter` | Insert a newline in the input box (Shift+Enter degrades to Alt+Enter on some terminals) |
| `Shift+Tab` / `y` | Approve and remember (fallback keys when Shift+Enter is not delivered by the terminal) |
| `Esc` | Cancel generation / deny approval / cancel running tools / close an overlay |
| `Tab` | Toggle the selected rule or skill in the rules/skills panel |
| `↑` `↓` | Navigate input history (or command list while typing `/`) |
| `←` `→` `Home` `End` | Move the cursor in the input box |
| `Ctrl+C` | Cancel generation or running tools; quit when idle |
| `Ctrl+D` | Quit |
| Mouse wheel | Scroll chat/panel content |

---

## ⚙️ Configuration

### 📂 Global vs project

Prognosis splits configuration between your machine and each project:

| Path | Holds | Scope | Commit to git? |
|---|---|---|---|
| `~/.prognosis/models.json` | Model entries **including API keys** | All your projects | Never |
| `~/.prognosis/rules/` | Your personal rules | All your projects | Never |
| `.prognosis/rules/` | Team rules | This project | ✅ Yes |
| `.prognosis/state.json` | Enabled/disabled toggles for rules & skills | This project | No |
| `.prognosis/history/` | Archived sessions | This project | No |
| `.prognosis/approvals.json` | Remembered tool approvals | This project | No |
| `.agents/skills/` | Project skills (SKILL.md folders) | This project | ✅ Yes (copy mode) |

**Model entries** live globally so API keys never enter a repository. `models.json` looks like:

```json
{
  "entries": [
    {
      "name": "deepseek-v4-flash",
      "kind": "deepseek",
      "base_url": "https://api.deepseek.com",
      "api_key": "sk-..."
    }
  ],
  "current": "deepseek-v4-flash"
}
```

### 📜 Rules

Rules are Markdown files with YAML frontmatter:

```markdown
---
description: "Use named exports in TypeScript modules"
globs: "src/**/*.ts"
alwaysApply: false
---

Always use named exports. Never use default exports in this project.
```

| Frontmatter | Meaning |
|---|---|
| `description` | When the rule should apply (used by the agent to decide) |
| `globs` | File patterns the rule attaches to (auto-attached when matching files are involved) |
| `regex` | Content pattern the rule attaches to |
| `alwaysApply` | `true` = always injected; `false` = agent decides |

- Global + project rules are **merged**; a project rule with the same name **overrides** the global one.
- The agent creates rules with the `create_rule_block` tool — pass `"scope": "global"` to store them in `~/.prognosis/rules`.
- Manage enable/disable in the TUI (`/rules` panel, `Tab` to toggle).

### 🛠️ Skills

Skills follow the open **Agent Skills** standard: a folder containing `SKILL.md` with YAML frontmatter (`name`, `description`, ...). Prognosis reads them from the shared ecosystem locations, so skills you installed for Codex, Cursor, or Gemini CLI are visible here too:

| Location | Scope |
|---|---|
| `.agents/skills/` | This project |
| `~/.agents/skills/` | Global (your machine) |
| `~/.config/agents/skills/` | Global (your machine) |

Install skills with the npm ecosystem CLI (target the **universal** agent group):

```bash
# 📁 Project-level: installs into .agents/skills/ (committed with the repo)
npx skills add <owner/repo> -a universal

# 🌍 Global-level: installs into ~/.config/agents/skills/
npx skills add <owner/repo> -g -a universal
```

Skills are listed in the `/skills` panel (name + description) and their full instructions load lazily via the `read_skill` tool only when the agent needs them — many skills cost almost nothing in context until used.

---

## ⌨️ Commands

Type `/` in the input box to open the command palette:

| Command | Purpose | Example |
|---|---|---|
| `/models` | List, switch, add, or remove models | `/models` → add DeepSeek key |
| `/rules` | List rules (project + `~/.prognosis`) | `/rules` → see all, `Tab` to toggle |
| `/skills` | List skills (`.agents/skills` + global) | `/skills` → see what's available |
| `/history` | List past sessions and load one fully | `/history` → pick a session to restore |
| `/continue` | Resume the most recent session | `/continue` → keep working where you left off |
| `/remember` | Inject an archived session summary | `/remember` → bring back context from an old session |
| `/resume` | Resume a session by id | `/resume 2026-08-09-...` |
| `/compact` | Compress the conversation into a summary | use when the thread is very long |
| `/approvals` | Manage remembered tool approvals | `/approvals` → forget a remembered tool |
| `/status` | Show live cognitive signals | RPE, modulators, working memory |
| `/trace` | Show recent cognitive trace records | see what the runtime just did |
| `/task` | List or cancel scheduled tasks | `/task` → cancel a monitor task |
| `/supervisor` | Toggle the supervisor judge on/off | `/supervisor` → disable gating |
| `/clear` | Clear the conversation | start fresh |
| `/help` | Show key bindings and commands | `/help` |

---

## 🧠 Cognitive architecture (brief)

The runtime runs cognitive actors over an internal event bus, in a continuous loop:

```
perception → attention → prediction → error computation → modulation → action selection
                    ↘            ↙
              working memory ← metacognition
```

- **🎯 Prediction** — the agent predicts your next message *before* you send it; the prediction error (RPE) is a learning signal that shapes future behavior. Visible as live RPE values in `/status`.
- **⚗️ Neuromodulation** — dopamine raises the action-selection GO threshold (persistence when things go well), norepinephrine holds attention on the current content, acetylcholine gates reasoning effort, serotonin damps negative salience.
- **👁️ Attention & working memory** — salient events (tool errors, denied calls, results) are flagged, stored in working-memory slots, and injected into the model as *semantic* reminders — never raw numbers.
- **🧩 Metacognition** — monitors uncertainty/confidence/conflict; when thresholds are crossed it injects advisory reminders ("verify every claim with evidence") instead of dumping numeric state.
- **🚦 Action selection** — a stochastic GO/NO-GO process over candidate actions, gated by an LLM supervisor judge that reviews each tool call before execution (approval, correction, or rejection).

---

## 🌍 Environment variables

| Variable | Purpose |
|---|---|
| `PROGNOSIS_LOG_LLM` | Set to `1` to log raw LLM requests for debugging |
| `PROGNOSIS_CONFIG_DIR` | Override the global config dir (default `~/.prognosis`) — used by tests and portable setups |
| `PROGNOSIS_SKILLS_HOME` | Override the home base for ecosystem skill dirs (testing) |

---

## 🛠️ Troubleshooting

**"linker 'cc' not found"**
The C compiler is missing. The installer handles this automatically; manually: `sudo apt-get install -y build-essential` (Debian/Ubuntu) or `xcode-select --install` (macOS).

**No prebuilt binary for my platform**
The installer falls back to a source build automatically. Build manually with the steps above.

**"Repository not found" during install**
The `PROGNOSIS_REPO` env var points at a repository that doesn't exist (or is private). Export the correct one and re-run.

**`npx` not found**
Skills can also be placed manually — each skill is just a folder containing a `SKILL.md`, dropped into any of the skill directories in the table above.

**API key not working**
Run `/models`, remove the entry, and re-add it with the correct key. Keys are read from `~/.prognosis/models.json` only — environment variables like `DEEPSEEK_API_KEY`/`OPENAI_API_KEY` are also honored as fallbacks.

**The TUI looks broken in a small window**
Resize to at least 80 columns; the layout adapts up to 120+ columns.

---

## 📦 Project layout

```
prognosis/
├── src/
│   ├── runtime/      # cognitive actors: perception, attention, prediction, modulation, ...
│   ├── app/          # App layer: tools, models, supervisor, scheduler, remember
│   ├── adapter/      # OpenAI / DeepSeek wire formats & HTTP clients
│   └── frontend/     # ratatui TUI: render, input, selectors, markdown
├── install.sh        # one-command installer
└── .github/workflows/release.yml  # tag → prebuilt binaries on GitHub Releases
```
