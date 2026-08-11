use std::path::PathBuf;
use std::time::Duration;

use prognosis::app::{App, AppConfig, AdapterKind};

fn pick_adapter(explicit: Option<String>) -> AdapterKind {
    match explicit.as_deref() {
        Some("openai") => return AdapterKind::OpenAi,
        Some("deepseek") => return AdapterKind::DeepSeek,
        Some(other) => {
            eprintln!("unknown adapter {other}, falling back to environment");
        }
        None => {}
    }
    if std::env::var("DEEPSEEK_API_KEY").is_ok() {
        AdapterKind::DeepSeek
    } else if std::env::var("OPENAI_API_KEY").is_ok() {
        AdapterKind::OpenAi
    } else {
        AdapterKind::DeepSeek
    }
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut adapter: Option<String> = None;
    let mut model: Option<String> = None;
    let mut supervisor: Option<String> = None;
    let mut resume_latest = false;

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--adapter" => adapter = args.next(),
            "--model" => model = args.next(),
            "--supervisor" => supervisor = args.next(),
            "--continue" => resume_latest = true,
            "--help" => {
                println!(
                    "usage: prognosis [OPTIONS]\n\
                     \n\
                     Options:\n\
                     \x20 --adapter <openai|deepseek>  LLM adapter (default: auto-detected from DEEPSEEK_API_KEY / OPENAI_API_KEY, falls back to deepseek)\n\
                     \x20 --model <NAME>               Initial model name (overrides the stored default model)\n\
                     \x20 --supervisor <on|off>        LLM supervisor that reviews tool calls before they run (default: on)\n\
                     \x20 --continue                   Resume the most recent session on startup (equivalent to /continue)\n\
                     \x20 --help                       Show this help and exit\n\
                     \n\
                     Interactive commands (type / inside the TUI):\n\
                     \x20 models  compact  approvals  status  task  rules  skills  history\n\
                     \x20 continue  remember  resume <id>  trace  supervisor  clear  help"
                );
                return Ok(());
            }
            other => {
                eprintln!("unknown argument: {other}");
            }
        }
    }

    let supervisor_enabled = !matches!(supervisor.as_deref(), Some("off"));

    let config = AppConfig {
        adapter: pick_adapter(adapter),
        model,
        supervisor_enabled,
        project_dir: PathBuf::from("."),
        tick_interval: Duration::from_millis(100),
    };
    let mut app = App::new(config)?;
    app.start().await;
    if resume_latest {
        match app.continue_session() {
            Ok(message) => app.startup_notice = Some(format!("(resumed) {message}")),
            Err(error) => app.startup_notice = Some(format!("(continue failed) {error}")),
        }
    }
    prognosis::frontend::run(app).await?;
    Ok(())
}
