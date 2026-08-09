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

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--adapter" => adapter = args.next(),
            "--model" => model = args.next(),
            "--supervisor" => supervisor = args.next(),
            "--help" => {
                println!(
                    "usage: prognosis [--adapter openai|deepseek] [--model NAME] [--supervisor on|off]"
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
    prognosis::frontend::run(app).await?;
    Ok(())
}
