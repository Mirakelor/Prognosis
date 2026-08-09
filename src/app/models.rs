use std::path::Path;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use futures::{Stream, StreamExt};
use std::pin::Pin;

use crate::adapter::error::AdapterError;
use crate::adapter::traits::LanguageModelAdapter;
use crate::adapter::types::{AdapterCapabilities, CompletionChunk, CompletionRequest, ModelInfo};

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ModelEntry {
    pub name: String,
    pub kind: String,
    #[serde(default)]
    pub base_url: String,
    #[serde(default)]
    pub api_key: String,
}

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct ModelsStore {
    #[serde(default)]
    pub entries: Vec<ModelEntry>,
    #[serde(default)]
    pub current: String,
}

impl ModelsStore {
    pub fn load(dir: &Path) -> Self {
        std::fs::read_to_string(dir.join("models.json"))
            .ok()
            .and_then(|text| serde_json::from_str(&text).ok())
            .unwrap_or_default()
    }

    pub fn save(&self, dir: &Path) {
        if let Some(parent) = dir.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        if let Ok(text) = serde_json::to_string_pretty(self) {
            let _ = std::fs::write(dir.join("models.json"), text);
        }
    }

    pub fn upsert(&mut self, entry: ModelEntry) {
        self.entries.retain(|e| e.name != entry.name);
        self.entries.push(entry);
    }

    pub fn get(&self, name: &str) -> Option<&ModelEntry> {
        self.entries.iter().find(|e| e.name == name)
    }

    pub fn remove(&mut self, name: &str) -> bool {
        let before = self.entries.len();
        self.entries.retain(|e| e.name != name);
        if self.current == name {
            self.current = self
                .entries
                .first()
                .map(|e| e.name.clone())
                .unwrap_or_default();
        }
        self.entries.len() != before
    }
}

pub fn migrate_from_project(project_dir: &Path) {
    let project_file = project_dir.join(".prognosis").join("models.json");
    if !project_file.is_file() {
        return;
    }
    let project_store = ModelsStore::load(&project_dir.join(".prognosis"));
    let global_dir = crate::app::tools::global_config_dir();
    let mut global_store = ModelsStore::load(&global_dir);
    for entry in project_store.entries {
        if let Some(existing) = global_store.entries.iter_mut().find(|e| e.name == entry.name) {
            if existing.api_key.is_empty() && !entry.api_key.is_empty() {
                existing.api_key = entry.api_key.clone();
            }
            if existing.base_url.is_empty() && !entry.base_url.is_empty() {
                existing.base_url = entry.base_url.clone();
            }
        } else {
            global_store.entries.push(entry);
        }
    }
    if global_store.current.is_empty() {
        global_store.current = project_store.current.clone();
    }
    global_store.save(&global_dir);
    let _ = std::fs::remove_file(&project_file);
}

pub struct SwitchableAdapter {
    inner: Mutex<Arc<dyn LanguageModelAdapter>>,
}

impl SwitchableAdapter {
    pub fn new(inner: Arc<dyn LanguageModelAdapter>) -> Self {
        Self {
            inner: Mutex::new(inner),
        }
    }

    pub fn switch(&self, inner: Arc<dyn LanguageModelAdapter>) {
        *self.inner.lock().unwrap() = inner;
    }

    pub fn current(&self) -> Arc<dyn LanguageModelAdapter> {
        self.inner.lock().unwrap().clone()
    }

}

pub fn build_client(entry: &ModelEntry) -> Result<Arc<dyn LanguageModelAdapter>, AdapterError> {
    match entry.kind.as_str() {
        "deepseek" => {
            let mut builder = crate::adapter::DeepSeekConfigBuilder::new()
                .default_model(entry.name.clone());
            if !entry.base_url.is_empty() {
                builder = builder.base_url(entry.base_url.clone());
            }
            if !entry.api_key.is_empty() {
                builder = builder.api_key(entry.api_key.clone());
            }
            Ok(Arc::new(crate::adapter::DeepSeekClient::new(builder.build()?)?))
        }
        _ => {
            let mut builder = crate::adapter::OpenAIConfigBuilder::new()
                .default_model(entry.name.clone());
            if !entry.base_url.is_empty() {
                builder = builder.base_url(entry.base_url.clone());
            }
            if !entry.api_key.is_empty() {
                builder = builder.api_key(entry.api_key.clone());
            }
            Ok(Arc::new(crate::adapter::OpenAIClient::new(builder.build()?)?))
        }
    }
}

pub fn default_model_name(kind: &str) -> String {
    match kind {
        "deepseek" => "deepseek-v4-flash".to_string(),
        _ => "gpt-4o".to_string(),
    }
}

pub fn build_from_env(kind: &str, model: Option<String>) -> Result<Arc<dyn LanguageModelAdapter>, AdapterError> {
    match kind {
        "deepseek" => {
            let mut builder = crate::adapter::DeepSeekConfigBuilder::new();
            if let Some(model) = model {
                builder = builder.default_model(model);
            }
            Ok(Arc::new(crate::adapter::DeepSeekClient::new(builder.build()?)?))
        }
        _ => {
            let mut builder = crate::adapter::OpenAIConfigBuilder::new();
            if let Some(model) = model {
                builder = builder.default_model(model);
            }
            Ok(Arc::new(crate::adapter::OpenAIClient::new(builder.build()?)?))
        }
    }
}

pub fn placeholder() -> Arc<dyn LanguageModelAdapter> {
    Arc::new(NoModelAdapter)
}

struct NoModelAdapter;

impl NoModelAdapter {
    const ERROR: &'static str =
        "no model configured: run /models in the TUI to add one (provider, model, API key)";
}

#[async_trait]
impl LanguageModelAdapter for NoModelAdapter {
    fn id(&self) -> &str {
        "no-model"
    }

    fn capabilities(&self) -> AdapterCapabilities {
        AdapterCapabilities::default()
    }

    async fn stream<'a>(
        &'a self,
        _request: CompletionRequest,
        _cancel: &'a tokio_util::sync::CancellationToken,
    ) -> Result<
        Pin<Box<dyn Stream<Item = Result<CompletionChunk, AdapterError>> + Send + 'a>>,
        AdapterError,
    > {
        Err(AdapterError::Config {
            adapter: "no-model".into(),
            message: Self::ERROR.into(),
        })
    }
}

pub fn switch_adapter(
    switchable: &SwitchableAdapter,
    store: &mut ModelsStore,
    dir: &Path,
    entry: &ModelEntry,
) -> Result<(), String> {
    let client = build_client(entry).map_err(|e| format!("model build failed: {e}"))?;
    switchable.switch(client);
    store.upsert(entry.clone());
    store.current = entry.name.clone();
    store.save(dir);
    Ok(())
}

#[async_trait]
impl LanguageModelAdapter for SwitchableAdapter {
    fn id(&self) -> &str {
        "switchable"
    }

    fn capabilities(&self) -> AdapterCapabilities {
        self.current().capabilities()
    }

    async fn models(&self) -> Result<Vec<ModelInfo>, AdapterError> {
        self.current().models().await
    }

    async fn model_info(&self, id: &str) -> Result<ModelInfo, AdapterError> {
        self.current().model_info(id).await
    }

    async fn stream<'a>(
        &'a self,
        request: CompletionRequest,
        cancel: &'a tokio_util::sync::CancellationToken,
    ) -> Result<
        Pin<Box<dyn Stream<Item = Result<CompletionChunk, AdapterError>> + Send + 'a>>,
        AdapterError,
    > {
        let adapter = self.current();
        let cancel = cancel.clone();
        Ok(Box::pin(async_stream::stream! {
            let mut stream = match adapter.stream(request, &cancel).await {
                Ok(stream) => stream,
                Err(err) => {
                    yield Err(err);
                    return;
                }
            };
            while let Some(item) = stream.next().await {
                yield item;
            }
        }))
    }
}

pub fn context_window(model: &str) -> usize {
    let lower = model.to_lowercase();
    if lower.contains("deepseek") {
        1_000_000
    } else if lower.contains("gpt-5") {
        400_000
    } else if lower.contains("o1") || lower.contains("o3") || lower.contains("o4") {
        200_000
    } else {
        128_000
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn store_roundtrip() {
        let dir = std::env::temp_dir().join(format!("prognosis_models_test_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let mut store = ModelsStore::load(&dir);
        store.upsert(ModelEntry {
            name: "m1".into(),
            kind: "openai".into(),
            base_url: String::new(),
            api_key: String::new(),
        });
        store.current = "m1".into();
        store.save(&dir);
        let loaded = ModelsStore::load(&dir);
        assert_eq!(loaded.current, "m1");
        assert_eq!(loaded.get("m1").map(|e| e.kind.as_str()), Some("openai"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn migrate_from_project_merges_key_into_global() {
        let _guard = crate::test_util::ENV_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let global = tempfile::TempDir::new().unwrap();
        unsafe {
            std::env::set_var("PROGNOSIS_CONFIG_DIR", global.path());
        }
        let project = tempfile::TempDir::new().unwrap();
        let store_dir = project.path().join(".prognosis");
        std::fs::create_dir_all(&store_dir).unwrap();
        let mut project_store = ModelsStore::default();
        project_store.upsert(ModelEntry {
            name: "deepseek-v4-flash".into(),
            kind: "deepseek".into(),
            base_url: "https://api.deepseek.com".into(),
            api_key: "sk-secret".into(),
        });
        project_store.current = "deepseek-v4-flash".into();
        project_store.save(&store_dir);

        migrate_from_project(project.path());

        assert!(!store_dir.join("models.json").exists(), "project file must be removed");
        let global_store = ModelsStore::load(global.path());
        assert_eq!(global_store.current, "deepseek-v4-flash");
        let entry = global_store.get("deepseek-v4-flash").unwrap();
        assert_eq!(entry.api_key, "sk-secret", "key must move to global");
        assert_eq!(entry.base_url, "https://api.deepseek.com");

        let mut global2 = ModelsStore::load(global.path());
        global2.upsert(ModelEntry {
            name: "m2".into(),
            kind: "openai".into(),
            base_url: String::new(),
            api_key: "sk-2".into(),
        });
        global2.save(global.path());
        std::fs::write(
            store_dir.join("models.json"),
            r#"{"entries":[{"name":"deepseek-v4-flash","kind":"deepseek","base_url":"","api_key":"sk-new"}],"current":""}"#,
        )
        .unwrap();
        migrate_from_project(project.path());
        let final_store = ModelsStore::load(global.path());
        assert_eq!(
            final_store.get("deepseek-v4-flash").unwrap().api_key,
            "sk-secret",
            "existing global key must not be overwritten by re-migration"
        );
        assert!(final_store.get("m2").is_some(), "existing global entries survive");
        unsafe {
            std::env::remove_var("PROGNOSIS_CONFIG_DIR");
        }
    }

    #[test]
    fn window_table() {
        assert_eq!(context_window("deepseek-v4-flash"), 1_000_000);
        assert_eq!(context_window("gpt-4o"), 128_000);
        assert_eq!(context_window("gpt-5"), 400_000);
    }

    #[tokio::test]
    async fn placeholder_errors_with_guidance() {
        let adapter = placeholder();
        assert_eq!(adapter.id(), "no-model");
        let err = match adapter
            .stream(
                CompletionRequest::new("deepseek-v4-flash", vec![crate::adapter::types::Message::user("hi")]),
                &tokio_util::sync::CancellationToken::new(),
            )
            .await
        {
            Err(err) => err,
            Ok(_) => panic!("placeholder stream should fail"),
        };
        match err {
            crate::adapter::error::AdapterError::Config { message, .. } => {
                assert!(message.contains("run /models"), "got: {message}");
            }
            other => panic!("expected Config error, got {other:?}"),
        }
    }
}
