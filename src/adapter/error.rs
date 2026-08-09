use std::time::Duration;

#[derive(Debug, thiserror::Error)]
pub enum AdapterError {
    #[error("[{adapter}] Setup failed: {message}")]
    Config {
        adapter: String,
        message: String,
    },
    #[error("[{adapter}] Invalid Request: {message}")]
    InvalidRequest {
        adapter: String,
        message: String,
    },
    #[error("[{adapter}] Permission denied: {message}")]
    PermissionDenied {
        adapter: String,
        message: String,
        request_id: Option<String>,
    },
    #[error("[{adapter}] Not found: {message}")]
    NotFound {
        adapter: String,
        message: String,
        request_id: Option<String>,
    },
    #[error("[{adapter}] Authentication failed: {message}")]
    Authentication {
        adapter: String,
        message: String,
        request_id: Option<String>,
    },
    #[error("[{adapter}] Rate limit exceeded: {message}")]
    RateLimit {
        adapter: String,
        message: String,
        retry_after: Option<Duration>,
        request_id: Option<String>,
    },
    #[error("[{adapter}] Server Error (HTTP {status}): {message}")]
    ServerError {
        adapter: String,
        status: u16,
        message: String,
        request_id: Option<String>,
    },
    #[error("[{adapter}] Timeout: {message}")]
    Timeout {
        adapter: String,
        message: String,
    },
    #[error("[{adapter}] Network Error: {message}")]
    Network {
        adapter: String,
        message: String,
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },
    #[error("[{adapter}] Decode Error: {message}")]
    Decode {
        adapter: String,
        message: String,
    },
    #[error("[{adapter}] Stream Error: {message}")]
    Stream {
        adapter: String,
        message: String,
    },
    #[error("[{adapter}] Unsupported Feature: {feature}")]
    Unsupported {
        adapter: String,
        feature: String,
    },
    #[error("[{adapter}] Request Cancelled")]
    Cancelled {
        adapter: String,
    },
}

impl AdapterError {
    pub fn config(adapter: impl Into<String>, message: impl Into<String>) -> Self {
        Self::Config {
            adapter: adapter.into(),
            message: message.into(),
        }
    }
    pub fn invalid_request(adapter: impl Into<String>, message: impl Into<String>) -> Self {
        Self::InvalidRequest {
            adapter: adapter.into(),
            message: message.into(),
        }
    }
    pub fn cancelled(adapter: impl Into<String>) -> Self {
        Self::Cancelled {
            adapter: adapter.into(),
        }
    }
    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            Self::Network { .. }
                | Self::Timeout { .. }
                | Self::RateLimit { .. }
                | Self::ServerError { .. }
        )
    }
    pub fn adapter_name(&self) -> &str {
        match self {
            Self::Config { adapter, .. }
            | Self::InvalidRequest { adapter, .. }
            | Self::Authentication { adapter, .. }
            | Self::PermissionDenied { adapter, .. }
            | Self::NotFound { adapter, .. }
            | Self::RateLimit { adapter, .. }
            | Self::ServerError { adapter, .. }
            | Self::Timeout { adapter, .. }
            | Self::Network { adapter, .. }
            | Self::Decode { adapter, .. }
            | Self::Stream { adapter, .. }
            | Self::Unsupported { adapter, .. }
            | Self::Cancelled { adapter } => adapter,
        }
    }
    pub fn retry_after(&self) -> Option<Duration> {
        match self {
            Self::RateLimit { retry_after, .. } => *retry_after,
            _ => None,
        }
    }
}

impl Clone for AdapterError {
    fn clone(&self) -> Self {
        match self {
            Self::Config { adapter, message } => Self::Config {
                adapter: adapter.clone(),
                message: message.clone(),
            },
            Self::InvalidRequest { adapter, message } => Self::InvalidRequest {
                adapter: adapter.clone(),
                message: message.clone(),
            },
            Self::Authentication { adapter, message, request_id } => Self::Authentication {
                adapter: adapter.clone(),
                message: message.clone(),
                request_id: request_id.clone(),
            },
            Self::PermissionDenied { adapter, message, request_id } => Self::PermissionDenied {
                adapter: adapter.clone(),
                message: message.clone(),
                request_id: request_id.clone(),
            },
            Self::NotFound { adapter, message, request_id } => Self::NotFound {
                adapter: adapter.clone(),
                message: message.clone(),
                request_id: request_id.clone(),
            },
            Self::RateLimit { adapter, message, retry_after, request_id } => Self::RateLimit {
                adapter: adapter.clone(),
                message: message.clone(),
                retry_after: *retry_after,
                request_id: request_id.clone(),
            },
            Self::ServerError { adapter, status, message, request_id } => Self::ServerError {
                adapter: adapter.clone(),
                status: *status,
                message: message.clone(),
                request_id: request_id.clone(),
            },
            Self::Timeout { adapter, message } => Self::Timeout {
                adapter: adapter.clone(),
                message: message.clone(),
            },
            Self::Network { adapter, message, source } => Self::Network {
                adapter: adapter.clone(),
                message: message.clone(),
                source: source.to_string().into(),
            },
            Self::Decode { adapter, message } => Self::Decode {
                adapter: adapter.clone(),
                message: message.clone(),
            },
            Self::Stream { adapter, message } => Self::Stream {
                adapter: adapter.clone(),
                message: message.clone(),
            },
            Self::Unsupported { adapter, feature } => Self::Unsupported {
                adapter: adapter.clone(),
                feature: feature.clone(),
            },
            Self::Cancelled { adapter } => Self::Cancelled {
                adapter: adapter.clone(),
            },
        }
    }
}

impl From<serde_json::Error> for AdapterError {
    fn from(err: serde_json::Error) -> Self {
        Self::Decode {
            adapter: "unknown".into(),
            message: err.to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retryable_classification() {
        let network = AdapterError::Network {
            adapter: "openai".into(),
            message: "connection refused".into(),
            source: "os error 111".into(),
        };
        assert!(network.is_retryable());

        let auth = AdapterError::Authentication {
            adapter: "openai".into(),
            message: "invalid key".into(),
            request_id: None,
        };
        assert!(!auth.is_retryable());
    }

    #[test]
    fn adapter_name_extraction() {
        let err = AdapterError::config("openai", "missing key");
        assert_eq!(err.adapter_name(), "openai");
    }

    #[test]
    fn rate_limit_retry_after() {
        let err = AdapterError::RateLimit {
            adapter: "openai".into(),
            message: "too many requests".into(),
            retry_after: Some(Duration::from_secs(30)),
            request_id: Some("req_abc".into()),
        };
        assert_eq!(err.retry_after(), Some(Duration::from_secs(30)));
    }

    #[test]
    fn clone_preserves_variant() {
        let err = AdapterError::ServerError {
            adapter: "openai".into(),
            status: 503,
            message: "overloaded".into(),
            request_id: Some("req_xyz".into()),
        };
        let cloned = err.clone();
        assert_eq!(cloned.adapter_name(), "openai");
        assert!(cloned.is_retryable());
    }

    #[test]
    fn display_format() {
        let err = AdapterError::config("openai", "Missing API Key");
        let text = err.to_string();
        assert!(text.contains("[openai]"));
        assert!(text.contains("Missing API Key"));
    }
}
