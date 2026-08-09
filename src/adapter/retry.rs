use std::future::Future;
use std::time::Duration;

use tokio_util::sync::CancellationToken;

use crate::adapter::config::RetryConfig;
use crate::adapter::error::AdapterError;

pub async fn retry<T, F, Fut>(
    config: &RetryConfig,
    cancel: &CancellationToken,
    mut operation: F,
) -> Result<T, AdapterError>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<T, AdapterError>>,
{
    let mut attempt = 0u32;
    loop {
        match operation().await {
            Ok(value) => return Ok(value),
            Err(err) if attempt + 1 >= config.max_attempts || !err.is_retryable() => {
                return Err(err);
            }
            Err(err) => {
                let delay = backoff_delay(config, attempt, err.retry_after(), fastrand::f64());
                attempt += 1;
                tokio::select! {
                    _ = tokio::time::sleep(delay) => {}
                    _ = cancel.cancelled() => return Err(AdapterError::cancelled(err.adapter_name())),
                }
            }
        }
    }
}

pub fn backoff_delay(
    config: &RetryConfig,
    attempt: u32,
    retry_after: Option<Duration>,
    jitter_sample: f64,
) -> Duration {
    if let Some(after) = retry_after {
        return after.min(config.max_delay);
    }
    let exponent = config.base_delay.saturating_mul(1u32 << attempt.min(16));
    let capped = exponent.min(config.max_delay);
    let ratio = 1.0 - config.jitter_ratio + 2.0 * config.jitter_ratio * jitter_sample.clamp(0.0, 1.0);
    Duration::from_secs_f64(capped.as_secs_f64() * ratio)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};

    fn retryable() -> AdapterError {
        AdapterError::ServerError {
            adapter: "test".into(),
            status: 500,
            message: "boom".into(),
            request_id: None,
        }
    }

    fn config() -> RetryConfig {
        RetryConfig::new(
            5,
            Duration::from_millis(100),
            Duration::from_millis(500),
            0.0,
        )
        .unwrap()
    }

    #[test]
    fn backoff_grows_exponentially_and_caps() {
        assert_eq!(
            backoff_delay(&config(), 0, None, 0.5),
            Duration::from_millis(100)
        );
        assert_eq!(
            backoff_delay(&config(), 1, None, 0.5),
            Duration::from_millis(200)
        );
        assert_eq!(
            backoff_delay(&config(), 2, None, 0.5),
            Duration::from_millis(400)
        );
        assert_eq!(
            backoff_delay(&config(), 3, None, 0.5),
            Duration::from_millis(500)
        );
        assert_eq!(
            backoff_delay(&config(), 10, None, 0.5),
            Duration::from_millis(500)
        );
    }

    #[test]
    fn retry_after_overrides_backoff() {
        assert_eq!(
            backoff_delay(&config(), 3, Some(Duration::from_millis(300)), 0.5),
            Duration::from_millis(300)
        );
        assert_eq!(
            backoff_delay(&config(), 3, Some(Duration::from_secs(60)), 0.5),
            Duration::from_millis(500)
        );
    }

    #[test]
    fn jitter_scales_delay() {
        let config = RetryConfig::new(
            5,
            Duration::from_millis(100),
            Duration::from_secs(10),
            1.0,
        )
        .unwrap();
        assert_eq!(backoff_delay(&config, 0, None, 0.0), Duration::ZERO);
        assert_eq!(
            backoff_delay(&config, 0, None, 1.0),
            Duration::from_millis(200)
        );
    }

    #[tokio::test]
    async fn retries_until_success() {
        let calls = AtomicU32::new(0);
        let config = RetryConfig::new(
            5,
            Duration::from_millis(1),
            Duration::from_millis(5),
            0.0,
        )
        .unwrap();
        let result = retry(&config, &CancellationToken::new(), || {
            let calls = &calls;
            async move {
                if calls.fetch_add(1, Ordering::SeqCst) < 2 {
                    Err(retryable())
                } else {
                    Ok("ok")
                }
            }
        })
        .await;
        assert_eq!(result.unwrap(), "ok");
        assert_eq!(calls.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn gives_up_after_max_attempts() {
        let config = RetryConfig::new(
            3,
            Duration::from_millis(1),
            Duration::from_millis(5),
            0.0,
        )
        .unwrap();
        let result = retry(&config, &CancellationToken::new(), || async {
            Err::<(), _>(retryable())
        })
        .await;
        assert!(matches!(result, Err(AdapterError::ServerError { .. })));
    }

    #[tokio::test]
    async fn non_retryable_returns_immediately() {
        let calls = AtomicU32::new(0);
        let config = RetryConfig::new(
            3,
            Duration::from_millis(1),
            Duration::from_millis(5),
            0.0,
        )
        .unwrap();
        let result = retry(&config, &CancellationToken::new(), || {
            let calls = &calls;
            async move {
                calls.fetch_add(1, Ordering::SeqCst);
                Err::<(), _>(AdapterError::invalid_request("test", "bad"))
            }
        })
        .await;
        assert!(matches!(result, Err(AdapterError::InvalidRequest { .. })));
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn cancellation_aborts_backoff() {
        let cancel = CancellationToken::new();
        cancel.cancel();
        let config = RetryConfig::new(
            5,
            Duration::from_secs(60),
            Duration::from_secs(60),
            0.0,
        )
        .unwrap();
        let result = retry(&config, &cancel, || async { Err::<(), _>(retryable()) }).await;
        assert!(matches!(result, Err(AdapterError::Cancelled { .. })));
    }
}
