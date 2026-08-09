use std::time::Duration;

use crate::adapter::error::AdapterError;

#[derive(Debug, Clone, PartialEq)]
pub struct RetryConfig {
    pub max_attempts: u32,
    pub base_delay: Duration,
    pub max_delay: Duration,
    pub jitter_ratio: f64,
}

impl RetryConfig {
    pub fn new(
        max_attempts: u32,
        base_delay: Duration,
        max_delay: Duration,
        jitter_ratio: f64,
    ) -> Result<Self, AdapterError> {
        if max_attempts == 0 {
            return Err(AdapterError::config(
                "config",
                "max_attempts must be at least 1",
            ));
        }
        if base_delay.is_zero() {
            return Err(AdapterError::config(
                "config",
                "base_delay must be positive",
            ));
        }
        if max_delay < base_delay {
            return Err(AdapterError::config(
                "config",
                "max_delay must be at least base_delay",
            ));
        }
        if !(0.0..=1.0).contains(&jitter_ratio) {
            return Err(AdapterError::config(
                "config",
                "jitter_ratio must be within [0, 1]",
            ));
        }
        Ok(Self {
            max_attempts,
            base_delay,
            max_delay,
            jitter_ratio,
        })
    }
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            base_delay: Duration::from_millis(250),
            max_delay: Duration::from_secs(10),
            jitter_ratio: 0.2,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TimeoutConfig {
    pub connect: Duration,
    pub request: Duration,
    pub stream_idle: Duration,
}

impl TimeoutConfig {
    pub fn new(
        connect: Duration,
        request: Duration,
        stream_idle: Duration,
    ) -> Result<Self, AdapterError> {
        if connect.is_zero() || request.is_zero() || stream_idle.is_zero() {
            return Err(AdapterError::config("config", "timeouts must be positive"));
        }
        Ok(Self {
            connect,
            request,
            stream_idle,
        })
    }
}

impl Default for TimeoutConfig {
    fn default() -> Self {
        Self {
            connect: Duration::from_secs(10),
            request: Duration::from_secs(120),
            stream_idle: Duration::from_secs(30),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retry_config_validation() {
        assert!(RetryConfig::new(
            0,
            Duration::from_millis(100),
            Duration::from_millis(500),
            0.2
        )
        .is_err());
        assert!(RetryConfig::new(
            3,
            Duration::ZERO,
            Duration::from_millis(500),
            0.2
        )
        .is_err());
        assert!(RetryConfig::new(
            3,
            Duration::from_millis(500),
            Duration::from_millis(100),
            0.2
        )
        .is_err());
        assert!(RetryConfig::new(
            3,
            Duration::from_millis(100),
            Duration::from_millis(500),
            1.5
        )
        .is_err());
        assert!(RetryConfig::new(
            3,
            Duration::from_millis(100),
            Duration::from_millis(500),
            0.2
        )
        .is_ok());
    }

    #[test]
    fn timeout_config_validation() {
        assert!(TimeoutConfig::new(
            Duration::ZERO,
            Duration::from_secs(10),
            Duration::from_secs(5)
        )
        .is_err());
        assert!(TimeoutConfig::new(
            Duration::from_secs(10),
            Duration::from_secs(60),
            Duration::from_secs(30)
        )
        .is_ok());
    }
}
