//! Append-only log of raw user messages for personalisation, stored as JSONL
//! (`<dir>/<user_id>.jsonl`). Each line is a JSON object with `timestamp` (RFC 3339)
//! and `message` fields.

use std::path::{Path, PathBuf};

use chrono::Utc;
use serde_json::json;

use crate::config;
use crate::memory::ensure_dir;

/// Handle to the per-user message log store.
#[derive(Clone)]
pub struct MessageLog {
    dir: PathBuf,
}

impl Default for MessageLog {
    fn default() -> Self {
        Self::new(config::data_dir().join("message_log"))
    }
}

impl MessageLog {
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        Self { dir: dir.into() }
    }

    fn path(&self, user_id: impl std::fmt::Display) -> PathBuf {
        self.dir.join(format!("{user_id}.jsonl"))
    }

    /// Append one user message to the log (fire-and-forget; errors are logged but not returned).
    pub async fn append(&self, user_id: impl std::fmt::Display, message: &str) {
        if let Err(e) = self.try_append(user_id, message).await {
            tracing::warn!(target: "housebot::message_log", "Failed to append message: {e}");
        }
    }

    async fn try_append(
        &self,
        user_id: impl std::fmt::Display,
        message: &str,
    ) -> std::io::Result<()> {
        ensure_dir(&self.dir).await?;
        let entry = json!({
            "timestamp": Utc::now().to_rfc3339(),
            "message": message,
        });
        let mut line = serde_json::to_string(&entry).unwrap_or_else(|_| "{}".into());
        line.push('\n');
        append_line(&self.path(user_id), &line).await
    }

    /// Delete all logged messages for a user. No-op when no log exists.
    pub async fn clear(&self, user_id: impl std::fmt::Display) -> std::io::Result<()> {
        match tokio::fs::remove_file(self.path(user_id)).await {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(e),
        }
    }
}

async fn append_line(path: &Path, line: &str) -> std::io::Result<()> {
    use tokio::io::AsyncWriteExt;
    let mut file = tokio::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .await?;
    file.write_all(line.as_bytes()).await?;
    file.flush().await
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn store() -> (TempDir, MessageLog) {
        let tmp = TempDir::new().unwrap();
        let log = MessageLog::new(tmp.path().join("message_log"));
        (tmp, log)
    }

    #[tokio::test]
    async fn append_and_clear() {
        let (_t, log) = store();
        log.append(1u64, "hello").await;
        log.append(1u64, "world").await;
        let path = log.path(1u64);
        let content = tokio::fs::read_to_string(&path).await.unwrap();
        assert_eq!(content.lines().count(), 2);
        assert!(content.contains("hello"));
        log.clear(1u64).await.unwrap();
        assert!(!path.exists());
    }

    #[tokio::test]
    async fn clear_noop_for_unknown_user() {
        let (_t, log) = store();
        log.clear(999u64).await.unwrap();
    }

    #[tokio::test]
    async fn entries_are_valid_json() {
        let (_t, log) = store();
        log.try_append(2u64, "test message").await.unwrap();
        let raw = tokio::fs::read_to_string(log.path(2u64)).await.unwrap();
        let val: serde_json::Value = serde_json::from_str(raw.trim()).unwrap();
        assert_eq!(val["message"], "test message");
        assert!(val["timestamp"].as_str().is_some());
    }
}
