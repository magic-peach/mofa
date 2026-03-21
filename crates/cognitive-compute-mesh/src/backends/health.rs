use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use crate::protocol::HealthStatus;

#[derive(Debug, Clone)]
pub struct BackendHealthRecord {
    pub status: HealthStatus,
    pub consecutive_failures: u32,
    pub last_check: std::time::Instant,
}

pub struct HealthTracker {
    records: Arc<RwLock<HashMap<String, BackendHealthRecord>>>,
}

impl HealthTracker {
    pub fn new() -> Self {
        Self {
            records: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub async fn record_success(&self, backend: &str) {
        let mut map = self.records.write().await;
        let record = map.entry(backend.to_string()).or_insert(BackendHealthRecord {
            status: HealthStatus::Healthy,
            consecutive_failures: 0,
            last_check: std::time::Instant::now(),
        });
        record.consecutive_failures = 0;
        record.status = HealthStatus::Healthy;
        record.last_check = std::time::Instant::now();
    }

    pub async fn record_failure(&self, backend: &str) {
        let mut map = self.records.write().await;
        let record = map.entry(backend.to_string()).or_insert(BackendHealthRecord {
            status: HealthStatus::Unhealthy,
            consecutive_failures: 0,
            last_check: std::time::Instant::now(),
        });
        record.consecutive_failures += 1;
        record.status = HealthStatus::Unhealthy;
        record.last_check = std::time::Instant::now();
    }

    pub async fn get_status(&self, backend: &str) -> HealthStatus {
        let map = self.records.read().await;
        map.get(backend)
            .map(|r| r.status.clone())
            .unwrap_or(HealthStatus::Unknown)
    }

    pub async fn is_circuit_open(&self, backend: &str) -> bool {
        let map = self.records.read().await;
        map.get(backend)
            .map(|r| r.consecutive_failures >= 3)
            .unwrap_or(false)
    }
}

impl Default for HealthTracker {
    fn default() -> Self {
        Self::new()
    }
}
