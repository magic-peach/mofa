use serde::Serialize;
use uuid::Uuid;
use chrono::{DateTime, Utc};
use crate::protocol::irp::{BackendScore, BackendMetrics, HealthStatus};

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum MeshEvent {
    RequestReceived {
        request_id: Uuid,
        timestamp: DateTime<Utc>,
        model: String,
    },
    RoutingDecision {
        request_id: Uuid,
        selected_backend: String,
        policy: String,
        reason: String,
        alternatives_considered: Vec<BackendScore>,
    },
    BackendAttempt {
        request_id: Uuid,
        backend: String,
        attempt: u32,
    },
    BackendSuccess {
        request_id: Uuid,
        backend: String,
        latency_ms: u64,
        tokens: u32,
        cost_usd: f64,
    },
    BackendFailure {
        request_id: Uuid,
        backend: String,
        error: String,
        failover_to: Option<String>,
    },
    ResponseComplete {
        request_id: Uuid,
        total_latency_ms: u64,
        total_cost_usd: f64,
    },
    BackendHealthChanged {
        backend: String,
        status: HealthStatus,
        timestamp: DateTime<Utc>,
    },
    MetricsUpdate {
        backends: Vec<BackendMetrics>,
        timestamp: DateTime<Utc>,
    },
}
