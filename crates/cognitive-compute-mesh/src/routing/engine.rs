use std::sync::Arc;
use tokio::sync::{RwLock, broadcast};
use anyhow::{Result, anyhow};
use crate::backends::{InferenceBackend, health::HealthTracker, metrics::MetricsStore};
use crate::protocol::*;

pub struct RoutingEngine {
    backends: Vec<Arc<dyn InferenceBackend>>,
    policy: Arc<RwLock<PolicyType>>,
    health_tracker: Arc<HealthTracker>,
    metrics: Arc<MetricsStore>,
    pub event_tx: broadcast::Sender<MeshEvent>,
}

impl RoutingEngine {
    pub fn new(backends: Vec<Arc<dyn InferenceBackend>>) -> Self {
        let (event_tx, _) = broadcast::channel(1000);
        Self {
            backends,
            policy: Arc::new(RwLock::new(PolicyType::Availability)),
            health_tracker: Arc::new(HealthTracker::new()),
            metrics: Arc::new(MetricsStore::new()),
            event_tx,
        }
    }

    pub async fn set_policy(&self, policy: PolicyType) {
        *self.policy.write().await = policy;
    }

    pub async fn get_policy(&self) -> PolicyType {
        self.policy.read().await.clone()
    }

    pub fn metrics(&self) -> Arc<MetricsStore> {
        Arc::clone(&self.metrics)
    }

    pub fn health_tracker(&self) -> Arc<HealthTracker> {
        Arc::clone(&self.health_tracker)
    }

    pub fn backends(&self) -> &[Arc<dyn InferenceBackend>] {
        &self.backends
    }

    pub async fn route(&self, request: &InferenceRequest) -> Result<Arc<dyn InferenceBackend>> {
        let policy = self.policy.read().await.clone();

        match policy {
            PolicyType::Availability => {
                // Return the first non-circuit-open backend
                for backend in &self.backends {
                    let circuit_open = self.health_tracker.is_circuit_open(backend.name()).await;
                    if !circuit_open {
                        return Ok(Arc::clone(backend));
                    }
                }
                // All circuits open — degraded mode, return first backend anyway
                self.backends
                    .first()
                    .map(Arc::clone)
                    .ok_or_else(|| anyhow!("No backends configured"))
            }

            PolicyType::Cost => {
                // Sort backends by estimated cost, return cheapest non-circuit-open one
                let mut candidates: Vec<(f64, &Arc<dyn InferenceBackend>)> = self
                    .backends
                    .iter()
                    .map(|b| (b.estimated_cost(request), b))
                    .collect();
                candidates.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

                for (_, backend) in &candidates {
                    let circuit_open = self.health_tracker.is_circuit_open(backend.name()).await;
                    if !circuit_open {
                        return Ok(Arc::clone(backend));
                    }
                }
                // All circuits open — degraded mode: return cheapest overall
                candidates
                    .first()
                    .map(|(_, b)| Arc::clone(b))
                    .ok_or_else(|| anyhow!("No backends configured"))
            }

            PolicyType::Latency => {
                // Sort backends by P50 latency (None treated as 0 to prefer untested backends)
                let metrics_snapshot = self.metrics.all_metrics().await;

                let mut candidates: Vec<(f64, &Arc<dyn InferenceBackend>)> = self
                    .backends
                    .iter()
                    .map(|b| {
                        let p50 = metrics_snapshot
                            .get(b.name())
                            .and_then(|m| m.latency_window.p50())
                            .unwrap_or(0.0);
                        (p50, b)
                    })
                    .collect();
                candidates.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

                for (_, backend) in &candidates {
                    let circuit_open = self.health_tracker.is_circuit_open(backend.name()).await;
                    if !circuit_open {
                        return Ok(Arc::clone(backend));
                    }
                }
                // All circuits open — degraded mode: return fastest overall
                candidates
                    .first()
                    .map(|(_, b)| Arc::clone(b))
                    .ok_or_else(|| anyhow!("No backends configured"))
            }
        }
    }

    pub async fn route_with_fallback(&self, request: &InferenceRequest) -> Result<InferenceResponse> {
        let start = std::time::Instant::now();

        // Emit RequestReceived
        let _ = self.event_tx.send(MeshEvent::RequestReceived {
            request_id: request.id,
            timestamp: chrono::Utc::now(),
            model: request.model.clone(),
        });

        // Select primary backend
        let primary = self.route(request).await?;
        let primary_name = primary.name().to_string();

        // Build alternatives_considered for RoutingDecision event
        let policy = self.get_policy().await;
        let metrics_snapshot = self.metrics.all_metrics().await;
        let alternatives_considered: Vec<BackendScore> = self
            .backends
            .iter()
            .map(|b| {
                let score = match policy {
                    PolicyType::Availability => {
                        // Score is 1.0 if healthy, 0.0 if circuit open (sync approximation)
                        1.0
                    }
                    PolicyType::Cost => b.estimated_cost(request),
                    PolicyType::Latency => metrics_snapshot
                        .get(b.name())
                        .and_then(|m| m.latency_window.p50())
                        .unwrap_or(0.0),
                };
                BackendScore {
                    backend: b.name().to_string(),
                    score,
                    reason: format!("{:?} policy score", policy),
                }
            })
            .collect();

        // Emit RoutingDecision
        let _ = self.event_tx.send(MeshEvent::RoutingDecision {
            request_id: request.id,
            selected_backend: primary_name.clone(),
            policy: format!("{:?}", policy),
            reason: format!("Selected by {:?} policy", policy),
            alternatives_considered,
        });

        // Build an ordered list: primary first, then remaining backends
        let mut ordered: Vec<Arc<dyn InferenceBackend>> = Vec::new();
        ordered.push(Arc::clone(&primary));
        for b in &self.backends {
            if b.name() != primary_name {
                ordered.push(Arc::clone(b));
            }
        }

        let mut attempt: u32 = 0;
        let mut last_error: Option<String> = None;

        for (i, backend) in ordered.iter().enumerate() {
            // Skip circuit-open backends (except the primary on first attempt)
            if i > 0 {
                let circuit_open = self.health_tracker.is_circuit_open(backend.name()).await;
                if circuit_open {
                    continue;
                }
            }

            attempt += 1;

            // Emit BackendAttempt
            let _ = self.event_tx.send(MeshEvent::BackendAttempt {
                request_id: request.id,
                backend: backend.name().to_string(),
                attempt,
            });

            match backend.infer(request).await {
                Ok(response) => {
                    // Record success
                    self.health_tracker.record_success(backend.name()).await;
                    self.metrics
                        .record_request(
                            backend.name(),
                            response.latency_ms,
                            response.usage.total_tokens,
                            response.cost_usd,
                        )
                        .await;

                    let _ = self.event_tx.send(MeshEvent::BackendSuccess {
                        request_id: request.id,
                        backend: backend.name().to_string(),
                        latency_ms: response.latency_ms,
                        tokens: response.usage.total_tokens,
                        cost_usd: response.cost_usd,
                    });

                    let total_latency_ms = start.elapsed().as_millis() as u64;
                    let _ = self.event_tx.send(MeshEvent::ResponseComplete {
                        request_id: request.id,
                        total_latency_ms,
                        total_cost_usd: response.cost_usd,
                    });

                    return Ok(response);
                }
                Err(e) => {
                    let err_str = e.to_string();
                    last_error = Some(err_str.clone());

                    // Record failure
                    self.health_tracker.record_failure(backend.name()).await;
                    self.metrics.record_error(backend.name()).await;

                    // Determine next backend to try (for failover_to field)
                    let failover_to = ordered.get(i + 1).map(|b| b.name().to_string());

                    let _ = self.event_tx.send(MeshEvent::BackendFailure {
                        request_id: request.id,
                        backend: backend.name().to_string(),
                        error: err_str,
                        failover_to,
                    });
                }
            }
        }

        Err(anyhow!(
            "All backends failed. Last error: {}",
            last_error.unwrap_or_else(|| "unknown".to_string())
        ))
    }

    pub async fn simulate_failure(&self, backend_name: &str) -> Result<()> {
        // Force 3 consecutive failures to trip the circuit breaker
        for _ in 0..3 {
            self.health_tracker.record_failure(backend_name).await;
        }
        let _ = self.event_tx.send(MeshEvent::BackendHealthChanged {
            backend: backend_name.to_string(),
            status: HealthStatus::Unhealthy,
            timestamp: chrono::Utc::now(),
        });
        Ok(())
    }
}
