use std::collections::VecDeque;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

pub struct LatencyWindow {
    samples: VecDeque<u64>,
    max_samples: usize,
}

impl LatencyWindow {
    pub fn new(max_samples: usize) -> Self {
        Self {
            samples: VecDeque::new(),
            max_samples,
        }
    }

    pub fn add(&mut self, latency_ms: u64) {
        if self.samples.len() >= self.max_samples {
            self.samples.pop_front();
        }
        self.samples.push_back(latency_ms);
    }

    pub fn p50(&self) -> Option<f64> {
        if self.samples.is_empty() {
            return None;
        }
        let mut sorted: Vec<u64> = self.samples.iter().cloned().collect();
        sorted.sort_unstable();
        let idx = sorted.len() / 2;
        Some(sorted[idx] as f64)
    }

    pub fn p99(&self) -> Option<f64> {
        if self.samples.is_empty() {
            return None;
        }
        let mut sorted: Vec<u64> = self.samples.iter().cloned().collect();
        sorted.sort_unstable();
        let idx = ((sorted.len() as f64 * 0.99) as usize).min(sorted.len() - 1);
        Some(sorted[idx] as f64)
    }
}

pub struct BackendMetricsRecord {
    pub total_requests: u64,
    pub total_errors: u64,
    pub total_cost_usd: f64,
    pub total_tokens: u64,
    pub latency_window: LatencyWindow,
}

pub struct MetricsStore {
    records: Arc<RwLock<HashMap<String, BackendMetricsRecord>>>,
}

impl MetricsStore {
    pub fn new() -> Self {
        Self {
            records: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub async fn record_request(&self, backend: &str, latency_ms: u64, tokens: u32, cost_usd: f64) {
        let mut map = self.records.write().await;
        let record = map.entry(backend.to_string()).or_insert(BackendMetricsRecord {
            total_requests: 0,
            total_errors: 0,
            total_cost_usd: 0.0,
            total_tokens: 0,
            latency_window: LatencyWindow::new(100),
        });
        record.total_requests += 1;
        record.total_tokens += tokens as u64;
        record.total_cost_usd += cost_usd;
        record.latency_window.add(latency_ms);
    }

    pub async fn record_error(&self, backend: &str) {
        let mut map = self.records.write().await;
        let record = map.entry(backend.to_string()).or_insert(BackendMetricsRecord {
            total_requests: 0,
            total_errors: 0,
            total_cost_usd: 0.0,
            total_tokens: 0,
            latency_window: LatencyWindow::new(100),
        });
        record.total_errors += 1;
    }

    pub async fn get_metrics(&self, backend: &str) -> Option<BackendMetricsRecord> {
        // We can't Clone BackendMetricsRecord easily due to LatencyWindow, so return None for now
        // Callers should use all_metrics() for a full snapshot
        let map = self.records.read().await;
        if map.contains_key(backend) {
            // Return a snapshot by reading underlying values
            // Since BackendMetricsRecord isn't Clone, we reconstruct
            let r = map.get(backend)?;
            let mut lw = LatencyWindow::new(100);
            for &s in &r.latency_window.samples {
                lw.add(s);
            }
            Some(BackendMetricsRecord {
                total_requests: r.total_requests,
                total_errors: r.total_errors,
                total_cost_usd: r.total_cost_usd,
                total_tokens: r.total_tokens,
                latency_window: lw,
            })
        } else {
            None
        }
    }

    pub async fn all_metrics(&self) -> HashMap<String, BackendMetricsRecord> {
        let map = self.records.read().await;
        let mut result = HashMap::new();
        for (k, r) in map.iter() {
            let mut lw = LatencyWindow::new(100);
            for &s in &r.latency_window.samples {
                lw.add(s);
            }
            result.insert(k.clone(), BackendMetricsRecord {
                total_requests: r.total_requests,
                total_errors: r.total_errors,
                total_cost_usd: r.total_cost_usd,
                total_tokens: r.total_tokens,
                latency_window: lw,
            });
        }
        result
    }
}

impl Default for MetricsStore {
    fn default() -> Self {
        Self::new()
    }
}
