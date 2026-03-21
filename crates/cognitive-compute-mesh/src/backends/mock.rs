use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use async_trait::async_trait;
use anyhow::{Result, anyhow};
use futures::Stream;
use rand::Rng;
use tokio::time::{sleep, Duration};
use uuid::Uuid;

use crate::protocol::{
    BackendCapabilities, BackendType, HealthStatus, InferenceRequest, InferenceResponse, TokenUsage,
};
use super::{InferenceBackend, StreamChunk};

pub struct MockBackend {
    pub name: String,
    pub backend_type: BackendType,
    pub simulated_latency_ms: u64,
    pub simulated_cost_per_token: f64,
    pub failure_rate: f32,
    pub responses: Vec<String>,
    pub response_index: Arc<AtomicUsize>,
}

impl MockBackend {
    pub fn new_local() -> Self {
        Self {
            name: "mock-local".to_string(),
            backend_type: BackendType::Local,
            simulated_latency_ms: 200,
            simulated_cost_per_token: 0.0,
            failure_rate: 0.0,
            responses: mock_responses(),
            response_index: Arc::new(AtomicUsize::new(0)),
        }
    }

    pub fn new_cloud() -> Self {
        Self {
            name: "mock-cloud".to_string(),
            backend_type: BackendType::Cloud,
            simulated_latency_ms: 800,
            simulated_cost_per_token: 0.002,
            failure_rate: 0.0,
            responses: mock_responses(),
            response_index: Arc::new(AtomicUsize::new(0)),
        }
    }
}

fn mock_responses() -> Vec<String> {
    vec![
        "I am a mock AI assistant. The Cognitive Compute Mesh routes requests intelligently across backends.".to_string(),
        "Mock response: This demonstrates local inference running at zero cost on your machine.".to_string(),
        "The routing engine supports three policies: availability-first, cost-optimized, and latency-optimized.".to_string(),
        "OminiX-MLX enables Apple Silicon acceleration for local model inference with zero API costs.".to_string(),
    ]
}

#[async_trait]
impl InferenceBackend for MockBackend {
    fn name(&self) -> &str {
        &self.name
    }

    fn backend_type(&self) -> BackendType {
        self.backend_type.clone()
    }

    async fn health_check(&self) -> Result<HealthStatus> {
        if self.failure_rate > 0.0 {
            let mut rng = rand::thread_rng();
            let roll: f32 = rng.r#gen();
            if roll < self.failure_rate {
                return Ok(HealthStatus::Unhealthy);
            }
        }
        Ok(HealthStatus::Healthy)
    }

    async fn infer(&self, request: &InferenceRequest) -> Result<InferenceResponse> {
        sleep(Duration::from_millis(self.simulated_latency_ms)).await;

        if self.failure_rate > 0.0 {
            let mut rng = rand::thread_rng();
            let roll: f32 = rng.r#gen();
            if roll < self.failure_rate {
                return Err(anyhow!("MockBackend simulated failure"));
            }
        }

        let idx = self.response_index.fetch_add(1, Ordering::Relaxed) % self.responses.len();
        let response_content = self.responses[idx].clone();

        let prompt_tokens = (request.messages.len() * 10) as u32;
        let completion_tokens = (response_content.len() / 4) as u32;
        let total_tokens = prompt_tokens + completion_tokens;
        let cost_usd = total_tokens as f64 * self.simulated_cost_per_token;

        Ok(InferenceResponse {
            id: Uuid::new_v4(),
            request_id: request.id,
            backend: self.name.clone(),
            content: response_content,
            usage: TokenUsage {
                prompt_tokens,
                completion_tokens,
                total_tokens,
            },
            latency_ms: self.simulated_latency_ms,
            cost_usd,
            model: request.model.clone(),
        })
    }

    async fn infer_stream(
        &self,
        request: &InferenceRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk>> + Send>>> {
        let idx = self.response_index.fetch_add(1, Ordering::Relaxed) % self.responses.len();
        let full_response = self.responses[idx].clone();
        let chunk_delay_ms = self.simulated_latency_ms / 10;

        // Collect characters into owned chunks of 5
        let chars: Vec<char> = full_response.chars().collect();
        let chunks: Vec<String> = chars
            .chunks(5)
            .map(|c| c.iter().collect::<String>())
            .collect();
        let total = chunks.len();

        let stream = futures::stream::unfold(
            (chunks, 0usize, chunk_delay_ms),
            move |(chunks, idx, delay)| async move {
                if idx >= chunks.len() {
                    return None;
                }
                sleep(Duration::from_millis(delay)).await;
                let content = chunks[idx].clone();
                let done = idx + 1 >= total;
                let chunk = StreamChunk { content, done };
                Some((Ok::<StreamChunk, anyhow::Error>(chunk), (chunks, idx + 1, delay)))
            },
        );

        Ok(Box::pin(stream))
    }

    fn estimated_cost(&self, request: &InferenceRequest) -> f64 {
        let total_chars: usize = request.messages.iter().map(|m| m.content.len()).sum();
        (total_chars as f64 / 4.0) * self.simulated_cost_per_token
    }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            streaming: true,
            tools: false,
            vision: false,
            max_context_tokens: 32768,
        }
    }
}
