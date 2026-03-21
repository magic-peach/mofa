use std::pin::Pin;
use async_trait::async_trait;
use anyhow::{Result, anyhow};
use futures::Stream;
use crate::backends::{InferenceBackend, StreamChunk};
use crate::protocol::*;

pub struct LocalBackend {
    client: reqwest::Client,
    base_url: String,
}

impl LocalBackend {
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::new(),
            base_url: std::env::var("OMINIX_URL")
                .unwrap_or_else(|_| "http://localhost:8080".to_string()),
        }
    }
}

#[async_trait]
impl InferenceBackend for LocalBackend {
    fn name(&self) -> &str { "ominix-local" }
    fn backend_type(&self) -> BackendType { BackendType::Local }

    async fn health_check(&self) -> Result<HealthStatus> {
        match self.client
            .get(format!("{}/health", self.base_url))
            .timeout(std::time::Duration::from_secs(2))
            .send()
            .await
        {
            Ok(resp) if resp.status().is_success() => Ok(HealthStatus::Healthy),
            Ok(_) => Ok(HealthStatus::Degraded),
            Err(_) => Ok(HealthStatus::Unhealthy),
        }
    }

    async fn infer(&self, request: &InferenceRequest) -> Result<InferenceResponse> {
        let start = std::time::Instant::now();

        let messages: Vec<serde_json::Value> = request.messages.iter()
            .map(|m| serde_json::json!({"role": m.role, "content": m.content}))
            .collect();

        let body = serde_json::json!({
            "model": request.model,
            "messages": messages,
            "max_tokens": request.max_tokens.unwrap_or(1000),
            "temperature": request.temperature.unwrap_or(0.7),
        });

        let response = self.client
            .post(format!("{}/v1/chat/completions", self.base_url))
            .json(&body)
            .timeout(std::time::Duration::from_secs(120))
            .send()
            .await
            .map_err(|e| anyhow!("OminiX-MLX server not running at {}: {}", self.base_url, e))?;

        let latency_ms = start.elapsed().as_millis() as u64;

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            return Err(anyhow!("OminiX-MLX error {}: {}", status, text));
        }

        let json: serde_json::Value = response.json().await?;

        let content = json["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or("")
            .to_string();

        let prompt_tokens = json["usage"]["prompt_tokens"].as_u64().unwrap_or(0) as u32;
        let completion_tokens = json["usage"]["completion_tokens"].as_u64().unwrap_or(0) as u32;
        let model = json["model"].as_str().unwrap_or("local").to_string();

        Ok(InferenceResponse {
            id: uuid::Uuid::new_v4(),
            request_id: request.id,
            backend: "ominix-local".to_string(),
            content,
            usage: TokenUsage {
                prompt_tokens,
                completion_tokens,
                total_tokens: prompt_tokens + completion_tokens,
            },
            latency_ms,
            cost_usd: 0.0, // Always free for local
            model,
        })
    }

    async fn infer_stream(&self, request: &InferenceRequest) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk>> + Send>>> {
        let response = self.infer(request).await?;
        Ok(Box::pin(futures::stream::iter(vec![Ok(StreamChunk { content: response.content, done: true })])))
    }

    fn estimated_cost(&self, _request: &InferenceRequest) -> f64 {
        0.0 // Always free
    }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            streaming: true,
            tools: false,
            vision: false,
            max_context_tokens: 4096,
        }
    }
}
