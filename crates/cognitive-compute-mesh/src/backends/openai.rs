use std::pin::Pin;
use async_trait::async_trait;
use anyhow::{Result, anyhow};
use futures::Stream;
use crate::backends::{InferenceBackend, StreamChunk};
use crate::protocol::*;

pub struct OpenAIBackend {
    client: reqwest::Client,
    api_key: Option<String>,
}

impl OpenAIBackend {
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::new(),
            api_key: std::env::var("OPENAI_API_KEY").ok(),
        }
    }
}

#[async_trait]
impl InferenceBackend for OpenAIBackend {
    fn name(&self) -> &str { "openai" }
    fn backend_type(&self) -> BackendType { BackendType::Cloud }

    async fn health_check(&self) -> Result<HealthStatus> {
        if self.api_key.is_none() {
            return Ok(HealthStatus::Unknown);
        }
        Ok(HealthStatus::Healthy)
    }

    async fn infer(&self, request: &InferenceRequest) -> Result<InferenceResponse> {
        let api_key = self.api_key.as_ref()
            .ok_or_else(|| anyhow!("Set OPENAI_API_KEY to enable OpenAI backend"))?;

        let start = std::time::Instant::now();

        let openai_messages: Vec<serde_json::Value> = request.messages.iter()
            .map(|m| serde_json::json!({"role": m.role, "content": m.content}))
            .collect();

        let body = serde_json::json!({
            "model": if request.model == "auto" { "gpt-3.5-turbo" } else { &request.model },
            "messages": openai_messages,
            "max_tokens": request.max_tokens.unwrap_or(1000),
            "temperature": request.temperature.unwrap_or(0.7),
        });

        let response = self.client
            .post("https://api.openai.com/v1/chat/completions")
            .header("Authorization", format!("Bearer {}", api_key))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await?;

        let latency_ms = start.elapsed().as_millis() as u64;

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            return Err(anyhow!("OpenAI API error {}: {}", status, text));
        }

        let json: serde_json::Value = response.json().await?;

        let content = json["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or("")
            .to_string();

        let prompt_tokens = json["usage"]["prompt_tokens"].as_u64().unwrap_or(0) as u32;
        let completion_tokens = json["usage"]["completion_tokens"].as_u64().unwrap_or(0) as u32;
        let total_tokens = json["usage"]["total_tokens"].as_u64().unwrap_or(0) as u32;
        let model = json["model"].as_str().unwrap_or("gpt-3.5-turbo").to_string();

        // Cost: ~$0.002 per 1k tokens for gpt-3.5-turbo
        let cost = total_tokens as f64 * 0.002 / 1000.0;

        Ok(InferenceResponse {
            id: uuid::Uuid::new_v4(),
            request_id: request.id,
            backend: "openai".to_string(),
            content,
            usage: TokenUsage { prompt_tokens, completion_tokens, total_tokens },
            latency_ms,
            cost_usd: cost,
            model,
        })
    }

    async fn infer_stream(&self, request: &InferenceRequest) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk>> + Send>>> {
        let response = self.infer(request).await?;
        let chunks: Vec<Result<StreamChunk>> = vec![
            Ok(StreamChunk { content: response.content, done: true }),
        ];
        Ok(Box::pin(futures::stream::iter(chunks)))
    }

    fn estimated_cost(&self, request: &InferenceRequest) -> f64 {
        let chars: usize = request.messages.iter().map(|m| m.content.len()).sum();
        (chars as f64 / 4.0) * 0.002 / 1000.0
    }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            streaming: true,
            tools: true,
            vision: false,
            max_context_tokens: 16385,
        }
    }
}
