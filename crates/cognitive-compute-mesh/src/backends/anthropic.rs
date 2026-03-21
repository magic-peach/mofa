use std::pin::Pin;
use async_trait::async_trait;
use anyhow::{Result, anyhow};
use futures::Stream;
use crate::backends::{InferenceBackend, StreamChunk};
use crate::protocol::*;

pub struct AnthropicBackend {
    client: reqwest::Client,
    api_key: Option<String>,
}

impl AnthropicBackend {
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::new(),
            api_key: std::env::var("ANTHROPIC_API_KEY").ok(),
        }
    }
}

#[async_trait]
impl InferenceBackend for AnthropicBackend {
    fn name(&self) -> &str { "anthropic" }
    fn backend_type(&self) -> BackendType { BackendType::Cloud }

    async fn health_check(&self) -> Result<HealthStatus> {
        if self.api_key.is_none() {
            return Ok(HealthStatus::Unknown);
        }
        Ok(HealthStatus::Healthy)
    }

    async fn infer(&self, request: &InferenceRequest) -> Result<InferenceResponse> {
        let api_key = self.api_key.as_ref()
            .ok_or_else(|| anyhow!("Set ANTHROPIC_API_KEY to enable Anthropic backend"))?;

        let start = std::time::Instant::now();

        // Convert IRP messages to Anthropic format
        // Anthropic requires: system message separate, user/assistant alternating
        let system = request.messages.iter()
            .find(|m| m.role == "system")
            .map(|m| m.content.clone());

        let messages: Vec<serde_json::Value> = request.messages.iter()
            .filter(|m| m.role != "system")
            .map(|m| serde_json::json!({"role": m.role, "content": m.content}))
            .collect();

        let model = if request.model == "auto" { "claude-3-haiku-20240307" } else { &request.model };

        let mut body = serde_json::json!({
            "model": model,
            "messages": messages,
            "max_tokens": request.max_tokens.unwrap_or(1000),
        });

        if let Some(sys) = system {
            body["system"] = serde_json::Value::String(sys);
        }

        let response = self.client
            .post("https://api.anthropic.com/v1/messages")
            .header("x-api-key", api_key)
            .header("anthropic-version", "2023-06-01")
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await?;

        let latency_ms = start.elapsed().as_millis() as u64;

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            return Err(anyhow!("Anthropic API error {}: {}", status, text));
        }

        // Convert Anthropic response -> IRP
        let json: serde_json::Value = response.json().await?;

        let content = json["content"][0]["text"]
            .as_str()
            .unwrap_or("")
            .to_string();

        let input_tokens = json["usage"]["input_tokens"].as_u64().unwrap_or(0) as u32;
        let output_tokens = json["usage"]["output_tokens"].as_u64().unwrap_or(0) as u32;
        let total_tokens = input_tokens + output_tokens;
        let model_name = json["model"].as_str().unwrap_or(model).to_string();

        // claude-3-haiku: $0.00025 per 1k input, $0.00125 per 1k output
        let cost = (input_tokens as f64 * 0.00025 + output_tokens as f64 * 0.00125) / 1000.0;

        Ok(InferenceResponse {
            id: uuid::Uuid::new_v4(),
            request_id: request.id,
            backend: "anthropic".to_string(),
            content,
            usage: TokenUsage { prompt_tokens: input_tokens, completion_tokens: output_tokens, total_tokens },
            latency_ms,
            cost_usd: cost,
            model: model_name,
        })
    }

    async fn infer_stream(&self, request: &InferenceRequest) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk>> + Send>>> {
        let response = self.infer(request).await?;
        Ok(Box::pin(futures::stream::iter(vec![Ok(StreamChunk { content: response.content, done: true })])))
    }

    fn estimated_cost(&self, request: &InferenceRequest) -> f64 {
        let chars: usize = request.messages.iter().map(|m| m.content.len()).sum();
        (chars as f64 / 4.0) * 0.00125 / 1000.0
    }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            streaming: true,
            tools: true,
            vision: true,
            max_context_tokens: 200000,
        }
    }
}
