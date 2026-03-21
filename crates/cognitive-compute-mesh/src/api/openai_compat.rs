use serde::{Deserialize, Serialize};
use uuid::Uuid;
use std::collections::HashMap;

use crate::protocol::{InferenceRequest, InferenceResponse, Message};

#[derive(Debug, Clone, Deserialize)]
pub struct OpenAIChatRequest {
    pub model: String,
    pub messages: Vec<OpenAIMessage>,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub stream: Option<bool>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OpenAIMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Serialize)]
pub struct OpenAIChatResponse {
    pub id: String,
    pub object: String,
    pub model: String,
    pub backend: String,
    pub choices: Vec<OpenAIChoice>,
    pub usage: OpenAIUsage,
    pub latency_ms: u64,
    pub cost_usd: f64,
}

#[derive(Debug, Serialize)]
pub struct OpenAIChoice {
    pub index: u32,
    pub message: OpenAIMessage,
    pub finish_reason: String,
}

#[derive(Debug, Serialize)]
pub struct OpenAIUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

pub fn openai_to_irp(req: OpenAIChatRequest) -> InferenceRequest {
    let messages: Vec<Message> = req
        .messages
        .into_iter()
        .map(|m| Message {
            role: m.role,
            content: m.content,
        })
        .collect();

    InferenceRequest {
        id: Uuid::new_v4(),
        model: req.model,
        messages,
        max_tokens: req.max_tokens,
        temperature: req.temperature,
        stream: req.stream.unwrap_or(false),
        tools: None,
        metadata: HashMap::new(),
    }
}

pub fn irp_to_openai(resp: InferenceResponse) -> OpenAIChatResponse {
    OpenAIChatResponse {
        id: resp.id.to_string(),
        object: "chat.completion".to_string(),
        model: resp.model.clone(),
        backend: resp.backend.clone(),
        choices: vec![OpenAIChoice {
            index: 0,
            message: OpenAIMessage {
                role: "assistant".to_string(),
                content: resp.content,
            },
            finish_reason: "stop".to_string(),
        }],
        usage: OpenAIUsage {
            prompt_tokens: resp.usage.prompt_tokens,
            completion_tokens: resp.usage.completion_tokens,
            total_tokens: resp.usage.total_tokens,
        },
        latency_ms: resp.latency_ms,
        cost_usd: resp.cost_usd,
    }
}
