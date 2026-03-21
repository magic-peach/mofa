use std::pin::Pin;
use async_trait::async_trait;
use anyhow::Result;
use futures::Stream;
use crate::protocol::{BackendType, BackendCapabilities, HealthStatus, InferenceRequest, InferenceResponse};

#[derive(Debug, Clone)]
pub struct StreamChunk {
    pub content: String,
    pub done: bool,
}

#[async_trait]
pub trait InferenceBackend: Send + Sync {
    fn name(&self) -> &str;
    fn backend_type(&self) -> BackendType;
    async fn health_check(&self) -> Result<HealthStatus>;
    async fn infer(&self, request: &InferenceRequest) -> Result<InferenceResponse>;
    async fn infer_stream(
        &self,
        request: &InferenceRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk>> + Send>>>;
    fn estimated_cost(&self, request: &InferenceRequest) -> f64;
    fn capabilities(&self) -> BackendCapabilities;
}

pub mod mock;
pub mod health;
pub mod metrics;
pub mod openai;
pub mod anthropic;
pub mod local;
