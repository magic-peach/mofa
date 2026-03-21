use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use async_trait::async_trait;
use anyhow::Result;
use serde_json::Value;
use crate::protocol::SearchResult;
use super::store::VectorStore;

#[derive(Clone)]
struct VectorEntry {
    id: String,
    vector: Vec<f32>,
    payload: Value,
}

pub struct MemoryVectorStore {
    entries: Arc<RwLock<HashMap<String, VectorEntry>>>,
}

impl MemoryVectorStore {
    pub fn new() -> Self {
        Self {
            entries: Arc::new(RwLock::new(HashMap::new())),
        }
    }
}

#[async_trait]
impl VectorStore for MemoryVectorStore {
    async fn upsert(&self, id: &str, vector: Vec<f32>, payload: Value) -> Result<()> {
        self.entries.write().await.insert(id.to_string(), VectorEntry {
            id: id.to_string(),
            vector,
            payload,
        });
        Ok(())
    }

    async fn search(&self, query: Vec<f32>, top_k: usize) -> Result<Vec<SearchResult>> {
        let entries = self.entries.read().await;
        let mut scores: Vec<(String, f32, Value)> = entries.values()
            .map(|e| {
                let score = cosine_similarity(&query, &e.vector);
                (e.id.clone(), score, e.payload.clone())
            })
            .collect();

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(top_k);

        Ok(scores.into_iter().map(|(id, score, payload)| SearchResult {
            id,
            score,
            payload,
        }).collect())
    }

    async fn delete(&self, id: &str) -> Result<()> {
        self.entries.write().await.remove(id);
        Ok(())
    }

    async fn count(&self) -> Result<usize> {
        Ok(self.entries.read().await.len())
    }
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() { return 0.0; }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 { 0.0 } else { dot / (norm_a * norm_b) }
}
