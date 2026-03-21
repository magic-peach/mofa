use async_trait::async_trait;
use anyhow::Result;

/// Embedding provider trait
#[async_trait]
pub trait EmbeddingProvider: Send + Sync {
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>>;
    fn dimensions(&self) -> usize;
    fn model_name(&self) -> &str;
}

/// Mock embeddings — deterministic per text (hash-seeded)
pub struct MockEmbeddings {
    dimensions: usize,
}

impl MockEmbeddings {
    pub fn new(dimensions: usize) -> Self {
        Self { dimensions }
    }
}

#[async_trait]
impl EmbeddingProvider for MockEmbeddings {
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        Ok(texts.iter().map(|text| deterministic_embedding(text, self.dimensions)).collect())
    }

    fn dimensions(&self) -> usize { self.dimensions }
    fn model_name(&self) -> &str { "mock-embeddings-v1" }
}

/// Generate a deterministic unit vector from text using a simple hash
fn deterministic_embedding(text: &str, dims: usize) -> Vec<f32> {
    // Use FNV-like hash to seed pseudo-random values
    let mut hash: u64 = 14695981039346656037;
    for byte in text.bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(1099511628211);
    }

    let mut vec: Vec<f32> = (0..dims).map(|i| {
        let seed = hash.wrapping_add(i as u64).wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        // Map to [-1.0, 1.0]
        ((seed >> 33) as f32 / (u32::MAX as f32)) * 2.0 - 1.0
    }).collect();

    // Normalize to unit vector
    let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        vec.iter_mut().for_each(|x| *x /= norm);
    }
    vec
}

/// Cosine similarity between two vectors
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 { 0.0 } else { dot / (norm_a * norm_b) }
}
