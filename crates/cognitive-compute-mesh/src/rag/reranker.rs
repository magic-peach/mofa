// LLM-based reranker stub — will be implemented with a real backend
use crate::protocol::RetrievedDocument;
use anyhow::Result;

pub struct Reranker;

impl Reranker {
    pub fn new() -> Self { Self }

    /// Rerank documents by relevance to query (stub: returns unchanged)
    pub async fn rerank(&self, _query: &str, docs: Vec<RetrievedDocument>) -> Result<Vec<RetrievedDocument>> {
        Ok(docs) // No-op stub
    }
}
