use std::collections::HashMap;
use std::sync::Arc;
use anyhow::Result;
use tokio::sync::RwLock;
use crate::protocol::{Document, RetrievedDocument};
use super::embeddings::{EmbeddingProvider, cosine_similarity};
use super::cache::EmbeddingCache;

/// BM25 index for sparse retrieval
pub struct BM25Index {
    documents: Vec<Document>,
    term_freq: HashMap<String, HashMap<usize, f32>>,   // term -> doc_id -> tf
    doc_freq: HashMap<String, usize>,                   // term -> number of docs containing it
    doc_lengths: Vec<usize>,                            // length of each doc
    avg_doc_len: f32,
    k1: f32,  // 1.5
    b: f32,   // 0.75
}

impl BM25Index {
    pub fn new() -> Self {
        Self {
            documents: Vec::new(),
            term_freq: HashMap::new(),
            doc_freq: HashMap::new(),
            doc_lengths: Vec::new(),
            avg_doc_len: 0.0,
            k1: 1.5,
            b: 0.75,
        }
    }

    fn tokenize(text: &str) -> Vec<String> {
        text.to_lowercase()
            .split_whitespace()
            .map(|w| w.trim_matches(|c: char| !c.is_alphabetic()).to_string())
            .filter(|w| !w.is_empty() && w.len() > 1)
            .collect()
    }

    pub fn add_document(&mut self, doc: Document) {
        let doc_id = self.documents.len();
        let tokens = Self::tokenize(&doc.content);
        let doc_len = tokens.len();

        // Count term frequencies
        let mut local_tf: HashMap<String, usize> = HashMap::new();
        for token in &tokens {
            *local_tf.entry(token.clone()).or_insert(0) += 1;
        }

        // Update global structures
        for (term, count) in &local_tf {
            let tf = *count as f32 / doc_len.max(1) as f32;
            self.term_freq.entry(term.clone()).or_default().insert(doc_id, tf);
            *self.doc_freq.entry(term.clone()).or_insert(0) += 1;
        }

        self.doc_lengths.push(doc_len);
        self.documents.push(doc);

        // Update avg doc len
        let total: usize = self.doc_lengths.iter().sum();
        self.avg_doc_len = total as f32 / self.documents.len() as f32;
    }

    pub fn search(&self, query: &str, top_k: usize) -> Vec<(usize, f32)> {
        let query_terms = Self::tokenize(query);
        let n = self.documents.len();
        if n == 0 { return vec![]; }

        let mut scores: HashMap<usize, f32> = HashMap::new();

        for term in &query_terms {
            let df = *self.doc_freq.get(term).unwrap_or(&0);
            if df == 0 { continue; }

            // IDF with smoothing
            let idf = ((n as f32 - df as f32 + 0.5) / (df as f32 + 0.5) + 1.0).ln();

            if let Some(postings) = self.term_freq.get(term) {
                for (&doc_id, &tf) in postings {
                    let doc_len = self.doc_lengths[doc_id] as f32;
                    let tf_norm = tf * (self.k1 + 1.0) /
                        (tf + self.k1 * (1.0 - self.b + self.b * doc_len / self.avg_doc_len));
                    *scores.entry(doc_id).or_insert(0.0) += idf * tf_norm;
                }
            }
        }

        let mut sorted: Vec<(usize, f32)> = scores.into_iter().collect();
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        sorted.truncate(top_k);
        sorted
    }

    pub fn get_document(&self, idx: usize) -> Option<&Document> {
        self.documents.get(idx)
    }

    pub fn len(&self) -> usize {
        self.documents.len()
    }
}

/// Hybrid retriever combining dense (vector) and sparse (BM25) retrieval
pub struct HybridRetriever {
    pub bm25_index: Arc<RwLock<BM25Index>>,
    pub embedding_provider: Arc<dyn EmbeddingProvider>,
    pub embedding_cache: Arc<EmbeddingCache>,
}

impl HybridRetriever {
    pub fn new(
        embedding_provider: Arc<dyn EmbeddingProvider>,
        embedding_cache: Arc<EmbeddingCache>,
    ) -> Self {
        Self {
            bm25_index: Arc::new(RwLock::new(BM25Index::new())),
            embedding_provider,
            embedding_cache,
        }
    }

    /// Add document to both BM25 index and get embedding (caller stores vector separately)
    pub async fn add_document(&self, doc: Document) -> Result<Vec<f32>> {
        let embedding = self.get_embedding(&doc.content).await?;
        self.bm25_index.write().await.add_document(doc);
        Ok(embedding)
    }

    async fn get_embedding(&self, text: &str) -> Result<Vec<f32>> {
        if let Some(cached) = self.embedding_cache.get(text).await {
            return Ok(cached);
        }
        let embeddings = self.embedding_provider.embed(&[text.to_string()]).await?;
        let embedding = embeddings.into_iter().next().unwrap_or_default();
        self.embedding_cache.set(text.to_string(), embedding.clone()).await;
        Ok(embedding)
    }

    /// Hybrid retrieval using Reciprocal Rank Fusion
    pub async fn retrieve_with_vectors(
        &self,
        query: &str,
        stored_embeddings: &[(String, Vec<f32>, Document)], // (doc_id, vector, doc)
        top_k: usize,
        dense_weight: f32,
        sparse_weight: f32,
    ) -> Result<Vec<RetrievedDocument>> {
        // 1. Dense retrieval
        let query_embedding = self.get_embedding(query).await?;
        let mut dense_scores: Vec<(usize, f32)> = stored_embeddings.iter().enumerate()
            .map(|(i, (_, vec, _))| (i, cosine_similarity(&query_embedding, vec)))
            .collect();
        dense_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // 2. Sparse (BM25) retrieval
        let bm25 = self.bm25_index.read().await;
        let sparse_scores = bm25.search(query, top_k * 2);
        drop(bm25);

        // 3. Reciprocal Rank Fusion (RRF)
        let k = 60.0f32; // RRF constant
        let mut rrf_scores: HashMap<usize, f32> = HashMap::new();

        for (rank, (idx, _)) in dense_scores.iter().take(top_k * 2).enumerate() {
            *rrf_scores.entry(*idx).or_insert(0.0) += dense_weight / (k + rank as f32 + 1.0);
        }

        for (rank, (idx, _)) in sparse_scores.iter().enumerate() {
            *rrf_scores.entry(*idx).or_insert(0.0) += sparse_weight / (k + rank as f32 + 1.0);
        }

        // 4. Sort by RRF score and return top_k
        let mut results: Vec<(usize, f32)> = rrf_scores.into_iter().collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(top_k);

        Ok(results.into_iter().filter_map(|(idx, score)| {
            stored_embeddings.get(idx).map(|(_, _, doc)| RetrievedDocument {
                document: doc.clone(),
                score,
                retrieval_method: "hybrid_rrf".to_string(),
            })
        }).collect())
    }
}
