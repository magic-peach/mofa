use cognitive_compute_mesh::{
    rag::{MockEmbeddings, EmbeddingProvider, EmbeddingCache, HybridRetriever},
    protocol::Document,
};
use std::sync::Arc;
use std::collections::HashMap;

fn make_doc(id: &str, content: &str) -> Document {
    Document {
        id: id.to_string(),
        content: content.to_string(),
        metadata: HashMap::new(),
    }
}

#[tokio::test]
async fn test_mock_embeddings_are_deterministic() {
    let embedder = MockEmbeddings::new(128);

    let texts = vec!["hello world".to_string(), "hello world".to_string()];
    let results = embedder.embed(&texts).await.unwrap();

    assert_eq!(results[0], results[1], "Same text must produce same embedding");

    // Different texts should produce different embeddings
    let texts2 = vec!["hello world".to_string(), "completely different text xyz".to_string()];
    let results2 = embedder.embed(&texts2).await.unwrap();
    assert_ne!(results2[0], results2[1], "Different texts must produce different embeddings");
}

#[tokio::test]
async fn test_mock_embeddings_are_unit_vectors() {
    let embedder = MockEmbeddings::new(64);
    let results = embedder.embed(&["test".to_string()]).await.unwrap();

    let norm: f32 = results[0].iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 0.001, "Embedding must be a unit vector, got norm={}", norm);
}

#[tokio::test]
async fn test_bm25_retrieval() {
    use cognitive_compute_mesh::rag::retrieval::BM25Index;

    let mut index = BM25Index::new();
    index.add_document(make_doc("1", "Rust is a systems programming language"));
    index.add_document(make_doc("2", "Python is used for data science and machine learning"));
    index.add_document(make_doc("3", "Rust ownership model prevents memory errors"));

    let results = index.search("Rust programming", 2);

    assert!(!results.is_empty(), "Should return results");
    // Rust docs should rank higher than Python doc
    let top_idx = results[0].0;
    assert!(top_idx == 0 || top_idx == 2, "Top result should be a Rust document, got idx={}", top_idx);
}

#[tokio::test]
async fn test_retrieval_latency_under_200ms() {
    let embedder = Arc::new(MockEmbeddings::new(64));
    let cache = Arc::new(EmbeddingCache::new(100));
    let retriever = HybridRetriever::new(Arc::clone(&embedder) as Arc<dyn EmbeddingProvider>, cache);

    // Ingest 50 documents
    let mut stored = Vec::new();
    for i in 0..50 {
        let doc = make_doc(&i.to_string(), &format!("Document {} about topic {}", i, i % 10));
        let embedding = retriever.add_document(doc.clone()).await.unwrap();
        stored.push((doc.id.clone(), embedding, doc));
    }

    // Run 10 queries and measure latency
    let mut latencies = Vec::new();
    for i in 0..10 {
        let start = std::time::Instant::now();
        let _results = retriever.retrieve_with_vectors(
            &format!("query about topic {}", i),
            &stored,
            3,
            0.7,
            0.3,
        ).await.unwrap();
        latencies.push(start.elapsed().as_millis());
    }

    let p99 = {
        let mut sorted = latencies.clone();
        sorted.sort();
        sorted[sorted.len() * 99 / 100]
    };

    assert!(p99 < 200, "P99 retrieval latency must be < 200ms, got {}ms", p99);
}
