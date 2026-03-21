use std::sync::Arc;
use axum::{
    Router,
    routing::{get, post},
    extract::{State, Path},
    response::Json,
    http::StatusCode,
};
use serde_json::{json, Value};
use tokio::sync::broadcast;
use crate::routing::RoutingEngine;
use crate::protocol::*;
use crate::rag::{HybridRetriever, EmbeddingProvider, MockEmbeddings, EmbeddingCache};
use crate::vector::memory::MemoryVectorStore;
use crate::vector::store::VectorStore;
use super::openai_compat::*;

pub struct RagStore {
    pub retriever: Arc<HybridRetriever>,
    pub vector_store: Arc<MemoryVectorStore>,
    pub stored: Arc<tokio::sync::RwLock<Vec<(String, Vec<f32>, Document)>>>,
}

pub struct AppState {
    pub engine: Arc<RoutingEngine>,
    pub event_tx: broadcast::Sender<MeshEvent>,
    pub rag: Arc<RagStore>,
}

impl AppState {
    pub fn new(engine: Arc<RoutingEngine>) -> Self {
        let event_tx = engine.event_tx.clone();
        let embedding_provider: Arc<dyn EmbeddingProvider> = Arc::new(MockEmbeddings::new(128));
        let embedding_cache = Arc::new(EmbeddingCache::new(1000));
        let retriever = Arc::new(HybridRetriever::new(embedding_provider, embedding_cache));
        let vector_store = Arc::new(MemoryVectorStore::new());
        let rag = Arc::new(RagStore {
            retriever,
            vector_store,
            stored: Arc::new(tokio::sync::RwLock::new(Vec::new())),
        });
        Self { engine, event_tx, rag }
    }
}

async fn health() -> Json<Value> {
    Json(json!({"status": "ok", "service": "cognitive-compute-mesh"}))
}

async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<OpenAIChatRequest>,
) -> Result<Json<OpenAIChatResponse>, (StatusCode, Json<Value>)> {
    let irp_req = openai_to_irp(req);
    match state.engine.route_with_fallback(&irp_req).await {
        Ok(resp) => Ok(Json(irp_to_openai(resp))),
        Err(e) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": e.to_string()})),
        )),
    }
}

async fn infer(
    State(state): State<Arc<AppState>>,
    Json(req): Json<InferenceRequest>,
) -> Result<Json<InferenceResponse>, (StatusCode, Json<Value>)> {
    match state.engine.route_with_fallback(&req).await {
        Ok(resp) => Ok(Json(resp)),
        Err(e) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": e.to_string()})),
        )),
    }
}

async fn list_backends(State(state): State<Arc<AppState>>) -> Json<Value> {
    let health_tracker = state.engine.health_tracker();
    let backends = state.engine.backends();

    let mut result = Vec::new();
    for backend in backends {
        let status = health_tracker.get_status(backend.name()).await;
        result.push(json!({
            "name": backend.name(),
            "backend_type": backend.backend_type(),
            "status": status,
            "capabilities": backend.capabilities(),
        }));
    }

    Json(json!(result))
}

async fn backend_health(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    let backends = state.engine.backends();
    let backend = backends.iter().find(|b| b.name() == name);

    match backend {
        Some(b) => {
            let health_tracker = state.engine.health_tracker();
            let status = health_tracker.get_status(b.name()).await;
            let circuit_open = health_tracker.is_circuit_open(b.name()).await;
            Ok(Json(json!({
                "name": b.name(),
                "status": status,
                "circuit_open": circuit_open,
            })))
        }
        None => Err((
            StatusCode::NOT_FOUND,
            Json(json!({"error": format!("Backend '{}' not found", name)})),
        )),
    }
}

async fn simulate_backend_failure(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
) -> Json<Value> {
    match state.engine.simulate_failure(&name).await {
        Ok(_) => Json(json!({"backend": name, "simulated": true})),
        Err(e) => Json(json!({"backend": name, "simulated": false, "error": e.to_string()})),
    }
}

async fn get_metrics(State(state): State<Arc<AppState>>) -> Json<Value> {
    let metrics_store = state.engine.metrics();
    let all = metrics_store.all_metrics().await;

    let mut result = serde_json::Map::new();
    for (name, record) in &all {
        let error_rate = if record.total_requests + record.total_errors > 0 {
            record.total_errors as f64 / (record.total_requests + record.total_errors) as f64
        } else {
            0.0
        };
        result.insert(name.clone(), json!({
            "total_requests": record.total_requests,
            "total_errors": record.total_errors,
            "total_cost_usd": record.total_cost_usd,
            "total_tokens": record.total_tokens,
            "p50_ms": record.latency_window.p50(),
            "p99_ms": record.latency_window.p99(),
            "error_rate": error_rate,
        }));
    }

    Json(Value::Object(result))
}

async fn metrics_compare(State(state): State<Arc<AppState>>) -> Json<Value> {
    let metrics_store = state.engine.metrics();
    let health_tracker = state.engine.health_tracker();
    let all_metrics = metrics_store.all_metrics().await;
    let backends = state.engine.backends();

    let mut comparison = Vec::new();
    for backend in backends {
        let status = health_tracker.get_status(backend.name()).await;
        let (p50_ms, p99_ms, total_requests, error_rate, cost_per_1k_tokens) =
            if let Some(record) = all_metrics.get(backend.name()) {
                let error_rate = if record.total_requests + record.total_errors > 0 {
                    record.total_errors as f64
                        / (record.total_requests + record.total_errors) as f64
                } else {
                    0.0
                };
                let cost_per_1k = if record.total_tokens > 0 {
                    (record.total_cost_usd / record.total_tokens as f64) * 1000.0
                } else {
                    0.0
                };
                (
                    record.latency_window.p50(),
                    record.latency_window.p99(),
                    record.total_requests,
                    error_rate,
                    cost_per_1k,
                )
            } else {
                (None, None, 0u64, 0.0f64, 0.0f64)
            };

        comparison.push(json!({
            "name": backend.name(),
            "status": status,
            "p50_ms": p50_ms,
            "p99_ms": p99_ms,
            "cost_per_1k_tokens": cost_per_1k_tokens,
            "total_requests": total_requests,
            "error_rate": error_rate,
        }));
    }

    Json(json!({"backends": comparison}))
}

async fn get_routing_policy(State(state): State<Arc<AppState>>) -> Json<Value> {
    let policy = state.engine.get_policy().await;
    Json(json!({"policy": policy}))
}

async fn set_routing_policy(
    State(state): State<Arc<AppState>>,
    Json(routing_policy): Json<RoutingPolicy>,
) -> Json<Value> {
    state.engine.set_policy(routing_policy.policy.clone()).await;
    Json(json!({"policy": routing_policy.policy, "updated": true}))
}

async fn simulate_routing(
    State(state): State<Arc<AppState>>,
    Json(req): Json<InferenceRequest>,
) -> Json<Value> {
    match state.engine.route(&req).await {
        Ok(backend) => Json(json!({
            "selected_backend": backend.name(),
            "backend_type": backend.backend_type(),
        })),
        Err(e) => Json(json!({"error": e.to_string()})),
    }
}

async fn rag_query(
    State(state): State<Arc<AppState>>,
    Json(body): Json<Value>,
) -> Json<Value> {
    let query = body.get("query").and_then(|v| v.as_str()).unwrap_or("").to_string();
    let top_k = body.get("top_k").and_then(|v| v.as_u64()).unwrap_or(5) as usize;

    if query.is_empty() {
        return Json(json!({"results": [], "query": query, "error": "empty query"}));
    }

    let stored = state.rag.stored.read().await;
    match state.rag.retriever.retrieve_with_vectors(&query, &stored, top_k, 0.7, 0.3).await {
        Ok(results) => {
            let items: Vec<Value> = results.into_iter().map(|r| json!({
                "id": r.document.id,
                "score": r.score,
                "content": r.document.content,
                "metadata": r.document.metadata,
            })).collect();
            Json(json!({"results": items, "query": query}))
        }
        Err(e) => Json(json!({"results": [], "query": query, "error": e.to_string()})),
    }
}

async fn rag_ingest(
    State(state): State<Arc<AppState>>,
    Json(body): Json<Value>,
) -> Json<Value> {
    let docs_val = match body.get("documents").and_then(|v| v.as_array()) {
        Some(arr) => arr.clone(),
        None => return Json(json!({"ingested": 0, "error": "missing documents array"})),
    };

    let mut ingested = 0usize;
    for doc_val in &docs_val {
        let id = doc_val.get("id").and_then(|v| v.as_str()).unwrap_or("").to_string();
        let content = doc_val.get("content").and_then(|v| v.as_str()).unwrap_or("").to_string();
        if id.is_empty() || content.is_empty() { continue; }

        let metadata = doc_val.get("metadata")
            .and_then(|v| v.as_object())
            .map(|m| m.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
            .unwrap_or_default();

        let doc = Document { id: id.clone(), content, metadata };
        match state.rag.retriever.add_document(doc.clone()).await {
            Ok(embedding) => {
                let payload = json!({"id": doc.id, "content": doc.content});
                let _ = state.rag.vector_store.upsert(&id, embedding.clone(), payload).await;
                state.rag.stored.write().await.push((id, embedding, doc));
                ingested += 1;
            }
            Err(_) => continue,
        }
    }

    let total = state.rag.stored.read().await.len();
    Json(json!({"ingested": ingested, "total": total}))
}

pub fn create_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/infer", post(infer))
        .route("/v1/backends", get(list_backends))
        .route("/v1/backends/:name/health", get(backend_health))
        .route("/v1/backends/:name/simulate-failure", post(simulate_backend_failure))
        .route("/v1/metrics", get(get_metrics))
        .route("/v1/metrics/compare", get(metrics_compare))
        .route("/v1/routing/policy", get(get_routing_policy).put(set_routing_policy))
        .route("/v1/routing/simulate", post(simulate_routing))
        .route("/v1/rag/query", post(rag_query))
        .route("/v1/rag/ingest", post(rag_ingest))
        .route("/ws/events", get(super::websocket::ws_handler))
        .with_state(state)
        .layer(tower_http::cors::CorsLayer::permissive())
}
