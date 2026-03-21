use cognitive_compute_mesh::{
    backends::{
        mock::MockBackend,
        openai::OpenAIBackend,
        anthropic::AnthropicBackend,
        local::LocalBackend,
        InferenceBackend,
    },
    routing::RoutingEngine,
    api::routes::{AppState, create_router},
};
use std::sync::Arc;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("cognitive_compute_mesh=info".parse().unwrap()),
        )
        .init();

    let backends: Vec<Arc<dyn InferenceBackend>> = vec![
        Arc::new(MockBackend::new_local()),
        Arc::new(MockBackend::new_cloud()),
        Arc::new(OpenAIBackend::new()),
        Arc::new(AnthropicBackend::new()),
        Arc::new(LocalBackend::new()),
    ];

    let engine = Arc::new(RoutingEngine::new(backends));
    let state = Arc::new(AppState::new(engine));
    let app = create_router(state);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:8090").await.unwrap();
    tracing::info!("Cognitive Compute Mesh server starting on port 8090");
    axum::serve(listener, app).await.unwrap();
}
