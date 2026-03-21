use axum::{routing::get, Json, Router};
use serde_json::{json, Value};
use tracing::info;

async fn health() -> Json<Value> {
    Json(json!({
        "status": "ok",
        "service": "cognitive-compute-mesh"
    }))
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    info!("Cognitive Compute Mesh server starting on port 8090");

    let app = Router::new().route("/health", get(health));

    let listener = tokio::net::TcpListener::bind("0.0.0.0:8090")
        .await
        .expect("Failed to bind to port 8090");

    axum::serve(listener, app)
        .await
        .expect("Server failed");
}
