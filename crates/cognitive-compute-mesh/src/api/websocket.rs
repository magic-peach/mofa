use axum::{
    extract::{State, WebSocketUpgrade},
    response::Response,
};
use axum::extract::ws::{Message, WebSocket};
use std::sync::Arc;
use tokio::time::{interval, Duration};
use futures_util::{SinkExt, StreamExt};
use super::routes::AppState;
use crate::protocol::{MeshEvent, BackendMetrics};

pub async fn ws_handler(
    ws: WebSocketUpgrade,
    State(state): State<Arc<AppState>>,
) -> Response {
    ws.on_upgrade(move |socket| handle_socket(socket, state))
}

async fn handle_socket(socket: WebSocket, state: Arc<AppState>) {
    let mut rx = state.event_tx.subscribe();
    let (mut sender, mut receiver) = socket.split();

    // Send initial MetricsUpdate on connect
    let initial_metrics = build_metrics_event(&state).await;
    if let Ok(json) = serde_json::to_string(&initial_metrics) {
        let _ = sender.send(Message::Text(json.into())).await;
    }

    // Spawn task to periodically emit MetricsUpdate events to the broadcast channel
    let state_clone = Arc::clone(&state);
    let metrics_task = tokio::spawn(async move {
        let mut ticker = interval(Duration::from_secs(5));
        loop {
            ticker.tick().await;
            let event = build_metrics_event(&state_clone).await;
            let _ = state_clone.event_tx.send(event);
        }
    });

    // Main event loop: forward broadcast events to websocket client
    loop {
        tokio::select! {
            // Event from broadcast channel
            result = rx.recv() => {
                match result {
                    Ok(event) => {
                        if let Ok(json) = serde_json::to_string(&event) {
                            if sender.send(Message::Text(json.into())).await.is_err() {
                                break; // Client disconnected
                            }
                        }
                    }
                    Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
                    Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => continue,
                }
            }
            // Client sent a message (or closed)
            result = receiver.next() => {
                match result {
                    Some(Ok(Message::Close(_))) | None => break,
                    _ => {} // Ignore other messages (ping/pong/text)
                }
            }
        }
    }

    metrics_task.abort();
}

async fn build_metrics_event(state: &Arc<AppState>) -> MeshEvent {
    let health_tracker = state.engine.health_tracker();
    let metrics_store = state.engine.metrics();
    let backends = state.engine.backends();
    let all_metrics = metrics_store.all_metrics().await;

    let mut backend_metrics = Vec::new();

    for backend in backends {
        let name = backend.name();
        let status = health_tracker.get_status(name).await;

        let (p50_ms, p99_ms, cost_per_1k_tokens, total_requests, error_rate) =
            if let Some(record) = all_metrics.get(name) {
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
                    cost_per_1k,
                    record.total_requests,
                    error_rate,
                )
            } else {
                (None, None, 0.0, 0u64, 0.0f64)
            };

        backend_metrics.push(BackendMetrics {
            backend: name.to_string(),
            status,
            p50_ms,
            p99_ms,
            cost_per_1k_tokens,
            total_requests,
            error_rate,
        });
    }

    MeshEvent::MetricsUpdate {
        backends: backend_metrics,
        timestamp: chrono::Utc::now(),
    }
}
