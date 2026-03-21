use axum::{
    extract::{State, WebSocketUpgrade},
    response::Response,
};
use std::sync::Arc;
use super::routes::AppState;

pub async fn ws_handler(
    ws: WebSocketUpgrade,
    State(_state): State<Arc<AppState>>,
) -> Response {
    ws.on_upgrade(|_socket| async {
        // WebSocket handling will be implemented in Task 6
    })
}
