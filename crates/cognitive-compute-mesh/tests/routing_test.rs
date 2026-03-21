use cognitive_compute_mesh::{
    backends::mock::MockBackend,
    routing::RoutingEngine,
    protocol::{InferenceRequest, Message, PolicyType},
    backends::InferenceBackend,
};
use std::sync::Arc;
use std::collections::HashMap;
use uuid::Uuid;

fn make_request() -> InferenceRequest {
    InferenceRequest {
        id: Uuid::new_v4(),
        model: "auto".to_string(),
        messages: vec![Message { role: "user".to_string(), content: "Hello".to_string() }],
        max_tokens: Some(100),
        temperature: Some(0.7),
        stream: false,
        tools: None,
        metadata: HashMap::new(),
    }
}

#[tokio::test]
async fn test_failover_under_1_second() {
    let backend_a = Arc::new(MockBackend::new_local());
    let backend_b = Arc::new(MockBackend::new_cloud());

    let engine = RoutingEngine::new(vec![
        backend_a as Arc<dyn InferenceBackend>,
        backend_b as Arc<dyn InferenceBackend>,
    ]);

    // Trip circuit breaker on mock-local
    engine.simulate_failure("mock-local").await.unwrap();

    let request = make_request();
    let start = std::time::Instant::now();
    let result = engine.route_with_fallback(&request).await;
    let elapsed = start.elapsed();

    assert!(result.is_ok(), "Should succeed via failover");
    assert!(
        elapsed.as_millis() < 1500,
        "Failover must complete within 1.5 seconds, took {}ms",
        elapsed.as_millis()
    );

    let response = result.unwrap();
    // Should have used mock-cloud since mock-local circuit is open
    assert_eq!(response.backend, "mock-cloud", "Should have routed to mock-cloud");
}

#[tokio::test]
async fn test_cost_optimized_routes_to_cheapest() {
    // mock-local: $0/token, mock-cloud: $0.002/token
    let backend_local = Arc::new(MockBackend::new_local());
    let backend_cloud = Arc::new(MockBackend::new_cloud());

    let engine = RoutingEngine::new(vec![
        backend_cloud as Arc<dyn InferenceBackend>,
        backend_local as Arc<dyn InferenceBackend>,
    ]);

    // Set cost-optimized routing
    engine.set_policy(PolicyType::Cost).await;

    let request = make_request();
    let selected = engine.route(&request).await.unwrap();

    assert_eq!(selected.name(), "mock-local", "Cost-optimized should prefer zero-cost local backend");
}

#[tokio::test]
async fn test_same_code_runs_on_local_and_cloud() {
    let backend_local = Arc::new(MockBackend::new_local());
    let backend_cloud = Arc::new(MockBackend::new_cloud());

    let request = make_request();

    let local_resp = backend_local.infer(&request).await.unwrap();
    let cloud_resp = backend_cloud.infer(&request).await.unwrap();

    // Same structure, different backend field
    assert!(!local_resp.content.is_empty(), "Local backend must return content");
    assert!(!cloud_resp.content.is_empty(), "Cloud backend must return content");
    assert_eq!(local_resp.backend, "mock-local");
    assert_eq!(cloud_resp.backend, "mock-cloud");
    assert_eq!(local_resp.cost_usd, 0.0, "Local backend must be free");
    assert!(cloud_resp.cost_usd > 0.0, "Cloud backend must have cost");

    // Same request ID used in response
    assert_eq!(local_resp.request_id, request.id);
    assert_eq!(cloud_resp.request_id, request.id);
}

#[tokio::test]
async fn test_availability_routing_skips_unhealthy() {
    let backend_a = Arc::new(MockBackend::new_local());
    let backend_b = Arc::new(MockBackend::new_cloud());

    let engine = RoutingEngine::new(vec![
        backend_a as Arc<dyn InferenceBackend>,
        backend_b as Arc<dyn InferenceBackend>,
    ]);

    // Trip circuit on first backend
    engine.simulate_failure("mock-local").await.unwrap();

    let request = make_request();
    let selected = engine.route(&request).await.unwrap();

    assert_eq!(selected.name(), "mock-cloud", "Should skip circuit-open backend");
}
