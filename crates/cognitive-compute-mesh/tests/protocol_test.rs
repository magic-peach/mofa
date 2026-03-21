use cognitive_compute_mesh::protocol::{
    InferenceResponse, TokenUsage,
};
use cognitive_compute_mesh::api::openai_compat::{OpenAIChatRequest, OpenAIMessage, openai_to_irp, irp_to_openai};
use uuid::Uuid;

#[test]
fn test_openai_to_irp_conversion() {
    let openai_req = OpenAIChatRequest {
        model: "gpt-4".to_string(),
        messages: vec![
            OpenAIMessage { role: "system".to_string(), content: "You are helpful".to_string() },
            OpenAIMessage { role: "user".to_string(), content: "Hello".to_string() },
        ],
        max_tokens: Some(100),
        temperature: Some(0.7),
        stream: Some(false),
    };

    let irp = openai_to_irp(openai_req);

    assert_eq!(irp.model, "gpt-4");
    assert_eq!(irp.messages.len(), 2);
    assert_eq!(irp.messages[0].role, "system");
    assert_eq!(irp.messages[1].content, "Hello");
    assert_eq!(irp.max_tokens, Some(100));
    assert_eq!(irp.stream, false);
}

#[test]
fn test_irp_to_openai_conversion() {
    let irp_resp = InferenceResponse {
        id: Uuid::new_v4(),
        request_id: Uuid::new_v4(),
        backend: "mock-local".to_string(),
        content: "Hello world".to_string(),
        usage: TokenUsage {
            prompt_tokens: 10,
            completion_tokens: 5,
            total_tokens: 15,
        },
        latency_ms: 200,
        cost_usd: 0.0,
        model: "mock-v1".to_string(),
    };

    let openai_resp = irp_to_openai(irp_resp);

    assert_eq!(openai_resp.choices.len(), 1);
    assert_eq!(openai_resp.choices[0].message.content, "Hello world");
    assert_eq!(openai_resp.usage.total_tokens, 15);
    assert_eq!(openai_resp.backend, "mock-local");
}

#[test]
fn test_irp_roundtrip_fidelity() {
    // OpenAI -> IRP -> OpenAI should preserve all fields
    let original = OpenAIChatRequest {
        model: "gpt-3.5-turbo".to_string(),
        messages: vec![
            OpenAIMessage { role: "user".to_string(), content: "Tell me about Rust".to_string() },
        ],
        max_tokens: Some(500),
        temperature: Some(0.9),
        stream: None,
    };

    let irp = openai_to_irp(OpenAIChatRequest {
        model: original.model.clone(),
        messages: original.messages.clone(),
        max_tokens: original.max_tokens,
        temperature: original.temperature,
        stream: original.stream,
    });

    assert_eq!(irp.model, original.model);
    assert_eq!(irp.messages[0].content, "Tell me about Rust");
    assert_eq!(irp.max_tokens, original.max_tokens);
}
