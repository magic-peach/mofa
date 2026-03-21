# Cognitive Compute Mesh

**A unified inference routing layer for distributed AI computation — GSoC 2024**

The Cognitive Compute Mesh is an intelligent middleware that sits between your application and multiple AI inference backends. It routes requests to the optimal backend based on cost, latency, or availability — with automatic failover, a hybrid RAG pipeline, and a real-time WebSocket dashboard.

---

## Quick Start

```bash
# 1. Start the server
cargo run --bin cognitive-compute-mesh

# 2. Open the dashboard (in a separate terminal)
cd crates/mofa-observatory/dashboard && npm run dev
```

Server: http://localhost:8090
Dashboard: http://localhost:5173

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Client Application                        │
│              (OpenAI-compatible REST API)                    │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                  Cognitive Compute Mesh                      │
│                                                              │
│  ┌─────────────────┐    ┌──────────────────────────────┐    │
│  │  IRP Protocol   │    │       Routing Engine         │    │
│  │  (Inference     │───▶│  • cost policy               │    │
│  │   Request       │    │  • latency policy            │    │
│  │   Protocol)     │    │  • availability policy       │    │
│  └─────────────────┘    └──────────────┬───────────────┘    │
│                                        │                     │
│  ┌─────────────────────────────────────▼──────────────────┐ │
│  │                   Backend Pool                          │ │
│  │  mock-local │ mock-cloud │ openai │ anthropic │ local  │ │
│  └─────────────────────────────────────────────────────────┘ │
│                                                              │
│  ┌──────────────────┐   ┌──────────────────────────────┐   │
│  │   RAG Pipeline   │   │    WebSocket Dashboard        │   │
│  │  • MockEmbeddings│   │    /ws/events                 │   │
│  │  • BM25 scoring  │   │    (real-time metrics)        │   │
│  │  • Hybrid fusion │   └──────────────────────────────┘   │
│  │  • HNSW vectors  │                                       │
│  └──────────────────┘                                       │
└─────────────────────────────────────────────────────────────┘
```

### Key Components

| Component | Description |
|-----------|-------------|
| **IRP Protocol** | Inference Request Protocol — a unified schema that normalizes requests/responses across all backends |
| **RoutingEngine** | Selects the optimal backend per request using pluggable policies; includes circuit breaker and health tracking |
| **Backends** | Pluggable adapters: mock-local, mock-cloud, OpenAI, Anthropic, and OminiX-MLX (local Apple Silicon) |
| **RAG Pipeline** | Hybrid retrieval combining dense vector search (128-dim embeddings) with BM25 sparse scoring |
| **WebSocket Dashboard** | Streams live `MeshEvent` JSON to the React dashboard at `/ws/events` |

---

## API Routes

| Method | Route | Description |
|--------|-------|-------------|
| `GET` | `/health` | Health check — returns `{"status":"ok"}` |
| `POST` | `/v1/chat/completions` | OpenAI-compatible chat completions endpoint |
| `POST` | `/v1/infer` | Raw IRP inference request |
| `GET` | `/v1/backends` | List all registered backends with health status |
| `GET` | `/v1/backends/:name/health` | Health + circuit breaker state for a specific backend |
| `POST` | `/v1/backends/:name/simulate-failure` | Inject a failure for demo/testing |
| `GET` | `/v1/metrics` | Per-backend latency, cost, and error metrics |
| `GET` | `/v1/metrics/compare` | Side-by-side backend performance comparison |
| `GET` | `/v1/routing/policy` | Get the current routing policy |
| `PUT` | `/v1/routing/policy` | Set routing policy: `cost`, `latency`, or `availability` |
| `POST` | `/v1/routing/simulate` | Dry-run routing — see which backend would be selected |
| `POST` | `/v1/rag/ingest` | Ingest documents into the hybrid retrieval store |
| `POST` | `/v1/rag/query` | Query ingested documents with hybrid scoring |
| `GET` | `/ws/events` | WebSocket stream of real-time mesh events |

---

## Running the Demo

The demo script runs all features top-to-bottom without any manual steps:

```bash
cd crates/cognitive-compute-mesh
./demo.sh
```

The demo walks through:
1. Server startup and readiness check
2. Backend discovery
3. Chat completion routed to mock-local (zero cost)
4. Switching to latency-optimized routing policy
5. Automatic failover after injecting a backend failure
6. RAG document ingestion (3 docs) + hybrid retrieval query
7. Backend performance comparison table

---

## Backend Comparison

| Backend | Type | Cost | Latency (p50) | Notes |
|---------|------|------|---------------|-------|
| `mock-local` | Mock | $0.000000 | ~5–50 ms | Simulated local inference, zero cost |
| `mock-cloud` | Mock | ~$0.000002/token | ~20–200 ms | Simulated cloud inference |
| `openai` | OpenAI API | per model pricing | variable | Requires `OPENAI_API_KEY` |
| `anthropic` | Anthropic API | per model pricing | variable | Requires `ANTHROPIC_API_KEY` |
| `local` | OminiX-MLX | $0.000000 | hardware-dependent | Apple Silicon local inference |

The routing engine automatically selects the best backend based on the active policy. With the default `cost` policy, `mock-local` (and `local` on Apple Silicon) are always preferred.

---

## Features

- **OpenAI-compatible API** — drop-in replacement for `POST /v1/chat/completions`
- **Three routing policies** — optimize for cost, latency, or availability
- **Automatic failover** — circuit breaker prevents requests to unhealthy backends
- **Hybrid RAG** — dense embedding search fused with BM25 keyword scoring
- **In-memory vector store** — approximate nearest-neighbor search (HNSW-style)
- **Real-time dashboard** — WebSocket event stream with React frontend
- **Failure injection** — `simulate-failure` endpoint for resilience testing
- **Per-backend metrics** — p50/p99 latency, cost tracking, error rates
- **Zero-dependency mocks** — works fully offline with `mock-local` and `mock-cloud`

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | No | Enables the OpenAI backend |
| `ANTHROPIC_API_KEY` | No | Enables the Anthropic backend |
| `RUST_LOG` | No | Log verbosity (e.g., `cognitive_compute_mesh=info`) |

The server starts and routes correctly even without any API keys — the mock backends handle all traffic in that case.
