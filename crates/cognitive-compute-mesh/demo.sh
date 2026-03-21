#!/bin/bash
set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo -e "${BLUE}╔═══════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   Cognitive Compute Mesh — MVP Demo       ║${NC}"
echo -e "${BLUE}║   GSoC 2024 Project Demo                  ║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════╝${NC}"

# Kill any existing server on port 8090
echo -e "\n${YELLOW}Cleaning up any existing server on port 8090...${NC}"
lsof -ti:8090 | xargs kill -9 2>/dev/null || true
sleep 1

# Start server in background
echo -e "\n${GREEN}[1/7] Starting Cognitive Compute Mesh server...${NC}"
cd "$WORKSPACE_ROOT"
RUST_LOG=cognitive_compute_mesh=info cargo run --bin cognitive-compute-mesh 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

# Wait for server to be ready
echo -e "Waiting for server to start..."
for i in $(seq 1 30); do
    if curl -s http://localhost:8090/health > /dev/null 2>&1; then
        echo -e "${GREEN}Server is ready!${NC}"
        break
    fi
    if [ $i -eq 30 ]; then
        echo -e "${RED}Server failed to start within 30 seconds${NC}"
        kill $SERVER_PID 2>/dev/null
        exit 1
    fi
    sleep 1
done

# Check backends
echo -e "\n${GREEN}[2/7] Checking registered backends...${NC}"
curl -s http://localhost:8090/v1/backends | python3 -m json.tool 2>/dev/null || \
  curl -s http://localhost:8090/v1/backends

# Demo 1: Route to local backend (zero cost)
echo -e "\n${GREEN}[3/7] Demo: Routing to local backend (zero cost)...${NC}"
echo -e "${YELLOW}Sending: 'What is the Cognitive Compute Mesh?'${NC}"
curl -s -X POST http://localhost:8090/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "auto",
    "messages": [{"role": "user", "content": "What is the Cognitive Compute Mesh?"}],
    "stream": false
  }' | python3 -m json.tool 2>/dev/null || \
  curl -s -X POST http://localhost:8090/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"auto","messages":[{"role":"user","content":"What is the Cognitive Compute Mesh?"}]}'

# Demo 2: Switch to latency-optimized routing
echo -e "\n${GREEN}[4/7] Switching to latency-optimized routing...${NC}"
curl -s -X PUT http://localhost:8090/v1/routing/policy \
  -H "Content-Type: application/json" \
  -d '{"policy": "latency"}' | python3 -m json.tool 2>/dev/null || echo "{}"

echo -e "${YELLOW}Sending request with latency-optimized policy...${NC}"
curl -s -X POST http://localhost:8090/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"auto","messages":[{"role":"user","content":"Hello from latency-optimized routing!"}]}' \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'Backend: {d[\"backend\"]}, Latency: {d[\"latency_ms\"]}ms, Cost: \${d[\"cost_usd\"]:.6f}')" 2>/dev/null || echo "Response received"

# Demo 3: Failover
echo -e "\n${GREEN}[5/7] Demo: Automatic failover...${NC}"
echo -e "${YELLOW}Simulating failure on mock-local backend...${NC}"
curl -s -X POST http://localhost:8090/v1/backends/mock-local/simulate-failure | python3 -m json.tool 2>/dev/null || echo "{}"
sleep 1

echo -e "${YELLOW}Sending request (should auto-failover to mock-cloud)...${NC}"
FAILOVER_RESP=$(curl -s -X POST http://localhost:8090/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"auto","messages":[{"role":"user","content":"Hello"}]}')
echo "$FAILOVER_RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'Routed to: {d[\"backend\"]} (failover successful)')" 2>/dev/null || echo "$FAILOVER_RESP"

# Demo 4: RAG pipeline
echo -e "\n${GREEN}[6/7] Demo: RAG pipeline — ingest + hybrid retrieval...${NC}"
echo -e "${YELLOW}Ingesting 3 documents...${NC}"
INGEST_RESP=$(curl -s -X POST http://localhost:8090/v1/rag/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "documents": [
      {"id": "1", "content": "Cognitive Compute Mesh uses IRP to unify inference backends"},
      {"id": "2", "content": "The routing engine supports cost, latency, and availability policies"},
      {"id": "3", "content": "OminiX-MLX runs models locally on Apple Silicon with zero cost"}
    ]
  }')
echo "$INGEST_RESP" | python3 -m json.tool 2>/dev/null || echo "$INGEST_RESP"

echo -e "${YELLOW}Running hybrid retrieval query...${NC}"
curl -s -X POST http://localhost:8090/v1/rag/query \
  -H "Content-Type: application/json" \
  -d '{"query": "how does routing work", "top_k": 2}' \
  | python3 -c "
import sys, json
d = json.load(sys.stdin)
for r in d.get('results', []):
    print(f'  score={r[\"score\"]:.4f}: {r[\"content\"]}')
" 2>/dev/null || curl -s -X POST http://localhost:8090/v1/rag/query \
    -H "Content-Type: application/json" \
    -d '{"query":"how does routing work","top_k":2}'

# Demo 5: Metrics comparison
echo -e "\n${GREEN}[7/7] Backend performance comparison...${NC}"
curl -s http://localhost:8090/v1/metrics/compare | python3 -m json.tool 2>/dev/null || \
  curl -s http://localhost:8090/v1/metrics/compare

echo -e "\n${GREEN}╔═══════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║   Demo complete!                          ║${NC}"
echo -e "${GREEN}║   Open http://localhost:5173 for dashboard ║${NC}"
echo -e "${GREEN}║   (run: cd crates/mofa-observatory/dashboard && npm run dev)${NC}"
echo -e "${GREEN}╚═══════════════════════════════════════════╝${NC}"

# Cleanup
echo -e "\n${YELLOW}Stopping server (PID: $SERVER_PID)...${NC}"
kill $SERVER_PID 2>/dev/null || true
echo -e "${GREEN}Done.${NC}"
