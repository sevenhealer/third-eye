# 0001 — Stack architecture review & staged adoption

**Status:** Accepted · **Date:** 2026-06-14

## Context
A full evaluation of the Third-Eye stack against modern alternatives across
inference serving, agents, memory, vector DBs, event processing, orchestration,
observability, knowledge graph, storage, and multimodal models. Reality anchor:
solo developer, one PCIe-x1 GPU box, single-camera testing, local-first /
privacy-first, most advanced layers still unbuilt.

## Decision
The current stack is ~80% right for a self-hosted privacy-first system. The risk
is **adding** technology, not missing it. Adopt in stages tied to scale, not all
at once. Guiding principles: self-hosted, privacy-first, ruthless about
complexity, stable interfaces + swappable implementations.

### Verdicts (condensed)
- **Inference:** Keep ONNX Runtime. Add TensorRT (now-ish, free 2–3×), Triton
  (production multi-camera), vLLM/SGLang (when LLM serving is real). Keep Ollama
  for home/MVP. Remove Ray Serve / KServe unless multi-node K8s enterprise.
- **Agents:** PydanticAI (default) + LangGraph (complex investigations). Remove
  CrewAI, AutoGen, Semantic Kernel, OpenAI Agents SDK (privacy), Google ADK.
- **Memory:** Redis (short-term) + Graphiti-on-Neo4j (long-term temporal KG) +
  pgvector (embeddings). Remove Mem0 / Zep / Letta / LangMem (redundant layers).
- **Vector DB:** pgvector now → Qdrant at scale. Remove Milvus / Weaviate /
  Chroma.
- **Event bus:** none at MVP (Redis pub/sub); Redpanda replaces Kafka at SMB+;
  RisingWave for streaming temporal analytics at production. Remove Flink unless
  enterprise.
- **Orchestration:** none at MVP; Temporal (durable investigation workflows) +
  Prefect *or* Dagster (ML/batch) at production. Remove Airflow.
- **Observability:** Keep Prometheus+Grafana. Add Loki (early), OTel+Tempo
  (production), Langfuse (when agents live). Skip Phoenix.
- **Knowledge graph:** Neo4j default; FalkorDB if a leaner self-hosted graph is
  needed. Remove ArangoDB / Memgraph.
- **Storage:** Keep Postgres+pgvector. **Add MinIO** (clips/evidence — biggest
  gap, see [0002]). TimescaleDB only when event volume justifies; ClickHouse at
  enterprise for forensic analytics.
- **Multimodal:** Replace CLIP with SigLIP2. Add Florence-2 (open-vocab,
  MIT) and Qwen2.5-VL (scene understanding / NL over footage). ColPali only if
  document-heavy.
- **Identity/reasoning (build, not buy):** cross-camera ReID, Global Identity
  Service, person embedding store, trajectory analytics; event fusion + temporal
  reasoning layers; investigation/incident/root-cause agents last, on a validated
  foundation, exposing detection/search/graph as MCP-style tools.

## Near-term order
1. Finish Sprint 3 live testing. 2. MinIO (footage storage). 3. TensorRT.
4. Decide TimescaleDB vs plain Postgres. Then, only as cameras/users grow:
Redpanda → Triton → Qdrant → Graphiti → agent stack.

## Consequences
Smaller surface area now; clear scale triggers for each addition. Deployment
tiers (home / SMB / enterprise) and MVP/production/future-proof topologies are
defined in the architecture review and tracked here as they're adopted.
