# Architecture Decision Records (ADRs)

Short, dated records of significant technical decisions: the context, what we
chose, what we rejected, and the consequences. One decision per file so each
can be superseded individually without rewriting history.

Why: most of this project's key calls were made in conversation. Capturing the
*reasoning* (not just the code) is how the architecture stays coherent as it
grows and as contributors join.

Format per ADR: **Status · Context · Decision · Alternatives rejected ·
Consequences.** Status is one of: Proposed, Accepted, Superseded, Deprecated.

| ADR | Decision | Status |
|-----|----------|--------|
| [0001](0001-stack-architecture-review.md) | Full stack review & staged adoption | Accepted |
| [0002](0002-object-storage-minio.md) | MinIO for clips/snapshots/evidence | Accepted |
| [0003](0003-message-bus-redpanda-over-kafka.md) | Redpanda over Kafka (none at MVP) | Accepted |
| [0004](0004-vector-store-pgvector-then-qdrant.md) | pgvector now → Qdrant at scale | Accepted |
| [0005](0005-memory-graphiti-on-neo4j.md) | Graphiti temporal-KG memory | Proposed |
| [0006](0006-agent-frameworks-pydanticai-langgraph.md) | PydanticAI + LangGraph only | Proposed |

Guiding principles (see [0001](0001-stack-architecture-review.md)): self-hosted,
privacy-first, **ruthless about complexity** — add infrastructure only when
scale forces it, prefer battle-tested self-hostable components, keep stable
interfaces with swappable implementations.
