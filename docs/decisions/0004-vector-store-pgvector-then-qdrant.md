# 0004 — pgvector now, Qdrant at scale

**Status:** Accepted · **Date:** 2026-06-14

## Context
Face/person embeddings power recognition and cross-camera ReID. The system
already runs PostgreSQL, and pgvector is in use for the face gallery.

## Decision
- **Now (MVP / home / SMB):** keep **pgvector**. One fewer system to operate,
  transactional consistency with the relational data, and it comfortably handles
  up to single-digit-millions of vectors — past our near-term needs.
- **At scale:** migrate the embedding workload to **Qdrant** when pgvector's
  recall/latency under heavy filtered search degrades (tens of millions of
  vectors, high QPS). Qdrant is the lightest dedicated self-hosted vector DB to
  operate (Rust, strong filtered search).

## Alternatives rejected
- **Milvus** — billion-scale, multi-component ops (etcd etc.); overkill until
  very large.
- **Weaviate** — capable but heavier to self-host than Qdrant.
- **Chroma** — prototype-grade; pgvector already beats it for our case
  (transactional, no extra service).

## Consequences
Keep embedding access behind a small interface (a person-embedding store API) so
the pgvector→Qdrant swap is an implementation change, not an app rewrite. Scale
trigger: filtered ANN latency/recall on the live ReID path.
