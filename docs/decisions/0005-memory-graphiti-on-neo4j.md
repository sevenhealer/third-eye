# 0005 — Graphiti temporal-KG memory on Neo4j

**Status:** Proposed · **Date:** 2026-06-14

## Context
Goals include long-term memory, temporal reasoning, and knowledge-graph
reasoning. These are often built as three separate systems. Neo4j is already in
the stack.

## Decision
Use **Graphiti** (temporal knowledge-graph memory) on top of Neo4j as the
long-term memory substrate. It unifies long-term memory + temporal relationships
+ graph reasoning in one place, on infrastructure we already run. Pair with:
- **Redis** for short-term/working state (already in use),
- **pgvector/Qdrant** for raw embedding similarity (see [0004]).

For a leaner home/SMB footprint, Graphiti can run on **FalkorDB** instead of
Neo4j (see [0001]); choose by deployment size.

## Alternatives rejected
- **Mem0 / Zep** — Zep is built on Graphiti anyway; use Graphiti directly. Mem0
  is lighter but lacks the temporal-graph model our domain needs.
- **Letta** — an agent runtime, overlaps LangGraph ([0006]); not a memory store.
- **LangMem** — only if going all-in on LangChain.
- **Bespoke memory layer** — keep custom code to the hot path only; don't rebuild
  a temporal KG.

## Consequences
One memory substrate instead of three. Proposed (not yet built) — depends on the
event/identity foundation being live-verified first. Access memory behind a
small interface so Neo4j↔FalkorDB and library choices stay swappable.
