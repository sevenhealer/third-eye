# 0006 — PydanticAI + LangGraph for the agent layer

**Status:** Proposed · **Date:** 2026-06-14

## Context
The roadmap includes autonomous investigation, incident-report, and root-cause
agents. A forensic system needs **controllable, auditable, deterministic** agent
behavior — not opaque autonomous role-play.

## Decision
Standardize on two complementary tools, both self-hostable libraries:
- **PydanticAI** as the default — type-safe, lightweight structured tool-calling
  that fits the existing Pydantic/FastAPI stack.
- **LangGraph** for complex, stateful, multi-step investigations needing
  explicit control flow, checkpoints, and human-in-the-loop gates.

Agents call detection/search/graph/memory through a stable **MCP-style tool
layer**, so the agent framework and the tools evolve independently.

## Alternatives rejected
- **CrewAI / AutoGen** — role-playing multi-agent abstractions; less control,
  heavier, churny — wrong for auditable forensics.
- **Semantic Kernel** — .NET heritage, no Python-side advantage.
- **OpenAI Agents SDK** — oriented to OpenAI hosted models/tools; conflicts with
  local-first/privacy.
- **Google ADK** — too new to bet on.

## Consequences
Two libraries, clear division of labor, no framework lock-in beyond a thin tool
interface. Proposed — build last, on a validated data/identity foundation;
observability via Langfuse ([0001]).
