# 0003 — Redpanda over Kafka (and no bus at MVP)

**Status:** Accepted (planned) · **Date:** 2026-06-14

## Context
The stack lists Kafka as the event bus. For a self-hosted system — especially
home/SMB on one or two boxes — Kafka's JVM footprint and coordination overhead
(ZooKeeper/KRaft) are heavy. At single-box MVP scale there is no producer/
consumer decoupling that needs a bus at all.

## Decision
- **MVP / home:** no message bus. Detect→event→alert runs in-process; Redis
  pub/sub carries alerts. (Current state — keep it.)
- **SMB and up:** when per-camera pipelines must decouple, use **Redpanda** —
  Kafka-API compatible, single binary, no JVM/ZooKeeper, far lighter to
  self-host. Drop-in for Kafka clients (`aiokafka` unchanged).
- **Kafka:** removed from the default plan; adopt only if an enterprise org
  mandates it.

## Alternatives rejected
- **Kafka** — operational weight unjustified at our scale; Redpanda is API-
  compatible and lighter.
- **Apache Flink for processing** — heavy JVM cluster CEP; only at large
  multi-camera scale. Use in-process logic now, RisingWave (streaming SQL) at
  production for temporal analytics.

## Consequences
Lower ops burden; a clean scale trigger (decoupling multiple camera pipelines /
services) for introducing Redpanda. Event-producing code should target the
Kafka API so the bus can appear later without app changes.
