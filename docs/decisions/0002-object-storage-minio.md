# 0002 — MinIO for clips, snapshots and evidence

**Status:** Accepted (implemented) · **Date:** 2026-06-14

## Context
A surveillance/forensic platform must persist visual artifacts — alert
snapshots, event clips, face/person crops, chain-of-custody evidence — yet the
stack had **no object storage at all**. The relational/vector/graph stores hold
metadata and embeddings, not media. This was the single biggest gap.

## Decision
Add **MinIO** (self-hosted, S3-compatible) as the object store. The application
talks to it through a thin boto3 wrapper (`src/storage/object_store.py`) so the
exact same code runs against MinIO on-prem or AWS S3 later — self-hostable now,
cloud-portable if ever needed. Keys are date-partitioned
(`<kind>/YYYY/MM/DD/<camera>/<ts>.<ext>`) for listable prefixes and simple
lifecycle/retention.

## Alternatives rejected
- **Local filesystem only** — no horizontal scale, no presigned sharing, no
  clean lifecycle/retention, awkward multi-node.
- **Store blobs in Postgres** — bloats the DB, terrible for large media.
- **Cloud S3 directly** — violates local-first/privacy; MinIO gives the same API
  on-prem, with S3 as a drop-in later.

## Consequences
- New compose service `minio` (API :9000, console :9001), `minio_data` volume,
  `S3_*` settings, `boto3` dependency.
- `ObjectStore` is disabled-safe: empty `s3_endpoint_url` makes it a guarded
  no-op so MVP/dev runs without it.
- Next: wire snapshot-on-alert and clip-on-event to write through `ObjectStore`,
  and store presigned URLs in alert payloads.
