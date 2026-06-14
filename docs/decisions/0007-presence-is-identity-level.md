# 0007 — Presence/occupancy is identity-level, not track-level

**Status:** Proposed · **Date:** 2026-06-14

## Context
Live multi-person testing exposed spurious events: a person who never left got
a second `PERSON_ENTERED` (no matching exit), and phantom `unknown` entries
appeared during crossing/edge events. Root cause is **track churn** — face-only
tracking drops and re-creates track IDs under occlusion, crossing, and
frame-edge dropouts. The presence monitor keys on `track_id`, so each new track
for the same physical person fires its own entry/exit.

## Decision
Model zone presence at the **resolved-identity level**, not the raw track level:
- A recognized identity is "present" once; additional track IDs that resolve to
  the same identity do **not** fire new ENTERED events.
- EXITED fires only when *all* tracks for that identity have been absent beyond
  the exit grace.
- Track IDs remain ephemeral evidence underneath; identity is the unit of
  presence (consistent with the track-level-identity model in run_live).

This pairs with **Sprint 6 body ReID**, which reduces churn at the source by
holding track IDs through occlusion/edge. Identity-level presence then absorbs
the residual churn for *recognized* people. Genuinely *unknown* churn (two
unknown strangers crossing) is only fully resolved once body ReID gives stable
IDs — accepted limitation until then.

## Alternatives rejected
- **Keep track-level presence** — faithfully records churn as duplicate/phantom
  events; unusable for multi-person occupancy and alerting.
- **Hacky per-track suppression** (raise grace / drop short tracks) — delays
  legitimate entries and still mis-handles crossings; fragile.

## Consequences
A redesign of `ZonePresenceMonitor` to track per-identity state with track sets.
Deferred to implement alongside Sprint 6 body ReID (synergistic). Until then,
Sprint 3 foundation is validated with single-person flows; multi-person churn is
documented as an accepted limitation (see live_testing/sprint3_live_tests.md).
