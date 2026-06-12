from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from src.core.logging import get_logger

logger = get_logger(__name__)

# the only condition forms the sprint-3 engine evaluates; rules using
# richer predicates (dwell time, multi-entry windows) are loaded but
# inactive until the temporal-reasoning sprint implements them
_SIMPLE_COND = re.compile(r"^\s*(event_type|zone_type)\s*==\s*'([^']+)'\s*$")


@dataclass
class AlertRule:
    name: str
    description: str
    severity: str
    cooldown_sec: float
    conditions: list[tuple[str, str]]      # (field, expected-value)
    supported: bool = True
    _last_fired: dict[str, float] = field(default_factory=dict)  # zone -> monotonic ts


@dataclass
class TriggeredAlert:
    rule_name: str
    event_type: str
    severity: str
    camera_id: str
    zone_id: str | None
    person_id: str | None
    description: str
    payload: dict


class AlertEngine:
    """
    Evaluates configs/alerting/alert_rules.yaml against the live event
    stream. Stateless per event except for per-(rule, zone) cooldowns —
    one alert per cooldown window regardless of how many matching events
    fire inside it.
    """

    def __init__(self, rules_path: str | Path, zone_types: dict[str, str]) -> None:
        self.zone_types = zone_types
        self.rules: list[AlertRule] = []
        raw = yaml.safe_load(Path(rules_path).read_text()) or {}
        for r in raw.get("rules", []):
            conditions: list[tuple[str, str]] = []
            supported = True
            for cond in r.get("trigger_conditions", []):
                m = _SIMPLE_COND.match(cond)
                if m:
                    conditions.append((m.group(1), m.group(2)))
                else:
                    supported = False
            rule = AlertRule(
                name=r["name"],
                description=r.get("description", r["name"]),
                severity=r.get("severity", "MEDIUM"),
                cooldown_sec=float(r.get("cooldown_sec", 60)),
                conditions=conditions,
                supported=supported,
            )
            if not supported:
                logger.info("alert_rule_inactive", rule=rule.name,
                            reason="uses predicates beyond sprint-3 engine")
            self.rules.append(rule)
        active = sum(1 for r in self.rules if r.supported)
        logger.info("alert_engine_loaded", rules=len(self.rules), active=active)

    def evaluate(
        self,
        *,
        event_type: str,
        camera_id: str,
        zone_id: str | None,
        person_id: str | None = None,
        person_name: str = "unknown",
        identity_state: str = "unknown",
    ) -> list[TriggeredAlert]:
        zone_type = self.zone_types.get(zone_id or "", "general")
        now = time.monotonic()
        fired: list[TriggeredAlert] = []

        for rule in self.rules:
            if not rule.supported or not rule.conditions:
                continue
            values = {"event_type": event_type, "zone_type": zone_type}
            if not all(values.get(f) == v for f, v in rule.conditions):
                continue
            key = zone_id or "_"
            last = rule._last_fired.get(key, -1e12)
            if now - last < rule.cooldown_sec:
                logger.info("alert_suppressed_cooldown", rule=rule.name, zone=zone_id)
                continue
            rule._last_fired[key] = now
            fired.append(TriggeredAlert(
                rule_name=rule.name,
                event_type=event_type,
                severity=rule.severity,
                camera_id=camera_id,
                zone_id=zone_id,
                person_id=person_id,
                description=f"{rule.description} — {person_name} in {zone_id}",
                payload={
                    "rule": rule.name,
                    "person": person_name,
                    "identity_state": identity_state,
                    "zone_type": zone_type,
                },
            ))
        return fired
