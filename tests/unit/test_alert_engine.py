import textwrap

from src.alerts.engine import AlertEngine

RULES_YAML = textwrap.dedent("""
    rules:
      - name: person_entered_restricted
        description: "Person entered a restricted zone"
        trigger_conditions:
          - "event_type == 'PERSON_ENTERED'"
          - "zone_type == 'restricted'"
        severity: HIGH
        cooldown_sec: 30

      - name: fancy_rule
        description: "Uses predicates the sprint-3 engine doesn't support"
        trigger_conditions:
          - "persons_in_zone('x') > 1"
        severity: HIGH
        cooldown_sec: 5
""")

ZONES = {"doorway": "restricted", "bedroom": "general"}


def _engine(tmp_path, yaml_text=RULES_YAML):
    p = tmp_path / "rules.yaml"
    p.write_text(yaml_text)
    return AlertEngine(p, ZONES)


def test_rule_fires_in_restricted_zone(tmp_path):
    eng = _engine(tmp_path)
    fired = eng.evaluate(event_type="PERSON_ENTERED", camera_id="cam0",
                         zone_id="doorway", person_name="Rohan")
    assert len(fired) == 1
    assert fired[0].severity == "HIGH"
    assert "Rohan" in fired[0].description


def test_rule_silent_in_general_zone(tmp_path):
    eng = _engine(tmp_path)
    assert eng.evaluate(event_type="PERSON_ENTERED", camera_id="cam0",
                        zone_id="bedroom") == []


def test_cooldown_suppresses_repeat(tmp_path):
    eng = _engine(tmp_path)
    assert eng.evaluate(event_type="PERSON_ENTERED", camera_id="cam0",
                        zone_id="doorway")
    assert eng.evaluate(event_type="PERSON_ENTERED", camera_id="cam0",
                        zone_id="doorway") == []


def test_cooldown_is_per_zone(tmp_path):
    eng = _engine(tmp_path)
    eng.zone_types["lab"] = "restricted"
    assert eng.evaluate(event_type="PERSON_ENTERED", camera_id="cam0",
                        zone_id="doorway")
    assert eng.evaluate(event_type="PERSON_ENTERED", camera_id="cam0",
                        zone_id="lab")


def test_unsupported_rule_inactive_not_crashing(tmp_path):
    eng = _engine(tmp_path)
    fancy = [r for r in eng.rules if r.name == "fancy_rule"][0]
    assert fancy.supported is False
    # an event that would match nothing supported → no fire, no error
    assert eng.evaluate(event_type="WHATEVER", camera_id="cam0",
                        zone_id="doorway") == []


def test_real_rules_file_parses():
    eng = AlertEngine("configs/alerting/alert_rules.yaml", ZONES)
    active = [r.name for r in eng.rules if r.supported]
    assert "person_entered_restricted" in active
    assert "unknown_person_restricted" in active
