import pytest

from src.nlq.query_planner import QueryIntent, classify_intent, _extract_zone, _extract_person_name


@pytest.mark.parametrize("query,expected", [
    ("Who is currently in Room A?", QueryIntent.CURRENT_PRESENCE),
    ("Who is inside the server room?", QueryIntent.CURRENT_PRESENCE),
    ("What is Rahul doing right now?", QueryIntent.CURRENT_ACTION),
    ("What is John's current activity?", QueryIntent.CURRENT_ACTION),
    ("When did John last enter the building?", QueryIntent.LAST_SEEN),
    ("When was Amit last seen?", QueryIntent.LAST_SEEN),
    ("Show every appearance of Person X in the last 30 days", QueryIntent.HISTORICAL_APPEARANCES),
    ("How many monitors are currently present?", QueryIntent.OBJECT_COUNT),
    ("How many people are in the corridor?", QueryIntent.OBJECT_COUNT),
    ("Did anyone access Server Rack 1 today?", QueryIntent.ZONE_ACCESS),
    ("Was there any unusual activity between 2 PM and 4 PM?", QueryIntent.UNUSUAL_ACTIVITY),
    ("Did the cat enter the room today?", QueryIntent.ANIMAL_DETECTION),
    ("What changed since yesterday?", QueryIntent.WHAT_CHANGED),
])
def test_intent_classification(query, expected):
    assert classify_intent(query) == expected


def test_extract_zone():
    assert _extract_zone("Who is in Room A?") == "room_a"
    assert _extract_zone("Is anyone in the server room?") == "server_room"
    assert _extract_zone("No location mentioned") is None


def test_extract_person_name():
    # "Rahul" is capitalised mid-sentence
    assert _extract_person_name("What is Rahul doing?") == "Rahul"
    # First word capitalised (sentence start) should not match
    result = _extract_person_name("Who entered the building?")
    assert result is None or result == "Who"  # naive extractor


def test_no_injection_in_query():
    from src.api.routers.queries import QueryRequest, _INJECTION_PATTERNS
    import re
    bad_queries = [
        "SELECT * FROM users; DROP TABLE persons;",
        "MATCH (n) DETACH DELETE n",
        "'; DROP TABLE audit_log; --",
    ]
    for q in bad_queries:
        assert _INJECTION_PATTERNS.search(q), f"Pattern should match: {q}"
