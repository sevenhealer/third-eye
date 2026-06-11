"""sqlalchemy text() does NOT recognise a bind param immediately followed by
a Postgres ``::`` cast — ``:param::jsonb`` reaches the server literally and
fails with "syntax error at or near :".  Use ``CAST(:param AS jsonb)``.
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_BIND_THEN_CAST = re.compile(r":[A-Za-z_]\w*::")


def test_no_bindparam_adjacent_casts():
    offenders = []
    for path in (ROOT / "src").rglob("*.py"):
        for lineno, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if _BIND_THEN_CAST.search(line):
                offenders.append(f"{path.relative_to(ROOT)}:{lineno}: {line.strip()}")
    assert not offenders, (
        "':param::type' is never bound by sqlalchemy text() — "
        "use CAST(:param AS type):\n" + "\n".join(offenders)
    )
