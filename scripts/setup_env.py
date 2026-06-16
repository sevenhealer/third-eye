#!/usr/bin/env python3
"""
Generate or top up .env with strong random secrets — idempotent, stdlib-only.

- No .env yet  : create from .env.example with every secret randomized
                 (DB URLs get the generated postgres password substituted in).
- .env exists  : add ONLY the keys missing from it (e.g. newly added S3_*),
                 generating secrets for secret keys. Existing values are never
                 touched — your running postgres/redis passwords stay intact.

Run:  python3 scripts/setup_env.py   (or: make setup)
"""
import base64
import os
import re
import secrets
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXAMPLE = ROOT / ".env.example"
ENV = ROOT / ".env"
LINE = re.compile(r"^([A-Z0-9_]+)=(.*)$")
PG_PLACEHOLDER = "change-me-strong-password"


def gen(key: str) -> str:
    if key == "JWT_SECRET_KEY":            # config requires >= 64 chars
        return secrets.token_urlsafe(64)
    if key == "APP_SECRET_KEY":            # >= 32 chars
        return secrets.token_urlsafe(36)
    if key == "EMBEDDING_ENCRYPTION_KEY":  # 32-byte base64 Fernet key
        return base64.urlsafe_b64encode(os.urandom(32)).decode()
    return secrets.token_urlsafe(24)


def split_comment(rest: str) -> tuple[str, str]:
    """Return (value, '  # inline comment') preserving any trailing comment."""
    if "#" in rest:
        i = rest.index("#")
        return rest[:i].rstrip(), "  " + rest[i:]
    return rest.strip(), ""


def parse_keys(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in text.splitlines():
        m = LINE.match(line.strip())
        if m:
            out[m.group(1)] = split_comment(m.group(2))[0]
    return out


def main() -> None:
    if not EXAMPLE.exists():
        print("ERROR: .env.example not found")
        sys.exit(1)

    if not ENV.exists():
        text = EXAMPLE.read_text()
        pg = gen("POSTGRES_PASSWORD")
        text = text.replace(PG_PLACEHOLDER, pg)   # also fixes DB URLs
        out = []
        for line in text.splitlines():
            m = LINE.match(line)
            if m and "change-me" in m.group(2):
                _, comment = split_comment(m.group(2))
                out.append(f"{m.group(1)}={gen(m.group(1))}{comment}")
            else:
                out.append(line)
        ENV.write_text("\n".join(out) + "\n")
        print(f"Created {ENV} with strong random secrets. Now: docker compose up -d")
        return

    existing = parse_keys(ENV.read_text())
    pg = existing.get("POSTGRES_PASSWORD", "")
    added, append_lines = [], []
    for line in EXAMPLE.read_text().splitlines():
        m = LINE.match(line)
        if not m or m.group(1) in existing:
            continue
        key, val, comment = m.group(1), *split_comment(m.group(2))
        if "change-me" in val:
            val = pg if (val == PG_PLACEHOLDER and pg) else gen(key)
        elif PG_PLACEHOLDER in val and pg:
            val = val.replace(PG_PLACEHOLDER, pg)
        append_lines.append(f"{key}={val}{comment}")
        added.append(key)

    if added:
        with ENV.open("a") as f:
            f.write("\n# ── added by setup_env.py ──\n")
            f.write("\n".join(append_lines) + "\n")
        print(f"Updated {ENV} — added {len(added)} missing key(s): {', '.join(added)}")
    else:
        print(f"{ENV} already complete — nothing to add")


if __name__ == "__main__":
    main()
