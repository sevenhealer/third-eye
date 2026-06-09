#!/usr/bin/env python3
"""
Verify the hash-chain integrity of the audit log.
Run via: make audit-verify  OR  python scripts/audit_log_verify.py
"""
import asyncio
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.core.database import get_db
from src.core.audit_log import verify_audit_chain


async def main() -> None:
    async for db in get_db():
        is_valid, first_invalid = await verify_audit_chain(db)
        if is_valid:
            print("✓ Audit log hash chain is INTACT.")
            sys.exit(0)
        else:
            print(f"✗ Audit log chain BROKEN at log_id={first_invalid}.")
            print("  This may indicate tampering. Preserve the database and investigate.")
            sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
