#!/usr/bin/env python3
"""
CLI tool to manually enroll an identity from the command line.
Requires an active admin session (JWT token in THIRDEYE_TOKEN env var).

Usage:
  python scripts/enroll_identity.py --name "John Doe" --role "engineer"
"""
import argparse
import asyncio
import json
import os
import sys

import httpx


async def main() -> None:
    parser = argparse.ArgumentParser(description="Enroll a new identity")
    parser.add_argument("--name", required=True, help="Person's display name")
    parser.add_argument("--role", default=None, help="Person's role (optional)")
    parser.add_argument("--notes", default=None, help="Notes (optional)")
    parser.add_argument(
        "--api-url",
        default=os.getenv("THIRDEYE_API_URL", "http://localhost:8000"),
    )
    args = parser.parse_args()

    token = os.getenv("THIRDEYE_TOKEN")
    if not token:
        print("ERROR: Set THIRDEYE_TOKEN env var to an admin JWT token.")
        sys.exit(1)

    payload = {"display_name": args.name, "role": args.role, "notes": args.notes}

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{args.api_url}/api/v1/identities",
            json=payload,
            headers={"Authorization": f"Bearer {token}"},
            timeout=30,
        )

    if resp.status_code == 201:
        data = resp.json()
        print(f"✓ Enrolled: {data['display_name']} (ID: {data['person_id']})")
    else:
        print(f"✗ Enrollment failed ({resp.status_code}): {resp.text}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
