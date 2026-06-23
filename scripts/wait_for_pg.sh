#!/usr/bin/env bash
# Block until the Postgres container accepts connections, so `make bootstrap`
# can run migrations against a freshly-started DB instead of racing its init.
set -u
echo -n "Waiting for Postgres "
for _ in $(seq 1 60); do
  if docker compose exec -T postgres pg_isready -q >/dev/null 2>&1; then
    echo " ready"
    exit 0
  fi
  echo -n "."
  sleep 2
done
echo " timed out after 120s"
exit 1
