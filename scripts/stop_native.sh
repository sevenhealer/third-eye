#!/usr/bin/env bash
# Stop the natively-run API server and any camera worker subprocesses.
# Used by `make stop` and `make fresh`. Infra containers are left running.
set -u

PATTERNS=("uvicorn src.api.main:app" "scripts/run_live.py")

found=0
for pat in "${PATTERNS[@]}"; do
  pids=$(pgrep -f "$pat" || true)
  if [ -n "$pids" ]; then
    found=1
    echo "  SIGTERM -> $pids  ($pat)"
    # shellcheck disable=SC2086
    kill -TERM $pids 2>/dev/null || true
  fi
done

if [ "$found" = 0 ]; then
  echo "  nothing running."
  exit 0
fi

# Graceful first (uvicorn now runs with --timeout-graceful-shutdown, so its
# lifespan stops the camera children), then force any stragglers.
sleep 4
for pat in "${PATTERNS[@]}"; do
  pids=$(pgrep -f "$pat" || true)
  # shellcheck disable=SC2086
  [ -n "$pids" ] && kill -KILL $pids 2>/dev/null || true
done
echo "  stopped."
