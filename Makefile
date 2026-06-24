.PHONY: help up down logs shell test lint format check-env setup install install-gpu \
        infra reset serve stop frontend-build fresh run certs bootstrap sign-models

COMPOSE = docker compose
APP_SERVICE = api

# Native dev run: infra (Postgres/Redis/MinIO/Neo4j) runs in Docker, the API
# server runs from .venv on the host (so it sees the local GPU directly).
INFRA_SERVICES = postgres redis minio neo4j
# Bound graceful shutdown so an open WebSocket can't hang uvicorn on SIGTERM
# (which used to orphan camera workers across restarts).
GRACEFUL_SHUTDOWN_S = 10

help:
	@echo "Third-Eye Visual Intelligence Platform"
	@echo ""
	@echo "  ── one-command dev (app runs natively, infra in Docker) ──"
	@echo "  make run          FRESH CLONE → fully set up → running (one command)"
	@echo "  make fresh        CLEAN SLATE: set up, stop, wipe all data, then run"
	@echo "  make bootstrap    One-time setup only (venv, .env, deps, models, DB, UI)"
	@echo "  make serve        Run only the API server (foreground; Ctrl+C stops it)"
	@echo "  make reset        DESTRUCTIVE: wipe all data, reseed admin/admin + zones"
	@echo "  make stop         Stop the native API server + camera workers"
	@echo "  make infra        Start just the infra containers"
	@echo "  make certs        Generate a self-signed cert so serve uses HTTPS"
	@echo "                    (needed for the webcam Enroll page over the LAN)"
	@echo "  make sign-models  (Re-)sign model weights into a SHA-256 manifest"
	@echo ""
	@echo "  ── full containerized stack / ops ──"
	@echo "  make setup        Generate/top-up .env with strong random secrets"
	@echo "  make up           Start all services (containerized)"
	@echo "  make down         Stop all services"
	@echo "  make logs         Follow logs from all services"
	@echo "  make logs s=api   Follow logs from a specific service"
	@echo "  make shell        Open shell in API container"
	@echo "  make test         Run all tests"
	@echo "  make lint         Run ruff linter"
	@echo "  make format       Auto-format with ruff"
	@echo "  make audit-verify Verify audit log hash chain"
	@echo "  make gpu-status   Show GPU VRAM allocation"
	@echo "  make models-pull  Pull Ollama models (mistral + llava)"

setup:
	@python3 scripts/setup_env.py

check-env:
	@test -f .env || (echo "ERROR: .env not found. Run 'make setup' first." && exit 1)

install:
	.venv/bin/pip install -e ".[dev]"

# Linux GPU: torch must come from the cu128 index BEFORE the editable install.
# PyPI torch ships CUDA 13 builds; stable onnxruntime-gpu is CUDA 12 — mixed
# runtimes break the CUDA provider (libcublasLt.so.12 not found, silent CPU
# fallback). Installing torch cu128 first pins the whole venv to CUDA 12.
install-gpu:
	.venv/bin/pip install torch torchvision torchaudio \
		--index-url https://download.pytorch.org/whl/cu128
	.venv/bin/pip install -e ".[dev]"

# Pick the dependency install path automatically from whether an NVIDIA GPU
# is present on this host.
HAS_GPU := $(shell command -v nvidia-smi >/dev/null 2>&1 && echo 1 || echo 0)

# ── First-run bootstrap: fresh clone → fully set up ───────────────────────────
# Idempotent — every step skips if already done, so it's safe to re-run and is
# what `make run`/`make fresh` call. Needs Docker, Node/npm, Python 3 and
# openssl already installed (it checks and tells you if one is missing).
bootstrap:
	@command -v docker  >/dev/null 2>&1 || { echo "ERROR: Docker not found — install Docker, then re-run."; exit 1; }
	@command -v node    >/dev/null 2>&1 || { echo "ERROR: Node/npm not found — install Node 18+, then re-run."; exit 1; }
	@command -v python3 >/dev/null 2>&1 || { echo "ERROR: python3 not found."; exit 1; }
	@test -d .venv || { echo "→ creating .venv"; python3 -m venv .venv; }
	@test -f .env  || { echo "→ generating .env (random secrets)"; python3 scripts/setup_env.py; }
	@.venv/bin/python -c "import fastapi, insightface, torch" 2>/dev/null \
		|| { echo "→ installing Python deps ($(if $(filter 1,$(HAS_GPU)),GPU,CPU))"; \
		     $(MAKE) $(if $(filter 1,$(HAS_GPU)),install-gpu,install); }
	@test -d models/weights/models/buffalo_l \
		|| { echo "→ downloading face models (~200 MB)"; .venv/bin/python scripts/download_models.py; }
	@test -f models/manifest.json \
		|| { echo "→ signing model weights (SHA-256 manifest)"; .venv/bin/python scripts/sign_models.py; }
	$(COMPOSE) up -d $(INFRA_SERVICES)
	@bash scripts/wait_for_pg.sh
	@echo "→ applying DB migrations"
	@.venv/bin/python -m alembic upgrade head
	@$(MAKE) frontend-build
	@echo "✓ bootstrap complete — login admin / admin"

# ── Native dev workflow ───────────────────────────────────────────────────────
infra: check-env
	$(COMPOSE) up -d $(INFRA_SERVICES)
	@echo "Infra up: Postgres, Redis, MinIO, Neo4j."

frontend-build:
	cd frontend/settings && npm install && npm run build

# The one command: fresh clone → fully set up → running. bootstrap is
# idempotent, so after the first (heavy) run this is just "bring it all up".
# Use `make fresh` instead when you want a wiped clean slate first. If port
# 8000 is already in use, `make stop` first.
run: bootstrap
	@echo ""
	@echo "  Backend + frontend up.  Dashboard: http://localhost:8000/settings/  (admin/admin)"
	@$(MAKE) serve

reset: infra
	@echo "Wiping ALL data (embeddings, persons, cameras, events, crops) ..."
	.venv/bin/python scripts/reset_db.py

stop:
	@bash scripts/stop_native.sh

certs:
	@bash scripts/gen_certs.sh

# (Re-)sign all present model weights into models/manifest.json. Run after a
# deliberate weights update; bootstrap calls this automatically on first run.
sign-models:
	@.venv/bin/python scripts/sign_models.py

# Serve HTTPS when a cert exists (so the webcam Enroll page works on the LAN),
# otherwise plain HTTP. Either way, bound graceful shutdown.
serve: check-env
	@if [ -f infrastructure/certs/server.crt ]; then \
		echo "API on https://localhost:8000  (dashboard: /settings/ , login admin/admin)"; \
		.venv/bin/python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 \
			--ssl-keyfile infrastructure/certs/server.key \
			--ssl-certfile infrastructure/certs/server.crt \
			--timeout-graceful-shutdown $(GRACEFUL_SHUTDOWN_S); \
	else \
		echo "API on http://localhost:8000  (HTTP — run 'make certs' for HTTPS / webcam enroll)"; \
		.venv/bin/python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 \
			--timeout-graceful-shutdown $(GRACEFUL_SHUTDOWN_S); \
	fi

# One command for a clean test slate: ensure everything's set up (bootstrap is
# idempotent), stop anything running, wipe every data store, then run.
fresh: bootstrap stop reset
	@echo ""
	@echo "──────────────────────────────────────────────"
	@echo "  Clean slate ready.  Login:  admin / admin"
	@echo "  Dashboard: http://localhost:8000/settings/"
	@echo "──────────────────────────────────────────────"
	@$(MAKE) serve

up: check-env
	$(COMPOSE) up -d --build
	@echo ""
	@echo "Services started. Endpoints:"
	@echo "  API:       http://localhost:8000/docs"
	@echo "  Grafana:   http://localhost:3000  (admin / see .env GRAFANA_PASSWORD)"
	@echo "  MLflow:    http://localhost:5000"
	@echo "  Neo4j:     http://localhost:7474"
	@echo "  Prometheus: http://localhost:9090"

down:
	$(COMPOSE) down

logs:
	@if [ -n "$(s)" ]; then \
		$(COMPOSE) logs -f $(s); \
	else \
		$(COMPOSE) logs -f; \
	fi

shell:
	$(COMPOSE) exec $(APP_SERVICE) /bin/bash

test:
	$(COMPOSE) exec $(APP_SERVICE) python -m pytest tests/ -v --cov=src --cov-report=term-missing

lint:
	python -m ruff check src/ tests/

format:
	python -m ruff format src/ tests/
	python -m ruff check --fix src/ tests/

audit-verify:
	$(COMPOSE) exec $(APP_SERVICE) python scripts/audit_log_verify.py

gpu-status:
	$(COMPOSE) exec $(APP_SERVICE) python -c "from src.core.gpu_manager import get_gpu_manager; import json; print(json.dumps(get_gpu_manager().status(), indent=2))"

models-pull:
	$(COMPOSE) exec ollama ollama pull mistral:7b-instruct-v0.3-q4_K_M
	$(COMPOSE) exec ollama ollama pull llava:7b-v1.5-q4_K_M

kafka-topics:
	$(COMPOSE) exec kafka kafka-topics.sh --bootstrap-server localhost:9092 --list

db-migrate:
	$(COMPOSE) exec $(APP_SERVICE) alembic upgrade head

neo4j-init:
	$(COMPOSE) exec neo4j cypher-shell -u neo4j -p "${NEO4J_PASSWORD}" \
		--file /init/init.cypher

enroll:
	@echo "Usage: make enroll NAME='John Doe' ROLE='engineer'"
	$(COMPOSE) exec $(APP_SERVICE) python scripts/enroll_identity.py \
		--name "$(NAME)" --role "$(ROLE)"
