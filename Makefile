.PHONY: help up down logs shell test lint format check-env setup install install-gpu \
        infra reset serve stop frontend-build fresh

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
	@echo "  make install      Install deps into .venv (Mac / CPU dev)"
	@echo "  make install-gpu  Install deps into .venv (Linux + NVIDIA GPU)"
	@echo "  make fresh        CLEAN SLATE: stop, wipe all data, rebuild UI, run"
	@echo "  make serve        Run the API server (foreground; Ctrl+C stops it)"
	@echo "  make reset        DESTRUCTIVE: wipe all data, reseed admin/admin + zones"
	@echo "  make stop         Stop the native API server + camera workers"
	@echo "  make infra        Start just the infra containers"
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

# ── Native dev workflow ───────────────────────────────────────────────────────
infra: check-env
	$(COMPOSE) up -d $(INFRA_SERVICES)
	@echo "Infra up: Postgres, Redis, MinIO, Neo4j."

frontend-build:
	cd frontend/settings && npm install && npm run build

reset: infra
	@echo "Wiping ALL data (embeddings, persons, cameras, events, crops) ..."
	.venv/bin/python scripts/reset_db.py

stop:
	@bash scripts/stop_native.sh

serve: check-env
	@echo "API on http://localhost:8000  (dashboard: /settings/ , login admin/admin)"
	.venv/bin/python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 \
		--timeout-graceful-shutdown $(GRACEFUL_SHUTDOWN_S)

# One command for a clean test slate: stop anything running, wipe every data
# store, rebuild the UI bundle, then run the server in the foreground.
fresh: stop infra reset frontend-build
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
