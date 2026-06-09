.PHONY: help up down logs shell test lint format check-env setup

COMPOSE = docker compose
APP_SERVICE = api

help:
	@echo "Third-Eye Visual Intelligence Platform"
	@echo ""
	@echo "  make setup        Copy .env.example → .env (first time)"
	@echo "  make up           Start all services"
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
	@test -f .env || (cp .env.example .env && echo "Created .env — edit it before running 'make up'")

check-env:
	@test -f .env || (echo "ERROR: .env not found. Run 'make setup' first." && exit 1)

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
		--file /var/lib/neo4j/import/init.cypher

enroll:
	@echo "Usage: make enroll NAME='John Doe' ROLE='engineer'"
	$(COMPOSE) exec $(APP_SERVICE) python scripts/enroll_identity.py \
		--name "$(NAME)" --role "$(ROLE)"
