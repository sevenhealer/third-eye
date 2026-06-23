from __future__ import annotations

from contextlib import asynccontextmanager

from pathlib import Path

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from prometheus_client import make_asgi_app

from src.core.config import get_settings
from src.core.database import close_all, get_neo4j_driver, get_redis
from src.core.exceptions import AuthorizationError, PromptInjectionError, ThirdEyeError
from src.core.logging import configure_logging, get_logger
from src.orchestration.supervisor import CameraSupervisor

settings = get_settings()
configure_logging(settings.app_log_level)
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("third_eye_starting", env=settings.app_env)
    from src.core.model_registry import startup_verify
    startup_verify(settings.model_manifest_path)
    # Warm up connections
    await get_redis()
    get_neo4j_driver()
    supervisor = CameraSupervisor()
    await supervisor.start()
    app.state.supervisor = supervisor
    logger.info("third_eye_ready")
    yield
    logger.info("third_eye_shutting_down")
    await supervisor.stop()
    await close_all()


app = FastAPI(
    title="Third-Eye Visual Intelligence API",
    version="0.1.0",
    docs_url="/docs" if not settings.is_production else None,
    redoc_url="/redoc" if not settings.is_production else None,
    openapi_url="/openapi.json" if not settings.is_production else None,
    lifespan=lifespan,
)

# ── CORS (restrict in production) ─────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"] if not settings.is_production else [],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Prometheus metrics endpoint ───────────────────────────────────────────────
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# ── Live camera snapshots (written by run_live.py/run_objects.py) ────────────
_SNAPSHOTS_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "snapshots"
_SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/stream", StaticFiles(directory=str(_SNAPSHOTS_DIR)), name="stream")

# ── Operator dashboard (static, no build tooling) ─────────────────────────────
_DASHBOARD_DIR = Path(__file__).resolve().parent.parent.parent / "static" / "dashboard"
if _DASHBOARD_DIR.is_dir():
    app.mount("/dashboard", StaticFiles(directory=str(_DASHBOARD_DIR), html=True), name="dashboard")

# ── Settings app (React+Vite build output; reads the same sessionStorage
# token the dashboard writes, no separate login) ──────────────────────────────
# Not a plain StaticFiles mount: this is a client-side-routed SPA (react-
# router, basename="/settings"), so a direct load/refresh of e.g.
# /settings/zones must still serve index.html and let the router take
# over - StaticFiles(html=True) only does that fallback for the exact
# mount root, not arbitrary sub-paths, which 404'd on every route but "/"
# until this was added.
_SETTINGS_DIR = Path(__file__).resolve().parent.parent.parent / "frontend" / "settings" / "dist"
if _SETTINGS_DIR.is_dir():
    @app.get("/settings")
    async def redirect_settings_root() -> RedirectResponse:
        return RedirectResponse(url="/settings/")

    @app.get("/settings/{full_path:path}")
    async def serve_settings_spa(full_path: str) -> FileResponse:
        candidate = _SETTINGS_DIR / full_path
        if full_path and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(_SETTINGS_DIR / "index.html")

# ── Exception handlers ────────────────────────────────────────────────────────

@app.exception_handler(AuthorizationError)
async def authorization_error_handler(
    request: Request, exc: AuthorizationError
) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_403_FORBIDDEN,
        content={"detail": str(exc)},
    )


@app.exception_handler(PromptInjectionError)
async def injection_error_handler(
    request: Request, exc: PromptInjectionError
) -> JSONResponse:
    logger.warning("prompt_injection_attempt", path=request.url.path)
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": "Query rejected: invalid input detected."},
    )


@app.exception_handler(ThirdEyeError)
async def thirdeye_error_handler(
    request: Request, exc: ThirdEyeError
) -> JSONResponse:
    logger.error("application_error", error=str(exc), path=request.url.path)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal system error. Check logs."},
    )


# ── Health check ──────────────────────────────────────────────────────────────

@app.get("/health", tags=["system"])
async def health() -> dict:
    return {"status": "ok", "version": "0.1.0"}


# ── Routers ───────────────────────────────────────────────────────────────────
from src.api.routers import auth, cameras, identities, events, queries, alerts, admin, zones, hardware  # noqa: E402

app.include_router(auth.router,        prefix="/api/v1/auth",        tags=["auth"])
app.include_router(cameras.router,     prefix="/api/v1/cameras",     tags=["cameras"])
app.include_router(identities.router,  prefix="/api/v1/identities",  tags=["identities"])
app.include_router(events.router,      prefix="/api/v1/events",       tags=["events"])
app.include_router(queries.router,     prefix="/api/v1/queries",      tags=["queries"])
app.include_router(alerts.router,      prefix="/api/v1/alerts",       tags=["alerts"])
app.include_router(admin.router,       prefix="/api/v1/admin",        tags=["admin"])
app.include_router(hardware.router,    prefix="/api/v1/hardware",     tags=["hardware"])
app.include_router(zones.router,       prefix="/api/v1/zones",        tags=["zones"])
