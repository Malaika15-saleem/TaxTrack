"""
TaxTrack - Vehicle Tax Verification System
FastAPI application entry point.

Usage:
    python main.py              # Runs with uvicorn
    uvicorn main:app --reload   # Alternative: direct uvicorn invocation
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

from config import settings, BASE_DIR
from backend.database import init_db
from backend.routes.vehicles import router as vehicles_router, tax_router, stats_router
from backend.routes.detection import router as detection_router
from backend.routes.camera import router as camera_router


# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────

logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Lifespan (startup / shutdown)
# ──────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: init DB on startup, cleanup on shutdown."""
    logger.info("Starting %s v%s", settings.app_name, settings.app_version)
    await init_db()
    logger.info("Database initialized")
    yield
    logger.info("Shutting down %s", settings.app_name)


# ──────────────────────────────────────────────
# App factory
# ──────────────────────────────────────────────

app = FastAPI(
    title=settings.app_name,
    description="Vehicle Tax Verification System with YOLO detection and PaddleOCR",
    version=settings.app_version,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS (allow all origins for development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
static_dir = BASE_DIR / "static"
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Templates
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# Register API routers
app.include_router(vehicles_router)
app.include_router(tax_router)
app.include_router(stats_router)
app.include_router(detection_router)
app.include_router(camera_router)


# ──────────────────────────────────────────────
# Dashboard (HTML)
# ──────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Serve the main web dashboard."""
    return templates.TemplateResponse(request=request, name="index.html")


# ──────────────────────────────────────────────
# Health check
# ──────────────────────────────────────────────

@app.get("/api/health", tags=["System"])
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "app": settings.app_name,
        "version": settings.app_version,
    }


# ──────────────────────────────────────────────
# Run with: python main.py
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )
