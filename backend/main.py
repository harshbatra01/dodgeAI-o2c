"""
main.py — FastAPI application initialization and startup

Bootstraps the backend server, instantiates GraphStore from graph.json,
and mounts API routers from the graph and chat modules.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from graph_store import GraphStore
from routes.graph import router as graph_router
from routes.chat import router as chat_router

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)

# Preload graph location (assumed adjacent to main.py)
GRAPH_FILE = Path(__file__).parent / "graph.json"

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle event for FastAPI. Initialize the state before requests arrive."""
    try:
        app.state.store = GraphStore(GRAPH_FILE)
        logger.info(f"Loaded graph state from {GRAPH_FILE} (O2C data store)")
    except FileNotFoundError:
        logger.error(f"FATAL: Graph database file not found at {GRAPH_FILE}")
        raise RuntimeError("Missing Phase 1 graph.json dataset.")
    yield


app = FastAPI(
    title="Dodge AI — O2C Entity Query System",
    description="Backend API for navigating large scale Order-to-Cash data graph representations.",
    version="1.0.0",
    lifespan=lifespan
)

cors_origins = [origin.strip() for origin in os.getenv("ALLOWED_ORIGINS", "*").split(",") if origin.strip()]
if not cors_origins:
    cors_origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(graph_router)
app.include_router(chat_router)


@app.exception_handler(Exception)
async def generic_exception_handler(request, exc):
    """Catch-all for exceptions missing standard handlers."""
    return JSONResponse({"detail": str(exc)}, status_code=500)
