"""
F1 2026 Race Strategy Optimizer — FastAPI Application Entry Point

Run with:
    uvicorn main:app --reload --port 8000

API docs at: http://localhost:8000/docs
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Add project root to path so imports work when run from any directory
sys.path.insert(0, str(Path(__file__).parent))

from src.api.routes import router
from config import REGULATIONS_2026, CURRENT_SEASON

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


app = FastAPI(
    title="F1 2026 Race Strategy Optimizer",
    description=(
        "Production-grade Formula 1 race strategy optimization API. "
        "Encodes all 2026 technical regulations: 50/50 hybrid PU split, "
        "Active Aerodynamics, Overtake Override Mode, Pirelli C1–C5 tyres, "
        "and the full 22-round season calendar. "
        "Uses Monte Carlo simulation (1000+ runs) to find optimal strategies."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")


@app.get("/", tags=["Health"])
def root() -> dict:
    return {
        "service": "F1 2026 Race Strategy Optimizer",
        "season": CURRENT_SEASON,
        "status": "operational",
        "regulations": "2026 Technical Regulations (Active Aero, OOM, no MGU-H)",
        "docs": "/docs",
    }


@app.get("/health", tags=["Health"])
def health() -> dict:
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
