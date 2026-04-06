"""
FastAPI Routes — F1 2026 Race Strategy Optimizer

Endpoints:
  GET  /circuits                    → list all 22 circuits
  GET  /circuits/{circuit_key}      → circuit detail
  GET  /teams                       → all teams + drivers
  GET  /compounds                   → tyre compound configs
  POST /simulate                    → run Monte Carlo strategy optimization
  POST /stint/analyze               → real-time stint analysis
  GET  /tyre/degradation            → deg curve for compound at circuit
  GET  /season/overview             → all circuits with predicted optimal strategy
  POST /oom/analyze                 → OOM decision for given race state
  GET  /historical/{circuit}/{year} → historical FastF1 lap data
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from config import (
    CIRCUITS, TEAMS, TYRE_COMPOUNDS, ALL_DRIVERS,
    POWER_UNITS, CIRCUIT_ORDER, REGULATIONS_2026,
)
from src.simulation import (
    MonteCarloSimulator, MonteCarloConfig,
    TyreDegradationModel, ERSModel, ERSState,
)

logger = logging.getLogger(__name__)
router = APIRouter()


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class SimulateRequest(BaseModel):
    circuit: str = Field(..., description="Circuit key e.g. 'australia'")
    team: str = Field(..., description="Team name e.g. 'Ferrari'")
    driver: str = Field(..., description="Driver full name")
    track_temp_c: float = Field(35.0, ge=10.0, le=65.0)
    starting_compound: Optional[str] = Field(None, description="Force starting compound")
    num_simulations: int = Field(1000, ge=100, le=5000)
    include_sc: bool = True
    max_stops: int = Field(3, ge=1, le=3)


class StintAnalysisRequest(BaseModel):
    circuit: str
    team: str
    current_lap: int = Field(..., ge=1)
    compound: str
    tyre_age: int = Field(..., ge=0)
    battery_fraction: float = Field(0.70, ge=0.0, le=1.0)
    gap_ahead_s: float = Field(999.0, ge=0.0)
    track_temp_c: float = Field(35.0, ge=10.0, le=65.0)


class OOMRequest(BaseModel):
    compound: str
    tyre_age: int
    battery_level_mj: float
    gap_to_car_ahead_s: float
    laps_remaining: int


class StrategyOut(BaseModel):
    compound_sequence: str
    full_sequence: str
    pit_laps: List[int]
    num_stops: int
    mean_time_s: float
    std_time_s: float
    stint_details: List[Dict[str, Any]]


class SimulateResponse(BaseModel):
    circuit: str
    team: str
    driver: str
    optimal_strategy: StrategyOut
    alternative_strategies: List[StrategyOut]
    sc_impact: Dict[str, float]
    undercut_windows: List[Dict[str, Any]]
    oom_recommendations: List[int]
    confidence_interval_95: List[float]
    simulation_count: int
    strategy_win_distribution: Dict[str, int]


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _strategy_to_out(strategy, mean_time: float, std_time: float) -> StrategyOut:
    return StrategyOut(
        compound_sequence=strategy.compound_sequence,
        full_sequence=strategy.full_compound_sequence,
        pit_laps=strategy.pit_laps,
        num_stops=strategy.num_stops,
        mean_time_s=round(mean_time, 3),
        std_time_s=round(std_time, 3),
        stint_details=[
            {
                "compound": s.compound,
                "start_lap": s.start_lap,
                "end_lap": s.end_lap,
                "length_laps": s.length,
                "push_level": s.push_level,
            }
            for s in strategy.stints
        ],
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/circuits", tags=["Reference"])
def list_circuits() -> List[Dict[str, Any]]:
    """Return all 22 circuits with key metadata."""
    return [
        {
            "key": key,
            "name": cfg.name,
            "circuit": cfg.circuit,
            "round": cfg.round_number,
            "total_laps": cfg.total_laps,
            "compounds": cfg.compounds,
            "circuit_type": cfg.circuit_type,
            "sc_probability": cfg.sc_probability,
            "is_sprint": cfg.is_sprint,
            "location": cfg.location,
        }
        for key, cfg in CIRCUITS.items()
    ]


@router.get("/circuits/{circuit_key}", tags=["Reference"])
def get_circuit(circuit_key: str) -> Dict[str, Any]:
    """Return full config for a single circuit."""
    if circuit_key not in CIRCUITS:
        raise HTTPException(status_code=404, detail=f"Circuit '{circuit_key}' not found")
    cfg = CIRCUITS[circuit_key]
    return {
        "key": circuit_key,
        "name": cfg.name,
        "circuit": cfg.circuit,
        "round": cfg.round_number,
        "total_laps": cfg.total_laps,
        "pit_loss_time": cfg.pit_loss_time,
        "compounds": cfg.compounds,
        "circuit_type": cfg.circuit_type,
        "sc_probability": cfg.sc_probability,
        "is_sprint": cfg.is_sprint,
        "base_lap_time": cfg.base_lap_time,
        "track_abrasiveness": cfg.track_abrasiveness,
        "fuel_per_lap": cfg.fuel_per_lap,
        "rain_probability": cfg.rain_probability,
        "overtake_difficulty": cfg.overtake_difficulty,
        "location": cfg.location,
        "num_drs_zones": cfg.num_drs_zones,
    }


@router.get("/teams", tags=["Reference"])
def list_teams() -> List[Dict[str, Any]]:
    """Return all 11 teams with drivers and performance data."""
    return [
        {
            "name": name,
            "short_name": cfg.short_name,
            "drivers": cfg.drivers,
            "pu_supplier": cfg.pu_supplier,
            "performance_tier": cfg.performance_tier,
            "base_lap_delta_s": cfg.base_lap_delta,
            "chassis_aero_efficiency": cfg.chassis_aero_efficiency,
        }
        for name, cfg in TEAMS.items()
    ]


@router.get("/compounds", tags=["Reference"])
def list_compounds() -> List[Dict[str, Any]]:
    """Return all tyre compound configurations."""
    return [
        {
            "name": name,
            "label": cfg.label,
            "color": cfg.color,
            "base_pace_offset_s": cfg.base_pace_offset,
            "linear_deg_rate": cfg.linear_deg_rate,
            "cliff_lap": cfg.cliff_lap,
            "cliff_exponent": cfg.cliff_exponent,
            "thermal_sensitivity": cfg.thermal_sensitivity,
            "max_viable_laps": cfg.max_viable_laps,
            "min_recommended_laps": cfg.min_recommended_laps,
        }
        for name, cfg in TYRE_COMPOUNDS.items()
    ]


@router.get("/regulations", tags=["Reference"])
def get_regulations() -> Dict[str, Any]:
    """Return 2026 regulation constants."""
    return REGULATIONS_2026


@router.post("/simulate", response_model=SimulateResponse, tags=["Strategy"])
def run_simulation(req: SimulateRequest) -> SimulateResponse:
    """
    Run Monte Carlo race strategy optimization.

    Simulates 1000+ race scenarios, sampling safety car timing,
    degradation variance, and weather to find the optimal strategy.
    """
    if req.circuit not in CIRCUITS:
        raise HTTPException(status_code=404, detail=f"Circuit '{req.circuit}' not found")
    if req.team not in TEAMS:
        raise HTTPException(status_code=404, detail=f"Team '{req.team}' not found")
    if req.driver not in ALL_DRIVERS:
        raise HTTPException(status_code=404, detail=f"Driver '{req.driver}' not found")

    config = MonteCarloConfig(
        num_simulations=req.num_simulations,
        include_sc=req.include_sc,
        max_stops=req.max_stops,
    )
    sim = MonteCarloSimulator(config)

    try:
        result = sim.run(
            circuit=req.circuit,
            team=req.team,
            driver=req.driver,
            track_temp_c=req.track_temp_c,
            starting_compound=req.starting_compound,
        )
    except Exception as exc:
        logger.error(f"Simulation error: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))

    return SimulateResponse(
        circuit=result.circuit,
        team=result.team,
        driver=result.driver,
        optimal_strategy=_strategy_to_out(
            result.optimal_strategy,
            result.optimal_strategy_mean_time,
            result.optimal_strategy_std_time,
        ),
        alternative_strategies=[
            _strategy_to_out(s, mt, st)
            for s, mt, st in result.alternative_strategies
        ],
        sc_impact={str(k): round(v, 3) for k, v in result.sc_impact_analysis.items()},
        undercut_windows=[
            {"lap": lap, "net_gain_s": round(gain, 3)}
            for lap, gain in result.undercut_windows
        ],
        oom_recommendations=list(result.oom_recommendations.keys()),
        confidence_interval_95=list(result.confidence_interval_95),
        simulation_count=result.simulation_count,
        strategy_win_distribution=result.strategy_win_distribution,
    )


@router.post("/stint/analyze", tags=["Strategy"])
def analyze_stint(req: StintAnalysisRequest) -> Dict[str, Any]:
    """
    Real-time stint analysis: given current lap, tyre, battery state —
    return pit window, undercut viability, and OOM recommendation.
    """
    if req.circuit not in CIRCUITS:
        raise HTTPException(status_code=404, detail=f"Circuit '{req.circuit}' not found")

    sim = MonteCarloSimulator()
    result = sim.analyze_current_stint(
        circuit=req.circuit,
        team=req.team,
        current_lap=req.current_lap,
        compound=req.compound,
        tyre_age=req.tyre_age,
        battery_fraction=req.battery_fraction,
        gap_ahead=req.gap_ahead_s,
        track_temp_c=req.track_temp_c,
    )
    return result


@router.get("/tyre/degradation", tags=["Tyres"])
def get_degradation_curve(
    circuit: str = Query(...),
    compound: str = Query(...),
    track_temp_c: float = Query(35.0),
    push_level: float = Query(0.80),
) -> Dict[str, Any]:
    """Return per-lap degradation data for a compound at a circuit."""
    if circuit not in CIRCUITS:
        raise HTTPException(status_code=404, detail=f"Circuit '{circuit}' not found")
    if compound not in TYRE_COMPOUNDS:
        raise HTTPException(status_code=404, detail=f"Compound '{compound}' not found")

    cfg_c = CIRCUITS[circuit]
    cfg_t = TYRE_COMPOUNDS[compound]
    model = TyreDegradationModel()

    max_laps = cfg_t.max_viable_laps
    deltas = model.compute_stint_time_deltas(
        compound=compound,
        num_laps=max_laps,
        track_abrasiveness=cfg_c.track_abrasiveness,
        track_temp_celsius=track_temp_c,
        push_level=push_level,
    )
    cliff_lap = model.compute_cliff_lap(compound, cfg_c.track_abrasiveness, track_temp_c)
    min_l, opt_l, max_l = model.estimate_optimal_stint_length(
        compound, cfg_c.track_abrasiveness, track_temp_c
    )

    return {
        "circuit": circuit,
        "compound": compound,
        "track_temp_c": track_temp_c,
        "push_level": push_level,
        "cliff_lap": cliff_lap,
        "optimal_stint_laps": opt_l,
        "max_stint_laps": max_l,
        "lap_deltas": [round(float(d), 4) for d in deltas],
    }


@router.post("/oom/analyze", tags=["Strategy"])
def analyze_oom(req: OOMRequest) -> Dict[str, Any]:
    """Return OOM strategic decision for given race state."""
    if req.compound not in TYRE_COMPOUNDS:
        raise HTTPException(status_code=404, detail=f"Compound '{req.compound}' not found")

    ers_model = ERSModel()
    state = ERSState(
        battery_level_mj=req.battery_level_mj,
        battery_capacity_mj=4.0,
    )
    use_oom, reason = ers_model.compute_oom_decision(
        ers_state=state,
        gap_to_car_ahead=req.gap_to_car_ahead_s,
        laps_remaining=req.laps_remaining,
        tyre_age=req.tyre_age,
        compound=req.compound,
    )
    return {
        "use_oom": use_oom,
        "reason": reason,
        "battery_level_mj": req.battery_level_mj,
        "battery_soc": round(req.battery_level_mj / 4.0, 3),
        "gap_to_car_ahead_s": req.gap_to_car_ahead_s,
        "detection_gap_s": 1.0,
        "within_detection_gap": req.gap_to_car_ahead_s <= 1.0,
    }


@router.get("/historical/{circuit_key}/{season}", tags=["Data"])
def get_historical_data(circuit_key: str, season: int) -> Dict[str, Any]:
    """Return cached historical lap data for a circuit/season."""
    if circuit_key not in CIRCUITS:
        raise HTTPException(status_code=404, detail=f"Circuit '{circuit_key}' not found")
    if season < 2022 or season > 2026:
        raise HTTPException(status_code=400, detail="Season must be 2022–2026")

    from src.data import FastF1Loader
    loader = FastF1Loader()
    df = loader.get_race_laps(circuit_key, season)
    stints = loader.get_stint_data(circuit_key, season)

    return {
        "circuit": circuit_key,
        "season": season,
        "lap_count": len(df),
        "drivers": sorted(df["driver"].unique().tolist()) if not df.empty else [],
        "compounds_used": sorted(df["compound"].dropna().unique().tolist()) if not df.empty else [],
        "stints": stints.to_dict(orient="records") if not stints.empty else [],
    }


@router.get("/season/overview", tags=["Season"])
def season_overview() -> List[Dict[str, Any]]:
    """Return optimal predicted strategy for every round of the 2026 season."""
    from config import COMPLETED_2026_ROUNDS
    overview = []
    for circuit_key in CIRCUIT_ORDER:
        cfg = CIRCUITS[circuit_key]
        # Quick single-run estimate (not full MC) for speed
        from src.simulation.strategy import StrategyGenerator
        gen = StrategyGenerator()
        strategies = gen.generate_all_strategies(circuit_key, max_stops=2)
        # Pick the first valid 1-stop as predicted
        one_stop = [s for s in strategies if s.num_stops == 1]
        two_stop = [s for s in strategies if s.num_stops == 2]

        overview.append({
            "round": cfg.round_number,
            "circuit": circuit_key,
            "name": cfg.name,
            "location": cfg.location,
            "is_sprint": cfg.is_sprint,
            "compounds": cfg.compounds,
            "sc_probability": cfg.sc_probability,
            "predicted_stops": 1 if one_stop else 2,
            "predicted_strategy": one_stop[0].full_compound_sequence if one_stop else (
                two_stop[0].full_compound_sequence if two_stop else "TBD"
            ),
            "data_available": circuit_key in COMPLETED_2026_ROUNDS,
        })
    return overview
