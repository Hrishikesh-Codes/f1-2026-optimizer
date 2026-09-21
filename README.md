# F1 2026 Race Strategy Optimizer

**Probabilistic Formula 1 race strategy engine built on the 2026 technical regulations — Monte Carlo simulation, real-time telemetry, and a full-stack deployment-ready dashboard.**

---

> **Running it:** the dashboard is a local Streamlit app — `streamlit run app.py`, no API keys required. FastF1 pulls telemetry from the official F1 timing feed on first run and caches it locally. Full setup is under [Getting Started](#getting-started).

---

## Overview

The F1 2026 Race Strategy Optimizer is a physics-based simulation engine that models every dimension of a Formula 1 race strategy under the new 2026 technical regulations. It runs up to 2,000 Monte Carlo simulations per strategy candidate, stochastically sampling tyre degradation variance, safety car deployments, track temperature, and fuel burn to produce statistically robust recommendations with 95% confidence intervals. The engine encodes the major 2026 regulation changes — the 50/50 ICE/ERS power split, elimination of the MGU-H, Active Aerodynamics replacing DRS, the new Overtake Override Mode (OOM), and Pirelli's revised C1–C5 tyre range — and is calibrated against ground-truth stint data from the first three 2026 Grands Prix. A seven-page Streamlit dashboard and an 11-endpoint FastAPI backend make the engine accessible both interactively and programmatically.

---

## Features

- **Monte Carlo Strategy Optimization** — evaluates 30–40 candidate strategies per race across 2,000 stochastic simulations; ranks by expected total race time with full win-distribution and confidence interval output
- **Physics-Based Tyre Model** — per-compound linear degradation with cliff-phase detection, track abrasiveness scaling, thermal sensitivity, and push-level modulation; calibrated on actual 2026 race data
- **2026 ERS & OOM Modeling** — 50/50 hybrid power split, 350 kW MGU-K, Super Clip vs Lift-off recovery modes, and Overtake Override Mode decision logic with per-lap battery-state tracking
- **Safety Car & VSC Simulation** — circuit-specific deployment probabilities, random timing/duration sampling, free-pit-window detection, and lap-time multipliers applied per lap
- **FastF1 Telemetry Integration** — fetches historical and live stint data via the FastF1 API, caches to SQLite, and feeds automated tyre-degradation calibration
- **Live Race Monitor** — real-time strategy recommendations per driver with tyre age tracking and pit-window alerting; supports live timing, cached replay, and synthetic fallback modes
- **FastAPI REST Backend** — 11 endpoints across strategy, tyre, season and reference data, with full Pydantic request/response validation
- **Seven-Page Streamlit Dashboard** — strategy optimizer, tyre degradation viewer, stint calculator, energy/OOM analyzer, live race monitor, season overview, and model accuracy tracker
- **Season Accuracy Tracking** — compares predicted stop-count and compound choices against actual 2026 results; 100% accurate across the first three completed rounds

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Dashboard | Streamlit 1.35+ |
| REST API | FastAPI 0.111+ / Uvicorn 0.29+ |
| Data / Telemetry | FastF1 3.3+, Pandas 2.2+ |
| Simulation | NumPy 1.26+, SciPy 1.13+ |
| Visualizations | Plotly 5.22+ |
| Validation | Pydantic 2.7+ |
| Reporting | ReportLab 4.1+ |
| Testing | Pytest 8.2+ / pytest-cov 5.0+ |
| Language | Python 3.11+ |

---

## Project Structure

```
f1-2026-optimizer/
├── app.py                          # Streamlit entry point + page router
├── main.py                         # FastAPI entry point
├── config.py                       # 2026 regulations, 22 circuits, teams, compounds (682 lines)
├── requirements.txt
│
├── src/simulation/                 # Core engine — 2,037 lines
│   ├── monte_carlo.py              # Stochastic optimizer, 2,000 runs/strategy (539)
│   ├── tyre.py                     # Degradation model, linear + cliff phases (361)
│   ├── strategy.py                 # Strategy generation & 2026 rules validation (320)
│   ├── ers.py                      # ERS/OOM model — 50/50 hybrid, Super Clip, Lift-off (291)
│   ├── safety_car.py               # SC/VSC deployment model (263)
│   └── laptime.py                  # Per-lap time assembly (247)
│
├── src/frontend/                   # Streamlit dashboard — 2,191 lines
│   ├── strategy_simulator.py       # Main optimizer page (322)
│   ├── oom_analyzer.py             # Overtake Override Mode analyzer (359)
│   ├── stint_calculator.py         # Stint/undercut calculator (312)
│   ├── live_dashboard.py           # Real-time race monitor (286)
│   ├── tyre_viewer.py              # Degradation curves (277)
│   ├── season_overview.py          # 22-round calendar (246)
│   ├── accuracy_tracker.py         # Prediction vs actual (178)
│   └── styles.py                   # Dark F1 theme (200)
│
├── src/api/
│   └── routes.py                   # 11 REST endpoints (420 lines)
│
├── src/data/
│   ├── fastf1_loader.py            # FastF1 integration + SQLite caching (615)
│   └── calibration/
│       ├── calibration_loader.py   # 2026 ground-truth stint data (616)
│       ├── deg_curves.json         # Tyre degradation per circuit
│       ├── pit_loss_2026.json      # Pit loss per circuit
│       └── sc_history.json         # Safety car probability history
│
└── tests/                          # 774 lines
    ├── test_oom.py (272)  test_monte_carlo.py (201)  test_tyre.py (201)
    └── test_calibration.py (59)    test_live.py (40)
```

---

## REST API

| Method | Route | Description |
|---|---|---|
| POST | `/simulate` | Run the Monte Carlo optimizer for a circuit/team |
| POST | `/stint/analyze` | Stint and undercut analysis |
| POST | `/oom/analyze` | Overtake Override Mode decision analysis |
| GET | `/tyre/degradation` | Degradation curve for a compound/circuit |
| GET | `/circuits` | All 22 circuits |
| GET | `/circuits/{circuit_key}` | Single circuit reference data |
| GET | `/teams` | Team reference data |
| GET | `/compounds` | Pirelli C1–C5 compound specs |
| GET | `/regulations` | 2026 regulation constants |
| GET | `/historical/{circuit_key}/{season}` | Historical stint data via FastF1 |
| GET | `/season/overview` | Season calendar and status |

Interactive docs at `http://localhost:8000/docs` once the server is running.

---

## Getting Started

### Prerequisites

- Python 3.11 or higher
- pip

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Hrishikesh-Codes/f1-2026-optimizer.git
cd f1-2026-optimizer

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Run the Streamlit Dashboard

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`.

### Run the FastAPI Backend

```bash
uvicorn main:app --reload
```

API at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

### Run Tests

```bash
pytest tests/ --cov=. --cov-report=term-missing
```

### Environment Setup

FastF1 caches race data locally to `data/cache/` by default. No API keys are required — FastF1 pulls from the official F1 timing feed and Ergast API automatically. On Streamlit Community Cloud, the cache writes to `/tmp/fastf1_cache` (ephemeral; repopulates on cold start).

---

## How It Works

### Monte Carlo Simulation

For a given circuit and team, the engine enumerates all valid pit stop strategies under 2026 regulations (1–3 stops, minimum 2 compounds, minimum 10 laps between stops). For each of the ~30–40 candidates it runs `N` simulations (default: 2,000). Each simulation independently samples:

- **Tyre degradation variance** — ±10% multiplier drawn from a normal distribution
- **Track temperature** — ±3°C variation affecting compound thermal sensitivity
- **Safety car events** — Poisson-sampled deployment lap and duration based on circuit-specific historical SC probabilities; a safety car applies a 1.42× lap-time multiplier and opens free-pit windows that may change the optimal strategy
- **Lap-time noise** — Gaussian noise (σ = 0.05s) per lap to represent real-world variation

Each simulation returns a total race time for every strategy. The optimizer ranks strategies by mean simulated time, computes 95% confidence intervals from the 2.5th/97.5th percentiles, and reports a win-probability distribution — the fraction of simulations each strategy wins — so the output is a probability distribution over outcomes rather than a single deterministic answer.

### Tyre Degradation Model

Lap-time penalty for a given compound at stint lap `l`:

```
Δt(l) = linear_rate × l × abrasiveness_factor × thermal_factor × push_factor
      + cliff_exponent × (l − cliff_lap)²    [once cliff lap is exceeded]
```

Compound parameters (base pace, linear rate, cliff lap, thermal sensitivity) are initialised from the 2026 Pirelli specifications and then refined via linear regression on actual stint data loaded from FastF1 — prioritising ground-truth 2026 results from Australia, China, and Japan. Higher track temperatures and more aggressive pushing advance the cliff lap onset.

### Pit Stop & Undercut Modeling

Pit loss times are circuit-specific, ranging from 21.5s (Austria) to 25.0s (Singapore), and calibrated from 2026 in-season data. Undercut profitability is computed by comparing the cumulative lap-time gain from a fresher tyre over a look-ahead window against the one-time pit-loss cost, yielding a lap-precise window in which an undercut is expected to net positive time against a rival who stays out.

### 2026 ERS & Overtake Override Mode

The ERS model tracks battery state (MJ) lap-by-lap under two recovery modes: Super Clip (charges at full throttle with no aerodynamic penalty) and Lift-off (charges under braking but disables Active Aerodynamics, costing ~0.18s/lap in drag). Overtake Override Mode — the 2026 replacement for DRS — is modelled as a per-lap binary decision: if the gap to the car ahead is ≤ 1.0s and battery ≥ 0.80 MJ, OOM is activated, granting +0.5 MJ extra capacity and approximately +0.15s effective power per lap.

---

## Model Accuracy (2026 Season)

| Round | GP | Predicted | Actual | Correct |
|-------|----|-----------|--------|---------|
| R1 | Australia | 1-stop | 1-stop (Piastri) | Yes |
| R2 | China | 1-stop | 1-stop (Norris) | Yes |
| R3 | Japan | 1-stop | 1-stop (Verstappen) | Yes |
| R4+ | ... | ... | *Season ongoing* | ... |

**Accuracy at time of calibration: 3/3 on stop-count prediction**, measured across Rounds 1–3 (Australia, China, Japan) — the rounds the tyre model was calibrated against. The figures above are a snapshot from that calibration run, not a live-updating season tracker; rerun the Accuracy Tracker page to score later rounds.

---

## 2026 Regulation Changes Encoded

| Rule | 2026 | vs 2025 |
|------|------|---------|
| ICE power | ~400 kW | Down from 550 kW |
| MGU-K | 350 kW | Up 3x from 120 kW |
| MGU-H | Eliminated | — |
| Hybrid split | 50/50 ICE/ERS | New |
| Aerodynamics | Active Aero (auto) | Replaces driver-activated DRS |
| Overtaking aid | Overtake Override Mode | Replaces DRS |
| Tyre range | C1–C5 only | C6 dropped |
| Rear tyre width | −30 mm | Narrower |
| Car weight | −30 kg | Lighter |

---

## License

MIT
