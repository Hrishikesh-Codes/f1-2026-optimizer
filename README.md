# F1 2026 Race Strategy Optimizer

**Probabilistic Formula 1 race strategy engine built on the 2026 technical regulations — Monte Carlo simulation, real-time telemetry, and a full-stack deployment-ready dashboard.**

---

## Live Demo

[Launch on Streamlit Community Cloud](https://your-app-name.streamlit.app) *(deploy and replace this link)*

---

## Overview

The F1 2026 Race Strategy Optimizer is a physics-based simulation engine that models every dimension of a Formula 1 race strategy under the new 2026 technical regulations. It runs up to 2,000 Monte Carlo simulations per strategy candidate, stochastically sampling tyre degradation variance, safety car deployments, track temperature, and fuel burn to produce statistically robust recommendations with 95% confidence intervals. The engine encodes the major 2026 regulation changes — the 50/50 ICE/ERS power split, elimination of the MGU-H, Active Aerodynamics replacing DRS, the new Overtake Override Mode (OOM), and Pirelli's revised C1–C5 tyre range — and is calibrated against ground-truth stint data from the first three 2026 Grands Prix. A six-page Streamlit dashboard and a FastAPI REST backend make the engine accessible both interactively and programmatically.

---

## Features

- **Monte Carlo Strategy Optimization** — evaluates 30–40 candidate strategies per race across 2,000 stochastic simulations; ranks by expected total race time with full win-distribution and confidence interval output
- **Physics-Based Tyre Model** — per-compound linear degradation with cliff-phase detection, track abrasiveness scaling, thermal sensitivity, and push-level modulation; calibrated on actual 2026 race data
- **2026 ERS & OOM Modeling** — 50/50 hybrid power split, 350 kW MGU-K, Super Clip vs Lift-off recovery modes, and Overtake Override Mode decision logic with per-lap battery-state tracking
- **Safety Car & VSC Simulation** — circuit-specific deployment probabilities, random timing/duration sampling, free-pit-window detection, and lap-time multipliers applied per lap
- **FastF1 Telemetry Integration** — fetches historical and live stint data via the FastF1 API, caches to SQLite, and feeds automated tyre-degradation calibration
- **Live Race Monitor** — real-time strategy recommendations per driver with tyre age tracking and pit-window alerting; supports live timing, cached replay, and synthetic fallback modes
- **FastAPI REST Backend** — `/simulate`, `/stint/analyze`, `/oom/analyze`, and reference-data endpoints with full Pydantic request/response validation
- **Six-Page Streamlit Dashboard** — strategy optimizer, tyre degradation viewer, energy/OOM analyzer, live monitor, season overview, and model accuracy tracker
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
f1_2026_optimizer/
├── app.py                        # Streamlit entry point
├── main.py                       # FastAPI entry point
├── config.py                     # 2026 regulations, all 22 circuits, teams, tyre compounds
├── requirements.txt
│
├── models/                       # Core simulation engine (~1,500 lines)
│   ├── monte_carlo.py            # Monte Carlo optimizer (2,000-run stochastic engine)
│   ├── tyre.py                   # Tyre degradation model (linear + cliff phases)
│   ├── ers.py                    # ERS/OOM model (50/50 hybrid, Super Clip, Lift-off)
│   ├── strategy.py               # Strategy generation & 2026 rules validation
│   ├── safety_car.py             # SC/VSC deployment model
│   ├── laptime.py                # Per-lap time assembly
│   └── __init__.py
│
├── ui/                           # Streamlit dashboard pages (~2,200 lines)
│   ├── app.py                    # Page router
│   ├── strategy_simulator.py     # Main strategy optimizer
│   ├── tyre_viewer.py            # Tyre degradation curves
│   ├── oom_analyzer.py           # Overtake Override Mode analyzer
│   ├── live_dashboard.py         # Real-time race monitor
│   ├── season_overview.py        # 22-round season calendar
│   ├── accuracy_tracker.py       # Prediction vs actual results
│   ├── styles.py                 # Custom dark F1 theme (CSS)
│   └── __init__.py
│
├── api/
│   ├── routes.py                 # REST API endpoints
│   └── __init__.py
│
├── data/
│   ├── fastf1_loader.py          # FastF1 API integration + SQLite caching
│   ├── calibration/
│   │   ├── calibration_loader.py # Loads 2026 ground-truth stint data
│   │   ├── deg_curves.json       # Tyre degradation per circuit
│   │   ├── pit_loss_2026.json    # Pit stop time loss per circuit
│   │   └── sc_history.json       # Safety car probability history
│   └── live/                     # Cached live race lap data
│
└── tests/                        # ~770 lines of test coverage
    ├── test_monte_carlo.py
    ├── test_tyre.py
    ├── test_oom.py
    ├── test_calibration.py
    └── test_live.py
```

---

## Getting Started

### Prerequisites

- Python 3.11 or higher
- pip

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-username/f1_2026_optimizer.git
cd f1_2026_optimizer

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

**Current accuracy: 3/3 (100%) on completed rounds** — strategy stop-count prediction.

---

## Screenshots

| Page | Preview |
|------|---------|
| Strategy Optimizer | ![Strategy Optimizer](docs/screenshots/strategy_optimizer.png) |
| Tyre Degradation Curves | ![Tyre Viewer](docs/screenshots/tyre_viewer.png) |
| Live Race Monitor | ![Live Dashboard](docs/screenshots/live_dashboard.png) |
| Energy & OOM Analyzer | ![OOM Analyzer](docs/screenshots/oom_analyzer.png) |
| Season Overview | ![Season Overview](docs/screenshots/season_overview.png) |
| Accuracy Tracker | ![Accuracy Tracker](docs/screenshots/accuracy_tracker.png) |

*Populate `docs/screenshots/` with captures from the running app to fill in this section.*

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
