"""
F1 2026 Race Strategy Optimizer — Streamlit Application

Run with:
    streamlit run app.py

Pages:
  1. Strategy Simulator    — Monte Carlo optimization
  2. Tyre Degradation      — C1-C5 deg curves + FastF1 overlay
  3. Energy & OOM          — ERS strategy + Overtake Override Mode
  4. LIVE Monitor          — Real-time race strategy recommendations
  5. Season Overview       — All 22 rounds
  6. Accuracy Tracker      — Model prediction accuracy vs actual results
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Ensure project root is on path so imports resolve correctly
sys.path.insert(0, str(Path(__file__).parent))

from config import UI_COLORS, CURRENT_SEASON
from src.frontend.styles import inject_custom_css


# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="F1 2026 Strategy Optimizer",
    page_icon="🏎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Global CSS — inject the F1 dark theme
# ---------------------------------------------------------------------------
inject_custom_css()

# ---------------------------------------------------------------------------
# Branded header banner
# ---------------------------------------------------------------------------
st.markdown(f"""
<div style="
    background: linear-gradient(135deg, #0A0A0A 0%, #1A0000 50%, #0A0A0A 100%);
    border-bottom: 2px solid {UI_COLORS['ferrari_red']};
    padding: 1rem 2rem;
    margin: -1.5rem -1rem 1.5rem -1rem;
    text-align: center;
">
    <h1 style="
        color: {UI_COLORS['ferrari_red']};
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.2rem;
        font-weight: 700;
        letter-spacing: 4px;
        margin: 0;
        text-transform: uppercase;
    ">F1 2026 STRATEGY OPTIMIZER</h1>
    <p style="
        color: {UI_COLORS['text_secondary']};
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.85rem;
        letter-spacing: 2px;
        margin: 0.3rem 0 0 0;
        text-transform: uppercase;
    ">Monte Carlo Race Strategy Engine &middot; Active Aero &middot; OOM &middot; 50/50 Hybrid</p>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(f"""
    <div style='text-align: center; padding: 1rem 0;'>
        <h1 style='color: {UI_COLORS["ferrari_red"]}; font-size: 2rem; margin: 0;'>F1</h1>
        <p style='color: {UI_COLORS["text_secondary"]}; margin: 0; font-size: 0.85rem;'>
            {CURRENT_SEASON} Strategy Optimizer
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    page = st.radio(
        "Navigation",
        options=[
            "Strategy Simulator",
            "Tyre Degradation",
            "Energy & OOM",
            "LIVE Monitor",
            "Season Overview",
            "Accuracy Tracker",
        ],
        key="nav_page",
    )

    st.divider()

    with st.expander("About"):
        st.markdown(f"""
        **F1 {CURRENT_SEASON} Race Strategy Optimizer**

        Encodes the full 2026 technical regulations:
        - 50/50 ICE/ERS hybrid split
        - Active Aerodynamics (replaces DRS)
        - Overtake Override Mode
        - Pirelli C1-C5 tyres (C6 dropped)
        - No MGU-H
        - 22-round season calendar

        **Monte Carlo simulation**: 2000 runs sampling SC timing,
        degradation variance +/-10%, and track temperature.

        **Data**: FastF1 integration with SQLite caching.
        Historical calibration: 2022-2025.

        Built with: Python, FastAPI, Streamlit, NumPy/SciPy, Plotly
        """)

    st.caption(f"Season {CURRENT_SEASON} | Active Aero | OOM | 50/50 Hybrid")


# ---------------------------------------------------------------------------
# Page routing
# ---------------------------------------------------------------------------
from src.frontend.strategy_simulator import render as render_strategy
from src.frontend.tyre_viewer import render as render_tyre
from src.frontend.stint_calculator import render as render_stint
from src.frontend.season_overview import render as render_season
from src.frontend.oom_analyzer import render as render_oom
from src.frontend.live_dashboard import render as render_live
from src.frontend.accuracy_tracker import render as render_accuracy

if "Strategy Simulator" in page:
    render_strategy()
elif "Tyre Degradation" in page:
    render_tyre()
elif "Energy & OOM" in page:
    render_oom()
elif "LIVE Monitor" in page:
    render_live()
elif "Season Overview" in page:
    render_season()
elif "Accuracy Tracker" in page:
    render_accuracy()
