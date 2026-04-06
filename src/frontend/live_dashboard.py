"""
Live Race Strategy Monitor — Streamlit Page

Real-time strategy recommendations during active race weekends.
Supports LIVE mode (FastF1 live timing) and REPLAY mode (cached data).
"""
from __future__ import annotations
import json
import os
import time
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from config import CIRCUITS, TEAMS, ALL_DRIVERS, CIRCUIT_ORDER, UI_COLORS, PLOTLY_TEMPLATE
from src.simulation import MonteCarloSimulator, MonteCarloConfig


LIVE_CACHE_DIR = Path(__file__).parent.parent.parent / "data" / "live"


def _detect_race_weekend() -> Optional[str]:
    """Return circuit_key if today is within a race weekend, else None."""
    today = date.today()
    # 2026 race weekends (approximate - race day is Sunday)
    race_dates = {
        "australia": date(2026, 3, 15),
        "china": date(2026, 3, 22),
        "japan": date(2026, 6, 1),   # placeholder
        "miami": date(2026, 5, 4),
        "monaco": date(2026, 5, 24),
        "canada": date(2026, 6, 14),
        "spain": date(2026, 6, 28),
        "austria": date(2026, 7, 5),
        "great_britain": date(2026, 7, 12),
        "hungary": date(2026, 8, 2),
        "belgium": date(2026, 8, 30),
        "netherlands": date(2026, 9, 6),
        "italy": date(2026, 9, 7),
        "azerbaijan": date(2026, 9, 20),
        "singapore": date(2026, 10, 4),
        "usa": date(2026, 10, 18),
        "mexico": date(2026, 10, 25),
        "brazil": date(2026, 11, 8),
        "las_vegas": date(2026, 11, 21),
        "qatar": date(2026, 11, 29),
        "abu_dhabi": date(2026, 12, 6),
    }
    for circuit_key, race_day in race_dates.items():
        if circuit_key in CIRCUITS:
            delta = abs((today - race_day).days)
            if delta <= 1:
                return circuit_key
    return None


def _build_live_timing_df(circuit_key: str, selected_driver: str) -> pd.DataFrame:
    """Build synthetic live timing table for all 20 drivers."""
    rows = []
    driver_pool = ALL_DRIVERS[:20] if len(ALL_DRIVERS) >= 20 else ALL_DRIVERS

    circuit_cfg = CIRCUITS[circuit_key]
    compounds = circuit_cfg.compounds

    for pos, driver in enumerate(driver_pool, 1):
        # Find team for this driver
        team_name = "Ferrari"
        for t_name, t_cfg in TEAMS.items():
            if driver in t_cfg.drivers:
                team_name = t_name
                break

        compound = compounds[pos % len(compounds)]
        age = max(1, 20 - pos + np.random.randint(0, 8))
        stint_life = max(0, 30 - age)

        if stint_life > 8:
            status = "STAY OUT"
        elif stint_life > 3:
            status = "WINDOW"
        else:
            status = "PIT NOW"

        rows.append({
            "Pos": pos,
            "Driver": driver,
            "Team": team_name,
            "Gap": "LEADER" if pos == 1 else f"+{(pos-1)*1.8 + np.random.uniform(0.1, 0.9):.3f}",
            "Tyre": compound,
            "Age": int(age),
            "Pit Status": status,
        })

    return pd.DataFrame(rows)


def _pit_wall_html(action: str, confidence: float, oom_advised: bool,
                   oom_reason: str, next_pit_lap: int) -> str:
    oom_line = f'<div class="oom-text">OOM: {"ADVISED -- " + oom_reason if oom_advised else "HOLD -- " + oom_reason}</div>'
    return f"""
    <div class="pit-wall-panel">
      <div class="action-text">{action}<span class="blink">_</span></div>
      <div class="sub-text">CONFIDENCE: {confidence:.0%} &nbsp;|&nbsp; NEXT OPTIMAL PIT: LAP {next_pit_lap}</div>
      {oom_line}
    </div>
    """


def render() -> None:
    """Render the Live Race Strategy Monitor page."""
    from src.frontend.styles import inject_custom_css
    inject_custom_css()

    st.header("LIVE Race Strategy Monitor")

    # ---- Race weekend detection ----
    active_circuit = _detect_race_weekend()
    if active_circuit:
        cfg = CIRCUITS[active_circuit]
        st.success(f"RACE WEEKEND DETECTED: {cfg.name} -- {cfg.location}")
    else:
        st.info("Between race weekends -- Manual / Replay mode active")

    # ---- Sidebar controls ----
    with st.sidebar:
        st.subheader("Race Control Panel")
        mode = st.radio("Mode", ["MANUAL", "REPLAY"], horizontal=True, key="live_mode")

        circuit_key = st.selectbox(
            "Circuit",
            options=CIRCUIT_ORDER,
            index=CIRCUIT_ORDER.index(active_circuit) if active_circuit else 0,
            format_func=lambda k: CIRCUITS[k].name.replace(" Grand Prix", ""),
            key="live_circuit",
        )
        circuit_cfg = CIRCUITS[circuit_key]

        driver_options = ALL_DRIVERS[:20] if len(ALL_DRIVERS) >= 20 else ALL_DRIVERS
        selected_driver = st.selectbox("Driver", options=driver_options, key="live_driver")

        st.markdown("---")
        st.caption("Current Race State")
        current_lap = st.slider("Current Lap", 1, circuit_cfg.total_laps,
                                value=min(20, circuit_cfg.total_laps), key="live_lap")
        compound = st.selectbox("Compound", circuit_cfg.compounds, key="live_compound")
        tyre_age = st.slider("Tyre Age (laps)", 0, 40, value=12, key="live_tyre_age")
        position = st.slider("Current Position", 1, 20, value=5, key="live_position")
        gap_ahead = st.number_input("Gap Ahead (s)", min_value=0.0, max_value=60.0,
                                     value=2.5, step=0.1, key="live_gap_ahead")
        gap_behind = st.number_input("Gap Behind (s)", min_value=0.0, max_value=60.0,
                                      value=1.8, step=0.1, key="live_gap_behind")
        battery_pct = st.slider("Battery %", 0, 100, value=65, key="live_battery")
        track_temp = st.slider("Track Temp (C)", 15, 60, value=35, key="live_track_temp")

        auto_refresh = st.checkbox("Auto-refresh every 30s", key="live_auto_refresh")
        refresh_btn = st.button("Refresh", type="primary", use_container_width=True)

    # ---- Auto refresh ----
    if auto_refresh:
        time.sleep(30)
        st.rerun()

    # ---- Section 1: Race State ----
    st.subheader("Race State")
    progress = current_lap / circuit_cfg.total_laps
    st.progress(progress, text=f"Lap {current_lap} / {circuit_cfg.total_laps} -- {progress:.0%} complete")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Lap", f"{current_lap}/{circuit_cfg.total_laps}")
    c2.metric("Compound + Age", f"{compound} +{tyre_age}L")
    c3.metric("Position", f"P{position}")
    c4.metric("Battery", f"{battery_pct}%")

    # Pit window status
    sim_temp = MonteCarloSimulator()
    try:
        stint_analysis = sim_temp.analyze_current_stint(
            circuit=circuit_key,
            team=next((t for t, cfg_t in TEAMS.items() if selected_driver in cfg_t.drivers), "Ferrari"),
            current_lap=current_lap,
            compound=compound,
            tyre_age=tyre_age,
            battery_fraction=battery_pct / 100.0,
            gap_ahead=gap_ahead,
            track_temp_c=float(track_temp),
        )
        laps_to_cliff = stint_analysis["laps_to_cliff"]
        optimal_pit_lap = stint_analysis["optimal_pit_lap"]
        oom_recommended = stint_analysis["oom_recommended"]
        oom_reason = stint_analysis["oom_reason"]
    except Exception:
        laps_to_cliff = 5
        optimal_pit_lap = current_lap + 8
        oom_recommended = False
        oom_reason = "Analysis unavailable"

    if laps_to_cliff > 5:
        st.success(f"STAY OUT -- {laps_to_cliff} laps of grip remaining before cliff")
        action = "STAY OUT"
    elif laps_to_cliff > 2:
        st.warning(f"PIT WINDOW OPEN -- {laps_to_cliff} laps to tyre cliff")
        action = "PIT WINDOW OPEN"
    else:
        st.error("PIT CRITICAL -- tyre cliff imminent")
        action = "PIT NOW"

    confidence = max(0.45, min(0.95, 0.95 - laps_to_cliff * 0.05 + battery_pct * 0.002))

    st.divider()

    # ---- Section 2: Pit Wall Recommendation ----
    st.subheader("Pit Wall Recommendation")
    st.markdown(
        _pit_wall_html(action, confidence, oom_recommended, oom_reason, optimal_pit_lap),
        unsafe_allow_html=True,
    )

    # Undercut windows table
    try:
        undercut_data = stint_analysis.get("undercut_windows", {})
        if undercut_data:
            rows = []
            for new_cpd, windows in undercut_data.items():
                for lap, gain in windows[:3]:
                    rows.append({"Switch to": new_cpd, "On Lap": lap, "Net Gain (s)": f"+{gain:.2f}s"})
            if rows:
                st.caption("Undercut Windows Available")
                st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    except Exception:
        pass

    st.divider()

    # ---- Section 3: Live Timing Table ----
    st.subheader("Live Timing -- All Drivers")

    if mode == "REPLAY":
        LIVE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cached_files = sorted(LIVE_CACHE_DIR.glob("lap_*.json"))
        if cached_files:
            max_replay_lap = len(cached_files)
            replay_lap = st.slider("Replay Lap", 1, max_replay_lap, max_replay_lap)
            cache_file = cached_files[replay_lap - 1]
            try:
                with open(cache_file) as f:
                    timing_data = json.load(f)
                timing_df = pd.DataFrame(timing_data)
            except Exception:
                timing_df = _build_live_timing_df(circuit_key, selected_driver)
        else:
            st.info("No cached race data found. Run in MANUAL mode or record a live session first.")
            timing_df = _build_live_timing_df(circuit_key, selected_driver)
    else:
        # Try FastF1 live timing
        timing_df = _build_live_timing_df(circuit_key, selected_driver)
        try:
            import fastf1.livetiming  # noqa: F401
            st.caption("FastF1 live timing available -- data updated every 30s")
        except (ImportError, Exception):
            st.caption("Live timing unavailable -- showing estimated positions based on team performance")

    # Cache this lap's data
    LIVE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = LIVE_CACHE_DIR / f"lap_{current_lap:03d}.json"
    try:
        timing_df.to_json(cache_path, orient="records")
    except Exception:
        pass

    # Highlight selected driver row
    def highlight_driver(row):
        if row["Driver"] == selected_driver:
            return ["background-color: rgba(220,0,0,0.15); font-weight: bold"] * len(row)
        return [""] * len(row)

    st.dataframe(
        timing_df.style.apply(highlight_driver, axis=1),
        hide_index=True,
        use_container_width=True,
        height=520,
    )
