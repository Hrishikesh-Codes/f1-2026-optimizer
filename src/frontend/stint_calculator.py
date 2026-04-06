"""
Stint Calculator — Streamlit Page

Real-time race engineer tool:
  - Input: current lap, compound, tyre age, battery level, gap ahead
  - Output: laps to cliff, optimal pit window, undercut viability, OOM decision
  - Live-updating as sliders change (no button press needed)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import (
    CIRCUITS, TEAMS, TYRE_COMPOUNDS, UI_COLORS, PLOTLY_TEMPLATE,
    CIRCUIT_ORDER, ERS_PARAMS, compound_label,
)
from src.simulation import TyreDegradationModel, ERSModel, ERSState
from src.simulation.monte_carlo import MonteCarloSimulator


COMPOUND_COLORS = {
    "C1": "#CCCCCC", "C2": "#AAAAAA",
    "C3": "#FFD700", "C4": "#FF6666", "C5": "#DC0000",
}


def _traffic_light(value: bool) -> str:
    return "🟢" if value else "🔴"


def render() -> None:
    """Render the Stint Calculator page."""
    from src.frontend.styles import inject_custom_css
    inject_custom_css()
    st.header("Stint Calculator")
    st.caption(
        "Live race-engineer tool. Adjust sliders to model current race state. "
        "All outputs update in real-time."
    )

    # ------------------------------------------------------------------ #
    # Inputs                                                                #
    # ------------------------------------------------------------------ #
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Race State")

        circuit_key = st.selectbox(
            "Circuit",
            options=CIRCUIT_ORDER,
            format_func=lambda k: f"R{CIRCUITS[k].round_number} — {CIRCUITS[k].name}",
            key="stint_circuit",
        )
        circuit_cfg = CIRCUITS[circuit_key]
        total_laps = circuit_cfg.total_laps

        team_name = st.selectbox("Team", options=sorted(TEAMS.keys()), key="stint_team")

        compound = st.selectbox(
            "Current Compound",
            options=circuit_cfg.compounds,
            format_func=lambda c: f"{compound_label(c, circuit_cfg.compounds)} ({c})",
            key="stint_compound",
        )

        current_lap = st.slider(
            "Current Race Lap", 1, total_laps - 1,
            value=min(25, total_laps - 10), key="stint_lap",
        )
        tyre_age = st.slider(
            "Tyre Age (laps on this set)",
            0, min(40, total_laps),
            value=min(18, current_lap), key="stint_age",
        )

    with col2:
        st.subheader("ERS & Track")

        battery_pct = st.slider(
            "Battery Level (%)", 0, 100, 65, step=5, key="stint_battery"
        )
        battery_frac = battery_pct / 100.0

        gap_ahead = st.slider(
            "Gap to Car Ahead (s)", 0.0, 10.0, 2.5, step=0.1, key="stint_gap"
        )

        track_temp = st.slider(
            "Track Temperature (°C)", 15, 60, 35, key="stint_temp"
        )

        push_level = st.slider(
            "Push Level", 0.0, 1.0, 0.80, step=0.05, key="stint_push"
        )

    st.divider()

    # ------------------------------------------------------------------ #
    # Compute outputs                                                        #
    # ------------------------------------------------------------------ #
    tyre_model = TyreDegradationModel()
    ers_model = ERSModel()
    compound_cfg = TYRE_COMPOUNDS[compound]

    cliff_lap = tyre_model.compute_cliff_lap(
        compound, circuit_cfg.track_abrasiveness, float(track_temp)
    )
    laps_to_cliff = max(0, cliff_lap - tyre_age)
    is_past_cliff = tyre_age >= cliff_lap
    min_l, opt_l, max_l = tyre_model.estimate_optimal_stint_length(
        compound, circuit_cfg.track_abrasiveness, float(track_temp)
    )
    optimal_pit_race_lap = current_lap + max(0, opt_l - tyre_age)
    laps_remaining = total_laps - current_lap

    # ERS state
    ers_state = ERSState(
        battery_level_mj=ERS_PARAMS["battery_capacity_mj"] * battery_frac,
    )
    oom_use, oom_reason = ers_model.compute_oom_decision(
        ers_state=ers_state,
        gap_to_car_ahead=float(gap_ahead),
        laps_remaining=laps_remaining,
        tyre_age=tyre_age,
        compound=compound,
    )

    # Undercut windows for all available alternative compounds
    undercut_data: dict = {}
    for alt_compound in circuit_cfg.compounds:
        if alt_compound == compound:
            continue
        window = tyre_model.compute_undercut_window(
            compound_old=compound,
            compound_new=alt_compound,
            laps_on_old=tyre_age,
            pit_loss_time=circuit_cfg.pit_loss_time,
            abrasiveness=circuit_cfg.track_abrasiveness,
            track_temp_c=float(track_temp),
            push_level=float(push_level),
            window_laps=min(15, laps_remaining),
        )
        undercut_data[alt_compound] = [(current_lap + offset, gain) for offset, gain in window]

    # ------------------------------------------------------------------ #
    # KPI cards                                                             #
    # ------------------------------------------------------------------ #
    k1, k2, k3, k4 = st.columns(4)
    k1.metric(
        "Laps to Cliff",
        f"{laps_to_cliff}" if not is_past_cliff else "PAST CLIFF ⚠️",
        delta=f"Cliff at stint lap {cliff_lap}",
        delta_color="off",
    )
    k2.metric(
        "Optimal Pit Lap",
        f"Lap {min(optimal_pit_race_lap, total_laps - 3)}",
        delta=f"In {max(0, optimal_pit_race_lap - current_lap)} laps",
        delta_color="off",
    )
    k3.metric(
        "Battery Level",
        f"{battery_pct}%  ({ers_state.battery_level_mj:.1f} MJ)",
        delta=f"Capacity: {ERS_PARAMS['battery_capacity_mj']:.1f} MJ",
        delta_color="off",
    )
    k4.metric(
        "OOM",
        f"{_traffic_light(oom_use)} {'USE' if oom_use else 'SAVE'}",
        delta=f"Gap: {gap_ahead:.1f}s (limit: {ERS_PARAMS['oom_detection_gap_s']:.1f}s)",
        delta_color="off",
    )

    st.caption(f"**OOM rationale:** {oom_reason}")

    if is_past_cliff:
        st.warning(
            f"⚠️  Tyre is {tyre_age - cliff_lap} laps past the degradation cliff. "
            "Significant pace loss expected — consider pitting this lap."
        )

    st.divider()

    # ------------------------------------------------------------------ #
    # Remaining stint pace projection                                        #
    # ------------------------------------------------------------------ #
    st.subheader("Remaining Stint Pace Projection")
    laps_ahead = min(laps_remaining, 25)
    future_laps_in_stint = [tyre_age + i for i in range(1, laps_ahead + 1)]
    future_race_laps = [current_lap + i for i in range(1, laps_ahead + 1)]
    future_deltas = [
        tyre_model.compute_lap_time_delta(
            compound, stint_lap,
            circuit_cfg.track_abrasiveness, float(track_temp), float(push_level)
        )
        for stint_lap in future_laps_in_stint
    ]
    base_time = circuit_cfg.base_lap_time + compound_cfg.base_pace_offset
    future_times = [base_time + d for d in future_deltas]

    cliff_marker_laps = [
        rl for rl, sl in zip(future_race_laps, future_laps_in_stint)
        if sl == cliff_lap
    ]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=future_race_laps,
        y=future_times,
        mode="lines+markers",
        name=f"{compound_label(compound, circuit_cfg.compounds)} ({compound}) pace",
        line=dict(color=COMPOUND_COLORS.get(compound, "#FFD700"), width=2.5),
    ))
    if cliff_marker_laps:
        fig.add_vline(
            x=cliff_marker_laps[0], line_dash="dash", line_color="#FF4444",
            annotation_text="Cliff", annotation_position="top right",
        )
    if optimal_pit_race_lap <= future_race_laps[-1]:
        fig.add_vline(
            x=optimal_pit_race_lap, line_dash="dot", line_color="#00FF88",
            annotation_text="Opt Pit", annotation_position="top left",
        )
    fig.update_layout(
        title=f"Projected Lap Times — {compound_label(compound, circuit_cfg.compounds)} ({compound}) from Current State",
        xaxis_title="Race Lap",
        yaxis_title="Lap Time (s)",
        template=PLOTLY_TEMPLATE,
        height=350,
    )
    st.plotly_chart(fig, use_container_width=True)

    # ------------------------------------------------------------------ #
    # Undercut viability chart                                              #
    # ------------------------------------------------------------------ #
    st.subheader("Undercut Viability")
    if undercut_data:
        fig_uc = go.Figure()
        for alt_cpd, window in undercut_data.items():
            if not window:
                continue
            laps_x = [l for l, _ in window]
            gains_y = [g for _, g in window]
            color = COMPOUND_COLORS.get(alt_cpd, "#888888")
            fig_uc.add_trace(go.Scatter(
                x=laps_x, y=gains_y,
                mode="lines+markers",
                name=f"Undercut onto {compound_label(alt_cpd, circuit_cfg.compounds)} ({alt_cpd})",
                line=dict(color=color, width=2),
            ))
        fig_uc.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.5)
        fig_uc.update_layout(
            title="Net Time Gain from Undercutting (positive = undercut pays off)",
            xaxis_title="Pit Lap",
            yaxis_title="Net Time Gain (s)",
            template=PLOTLY_TEMPLATE,
            height=320,
        )
        st.plotly_chart(fig_uc, use_container_width=True)

        # Undercut verdict
        best_uc = max(
            [(cpd, max(g for _, g in w)) for cpd, w in undercut_data.items() if w],
            key=lambda x: x[1],
            default=None,
        )
        if best_uc and best_uc[1] > 0:
            st.success(
                f"✅ **Undercut viable** onto {compound_label(best_uc[0], circuit_cfg.compounds)} ({best_uc[0]}): "
                f"+{best_uc[1]:.2f}s net gain if pitting this lap range."
            )
        else:
            st.info("Undercut currently net negative — track position value outweighs tyre delta.")
    else:
        st.info("Only one compound available — undercut analysis not applicable.")

    # ------------------------------------------------------------------ #
    # ERS deployment table                                                  #
    # ------------------------------------------------------------------ #
    st.subheader("ERS Strategy Options")
    ers_rows = [
        {
            "Mode": "🔋 Super Clipping",
            "Recovery": f"{ERS_PARAMS['super_clip_recovery_mj_per_lap']:.1f} MJ/lap",
            "Aero Penalty": "None",
            "Best For": "Recovering battery without sacrificing straight-line speed",
        },
        {
            "Mode": "🛑 Lift-Off Regen",
            "Recovery": f"{ERS_PARAMS['lift_off_recovery_mj_per_lap']:.1f} MJ/lap",
            "Aero Penalty": f"{ERS_PARAMS['lift_off_aero_time_penalty_s']:.2f}s/lap",
            "Best For": "Rapidly charging battery when position gap is comfortable",
        },
        {
            "Mode": "⚡ Boost Deployment",
            "Recovery": f"-{ERS_PARAMS['boost_cost_mj']:.1f} MJ/lap",
            "Aero Penalty": "—",
            "Best For": f"Attack: +{ERS_PARAMS['boost_lap_time_gain_s']:.2f}s lap-time gain",
        },
        {
            "Mode": "🚀 Overtake Override Mode",
            "Recovery": f"+{ERS_PARAMS['oom_extra_capacity_mj']:.1f} MJ capacity",
            "Aero Penalty": "—",
            "Best For": f"Closing gap (requires ≤{ERS_PARAMS['oom_detection_gap_s']:.1f}s to car ahead)",
        },
    ]
    st.dataframe(pd.DataFrame(ers_rows), hide_index=True, use_container_width=True, height=160)
