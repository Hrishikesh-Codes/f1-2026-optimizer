"""
Tyre Degradation Viewer — Streamlit Page

Shows:
  - Degradation curves for all 3 allocated compounds at selected circuit
  - Cliff lap marker per compound
  - Historical FastF1 data overlay (2022–2025 + 2026 if available)
  - Temperature sensitivity: how deg changes with track temp
  - Compound comparison: pace vs durability trade-off
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from config import (
    CIRCUITS, TYRE_COMPOUNDS, UI_COLORS, PLOTLY_TEMPLATE,
    CIRCUIT_ORDER, HISTORICAL_SEASONS, compound_label,
)
from src.simulation import TyreDegradationModel
from src.data import FastF1Loader


COMPOUND_COLORS = {
    "C1": "#CCCCCC",
    "C2": "#AAAAAA",
    "C3": "#FFD700",
    "C4": "#FF6666",
    "C5": "#DC0000",
}
COMPOUND_DASH = {
    "C1": "dot",
    "C2": "dashdot",
    "C3": "dash",
    "C4": "solid",
    "C5": "solid",
}


def render() -> None:
    """Render the Tyre Degradation Viewer page."""
    from src.frontend.styles import inject_custom_css
    inject_custom_css()
    st.header("Tyre Degradation Viewer")
    st.caption(
        "Pirelli C1–C5 degradation curves for 2026 (narrower tyres, lighter cars = "
        "higher durability vs 2022–2025). Overlays historical FastF1 observed data."
    )

    # ------------------------------------------------------------------ #
    # Controls                                                              #
    # ------------------------------------------------------------------ #
    col1, col2, col3 = st.columns(3)

    with col1:
        circuit_key = st.selectbox(
            "Circuit",
            options=CIRCUIT_ORDER,
            format_func=lambda k: f"R{CIRCUITS[k].round_number} — {CIRCUITS[k].name}",
            key="tyre_circuit",
        )
    circuit_cfg = CIRCUITS[circuit_key]

    with col2:
        track_temp = st.slider(
            "Track Temperature (°C)", 15, 60, 35, key="tyre_temp"
        )
    with col3:
        push_level = st.slider(
            "Push Level", 0.0, 1.0, 0.80, step=0.05, key="tyre_push"
        )

    show_historical = st.checkbox("Show historical FastF1 data overlay", value=True,
                                  key="tyre_historical")

    st.divider()

    compounds = circuit_cfg.compounds
    tyre_model = TyreDegradationModel()

    # ------------------------------------------------------------------ #
    # Main degradation curve chart                                          #
    # ------------------------------------------------------------------ #
    fig = go.Figure()
    max_lap_all = 0

    for compound in compounds:
        cfg = TYRE_COMPOUNDS[compound]
        max_laps = cfg.max_viable_laps
        max_lap_all = max(max_lap_all, max_laps)

        deltas = tyre_model.compute_stint_time_deltas(
            compound=compound,
            num_laps=max_laps,
            track_abrasiveness=circuit_cfg.track_abrasiveness,
            track_temp_celsius=float(track_temp),
            push_level=float(push_level),
        )
        laps = np.arange(1, max_laps + 1)
        cliff_lap = tyre_model.compute_cliff_lap(
            compound, circuit_cfg.track_abrasiveness, float(track_temp)
        )

        color = COMPOUND_COLORS.get(compound, "#888888")
        cpd_label = compound_label(compound, compounds)
        trace_label = f"{cpd_label} ({compound})"

        fig.add_trace(go.Scatter(
            x=laps.tolist(),
            y=deltas.tolist(),
            mode="lines",
            name=trace_label,
            line=dict(color=color, width=2.5, dash=COMPOUND_DASH.get(compound, "solid")),
        ))

        # Cliff marker
        if cliff_lap <= max_laps:
            cliff_delta = float(deltas[cliff_lap - 1]) if cliff_lap <= len(deltas) else 0
            fig.add_trace(go.Scatter(
                x=[cliff_lap],
                y=[cliff_delta],
                mode="markers",
                marker=dict(symbol="x", size=12, color=color),
                name=f"{cpd_label} cliff (lap {cliff_lap})",
                showlegend=True,
            ))

    # Historical overlay
    if show_historical:
        loader = FastF1Loader()
        for compound in compounds:
            hist_df = loader.get_historical_deg_curves(
                circuit_key, compound, seasons=HISTORICAL_SEASONS
            )
            if not hist_df.empty and "mean_delta_s" in hist_df.columns:
                valid = hist_df[hist_df["sample_count"] > 0] if "sample_count" in hist_df.columns else hist_df
                if not valid.empty:
                    color = COMPOUND_COLORS.get(compound, "#888888")
                    fig.add_trace(go.Scatter(
                        x=valid["lap_in_stint"].tolist(),
                        y=valid["mean_delta_s"].tolist(),
                        mode="markers",
                        marker=dict(color=color, size=4, opacity=0.45, symbol="circle"),
                        name=f"{compound} — observed",
                        showlegend=True,
                    ))
                    # Error band
                    if "std_delta_s" in valid.columns:
                        upper = (valid["mean_delta_s"] + valid["std_delta_s"]).tolist()
                        lower = (valid["mean_delta_s"] - valid["std_delta_s"]).tolist()
                        fig.add_trace(go.Scatter(
                            x=valid["lap_in_stint"].tolist() + valid["lap_in_stint"].tolist()[::-1],
                            y=upper + lower[::-1],
                            fill="toself",
                            fillcolor=color.replace("#", "rgba(").rstrip(")") + ",0.10)",
                            line=dict(color="rgba(0,0,0,0)"),
                            showlegend=False,
                            name=f"{compound} ±1σ",
                        ))

    fig.update_layout(
        title=f"Tyre Degradation — {circuit_cfg.name} (Track: {track_temp}°C, Push: {push_level:.0%})",
        xaxis_title="Lap in Stint",
        yaxis_title="Additional Time vs Fresh Tyre (s)",
        template=PLOTLY_TEMPLATE,
        height=480,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ------------------------------------------------------------------ #
    # Compound stats table                                                  #
    # ------------------------------------------------------------------ #
    st.subheader("Compound Summary")
    rows = []
    for compound in compounds:
        cfg = TYRE_COMPOUNDS[compound]
        cliff = tyre_model.compute_cliff_lap(
            compound, circuit_cfg.track_abrasiveness, float(track_temp)
        )
        min_l, opt_l, max_l = tyre_model.estimate_optimal_stint_length(
            compound, circuit_cfg.track_abrasiveness, float(track_temp)
        )
        color = COMPOUND_COLORS.get(compound, "#888888")
        rows.append({
            "Compound": f"<span style='color:{color};font-weight:700'>{compound_label(compound, compounds)} ({compound})</span>",
            "Pace vs Baseline": f"{cfg.base_pace_offset:+.2f}s",
            "Cliff Lap": cliff,
            "Optimal Stint": f"{min_l}–{opt_l} laps",
            "Max Stint": f"{max_l} laps",
            "Deg Rate (s/lap)": f"{cfg.linear_deg_rate:.3f}",
            "Thermal Sensitivity": f"{cfg.thermal_sensitivity:.2f}",
        })
    # Render as HTML table to avoid dark-theme CSS conflicts with st.dataframe
    cols = ["Compound", "Pace vs Baseline", "Cliff Lap",
            "Optimal Stint", "Max Stint", "Deg Rate (s/lap)", "Thermal Sensitivity"]
    header_html = "".join(
        f"<th style='background:#1a1a1a;color:#888888;font-size:11px;text-transform:uppercase;"
        f"letter-spacing:1px;padding:8px 12px;border-bottom:1px solid #333;'>{c}</th>"
        for c in cols
    )
    body_html = ""
    for row in rows:
        cells = "".join(
            f"<td style='padding:8px 12px;border-bottom:1px solid #1e1e1e;color:#ffffff;'>"
            f"{row[c]}</td>"
            for c in cols
        )
        body_html += f"<tr>{cells}</tr>"
    st.markdown(
        f"<div style='overflow-x:auto;background:#141414;border:1px solid #222;"
        f"border-radius:4px;margin:8px 0;'>"
        f"<table style='width:100%;border-collapse:collapse;font-size:13px;"
        f"font-family:monospace;'>"
        f"<thead><tr>{header_html}</tr></thead>"
        f"<tbody>{body_html}</tbody>"
        f"</table></div>",
        unsafe_allow_html=True,
    )

    # ------------------------------------------------------------------ #
    # Temperature sensitivity chart                                         #
    # ------------------------------------------------------------------ #
    st.subheader("Temperature Sensitivity")
    st.caption("How degradation rate changes with track temperature at lap 20.")

    temps = list(range(20, 61, 5))
    fig_temp = go.Figure()
    for compound in compounds:
        deltas_at_lap20 = []
        for t in temps:
            d = tyre_model.compute_lap_time_delta(
                compound, 20, circuit_cfg.track_abrasiveness, float(t), float(push_level)
            )
            deltas_at_lap20.append(d)
        color = COMPOUND_COLORS.get(compound, "#888888")
        fig_temp.add_trace(go.Scatter(
            x=temps,
            y=deltas_at_lap20,
            mode="lines+markers",
            name=f"{compound_label(compound, compounds)} ({compound})",
            line=dict(color=color, width=2),
        ))

    fig_temp.add_vline(
        x=track_temp, line_dash="dash", line_color="white",
        annotation_text=f"Selected: {track_temp}°C",
    )
    fig_temp.update_layout(
        title="Degradation at Lap 20 vs Track Temperature",
        xaxis_title="Track Temperature (°C)",
        yaxis_title="Deg Penalty at Lap 20 (s)",
        template=PLOTLY_TEMPLATE,
        height=350,
    )
    st.plotly_chart(fig_temp, use_container_width=True)

    # ------------------------------------------------------------------ #
    # 2026 regulation callout                                               #
    # ------------------------------------------------------------------ #
    with st.expander("2026 Tyre Regulation Changes"):
        st.markdown("""
        **Key 2026 changes vs 2025:**
        - **Narrower footprint**: Front -25mm, Rear -30mm tread width
        - **Smaller diameter**: Front -15mm, Rear -10mm
        - **C6 dropped**: Only C1–C5 available
        - **Higher durability**: Lighter, shorter cars generate less lateral load →
          cliff laps ~10–15% later than 2025 equivalent compound
        - **Colour coding unchanged**: Hard = white, Medium = yellow, Soft = red
        - **Minimum 2 dry compounds** must be used per race (≥1 pit stop)
        - **13 sets per weekend** (12 for Sprint weekends)
        """)
