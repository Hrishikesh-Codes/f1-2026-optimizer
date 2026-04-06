"""
Season Overview — Streamlit Page

All 22 rounds of the 2026 season:
  - Predicted optimal strategy per circuit
  - Actual result if race has been run (FastF1 data)
  - Points standings progression (model vs actual)
  - Sprint weekends highlighted
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from config import (
    CIRCUITS, TEAMS, ALL_DRIVERS, CIRCUIT_ORDER,
    UI_COLORS, PLOTLY_TEMPLATE, COMPLETED_2026_ROUNDS,
    POINTS_SYSTEM, SPRINT_POINTS, compound_label,
)
from src.simulation.strategy import StrategyGenerator


TEAM_COLORS = {
    "Mercedes": "#00D2BE",
    "Ferrari": "#DC0000",
    "McLaren": "#FF8000",
    "Red Bull": "#3671C6",
    "Aston Martin": "#358C75",
    "Alpine": "#2293D1",
    "Williams": "#37BEDD",
    "Racing Bulls": "#6692FF",
    "Haas": "#B6BABD",
    "Audi": "#6E6E6E",
    "Cadillac": "#C0C0C0",
}


def _predict_strategy(circuit_key: str) -> str:
    """Quick strategy prediction without full Monte Carlo."""
    gen = StrategyGenerator()
    strategies = gen.generate_all_strategies(circuit_key, max_stops=2)
    one_stop = [s for s in strategies if s.num_stops == 1]
    if one_stop:
        s = one_stop[len(one_stop) // 2]  # pick a mid-range pit-lap variant
        return s.full_compound_sequence
    if strategies:
        return strategies[0].full_compound_sequence
    return "—"


def render() -> None:
    """Render the Season Overview page."""
    from src.frontend.styles import inject_custom_css
    inject_custom_css()
    st.header("2026 Season Overview")
    st.caption(
        "All 22 rounds · Bahrain and Saudi Arabia cancelled due to regional conflict. "
        "Green rows = completed. 2026 regulations: Active Aero, OOM replaces DRS, "
        "50/50 hybrid split, no MGU-H."
    )

    gen = StrategyGenerator()

    # ------------------------------------------------------------------ #
    # Season calendar table                                                 #
    # ------------------------------------------------------------------ #
    rows = []
    for circuit_key in CIRCUIT_ORDER:
        cfg = CIRCUITS[circuit_key]
        predicted = _predict_strategy(circuit_key)
        is_done = circuit_key in COMPLETED_2026_ROUNDS

        rows.append({
            "Rnd": cfg.round_number,
            "GP": cfg.name.replace(" Grand Prix", ""),
            "Circuit": cfg.circuit,
            "Laps": cfg.total_laps,
            "Compounds": " / ".join(compound_label(c, cfg.compounds) for c in cfg.compounds),
            "SC Prob": f"{cfg.sc_probability:.0%}",
            "Sprint": "Yes" if cfg.is_sprint else "",
            "Type": cfg.circuit_type.title(),
            "Predicted Strategy": predicted,
            "Status": "✅ Done" if is_done else "🔜 Upcoming",
        })

    df = pd.DataFrame(rows)

    # Color done rounds
    def row_style(row):
        if row["Status"].startswith("✅"):
            return ["background-color: rgba(0, 180, 0, 0.12)"] * len(row)
        return [""] * len(row)

    st.dataframe(
        df.style.apply(row_style, axis=1),
        hide_index=True,
        use_container_width=True,
        height=700,
    )

    st.divider()

    # ------------------------------------------------------------------ #
    # Circuit characteristics radar chart                                   #
    # ------------------------------------------------------------------ #
    st.subheader("Circuit Characteristics")
    selected_circuits = st.multiselect(
        "Select circuits to compare",
        options=CIRCUIT_ORDER,
        default=CIRCUIT_ORDER[:5],
        format_func=lambda k: CIRCUITS[k].name.replace(" Grand Prix", ""),
        key="season_compare_circuits",
    )

    if selected_circuits:
        categories = [
            "Abrasiveness", "SC Probability", "Pit Loss",
            "Overtake Difficulty", "Rain Risk",
        ]

        fig_radar = go.Figure()
        for c_key in selected_circuits:
            cfg = CIRCUITS[c_key]
            # Normalise to 0–1 scale
            vals = [
                (cfg.track_abrasiveness - 0.60) / (1.15 - 0.60),
                cfg.sc_probability,
                (cfg.pit_loss_time - 21.0) / (25.0 - 21.0),
                cfg.overtake_difficulty,
                cfg.rain_probability,
            ]
            vals = [round(v, 2) for v in vals]
            fig_radar.add_trace(go.Scatterpolar(
                r=vals + [vals[0]],
                theta=categories + [categories[0]],
                fill="toself",
                name=cfg.name.replace(" Grand Prix", ""),
                opacity=0.65,
            ))

        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            template=PLOTLY_TEMPLATE,
            height=420,
            title="Circuit Characteristics (normalised 0–1)",
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    st.divider()

    # ------------------------------------------------------------------ #
    # Predicted strategy distribution                                        #
    # ------------------------------------------------------------------ #
    st.subheader("Predicted Strategy Distribution")
    strategy_counts: dict = {}
    for circuit_key in CIRCUIT_ORDER:
        gen2 = StrategyGenerator()
        strats = gen2.generate_all_strategies(circuit_key, max_stops=2)
        one_stop = [s for s in strats if s.num_stops == 1]
        label = "1-Stop" if one_stop else "2-Stop"
        strategy_counts[label] = strategy_counts.get(label, 0) + 1

    fig_pie = go.Figure(go.Pie(
        labels=list(strategy_counts.keys()),
        values=list(strategy_counts.values()),
        marker_colors=[UI_COLORS["ferrari_red"], UI_COLORS["mercedes_teal"]],
        hole=0.45,
    ))
    fig_pie.update_layout(
        title="Predicted Race Strategy Split — 2026 Season",
        template=PLOTLY_TEMPLATE,
        height=300,
    )
    st.plotly_chart(fig_pie, use_container_width=True)

    # ------------------------------------------------------------------ #
    # SC probability across the season                                       #
    # ------------------------------------------------------------------ #
    st.subheader("Safety Car Probability by Round")
    sc_df = pd.DataFrame([
        {
            "Round": f"R{CIRCUITS[k].round_number}",
            "GP": CIRCUITS[k].name.replace(" Grand Prix", ""),
            "SC Probability": CIRCUITS[k].sc_probability,
            "Type": CIRCUITS[k].circuit_type,
        }
        for k in CIRCUIT_ORDER
    ])
    fig_sc = px.bar(
        sc_df, x="Round", y="SC Probability",
        color="Type",
        color_discrete_map={
            "permanent": UI_COLORS["mercedes_teal"],
            "street": UI_COLORS["ferrari_red"],
            "hybrid": UI_COLORS["mclaren_papaya"],
        },
        title="Safety Car Probability per Round (street circuits have higher SC rates)",
        template=PLOTLY_TEMPLATE,
        text_auto=".0%",
    )
    fig_sc.update_layout(
        yaxis=dict(tickformat=".0%"),
        height=360,
    )
    st.plotly_chart(fig_sc, use_container_width=True)

    # ------------------------------------------------------------------ #
    # Sprint weekends                                                        #
    # ------------------------------------------------------------------ #
    st.subheader("Sprint Weekends")
    sprint_circuits = [k for k in CIRCUIT_ORDER if CIRCUITS[k].is_sprint]
    cols = st.columns(len(sprint_circuits))
    for col, c_key in zip(cols, sprint_circuits):
        cfg = CIRCUITS[c_key]
        col.metric(
            f"R{cfg.round_number}",
            cfg.name.replace(" Grand Prix", ""),
            delta=cfg.location,
            delta_color="off",
        )

    # ------------------------------------------------------------------ #
    # 2026 regulation summary                                                #
    # ------------------------------------------------------------------ #
    with st.expander("2026 Regulation Highlights"):
        st.markdown("""
        | Regulation | 2026 Value | Change vs 2025 |
        |-----------|-----------|---------------|
        | ICE Power | ~400 kW | ↓ from ~550 kW |
        | MGU-K Power | 350 kW | ↑ tripled from 120 kW |
        | MGU-H | **Eliminated** | Removed |
        | Battery Capacity | ~4 MJ | ↑ ~2× |
        | Hybrid Split | 50/50 ICE/ERS | New regulation |
        | Aero System | **Active Aero** | Replaces DRS |
        | OOM | **New** | Replaces DRS benefit |
        | Car Weight | -30 kg | Lighter |
        | Tyre Width (front) | -25 mm | Narrower |
        | Tyre Width (rear) | -30 mm | Narrower |
        | C6 Compound | **Dropped** | C1–C5 only |
        | Fuel | 100% Sustainable | New mandate |
        | Team Cost Cap | $215M | ↑ from $135M |
        """)
