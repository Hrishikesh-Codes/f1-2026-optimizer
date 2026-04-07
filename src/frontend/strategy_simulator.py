"""
Race Strategy Simulator — Streamlit Page

Allows users to:
  - Select circuit, team, driver, starting compound
  - Configure track temperature and SC probability override
  - Run Monte Carlo simulation (1000+ runs)
  - View ranked strategy table with confidence intervals
  - View SC impact analysis (how does strategy change at lap 10/20/30/40?)
  - View undercut window chart
"""

from __future__ import annotations

import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from config import (
    CIRCUITS, TEAMS, TYRE_COMPOUNDS, CIRCUIT_ORDER, UI_COLORS, PLOTLY_TEMPLATE,
    compound_label, compound_label_with_code,
)
from src.simulation import MonteCarloSimulator, MonteCarloConfig, MonteCarloResult


COMPOUND_COLORS = {
    "C1": "#DDDDDD",
    "C2": "#CCCCCC",
    "C3": "#FFD700",
    "C4": "#FF4444",
    "C5": "#DC0000",
}


def _fmt_time(seconds: float) -> str:
    """Format seconds as M:SS.mmm."""
    m = int(seconds // 60)
    s = seconds % 60
    return f"{m}:{s:06.3f}"


def _label_sequence(sequence: str, compounds: list) -> str:
    """Replace compound codes in a display string with Hard/Medium/Soft labels.

    e.g. "C4 (28L) → C5 (30L)" → "Soft (28L) → Soft (30L)"
    """
    result = sequence
    # Sort longest codes first to avoid partial replacement issues
    for code in sorted(set(compounds), key=lambda c: -len(c)):
        lbl = compound_label(code, compounds)
        result = result.replace(code, lbl)
    return result


def render() -> None:
    """Render the Race Strategy Simulator page."""
    from src.frontend.styles import inject_custom_css
    inject_custom_css()
    st.header("Race Strategy Simulator")
    st.caption(
        "Monte Carlo simulation (1000 runs) sampling SC timing, "
        "deg variance ±10%, and rival strategies to find the optimal race strategy."
    )

    # ------------------------------------------------------------------ #
    # Sidebar controls                                                      #
    # ------------------------------------------------------------------ #
    with st.sidebar:
        st.subheader("Simulation Parameters")

        # Circuit selector
        circuit_labels = {
            k: f"R{CIRCUITS[k].round_number} — {CIRCUITS[k].name}"
            for k in CIRCUIT_ORDER
        }
        circuit_key = st.selectbox(
            "Circuit",
            options=CIRCUIT_ORDER,
            format_func=lambda k: circuit_labels[k],
            key="sim_circuit",
        )
        circuit_cfg = CIRCUITS[circuit_key]

        # Team / driver
        team_name = st.selectbox("Team", options=sorted(TEAMS.keys()), key="sim_team")
        team_cfg = TEAMS[team_name]
        driver = st.selectbox("Driver", options=team_cfg.drivers, key="sim_driver")

        # Starting compound
        compounds = circuit_cfg.compounds
        starting_compound = st.selectbox(
            "Starting Compound",
            options=["Auto (model selects)"] + compounds,
            format_func=lambda c: c if c == "Auto (model selects)"
                else compound_label_with_code(c, compounds),
            key="sim_start_compound",
        )
        start_cpd = None if starting_compound == "Auto (model selects)" else starting_compound

        # Track conditions
        st.markdown("---")
        track_temp = st.slider(
            "Track Temperature (°C)", min_value=15, max_value=60,
            value=35, step=1, key="sim_track_temp",
        )
        sc_override = st.checkbox(
            f"Override SC probability (default {circuit_cfg.sc_probability:.0%})",
            key="sim_sc_override",
        )
        if sc_override:
            sc_prob = st.slider("SC Probability", 0.0, 1.0,
                                value=circuit_cfg.sc_probability, step=0.05,
                                key="sim_sc_prob")
        else:
            sc_prob = circuit_cfg.sc_probability

        num_sims = st.select_slider(
            "Simulations", options=[200, 500, 1000, 2000],
            value=1000, key="sim_count",
        )
        max_stops = st.radio("Max Pit Stops", [1, 2, 3], index=1, horizontal=True,
                             key="sim_max_stops")

        run_btn = st.button("▶  Run Simulation", type="primary", use_container_width=True)

    # ------------------------------------------------------------------ #
    # Circuit info card                                                     #
    # ------------------------------------------------------------------ #
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Laps", circuit_cfg.total_laps)
    col2.metric("Pit Loss", f"{circuit_cfg.pit_loss_time:.1f}s")
    col3.metric("SC Probability", f"{sc_prob:.0%}")
    col4.metric("Compounds", " / ".join(
        compound_label(c, circuit_cfg.compounds) for c in circuit_cfg.compounds
    ))

    # ------------------------------------------------------------------ #
    # Run simulation                                                        #
    # ------------------------------------------------------------------ #
    if run_btn:
        with st.spinner(f"Running {num_sims} Monte Carlo simulations…"):
            t0 = time.perf_counter()
            config = MonteCarloConfig(
                num_simulations=num_sims,
                include_sc=True,
                max_stops=int(max_stops),
            )
            sim = MonteCarloSimulator(config)
            try:
                result = sim.run(
                    circuit=circuit_key,
                    team=team_name,
                    driver=driver,
                    track_temp_c=float(track_temp),
                    starting_compound=start_cpd,
                )
                st.session_state["sim_result"] = result
                elapsed = time.perf_counter() - t0
                st.success(f"Simulation complete in {elapsed:.1f}s — {num_sims} runs evaluated.")
            except Exception as exc:
                st.error(f"Simulation error: {exc}")
                return

    # ------------------------------------------------------------------ #
    # Display results                                                       #
    # ------------------------------------------------------------------ #
    result: MonteCarloResult | None = st.session_state.get("sim_result")
    if result is None or result.circuit != circuit_key:
        st.info("Configure parameters above and click **Run Simulation** to begin.")
        return

    st.divider()

    # --- Optimal strategy banner ---
    opt = result.optimal_strategy
    ci_lo, ci_hi = result.confidence_interval_95
    st.subheader("Optimal Strategy")
    bcol1, bcol2, bcol3, bcol4 = st.columns(4)
    bcol1.metric("Strategy", "-".join(
        compound_label(s.compound, circuit_cfg.compounds)[0]
        for s in opt.stints
    ))
    bcol2.metric("Stops", opt.num_stops)
    bcol3.metric("Pit Laps", ", ".join(str(l) for l in opt.pit_laps) or "—")
    bcol4.metric("Expected Race Time", _fmt_time(result.optimal_strategy_mean_time))

    cpds = circuit_cfg.compounds
    st.caption(
        f"95% CI: {_fmt_time(ci_lo)} – {_fmt_time(ci_hi)}  |  "
        f"Std Dev: {result.optimal_strategy_std_time:.2f}s  |  "
        f"Strategy: {_label_sequence(opt.full_compound_sequence, cpds)}"
    )

    # --- Strategy comparison table ---
    st.subheader("Strategy Comparison")
    all_rows = []
    all_rows.append({
        "Rank": 1,
        "Sequence": _label_sequence(opt.full_compound_sequence, cpds),
        "Stops": opt.num_stops,
        "Pit Laps": ", ".join(str(l) for l in opt.pit_laps),
        "Mean Time": _fmt_time(result.optimal_strategy_mean_time),
        "Std Dev (s)": f"{result.optimal_strategy_std_time:.2f}",
        "Δ vs Optimal": "—",
        "Tag": "✅ Optimal",
    })
    for rank, (strat, mean_t, std_t) in enumerate(result.alternative_strategies, start=2):
        delta = mean_t - result.optimal_strategy_mean_time
        all_rows.append({
            "Rank": rank,
            "Sequence": _label_sequence(strat.full_compound_sequence, cpds),
            "Stops": strat.num_stops,
            "Pit Laps": ", ".join(str(l) for l in strat.pit_laps),
            "Mean Time": _fmt_time(mean_t),
            "Std Dev (s)": f"{std_t:.2f}",
            "Δ vs Optimal": f"+{delta:.2f}s",
            "Tag": "🔄 Alternative",
        })

    df_strats = pd.DataFrame(all_rows)
    st.table(df_strats)

    # --- Strategy win distribution ---
    st.subheader("Strategy Win Distribution (across all simulations)")
    if result.strategy_win_distribution:
        win_df = pd.DataFrame([
            {"Strategy": _label_sequence(k, cpds), "Wins": v}
            for k, v in sorted(result.strategy_win_distribution.items(),
                               key=lambda x: -x[1])
        ])
        fig_wins = px.bar(
            win_df, x="Strategy", y="Wins",
            title="How often each strategy was optimal across simulations",
            template=PLOTLY_TEMPLATE,
            color_discrete_sequence=[UI_COLORS["ferrari_red"]],
        )
        fig_wins.update_layout(xaxis_tickangle=-30)
        st.plotly_chart(fig_wins, use_container_width=True)

    # --- SC impact analysis ---
    st.subheader("Safety Car Impact Analysis")
    st.caption(
        "Strategic time gain/loss vs the field when SC deploys at each lap. "
        "Negative (red) = SC helps — your pit window aligns with the free-pit window. "
        "Positive (grey) = SC hurts — rivals can pit free while you cannot."
    )
    if result.sc_impact_analysis:
        sc_rows = [
            {"SC Lap": f"Lap {lap}", "Time Delta (s)": round(delta, 2)}
            for lap, delta in sorted(result.sc_impact_analysis.items())
        ]
        sc_df = pd.DataFrame(sc_rows)
        fig_sc = go.Figure()
        colors = [
            UI_COLORS["ferrari_red"] if d < 0 else "#888888"
            for d in sc_df["Time Delta (s)"]
        ]
        vals = sc_df["Time Delta (s)"]
        y_min = min(vals.min() * 1.3 if vals.min() < 0 else -5, -5)
        y_max = max(vals.max() * 1.3 if vals.max() > 0 else 5, 5)
        fig_sc.add_trace(go.Bar(
            x=sc_df["SC Lap"],
            y=vals,
            marker_color=colors,
            text=[f"{v:+.1f}s" for v in vals],
            textposition="auto",
            textfont=dict(size=13, color="white"),
        ))
        fig_sc.update_layout(
            title="SC Strategic Impact (negative = SC helps this strategy)",
            yaxis_title="Δ Time vs Field (s)",
            yaxis=dict(range=[y_min, y_max], zeroline=True, zerolinecolor="white", zerolinewidth=1),
            template=PLOTLY_TEMPLATE,
            height=380,
            margin=dict(t=50, b=40),
        )
        st.plotly_chart(fig_sc, use_container_width=True)

    # --- Undercut windows ---
    st.subheader("Undercut Window")
    if result.undercut_windows:
        uc_df = pd.DataFrame([
            {"Race Lap": lap, "Net Time Gain (s)": round(gain, 2)}
            for lap, gain in result.undercut_windows
        ])
        fig_uc = go.Figure()
        fig_uc.add_trace(go.Scatter(
            x=uc_df["Race Lap"],
            y=uc_df["Net Time Gain (s)"],
            mode="lines+markers",
            fill="tozeroy",
            line=dict(color=UI_COLORS["ferrari_red"], width=2),
            name="Undercut gain",
        ))
        fig_uc.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.4)
        fig_uc.update_layout(
            title="Undercut Viability (positive = undercutting gains time)",
            xaxis_title="Race Lap",
            yaxis_title="Net Time Gain (s)",
            template=PLOTLY_TEMPLATE,
            height=320,
        )
        st.plotly_chart(fig_uc, use_container_width=True)
    else:
        st.info("No undercut window data — strategy may be 1-stop with no viable undercut.")

    # --- OOM recommendations ---
    st.subheader("Overtake Override Mode (OOM) Lap Recommendations")
    if result.oom_recommendations:
        oom_laps = sorted(result.oom_recommendations.keys())
        st.markdown(
            "**Recommended OOM laps:** " +
            ", ".join(f"Lap {l}" for l in oom_laps[:12])
        )
        with st.expander("Full OOM reasoning per lap"):
            for lap, reason in sorted(result.oom_recommendations.items()):
                st.text(f"Lap {lap:3d}: {reason}")
    else:
        st.info("No OOM recommendations — check stint structure.")
