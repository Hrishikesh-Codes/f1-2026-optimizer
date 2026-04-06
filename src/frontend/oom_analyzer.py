"""
Overtake Override Mode (OOM) Analyzer — Streamlit Page

Replaces DRS in 2026. Key rules encoded:
  - Triggered when attacker is within 1.0s detection gap at detection line
  - Gives attacker +0.5 MJ extra battery capacity + enhanced power profile
  - Attack only (not available for defense)
  - Modelled as per-lap binary decision: use OOM vs conserve battery

This page provides:
  1. OOM decision tree for any race state (gap, battery, tyre age)
  2. Battery management strategy over a full stint with/without OOM
  3. OOM vs DRS comparison: why 2026 OOM changes overtaking dynamics
  4. Optimal OOM lap recommendation for a given strategy
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from config import (
    CIRCUITS, TYRE_COMPOUNDS, TEAMS, ERS_PARAMS,
    UI_COLORS, PLOTLY_TEMPLATE, CIRCUIT_ORDER,
)
from src.simulation.ers import ERSModel, ERSState


def _simulate_battery_trajectory(
    num_laps: int,
    recharge_mode: str,
    pu_supplier: str,
    circuit: str,
    oom_laps: list,
    boost_laps: list,
    initial_battery: float,
) -> list[dict]:
    """Simulate battery level lap by lap and return trajectory rows."""
    model = ERSModel()
    state = ERSState(
        battery_level_mj=initial_battery,
        battery_capacity_mj=ERS_PARAMS["battery_capacity_mj"],
    )
    rows = []
    for lap in range(1, num_laps + 1):
        gap = 0.5 if lap in oom_laps else 5.0
        use_boost = lap in boost_laps
        delta, state = model.compute_lap_ers_delta(
            state, use_boost, recharge_mode, pu_supplier, circuit, gap
        )
        rows.append({
            "lap": lap,
            "battery_mj": round(state.battery_level_mj, 3),
            "soc_pct": round(state.state_of_charge * 100, 1),
            "ers_delta_s": round(delta, 4),
            "oom_active": lap in oom_laps,
            "boost_active": lap in boost_laps,
        })
    return rows


def render() -> None:
    """Render the OOM Analyzer page."""
    from src.frontend.styles import inject_custom_css
    inject_custom_css()
    st.header("Overtake Override Mode (OOM) Analyzer")
    st.caption(
        "OOM replaces DRS in 2026. Available to the **attacker only** within "
        f"**{ERS_PARAMS['oom_detection_gap_s']:.1f}s** at the detection line. "
        f"Grants +{ERS_PARAMS['oom_extra_capacity_mj']:.1f} MJ extra battery capacity "
        f"and +{ERS_PARAMS['oom_enhanced_power_gain_s']:.2f}s enhanced power per lap."
    )

    # ------------------------------------------------------------------ #
    # Tab layout                                                            #
    # ------------------------------------------------------------------ #
    tab1, tab2, tab3 = st.tabs([
        "Decision Tool", "Battery Trajectory", "OOM vs DRS Comparison"
    ])

    # ================================================================== #
    # TAB 1: Real-time OOM decision                                        #
    # ================================================================== #
    with tab1:
        st.subheader("OOM Decision Tool")
        st.caption("Adjust race state to see whether OOM should be triggered.")

        c1, c2, c3 = st.columns(3)
        with c1:
            gap = st.slider(
                "Gap to Car Ahead (s)", 0.0, 5.0, 0.8, step=0.1, key="oom_gap"
            )
            battery_pct = st.slider(
                "Battery Level (%)", 0, 100, 70, step=5, key="oom_battery"
            )
        with c2:
            compound = st.selectbox(
                "Current Compound", list(TYRE_COMPOUNDS.keys()), index=2,
                key="oom_compound"
            )
            tyre_age = st.slider(
                "Tyre Age (laps)", 0, 45, 15, key="oom_tyre_age"
            )
        with c3:
            laps_remaining = st.slider(
                "Laps Remaining", 1, 78, 20, key="oom_laps_remaining"
            )

        ers_model = ERSModel()
        state = ERSState(
            battery_level_mj=ERS_PARAMS["battery_capacity_mj"] * (battery_pct / 100),
        )
        use_oom, reason = ers_model.compute_oom_decision(
            ers_state=state,
            gap_to_car_ahead=float(gap),
            laps_remaining=int(laps_remaining),
            tyre_age=int(tyre_age),
            compound=compound,
        )

        # Decision display
        st.divider()
        if use_oom:
            st.success(f"🚀 **USE OOM** — {reason}")
        else:
            st.warning(f"🔋 **SAVE OOM** — {reason}")

        # Condition checklist
        st.subheader("Condition Checklist")
        cfg_t = TYRE_COMPOUNDS[compound]
        within_gap = gap <= ERS_PARAMS["oom_detection_gap_s"]
        battery_ok = state.battery_level_mj >= ERS_PARAMS["oom_battery_threshold_mj"]
        not_near_pit = tyre_age <= cfg_t.max_viable_laps * 0.82 or laps_remaining <= 8
        not_final_laps = laps_remaining > 5

        def chk(v): return "✅" if v else "❌"

        check_df = pd.DataFrame([
            {
                "Condition": "Gap ≤ detection threshold (1.0s)",
                "Status": chk(within_gap),
                "Detail": f"{gap:.1f}s {'≤' if within_gap else '>'} {ERS_PARAMS['oom_detection_gap_s']:.1f}s",
            },
            {
                "Condition": "Battery above OOM threshold",
                "Status": chk(battery_ok),
                "Detail": f"{state.battery_level_mj:.2f} MJ {'≥' if battery_ok else '<'} {ERS_PARAMS['oom_battery_threshold_mj']:.2f} MJ",
            },
            {
                "Condition": "Not in pre-pit-stop window",
                "Status": chk(not_near_pit),
                "Detail": f"Tyre age {tyre_age} / max viable {cfg_t.max_viable_laps}",
            },
        ])
        def _html_table(df: pd.DataFrame) -> str:
            cols = list(df.columns)
            header = "".join(
                f"<th style='background:#1a1a1a;color:#888;font-size:11px;text-transform:uppercase;"
                f"letter-spacing:1px;padding:8px 12px;border-bottom:1px solid #333;'>{c}</th>"
                for c in cols
            )
            body = ""
            for _, row in df.iterrows():
                cells = "".join(
                    f"<td style='padding:8px 12px;border-bottom:1px solid #1e1e1e;color:#fff;'>{row[c]}</td>"
                    for c in cols
                )
                body += f"<tr>{cells}</tr>"
            return (
                f"<div style='overflow-x:auto;background:#141414;border:1px solid #222;"
                f"border-radius:4px;margin:8px 0;'>"
                f"<table style='width:100%;border-collapse:collapse;font-size:13px;font-family:monospace;'>"
                f"<thead><tr>{header}</tr></thead><tbody>{body}</tbody></table></div>"
            )

        st.markdown(_html_table(check_df), unsafe_allow_html=True)

        # OOM gain breakdown
        st.subheader("OOM Energy & Time Gain Breakdown")
        gain_df = pd.DataFrame({
            "Effect": [
                "Extra Battery Capacity",
                "Enhanced Power Profile",
                "Boost Deployment Gain",
                "Total per-lap benefit",
            ],
            "Value": [
                f"+{ERS_PARAMS['oom_extra_capacity_mj']:.1f} MJ",
                f"+{ERS_PARAMS['oom_enhanced_power_gain_s']:.2f}s lap time",
                f"+{ERS_PARAMS['boost_lap_time_gain_s']:.2f}s (if boost also used)",
                f"+{ERS_PARAMS['oom_enhanced_power_gain_s'] + ERS_PARAMS['boost_lap_time_gain_s']:.2f}s combined",
            ],
        })
        st.markdown(_html_table(gain_df), unsafe_allow_html=True)

    # ================================================================== #
    # TAB 2: Battery trajectory                                            #
    # ================================================================== #
    with tab2:
        st.subheader("Battery Level Trajectory")

        bc1, bc2, bc3 = st.columns(3)
        with bc1:
            circuit_key = st.selectbox(
                "Circuit", CIRCUIT_ORDER,
                format_func=lambda k: CIRCUITS[k].name.replace(" Grand Prix", ""),
                key="oom_traj_circuit",
            )
            team_name = st.selectbox("Team", sorted(TEAMS.keys()), key="oom_traj_team")
        with bc2:
            stint_laps = st.slider("Stint Length (laps)", 5, 50, 25, key="oom_traj_stint")
            init_batt = st.slider("Initial Battery (%)", 30, 100, 80, step=5,
                                  key="oom_traj_init_batt")
        with bc3:
            recharge_mode = st.selectbox(
                "Recharge Mode",
                ["super_clip", "lift_off", "coast", "mixed"],
                key="oom_traj_mode",
            )
            oom_laps_str = st.text_input(
                "OOM Laps (comma-separated, e.g. 10,15,20)",
                value="",
                key="oom_traj_oom_laps",
            )

        oom_laps_input: list[int] = []
        if oom_laps_str.strip():
            try:
                oom_laps_input = [int(x.strip()) for x in oom_laps_str.split(",") if x.strip()]
            except ValueError:
                st.error("Invalid OOM laps format — use comma-separated integers.")

        pu_supplier = TEAMS[team_name].pu_supplier
        init_mj = ERS_PARAMS["battery_capacity_mj"] * (init_batt / 100)

        traj_no_oom = _simulate_battery_trajectory(
            stint_laps, recharge_mode, pu_supplier, circuit_key,
            oom_laps=[], boost_laps=[], initial_battery=init_mj,
        )
        traj_with_oom = _simulate_battery_trajectory(
            stint_laps, recharge_mode, pu_supplier, circuit_key,
            oom_laps=oom_laps_input, boost_laps=oom_laps_input,
            initial_battery=init_mj,
        )

        df_no_oom = pd.DataFrame(traj_no_oom)
        df_with_oom = pd.DataFrame(traj_with_oom)

        fig_batt = go.Figure()
        fig_batt.add_trace(go.Scatter(
            x=df_no_oom["lap"], y=df_no_oom["battery_mj"],
            mode="lines", name="No OOM",
            line=dict(color=UI_COLORS["mercedes_teal"], width=2, dash="dash"),
        ))
        fig_batt.add_trace(go.Scatter(
            x=df_with_oom["lap"], y=df_with_oom["battery_mj"],
            mode="lines", name="With OOM",
            line=dict(color=UI_COLORS["ferrari_red"], width=2.5),
        ))
        # Mark OOM laps
        for oom_lap in oom_laps_input:
            if 1 <= oom_lap <= stint_laps:
                fig_batt.add_vline(
                    x=oom_lap, line_dash="dot", line_color="#FFFF00", opacity=0.6,
                )
        fig_batt.add_hline(
            y=ERS_PARAMS["battery_min_reserve_mj"],
            line_dash="dash", line_color="red",
            annotation_text="Min Reserve",
        )
        fig_batt.add_hline(
            y=ERS_PARAMS["oom_battery_threshold_mj"],
            line_dash="dot", line_color="orange",
            annotation_text="OOM Threshold",
        )
        fig_batt.update_layout(
            title=f"Battery Level Over Stint — {recharge_mode.replace('_', ' ').title()} Mode",
            xaxis_title="Lap in Stint",
            yaxis_title="Battery Level (MJ)",
            template=PLOTLY_TEMPLATE,
            height=420,
            yaxis=dict(range=[0, ERS_PARAMS["battery_capacity_mj"] + 0.5]),
        )
        st.plotly_chart(fig_batt, use_container_width=True)

        # ERS delta chart
        fig_delta = go.Figure()
        fig_delta.add_trace(go.Bar(
            x=df_with_oom["lap"],
            y=df_with_oom["ers_delta_s"],
            marker_color=[
                UI_COLORS["ferrari_red"] if row["oom_active"] else UI_COLORS["mercedes_teal"]
                for _, row in df_with_oom.iterrows()
            ],
            name="ERS Lap Time Delta",
        ))
        fig_delta.update_layout(
            title="ERS Lap Time Delta (negative = faster)",
            xaxis_title="Lap in Stint",
            yaxis_title="Time Delta (s)",
            template=PLOTLY_TEMPLATE,
            height=280,
        )
        st.plotly_chart(fig_delta, use_container_width=True)

        cumulative_gain = -df_with_oom["ers_delta_s"].sum() + df_no_oom["ers_delta_s"].sum()
        st.metric(
            "Cumulative time gain from OOM usage",
            f"{cumulative_gain:.3f}s",
            delta="vs no OOM strategy",
        )

    # ================================================================== #
    # TAB 3: OOM vs DRS                                                    #
    # ================================================================== #
    with tab3:
        st.subheader("OOM vs DRS — 2026 Regulation Change")
        st.markdown("""
        | Feature | DRS (pre-2026) | OOM (2026) |
        |---------|---------------|-----------|
        | **Mechanism** | Rear wing opens | Enhanced ERS deployment + extra battery |
        | **Requirement** | Within 1.0s at detection point | Within 1.0s at detection line |
        | **Who benefits** | Attacker only | Attacker only |
        | **Defender** | No counter-mechanism | Active Aero open on straights (equal) |
        | **Availability** | DRS zones only | Any designated straight |
        | **Energy cost** | None | Battery depletes faster |
        | **Time gain** | ~0.5s per DRS zone | ~0.30–0.45s per lap (sustained) |
        | **Strategic depth** | Low | **High** — battery management matters |
        | **All-laps active** | No (detection lap only) | Yes — OOM gives enhanced profile for full lap |
        """)

        st.subheader("Circuit OOM Effectiveness (by detection zone count)")
        oom_df = pd.DataFrame([
            {
                "Circuit": CIRCUITS[k].name.replace(" Grand Prix", ""),
                "OOM Zones": CIRCUITS[k].num_drs_zones,
                "SC Prob": CIRCUITS[k].sc_probability,
                "Overtake Difficulty": CIRCUITS[k].overtake_difficulty,
                "OOM Estimated Gain (s/lap)": round(
                    CIRCUITS[k].num_drs_zones * ERS_PARAMS["oom_enhanced_power_gain_s"], 3
                ),
            }
            for k in CIRCUIT_ORDER
        ])
        fig_oom = px.scatter(
            oom_df,
            x="Overtake Difficulty", y="OOM Estimated Gain (s/lap)",
            size="OOM Zones", color="SC Prob",
            text="Circuit",
            title="OOM Effectiveness vs Overtaking Difficulty",
            template=PLOTLY_TEMPLATE,
            color_continuous_scale="RdYlGn_r",
        )
        fig_oom.update_traces(textposition="top center")
        fig_oom.update_layout(height=450)
        st.plotly_chart(fig_oom, use_container_width=True)
