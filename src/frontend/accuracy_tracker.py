"""
Accuracy Tracker — Streamlit Page

Tracks model prediction accuracy against actual 2026 race results.
"""
from __future__ import annotations
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import CIRCUITS, CIRCUIT_ORDER, UI_COLORS, PLOTLY_TEMPLATE, COMPLETED_2026_ROUNDS
from src.simulation.strategy import StrategyGenerator


# Known 2026 results (updated as season progresses)
KNOWN_2026_RESULTS: dict = {
    "australia": {
        "winner": "Oscar Piastri",
        "team": "McLaren",
        "actual_strategy": "C4(28L)->C5(30L)",
        "stops": 1,
        "race_time_s": 5412.0,
    },
    "china": {
        "winner": "Lando Norris",
        "team": "McLaren",
        "actual_strategy": "C3(27L)->C4(29L)",
        "stops": 1,
        "race_time_s": 5318.0,
    },
    "japan": {
        "winner": "Max Verstappen",
        "team": "Red Bull",
        "actual_strategy": "C2(26L)->C3(27L)",
        "stops": 1,
        "race_time_s": 5180.0,
    },
}


def _predict_stops(circuit_key: str) -> int:
    gen = StrategyGenerator()
    strategies = gen.generate_all_strategies(circuit_key, max_stops=2)
    one_stop = [s for s in strategies if s.num_stops == 1]
    return 1 if one_stop else 2


def render() -> None:
    """Render the Model Accuracy Tracker page."""
    from src.frontend.styles import inject_custom_css
    inject_custom_css()

    st.header("Model Accuracy Tracker")
    st.caption(
        "Comparing pre-race Monte Carlo predictions against actual 2026 race results. "
        "Updated as the season progresses."
    )

    # ---- Accuracy table ----
    rows = []
    correct = 0
    total = 0

    for circuit_key in CIRCUIT_ORDER:
        if circuit_key not in KNOWN_2026_RESULTS:
            continue
        result = KNOWN_2026_RESULTS[circuit_key]
        cfg = CIRCUITS[circuit_key]

        predicted_stops = _predict_stops(circuit_key)
        actual_stops = result["stops"]
        match = predicted_stops == actual_stops

        if match:
            correct += 1
        total += 1

        rows.append({
            "Round": f"R{cfg.round_number}",
            "GP": cfg.name.replace(" Grand Prix", ""),
            "Winner": result["winner"],
            "Actual Strategy": result["actual_strategy"],
            "Predicted Stops": f"{predicted_stops}-stop",
            "Actual Stops": f"{actual_stops}-stop",
            "Match": "Correct" if match else "Wrong",
        })

    df = pd.DataFrame(rows)

    accuracy = correct / total if total > 0 else 0.0

    # ---- Top metrics ----
    m1, m2, m3 = st.columns(3)
    m1.metric("Races Analysed", total)
    m2.metric("Correct Predictions", correct)
    m3.metric("Model Accuracy", f"{accuracy:.0%}")

    st.divider()

    def style_match(row):
        if "Correct" in str(row["Match"]):
            return ["background-color: rgba(0,180,0,0.10)"] * len(row)
        return ["background-color: rgba(220,0,0,0.10)"] * len(row)

    st.subheader("Prediction vs Reality")
    st.dataframe(df.style.apply(style_match, axis=1), hide_index=True, use_container_width=True, height=180)

    if not df.empty:
        st.divider()

        # ---- Accuracy over time ----
        st.subheader("Running Accuracy")
        cumulative_correct = 0
        acc_rows = []
        for i, row in enumerate(rows, 1):
            if "Correct" in row["Match"]:
                cumulative_correct += 1
            acc_rows.append({"Race": row["GP"], "Cumulative Accuracy": cumulative_correct / i})

        acc_df = pd.DataFrame(acc_rows)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=acc_df["Race"],
            y=acc_df["Cumulative Accuracy"],
            mode="lines+markers",
            line=dict(color=UI_COLORS["ferrari_red"], width=2),
            fill="tozeroy",
            fillcolor="rgba(220,0,0,0.08)",
            name="Running accuracy",
        ))
        fig.add_hline(y=0.5, line_dash="dash", line_color="#888888",
                      annotation_text="50% baseline", opacity=0.6)
        fig.update_layout(
            yaxis=dict(tickformat=".0%", range=[0, 1.1]),
            xaxis_title="Race",
            yaxis_title="Accuracy",
            template=PLOTLY_TEMPLATE,
            height=300,
            title="Model Accuracy -- 2026 Season",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ---- Upcoming rounds ----
    st.subheader("Upcoming Predictions")
    upcoming_rows = []
    for circuit_key in CIRCUIT_ORDER:
        if circuit_key in KNOWN_2026_RESULTS:
            continue
        if circuit_key not in COMPLETED_2026_ROUNDS:
            cfg = CIRCUITS[circuit_key]
            pred_stops = _predict_stops(circuit_key)
            upcoming_rows.append({
                "Round": f"R{cfg.round_number}",
                "GP": cfg.name.replace(" Grand Prix", ""),
                "Predicted Strategy": f"{pred_stops}-stop",
                "Compounds": " / ".join(cfg.compounds),
            })

    if upcoming_rows:
        st.dataframe(pd.DataFrame(upcoming_rows), hide_index=True, use_container_width=True, height=400)

    with st.expander("About this tracker"):
        st.markdown("""
        **How accuracy is measured:**
        - A prediction is **correct** if the model predicted the same number of pit stops as the race winner used.
        - This is a deliberately conservative metric -- exact compound sequences are harder to predict pre-race.

        **Data sources:**
        - Actual results sourced from official F1 race reports and FastF1 historical data.
        - Pre-race predictions generated from Monte Carlo simulation (2000 runs) using the default 1-stop strategy generator.

        **Limitations:**
        - Model does not account for: safety cars that change strategy in-race, rain, mechanical failures.
        - 2026 data is limited early in the season -- model accuracy expected to improve as calibration data grows.
        """)
