"""
Lap Time Model — F1 2026

Assembles per-lap race times from all contributing factors:

    lap_time = base_lap_time
             + fuel_load_penalty          (0.032 s/kg × remaining fuel)
             + tyre_deg_penalty           (compound age + abrasiveness)
             + tyre_compound_offset       (C1 slowest → C5 fastest baseline)
             + ers_delta                  (boost gain / lift-off drag penalty)
             + team_performance_delta     (tier offset)
             + track_evolution_benefit    (rubber laid down)
             + gaussian_noise             (σ = 0.05 s)

SC laps are handled by multiplying by SC_LAP_TIME_MULTIPLIER in the SC model.
This model focuses purely on green-flag racing laps.
"""

from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional

from config import (
    CIRCUITS, TYRE_COMPOUNDS, TEAMS, FUEL_PARAMS,
    SIM_PARAMS, ERS_PARAMS,
)
from .tyre import TyreDegradationModel, TyreState
from .ers import ERSModel, ERSState
from .safety_car import SCEvent


@dataclass
class LapTimeInputs:
    """All inputs required to compute a single lap time."""
    circuit: str
    lap_number: int            # race lap (1-indexed)
    compound: str
    tyre_age: int              # laps on this stint's tyres
    fuel_kg: float             # fuel remaining at start of lap
    ers_state: ERSState
    track_temp_c: float
    push_level: float          # 0.0 conservative → 1.0 maximum attack
    use_boost: bool
    recharge_mode: str         # "super_clip" | "lift_off" | "coast"
    pu_supplier: str
    team: str
    deg_multiplier: float = 1.0   # Monte Carlo variance
    gap_to_car_ahead: float = 999.0


class LapTimeModel:
    """
    Computes green-flag lap times and advances car state.
    """

    def __init__(self) -> None:
        self.tyre_model = TyreDegradationModel()
        self.ers_model = ERSModel()

    def compute_lap_time(
        self,
        inputs: LapTimeInputs,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[float, ERSState]:
        """
        Compute a single lap time and return the updated ERS state.

        Returns
        -------
        (lap_time_seconds, new_ers_state)
        """
        circuit_cfg = CIRCUITS[inputs.circuit]
        team_cfg = TEAMS[inputs.team]
        compound_cfg = TYRE_COMPOUNDS[inputs.compound]

        # ---- Base lap time ----
        lap_time = circuit_cfg.base_lap_time

        # ---- Fuel load penalty ----
        # Lap starts with inputs.fuel_kg; contribution falls as fuel burns
        fuel_penalty = inputs.fuel_kg * FUEL_PARAMS["fuel_effect_per_kg_s"]
        lap_time += fuel_penalty

        # ---- Tyre compound baseline offset ----
        lap_time += compound_cfg.base_pace_offset

        # ---- Tyre degradation penalty ----
        deg_penalty = self.tyre_model.compute_lap_time_delta(
            compound=inputs.compound,
            lap_in_stint=inputs.tyre_age + 1,
            track_abrasiveness=circuit_cfg.track_abrasiveness,
            track_temp_celsius=inputs.track_temp_c,
            push_level=inputs.push_level,
            circuit=inputs.circuit,
        ) * inputs.deg_multiplier
        lap_time += deg_penalty

        # ---- ERS delta ----
        ers_delta, new_ers_state = self.ers_model.compute_lap_ers_delta(
            ers_state=inputs.ers_state,
            use_boost=inputs.use_boost,
            recharge_mode=inputs.recharge_mode,
            pu_supplier=inputs.pu_supplier,
            circuit=inputs.circuit,
            gap_to_car_ahead=inputs.gap_to_car_ahead,
        )
        lap_time += ers_delta

        # ---- Team performance offset ----
        lap_time += team_cfg.base_lap_delta

        # ---- Track evolution (rubber = faster as laps go by) ----
        # Exponential approach: most benefit in first 25 laps but continues
        # to improve — replaces the hard cap at lap 25
        track_evo = -circuit_cfg.track_evolution_rate * 25.0 * (
            1.0 - math.exp(-inputs.lap_number / 25.0)
        )
        lap_time += track_evo

        # ---- Gaussian noise ----
        if rng is not None:
            lap_time += rng.normal(0.0, SIM_PARAMS["lap_time_noise_sigma"])

        return max(lap_time, circuit_cfg.base_lap_time * 0.90), new_ers_state

    def compute_race_lap_times(
        self,
        circuit: str,
        stints: List[Tuple[str, int, int, float]],  # (compound, start_lap, end_lap, push_level)
        team: str,
        track_temp_c: float,
        sc_events: List[SCEvent],
        pit_loss_time: float,
        deg_multiplier: float = 1.0,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[np.ndarray, float]:
        """
        Simulate a full race given a list of stints and return lap-time array + total time.

        Parameters
        ----------
        stints     : list of (compound, start_lap, end_lap, push_level)
                     start_lap and end_lap are 1-indexed race laps.
                     end_lap is the lap the pit stop occurs (final stint: race end lap).
        pit_loss_time : net time added per pit stop

        Returns
        -------
        (lap_times_array, total_race_time_seconds)
        """
        circuit_cfg = CIRCUITS[circuit]
        team_cfg = TEAMS[team]
        total_laps = circuit_cfg.total_laps

        lap_times = np.zeros(total_laps, dtype=np.float64)

        # Start with fresh ERS at 80% charge
        ers_state = ERSModel.fresh_state(0.80)
        fuel_kg = FUEL_PARAMS["fuel_load_kg"]
        fuel_per_lap = circuit_cfg.fuel_per_lap

        # Add fuel consumption variance
        if rng is not None:
            fuel_variance = rng.normal(1.0, FUEL_PARAMS["fuel_burn_variance"])
            fuel_per_lap *= fuel_variance

        total_time = 0.0
        current_race_lap = 1

        for stint_idx, (compound, start_lap, end_lap, push_level) in enumerate(stints):
            tyre_age = 0

            for race_lap in range(start_lap, end_lap + 1):
                if race_lap > total_laps:
                    break

                # Get SC multiplier
                sc_multiplier = 1.0
                from .safety_car import SafetyCarModel
                sc_model = SafetyCarModel()
                sc_multiplier = sc_model.get_lap_multiplier(race_lap, sc_events)

                # Use boost on laps with healthy battery; save near stint end
                use_boost = (
                    ers_state.can_boost
                    and ers_state.state_of_charge > 0.55
                    and tyre_age < TYRE_COMPOUNDS_IMPORT(compound).max_viable_laps * 0.7
                    and sc_multiplier == 1.0   # don't burn boost under SC
                )

                # Prefer super_clip; switch to lift_off if battery nearly full
                mode = "super_clip"
                if ers_state.state_of_charge > 0.88:
                    mode = "lift_off"

                inputs = LapTimeInputs(
                    circuit=circuit,
                    lap_number=race_lap,
                    compound=compound,
                    tyre_age=tyre_age,
                    fuel_kg=max(0.0, fuel_kg),
                    ers_state=ers_state,
                    track_temp_c=track_temp_c,
                    push_level=push_level if sc_multiplier == 1.0 else 0.4,
                    use_boost=use_boost and sc_multiplier == 1.0,
                    recharge_mode=mode,
                    pu_supplier=team_cfg.pu_supplier,
                    team=team,
                    deg_multiplier=deg_multiplier,
                )

                base_time, ers_state = self.compute_lap_time(inputs, rng)
                lap_time = base_time * sc_multiplier

                lap_times[race_lap - 1] = lap_time
                total_time += lap_time

                tyre_age += 1
                fuel_kg -= fuel_per_lap
                current_race_lap += 1

            # Add pit stop time loss (not on the last stint)
            if stint_idx < len(stints) - 1:
                total_time += pit_loss_time
                # After pit stop, ERS partially recovered during pit lane
                ers_state = ERSState(
                    battery_level_mj=min(
                        ers_state.battery_level_mj + 0.3,
                        ers_state.effective_capacity_mj,
                    ),
                    battery_capacity_mj=ers_state.battery_capacity_mj,
                )

        return lap_times, total_time


# Helper so laptime.py doesn't need to import at module level (avoid circular)
def TYRE_COMPOUNDS_IMPORT(compound: str):
    from config import TYRE_COMPOUNDS
    return TYRE_COMPOUNDS[compound]


# Re-export ERSState for callers that only import from this module
from .ers import ERSState  # noqa: E402, F401
