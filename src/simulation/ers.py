"""
ERS Model — F1 2026

Encodes the radical ERS changes for 2026:
  - ICE: ~400 kW (down from ~550 kW 2025)
  - MGU-K: 350 kW (tripled from 120 kW 2025)
  - MGU-H: ELIMINATED
  - Battery capacity: ~4 MJ (approximately double 2025)
  - Energy recovery modes:
      * Super Clipping: full-throttle recovery, active aero stays open → preferred
      * Lift-off Regeneration: braking/coasting, DISABLES active aero on straights
        → drag penalty (~0.18s per lap)
  - Deployment modes:
      * Boost button: max power burst (~0.30s lap-time gain, costs 0.5 MJ)
      * Overtake Override Mode (OOM): replaces DRS
        - Triggered within 1.0s gap at detection line
        - Gives +0.5 MJ extra battery capacity + enhanced power profile
        - Attack only (not available on defense)
  - No recharging between stint boundary (battery state carries across stints)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, List, Optional

from config import ERS_PARAMS, POWER_UNITS, CIRCUITS


@dataclass
class ERSState:
    """Encapsulates battery and deployment state at any point in the race."""
    battery_level_mj: float          # current charge level
    battery_capacity_mj: float = ERS_PARAMS["battery_capacity_mj"]
    oom_extra_capacity_mj: float = 0.0   # bonus capacity from OOM trigger
    oom_active: bool = False
    oom_used_this_lap: bool = False
    recharge_mode: str = "super_clip"    # "super_clip" | "lift_off" | "coast"
    boost_used_this_lap: bool = False
    laps_since_oom: int = 0              # laps since last OOM use

    @property
    def effective_capacity_mj(self) -> float:
        return self.battery_capacity_mj + self.oom_extra_capacity_mj

    @property
    def state_of_charge(self) -> float:
        """0.0 (empty) to 1.0 (full)."""
        return self.battery_level_mj / self.effective_capacity_mj

    @property
    def can_boost(self) -> bool:
        return (
            self.battery_level_mj >= ERS_PARAMS["boost_cost_mj"]
            and not self.boost_used_this_lap
        )

    @property
    def can_trigger_oom(self) -> bool:
        return (
            self.battery_level_mj >= ERS_PARAMS["oom_battery_threshold_mj"]
            and not self.oom_used_this_lap
        )

    def reset_lap_flags(self) -> "ERSState":
        """Reset per-lap flags. Call at the start of each new lap."""
        return ERSState(
            battery_level_mj=self.battery_level_mj,
            battery_capacity_mj=self.battery_capacity_mj,
            oom_extra_capacity_mj=self.oom_extra_capacity_mj,
            oom_active=self.oom_active,
            oom_used_this_lap=False,
            recharge_mode=self.recharge_mode,
            boost_used_this_lap=False,
            laps_since_oom=self.laps_since_oom + 1,
        )


class ERSModel:
    """
    Computes per-lap ERS contributions to lap time and advances battery state.

    Design principles:
    - Super clipping is preferred: recharges without aero penalty
    - Lift-off recharge trades drag (slower straights) for battery recovery
    - Boost deployment: short-term gain, battery investment
    - OOM: attack window tool; modelled as discrete per-lap decision
    """

    def _recovery_amount(
        self,
        recharge_mode: str,
        pu_supplier: str,
        current_battery: float,
        capacity: float,
    ) -> float:
        """
        Compute MJ recovered this lap given mode and PU efficiency.
        Clamps so battery does not exceed capacity.
        """
        pu = POWER_UNITS[pu_supplier]
        base_rates = {
            "super_clip": ERS_PARAMS["super_clip_recovery_mj_per_lap"],
            "lift_off": ERS_PARAMS["lift_off_recovery_mj_per_lap"],
            "coast": ERS_PARAMS["coast_recovery_mj_per_lap"],
        }
        raw = base_rates.get(recharge_mode, ERS_PARAMS["super_clip_recovery_mj_per_lap"])
        recovered = raw * pu.energy_recovery_efficiency
        # Can't charge beyond capacity
        space = max(0.0, capacity - current_battery)
        return min(recovered, space)

    def compute_lap_ers_delta(
        self,
        ers_state: ERSState,
        use_boost: bool,
        recharge_mode: str,
        pu_supplier: str,
        circuit: str,
        gap_to_car_ahead: float = 999.0,
    ) -> Tuple[float, ERSState]:
        """
        Compute net ERS lap-time delta and advance battery state.

        Positive delta = slower (drag penalty).
        Negative delta = faster (boost gain).

        Returns
        -------
        (lap_time_delta_s, new_ers_state)
        """
        new_state = ers_state.reset_lap_flags()
        new_state.recharge_mode = recharge_mode
        pu = POWER_UNITS[pu_supplier]
        lap_delta = 0.0

        # ---------- Active Aero drag penalty (lift-off recharge) ----------
        if recharge_mode == "lift_off":
            lap_delta += ERS_PARAMS["lift_off_aero_time_penalty_s"]

        # ---------- Boost deployment ----------
        if use_boost and new_state.can_boost:
            lap_delta -= ERS_PARAMS["boost_lap_time_gain_s"] * pu.ers_lap_time_bonus
            new_state.battery_level_mj = max(
                0.0,
                new_state.battery_level_mj - ERS_PARAMS["boost_cost_mj"]
            )
            new_state.boost_used_this_lap = True

        # ---------- OOM check ----------
        oom_triggered = False
        if (
            gap_to_car_ahead <= ERS_PARAMS["oom_detection_gap_s"]
            and new_state.can_trigger_oom
        ):
            oom_triggered = True
            new_state.oom_active = True
            new_state.oom_used_this_lap = True
            new_state.oom_extra_capacity_mj = ERS_PARAMS["oom_extra_capacity_mj"]
            # OOM gives enhanced power: extra lap-time benefit
            lap_delta -= ERS_PARAMS["oom_enhanced_power_gain_s"]
            new_state.laps_since_oom = 0

        # Clear OOM extra capacity when gap is too large again
        if not oom_triggered and new_state.laps_since_oom > 1:
            new_state.oom_extra_capacity_mj = 0.0
            new_state.oom_active = False

        # ---------- Energy recovery ----------
        recovered = self._recovery_amount(
            recharge_mode, pu_supplier,
            new_state.battery_level_mj,
            new_state.effective_capacity_mj,
        )
        new_state.battery_level_mj = min(
            new_state.battery_level_mj + recovered,
            new_state.effective_capacity_mj,
        )

        # Ensure battery doesn't go below reserve
        new_state.battery_level_mj = max(
            new_state.battery_level_mj,
            ERS_PARAMS["battery_min_reserve_mj"],
        )

        return lap_delta, new_state

    def compute_oom_decision(
        self,
        ers_state: ERSState,
        gap_to_car_ahead: float,
        laps_remaining: int,
        tyre_age: int,
        compound: str,
    ) -> Tuple[bool, str]:
        """
        Strategic OOM decision logic.

        Returns (should_trigger_oom, reasoning_string)

        Decision tree:
        1. If gap > detection gap → cannot trigger → False
        2. If battery below threshold → save → False
        3. If within 3 laps of likely pit stop → save battery for after stop → False
        4. If tyre on cliff (age > 80% of expected) → OOM likely wasted → conditional
        5. If laps_remaining < 5 → use everything → True
        6. Otherwise → trigger if battery > 60% charge
        """
        from config import TYRE_COMPOUNDS

        if gap_to_car_ahead > ERS_PARAMS["oom_detection_gap_s"]:
            return False, "Gap too large — OOM not available"

        if not ers_state.can_trigger_oom:
            return False, f"Battery {ers_state.battery_level_mj:.2f} MJ below threshold"

        if laps_remaining <= 5:
            return True, "Final laps — deploy all available energy"

        # Check if nearing pit stop (heuristic: if tyre age > max_viable * 0.85)
        cfg = TYRE_COMPOUNDS[compound]
        near_pit = tyre_age > cfg.max_viable_laps * 0.82
        if near_pit and laps_remaining > 8:
            return False, "Near pit stop window — conserve battery for fresh tyres"

        # Use OOM if battery is healthy (> 60% charge)
        if ers_state.state_of_charge >= 0.60:
            return True, f"Battery {ers_state.state_of_charge:.0%} — OOM advised to close gap"

        return False, f"Battery {ers_state.state_of_charge:.0%} — conserve for subsequent laps"

    def simulate_stint_ers(
        self,
        num_laps: int,
        pu_supplier: str,
        circuit: str,
        oom_laps: Optional[List[int]] = None,
        boost_laps: Optional[List[int]] = None,
        recharge_strategy: str = "super_clip",
        initial_battery_fraction: float = 0.80,
    ) -> Tuple[List[float], List[ERSState]]:
        """
        Simulate ERS over a full stint.

        Parameters
        ----------
        oom_laps         : 1-indexed lap numbers within stint where OOM is triggered
        boost_laps       : 1-indexed lap numbers within stint where boost is used
        recharge_strategy: "super_clip" | "lift_off" | "coast" | "mixed"
        initial_battery_fraction: battery SOC at stint start (0–1)

        Returns
        -------
        (list_of_lap_time_deltas, list_of_ers_states)
        """
        if oom_laps is None:
            oom_laps = []
        if boost_laps is None:
            boost_laps = []

        state = ERSState(
            battery_level_mj=ERS_PARAMS["battery_capacity_mj"] * initial_battery_fraction,
        )

        deltas: List[float] = []
        states: List[ERSState] = []

        for lap in range(1, num_laps + 1):
            use_boost = lap in boost_laps
            gap = 0.5 if lap in oom_laps else 999.0  # simulate gap scenario

            # Alternate recharge mode for "mixed" strategy
            mode = recharge_strategy
            if recharge_strategy == "mixed":
                mode = "super_clip" if state.state_of_charge < 0.70 else "lift_off"

            delta, state = self.compute_lap_ers_delta(
                state, use_boost, mode, pu_supplier, circuit, gap_to_car_ahead=gap
            )
            deltas.append(delta)
            states.append(state)

        return deltas, states

    @staticmethod
    def fresh_state(fraction: float = 0.80) -> ERSState:
        """Create a fresh ERSState at a given state of charge fraction."""
        return ERSState(
            battery_level_mj=ERS_PARAMS["battery_capacity_mj"] * fraction,
        )
