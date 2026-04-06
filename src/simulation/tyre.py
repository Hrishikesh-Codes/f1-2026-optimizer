"""
Tyre Degradation Model — F1 2026

Models Pirelli C1–C5 degradation with:
  - Linear phase: constant time loss per lap
  - Cliff phase: exponential blow-up after compound-specific cliff lap
  - Thermal degradation: temperature sensitivity per compound
  - Abrasiveness scaling per circuit
  - Push-level effect (saving tyres = slower but longer stints)

2026 tyre changes vs 2025:
  - Narrower footprint (front -25mm, rear -30mm)
  - Smaller diameter (front -15mm, rear -10mm)
  - Less lateral load from lighter/shorter car → higher durability baseline
  - C6 dropped; C1–C5 remain
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, List

from config import TYRE_COMPOUNDS, SIM_PARAMS


@dataclass
class TyreState:
    """Mutable state of a tyre during a stint."""
    compound: str
    age: int                      # laps completed on this set
    cumulative_linear_deg: float  # seconds accumulated from linear phase
    cumulative_cliff_deg: float   # seconds accumulated from cliff phase
    thermal_deg: float            # seconds from thermal effects
    is_past_cliff: bool = False
    surface_temp_celsius: float = 90.0   # tyre surface temperature
    graining: float = 0.0                # 0-1 graining severity (cold laps or over-push)

    @property
    def total_deg_penalty(self) -> float:
        """Total additional time vs fresh tyre of the same compound."""
        return self.cumulative_linear_deg + self.cumulative_cliff_deg + self.thermal_deg

    @property
    def effective_compound_age(self) -> int:
        """Age adjusted for stint conditions; used by some model paths."""
        return self.age


class TyreDegradationModel:
    """
    Physics-based tyre degradation model calibrated to 2026 Pirelli compounds.

    The degradation function per lap in stint (l) is:

        Δt(l) = linear_rate × l × A × T_factor × P_factor
               + cliff_term(l)
               + graining_term(l)

    where:
        A        = track abrasiveness factor
        T_factor = 1 + thermal_sensitivity × max(0, (temp_c - 35) / 10)
        P_factor = 0.6 + 0.4 × push_level   (save=0.6×, push=1.0×)
        cliff_term(l) = cliff_exp × (l - cliff_lap_adj)^2  if l > cliff_lap_adj else 0
    """

    # Temperature above which thermal deg kicks in, in °C
    THERMAL_BASELINE_C: float = 35.0

    def __init__(self, calibration: dict = None) -> None:
        self.calibration = calibration or {}

    def _adjusted_cliff_lap(
        self,
        compound: str,
        abrasiveness: float,
        track_temp_c: float,
        push_level: float,
    ) -> int:
        """
        Compute the adjusted cliff lap for given circuit/weather/push conditions.

        Higher abrasiveness, higher temperature, and harder pushing all bring
        the cliff earlier.
        """
        cfg = TYRE_COMPOUNDS[compound]
        temp_factor = 1.0 + cfg.thermal_sensitivity * max(0.0, (track_temp_c - self.THERMAL_BASELINE_C) / 10.0)
        push_factor = 0.6 + 0.4 * push_level
        # Cliff lap shrinks (earlier cliff) when abrasion, temp, or push are high
        adjusted = cfg.cliff_lap / (abrasiveness * temp_factor * push_factor)
        return max(5, int(adjusted))

    def compute_lap_time_delta(
        self,
        compound: str,
        lap_in_stint: int,
        track_abrasiveness: float,
        track_temp_celsius: float,
        push_level: float,
        circuit: str = None,
    ) -> float:
        """
        Return the additional lap time (seconds) caused by tyre wear for a
        given lap within the stint, vs a lap-0 (new tyre) baseline.

        Parameters
        ----------
        compound          : tyre compound key e.g. "C3"
        lap_in_stint      : 1-indexed lap number within this stint
        track_abrasiveness: circuit factor (0.65 smooth → 1.15 abrasive)
        track_temp_celsius: track surface temperature
        push_level        : 0.0 (heavy conservation) → 1.0 (flat-out push)
        circuit           : optional circuit key for calibration lookup
        """
        if lap_in_stint <= 0:
            return 0.0

        cfg = TYRE_COMPOUNDS[compound]

        # Check for calibration overrides for this circuit+compound
        cal_linear_rate = cfg.linear_deg_rate
        cal_cliff_lap = cfg.cliff_lap
        cal_cliff_exp = cfg.cliff_exponent

        if circuit and self.calibration:
            deg_curves = self.calibration.get("deg_curves", {})
            circuit_cal = deg_curves.get(circuit, {})
            compound_cal = circuit_cal.get(compound, {})
            if compound_cal:
                cal_linear_rate = compound_cal.get("linear_rate", cal_linear_rate)
                cal_cliff_lap = compound_cal.get("cliff_lap", cal_cliff_lap)
                cal_cliff_exp = compound_cal.get("cliff_exp", cal_cliff_exp)

        # Thermal multiplier
        temp_factor = 1.0 + cfg.thermal_sensitivity * max(
            0.0, (track_temp_celsius - self.THERMAL_BASELINE_C) / 10.0
        )
        # Push multiplier (saving = 0.6×, attacking = 1.0×)
        push_factor = 0.6 + 0.4 * push_level

        # ---------- Linear degradation phase ----------
        linear_delta = (
            cal_linear_rate
            * lap_in_stint
            * track_abrasiveness
            * temp_factor
            * push_factor
        )

        # ---------- Cliff degradation phase ----------
        # Use calibrated cliff_lap in the adjustment calculation
        cliff_lap_adj = self._adjusted_cliff_lap(
            compound, track_abrasiveness, track_temp_celsius, push_level
        )
        # If we have calibration, also adjust based on ratio
        if circuit and self.calibration and compound_cal:
            ratio = cal_cliff_lap / max(1, cfg.cliff_lap)
            cliff_lap_adj = max(5, int(cliff_lap_adj * ratio))

        cliff_delta = 0.0
        if lap_in_stint > cliff_lap_adj:
            laps_past_cliff = lap_in_stint - cliff_lap_adj
            cliff_delta = (
                cal_cliff_exp
                * (laps_past_cliff ** 2)
                * track_abrasiveness
                * temp_factor
            )

        return linear_delta + cliff_delta

    def compute_stint_time_deltas(
        self,
        compound: str,
        num_laps: int,
        track_abrasiveness: float,
        track_temp_celsius: float,
        push_level: float,
        deg_multiplier: float = 1.0,
    ) -> np.ndarray:
        """
        Compute per-lap time deltas for an entire stint.

        Returns an array of length num_laps where element i is the
        additional time (seconds) on lap i+1 of the stint.

        Parameters
        ----------
        deg_multiplier : Monte Carlo variance multiplier (e.g. 0.9–1.1)
        """
        deltas = np.array([
            self.compute_lap_time_delta(
                compound, lap + 1, track_abrasiveness,
                track_temp_celsius, push_level
            )
            for lap in range(num_laps)
        ], dtype=np.float64)
        return deltas * deg_multiplier

    def compute_cliff_lap(
        self,
        compound: str,
        abrasiveness: float,
        track_temp_c: float,
        push_level: float = 0.8,
    ) -> int:
        """Return adjusted cliff lap for display / UI purposes."""
        return self._adjusted_cliff_lap(compound, abrasiveness, track_temp_c, push_level)

    def estimate_optimal_stint_length(
        self,
        compound: str,
        abrasiveness: float,
        track_temp_c: float,
        push_level: float = 0.8,
        time_tolerance_s: float = 3.0,
    ) -> Tuple[int, int, int]:
        """
        Estimate (min, optimal, max) stint length for a compound at given conditions.

        Optimal = lap before cliff kicks in significantly (< 0.5s/lap added).
        Max = where cumulative cliff penalty exceeds time_tolerance_s.

        Returns
        -------
        (min_laps, optimal_laps, max_laps)
        """
        cfg = TYRE_COMPOUNDS[compound]
        min_laps = cfg.min_recommended_laps
        cliff_lap = self._adjusted_cliff_lap(compound, abrasiveness, track_temp_c, push_level)
        optimal_laps = max(min_laps, cliff_lap - 2)

        # Find max lap where PER-LAP degradation penalty exceeds time_tolerance_s
        # (proxy for cliff blow-up — single-lap delta grows rapidly past cliff)
        max_laps = optimal_laps
        for lap in range(optimal_laps, cfg.max_viable_laps + 1):
            per_lap_delta = self.compute_lap_time_delta(
                compound, lap, abrasiveness, track_temp_c, push_level
            )
            if per_lap_delta > time_tolerance_s:
                break
            max_laps = lap

        max_laps = min(max_laps, cfg.max_viable_laps)
        return min_laps, optimal_laps, max_laps

    def compute_undercut_window(
        self,
        compound_old: str,
        compound_new: str,
        laps_on_old: int,
        pit_loss_time: float,
        abrasiveness: float,
        track_temp_c: float,
        push_level: float = 0.9,
        window_laps: int = 10,
    ) -> List[Tuple[int, float]]:
        """
        Compute the undercut potential over the next window_laps.

        For each future lap, calculate:
          pace_delta = (old_compound_at_age + laps_on_old + n)
                     - (new_compound_at_age_n)
                     - pit_loss_time / remaining_window

        Returns list of (lap_offset, cumulative_time_gain) tuples.
        A positive value means undercutting NOW gains time vs staying out.
        """
        results: List[Tuple[int, float]] = []
        cumulative_gain = 0.0

        for n in range(1, window_laps + 1):
            # Pace of old compound at lap (laps_on_old + n) vs fresh new compound at lap n
            old_delta = self.compute_lap_time_delta(
                compound_old, laps_on_old + n, abrasiveness, track_temp_c, push_level
            )
            new_delta = self.compute_lap_time_delta(
                compound_new, n, abrasiveness, track_temp_c, push_level
            )
            # Old pace offset from baseline
            old_base = TYRE_COMPOUNDS[compound_old].base_pace_offset
            new_base = TYRE_COMPOUNDS[compound_new].base_pace_offset

            # Net pace advantage of new compound per lap (negative = new is faster)
            pace_delta_per_lap = (old_base + old_delta) - (new_base + new_delta)
            cumulative_gain += pace_delta_per_lap

            # First lap: must recoup pit stop loss
            if n == 1:
                net = cumulative_gain - pit_loss_time
            else:
                net = cumulative_gain - pit_loss_time

            results.append((n, net))

        return results

    def build_tyre_state(self, compound: str) -> TyreState:
        """Create a fresh TyreState for a new set of tyres."""
        return TyreState(
            compound=compound,
            age=0,
            cumulative_linear_deg=0.0,
            cumulative_cliff_deg=0.0,
            thermal_deg=0.0,
            is_past_cliff=False,
        )

    def advance_tyre_state(
        self,
        state: TyreState,
        track_abrasiveness: float,
        track_temp_c: float,
        push_level: float,
        deg_multiplier: float = 1.0,
    ) -> TyreState:
        """
        Advance TyreState by one lap.
        Returns a new TyreState (immutable-style update).
        """
        new_age = state.age + 1
        cfg = TYRE_COMPOUNDS[state.compound]

        temp_factor = 1.0 + cfg.thermal_sensitivity * max(
            0.0, (track_temp_c - self.THERMAL_BASELINE_C) / 10.0
        )
        push_factor = 0.6 + 0.4 * push_level

        # This lap's linear increment
        linear_inc = (
            cfg.linear_deg_rate
            * track_abrasiveness
            * temp_factor
            * push_factor
            * deg_multiplier
        )

        cliff_lap_adj = self._adjusted_cliff_lap(
            state.compound, track_abrasiveness, track_temp_c, push_level
        )
        is_past_cliff = new_age > cliff_lap_adj
        cliff_inc = 0.0
        if is_past_cliff:
            laps_past = new_age - cliff_lap_adj
            cliff_inc = (
                cfg.cliff_exponent
                * (laps_past ** 2 - max(0, laps_past - 1) ** 2)
                * track_abrasiveness
                * temp_factor
                * deg_multiplier
            )

        return TyreState(
            compound=state.compound,
            age=new_age,
            cumulative_linear_deg=state.cumulative_linear_deg + linear_inc,
            cumulative_cliff_deg=state.cumulative_cliff_deg + cliff_inc,
            thermal_deg=state.thermal_deg,
            is_past_cliff=is_past_cliff,
            surface_temp_celsius=state.surface_temp_celsius,
        )
