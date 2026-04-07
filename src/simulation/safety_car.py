"""
Safety Car / VSC Model — F1 2026

Models Safety Car and Virtual Safety Car deployment during races.

Key behaviours encoded:
  - SC probability per circuit (from config)
  - SC vs VSC split (70/30)
  - Timing distribution: early (40%) / mid (35%) / late (25%)
  - Duration: 3-6 laps (SC), 2-4 laps (VSC)
  - Field compression under SC (~30 seconds)
  - Free pit-stop window: first 3 laps after SC deployment
  - Impact on undercut / overcut strategy
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, TYPE_CHECKING

from config import SC_PARAMS, CIRCUITS

if TYPE_CHECKING:
    from .strategy import Strategy


@dataclass
class SCEvent:
    """Represents a single Safety Car or VSC deployment."""
    deploy_lap: int
    end_lap: int
    event_type: str      # "SC" | "VSC"

    @property
    def duration_laps(self) -> int:
        return self.end_lap - self.deploy_lap

    @property
    def is_full_sc(self) -> bool:
        return self.event_type == "SC"

    @property
    def free_pit_window_end(self) -> int:
        """Last lap in the 'free pit stop' window under SC."""
        return self.deploy_lap + SC_PARAMS["free_pit_window_laps"]


class SafetyCarModel:
    """
    Samples and applies SC/VSC events during race simulation.
    """

    def sample_sc_events(
        self,
        total_laps: int,
        sc_probability: float,
        rng: np.random.Generator,
    ) -> List[SCEvent]:
        """
        Probabilistically sample SC/VSC events for a race.

        Parameters
        ----------
        total_laps      : race distance in laps
        sc_probability  : probability of at least one SC event (from CircuitConfig)
        rng             : seeded random number generator for reproducibility

        Returns
        -------
        Ordered list of SCEvent (may be empty).
        """
        events: List[SCEvent] = []

        # Does at least one event happen?
        if rng.random() > sc_probability:
            return events

        # Determine timing region: early / mid / late
        timing_weights = SC_PARAMS["sc_timing_weights"]
        boundaries = SC_PARAMS["sc_timing_boundaries"]

        region = rng.choice(["early", "mid", "late"], p=timing_weights)
        if region == "early":
            lap_start = max(2, int(total_laps * 0.05))
            lap_end_limit = int(total_laps * boundaries[0])
        elif region == "mid":
            lap_start = int(total_laps * boundaries[0]) + 1
            lap_end_limit = int(total_laps * boundaries[1])
        else:
            lap_start = int(total_laps * boundaries[1]) + 1
            lap_end_limit = max(lap_start + 1, total_laps - 6)

        # Clamp
        lap_start = max(2, min(lap_start, total_laps - 6))
        lap_end_limit = max(lap_start, min(lap_end_limit, total_laps - 4))

        deploy_lap = int(rng.integers(lap_start, lap_end_limit + 1))

        # SC vs VSC split
        is_vsc = rng.random() < SC_PARAMS["vsc_fraction_of_sc"]
        event_type = "VSC" if is_vsc else "SC"

        dur_range = SC_PARAMS["vsc_typical_duration_laps" if is_vsc else "sc_typical_duration_laps"]
        duration = int(rng.integers(dur_range[0], dur_range[1] + 1))
        end_lap = min(deploy_lap + duration, total_laps - 1)

        events.append(SCEvent(deploy_lap=deploy_lap, end_lap=end_lap, event_type=event_type))

        # Small chance of a second SC event (e.g. multi-incident race)
        # Guard: need end_lap + 5 < total_laps - 4 for valid rng.integers range
        if rng.random() < 0.15 and end_lap + 10 < total_laps:
            second_deploy = int(rng.integers(end_lap + 5, total_laps - 4))
            second_dur = int(rng.integers(dur_range[0], dur_range[1] + 1))
            second_end = min(second_deploy + second_dur, total_laps - 1)
            is_vsc2 = rng.random() < SC_PARAMS["vsc_fraction_of_sc"]
            events.append(SCEvent(
                deploy_lap=second_deploy,
                end_lap=second_end,
                event_type="VSC" if is_vsc2 else "SC",
            ))

        return sorted(events, key=lambda e: e.deploy_lap)

    def get_lap_multiplier(
        self,
        lap: int,
        sc_events: List[SCEvent],
    ) -> float:
        """
        Return lap-time multiplier for a given lap.
        1.0 = normal racing; > 1.0 = SC/VSC slow-down.
        """
        for event in sc_events:
            if event.deploy_lap <= lap <= event.end_lap:
                if event.event_type == "SC":
                    return SC_PARAMS["sc_lap_time_multiplier"]
                else:
                    return SC_PARAMS["vsc_lap_time_multiplier"]
        return 1.0

    def apply_sc_to_lap_times(
        self,
        lap_times: np.ndarray,
        sc_events: List[SCEvent],
    ) -> np.ndarray:
        """
        Apply SC/VSC time multipliers to an array of lap times.

        Parameters
        ----------
        lap_times : array of shape (total_laps,), 1-indexed lap 1 = index 0
        sc_events : list of SCEvent objects

        Returns
        -------
        Modified lap times array.
        """
        modified = lap_times.copy()
        for event in sc_events:
            for lap in range(event.deploy_lap, event.end_lap + 1):
                if 1 <= lap <= len(lap_times):
                    multiplier = (
                        SC_PARAMS["sc_lap_time_multiplier"]
                        if event.event_type == "SC"
                        else SC_PARAMS["vsc_lap_time_multiplier"]
                    )
                    modified[lap - 1] = lap_times[lap - 1] * multiplier
        return modified

    def is_free_pit_window(
        self,
        lap: int,
        sc_events: List[SCEvent],
    ) -> bool:
        """
        Returns True if this lap falls inside the free-pit-stop window
        of any active SC event (first 3 laps after SC deployment).
        """
        for event in sc_events:
            if event.event_type == "SC":
                if event.deploy_lap <= lap <= event.free_pit_window_end:
                    return True
        return False

    def load_empirical_sc_rates(self) -> dict:
        """
        Read sc_history.json and return {circuit_key: sc_rate} mapping.
        Falls back to {} on any error.
        """
        try:
            import json
            from pathlib import Path
            sc_path = Path(__file__).parent.parent / "data" / "calibration" / "sc_history.json"
            if sc_path.exists():
                with open(sc_path) as f:
                    data = json.load(f)
                result = {}
                for circuit_key, info in data.items():
                    if isinstance(info, dict) and "sc_rate" in info:
                        result[circuit_key] = float(info["sc_rate"])
                    elif isinstance(info, (int, float)):
                        result[circuit_key] = float(info)
                return result
        except Exception:
            pass
        return {}

    def compute_strategic_impact(
        self,
        strategy: "Strategy",
        sc_events: List[SCEvent],
        pit_loss_time: float,
    ) -> float:
        """
        Estimate time gain or loss from SC events given this strategy's pit timing.

        Logic:
        - If a pit stop falls in the SC free window → gain ≈ pit_loss_time - SC_compression
        - If SC compresses field while car is in pit lane → potential position loss
        - VSC gives ~half the compression benefit of SC

        Returns
        -------
        Net time delta (negative = gain, positive = loss).
        """
        if not sc_events:
            return 0.0

        net_delta = 0.0
        pit_laps = set(strategy.pit_laps)

        for event in sc_events:
            if event.event_type == "SC":
                # Full SC compresses the field — worth ~30s if pitting under it
                compression = SC_PARAMS["sc_field_compression_s"]
                free_window = event.free_pit_window_end
            else:
                # VSC does NOT compress the field (cars hold their gaps).
                # Strategic benefit is reduced pit-lane time loss only:
                # field is slow so your pit-lane speed limit costs ~60% less
                compression = 0.0
                free_window = event.free_pit_window_end

            this_strategy_pits_free = any(
                event.deploy_lap <= pl <= free_window
                for pl in pit_laps
            )

            if this_strategy_pits_free:
                if event.event_type == "SC":
                    # Free pit under SC: save ~80% of pit loss vs staying out
                    net_delta -= pit_loss_time * 0.80
                else:
                    # VSC pit: save ~35% of pit loss (pit lane still costs time
                    # since VSC speed is higher than SC speed)
                    net_delta -= pit_loss_time * 0.35
            else:
                if event.event_type == "SC":
                    # Rivals can take a free SC pit; we lose ground vs field
                    net_delta += compression * 0.35

        return net_delta
