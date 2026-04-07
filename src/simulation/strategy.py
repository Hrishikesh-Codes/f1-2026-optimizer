"""
Strategy Representation & Generator — F1 2026

Handles:
  - Strategy and Stint dataclasses
  - Validation against 2026 regulations (≥2 compounds, ≥1 pit stop)
  - Generating all valid 1-stop, 2-stop, 3-stop strategies for a circuit
  - Undercut / overcut window calculation
  - OOM lap recommendations
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import numpy as np

from config import CIRCUITS, TYRE_COMPOUNDS, SIM_PARAMS


@dataclass
class Stint:
    """One continuous period on a single tyre compound."""
    compound: str
    start_lap: int      # 1-indexed race lap where this stint begins
    end_lap: int        # last racing lap of this stint (pit happens after this lap)
    push_level: float = 0.80   # 0.0 = heavy save, 1.0 = flat-out

    @property
    def length(self) -> int:
        return self.end_lap - self.start_lap + 1

    def __repr__(self) -> str:
        return (
            f"Stint({self.compound}, laps {self.start_lap}–{self.end_lap}, "
            f"push={self.push_level:.1f})"
        )


@dataclass
class Strategy:
    """Full race strategy: ordered list of stints."""
    stints: List[Stint]

    @property
    def pit_laps(self) -> List[int]:
        """Race laps on which pit stops occur (after each stint except the last)."""
        return [s.end_lap for s in self.stints[:-1]]

    @property
    def num_stops(self) -> int:
        return len(self.stints) - 1

    @property
    def compound_sequence(self) -> str:
        """Human-readable compound sequence using raw codes, e.g. 'C2-C3'."""
        return "-".join(s.compound for s in self.stints)

    @property
    def full_compound_sequence(self) -> str:
        return " → ".join(
            f"{s.compound} ({s.length}L)" for s in self.stints
        )

    def uses_n_compounds(self) -> int:
        return len({s.compound for s in self.stints})

    def is_valid(self, circuit_key: str) -> bool:
        """
        Validate against 2026 regulations and circuit constraints.

        Rules:
        - Must use ≥ 2 different dry compounds
        - All compounds must be in the circuit's allocated set
        - Minimum 1 pit stop
        - No stint shorter than MIN_STOP_SPACING_LAPS
        - Total laps must equal circuit total_laps
        """
        cfg = CIRCUITS[circuit_key]

        if self.num_stops < 1:
            return False
        if self.uses_n_compounds() < 2:
            return False
        allowed = set(cfg.compounds)
        if any(s.compound not in allowed for s in self.stints):
            return False
        if any(s.length < SIM_PARAMS["min_stop_spacing_laps"] for s in self.stints):
            return False
        # Total laps covered
        last_lap = self.stints[-1].end_lap
        if last_lap != cfg.total_laps:
            return False
        return True

    def __repr__(self) -> str:
        return f"Strategy({self.full_compound_sequence})"


@dataclass
class StrategyResult:
    """Result from simulating a single strategy."""
    strategy: Strategy
    total_time: float
    lap_times: np.ndarray
    sc_events: list        # List[SCEvent]
    mean_time: float = 0.0
    std_time: float = 0.0
    undercut_windows: List[Tuple[int, int]] = field(default_factory=list)
    oom_recommendations: List[int] = field(default_factory=list)


class StrategyGenerator:
    """
    Generates all valid F1 race strategies for a given circuit.

    Approach:
    1. Enumerate all compound permutations for 1, 2, and 3 pit-stop races.
    2. For each compound sequence, generate ~5-8 candidate pit-stop lap timings.
    3. Validate each strategy against 2026 rules.
    4. Return deduplicated list.
    """

    MIN_STINT_LAPS = SIM_PARAMS["min_stop_spacing_laps"]

    def _compound_sequences(
        self, available: List[str], num_stints: int
    ) -> List[Tuple[str, ...]]:
        """
        Return all ordered compound sequences of length num_stints
        that use at least 2 different compounds from available.
        """
        seqs = list(itertools.product(available, repeat=num_stints))
        return [
            seq for seq in seqs
            if len(set(seq)) >= 2
        ]

    def _candidate_pit_laps(
        self, total_laps: int, num_stops: int
    ) -> List[Tuple[int, ...]]:
        """
        Generate candidate pit-stop lap combinations for num_stops stops.
        Samples ~5-8 evenly-spaced options plus early/late variants.
        """
        min_l = self.MIN_STINT_LAPS
        candidates: List[Tuple[int, ...]] = []

        if num_stops == 1:
            # Real 2026 data: 1-stop pits cluster at 44–54% of race distance.
            # Widen slightly to 38–62% to capture undercut/overcut variants.
            lo = max(min_l, int(total_laps * 0.38))
            hi = min(total_laps - min_l, int(total_laps * 0.62))
            step = max(1, (hi - lo) // 10)
            for p in range(lo, hi + 1, step):
                candidates.append((p,))

        elif num_stops == 2:
            lo1 = max(min_l, int(total_laps * 0.22))
            hi1 = int(total_laps * 0.45)
            lo2 = int(total_laps * 0.50)
            hi2 = min(total_laps - min_l, int(total_laps * 0.75))
            step = max(1, (hi1 - lo1) // 4)
            for p1 in range(lo1, hi1 + 1, step):
                step2 = max(1, (hi2 - lo2) // 4)
                for p2 in range(lo2, hi2 + 1, step2):
                    if p2 > p1 + min_l:
                        candidates.append((p1, p2))

        elif num_stops == 3:
            spacing = total_laps // 4
            base_pits = [spacing, 2 * spacing, 3 * spacing]
            offsets = [-3, 0, 3]
            for o1 in offsets:
                for o2 in offsets:
                    for o3 in offsets:
                        pits = (
                            base_pits[0] + o1,
                            base_pits[1] + o2,
                            base_pits[2] + o3,
                        )
                        if (
                            pits[0] >= min_l
                            and pits[1] > pits[0] + min_l
                            and pits[2] > pits[1] + min_l
                            and pits[2] < total_laps - min_l
                        ):
                            candidates.append(pits)

        return list(dict.fromkeys(candidates))   # deduplicate while preserving order

    def generate_all_strategies(
        self,
        circuit_key: str,
        max_stops: int = 3,
    ) -> List[Strategy]:
        """
        Generate all valid race strategies for a circuit.

        Parameters
        ----------
        circuit_key : key in CIRCUITS dict
        max_stops   : maximum pit stops to consider (1, 2, or 3)

        Returns
        -------
        List of valid Strategy objects, sorted by num_stops then first pit lap.
        """
        cfg = CIRCUITS[circuit_key]
        total_laps = cfg.total_laps
        available = cfg.compounds

        strategies: List[Strategy] = []
        seen: set = set()

        for num_stops in range(1, max_stops + 1):
            num_stints = num_stops + 1
            compound_seqs = self._compound_sequences(available, num_stints)
            pit_lap_combos = self._candidate_pit_laps(total_laps, num_stops)

            for compound_seq in compound_seqs:
                for pit_laps in pit_lap_combos:
                    # Build stints
                    lap_boundaries = [0] + list(pit_laps) + [total_laps]
                    stints = []
                    valid = True

                    for i, compound in enumerate(compound_seq):
                        start_lap = lap_boundaries[i] + 1
                        end_lap = lap_boundaries[i + 1]
                        if start_lap > end_lap:
                            valid = False
                            break
                        stints.append(Stint(
                            compound=compound,
                            start_lap=start_lap,
                            end_lap=end_lap,
                        ))

                    if not valid:
                        continue

                    strategy = Strategy(stints=stints)
                    if not strategy.is_valid(circuit_key):
                        continue

                    key = (compound_seq, tuple(pit_laps))
                    if key not in seen:
                        seen.add(key)
                        strategies.append(strategy)

        return sorted(strategies, key=lambda s: (s.num_stops, s.pit_laps[0] if s.pit_laps else 0))

    def compute_undercut_windows(
        self,
        circuit_key: str,
        compound_old: str,
        compound_new: str,
        laps_on_old: int,
        current_lap: int,
        push_level: float = 0.85,
    ) -> List[Tuple[int, float]]:
        """
        Compute undercut viability over the next 15 laps.

        Returns list of (future_lap, net_time_gain) where positive
        gain means undercutting on that lap saves time overall.
        """
        from .tyre import TyreDegradationModel
        tyre_model = TyreDegradationModel()
        cfg = CIRCUITS[circuit_key]

        results = tyre_model.compute_undercut_window(
            compound_old=compound_old,
            compound_new=compound_new,
            laps_on_old=laps_on_old,
            pit_loss_time=cfg.pit_loss_time,
            abrasiveness=cfg.track_abrasiveness,
            track_temp_c=35.0,
            push_level=push_level,
            window_laps=min(15, cfg.total_laps - current_lap),
        )

        return [(current_lap + offset, gain) for offset, gain in results]

    def recommend_oom_laps(
        self,
        strategy: Strategy,
        circuit_key: str,
        team: str,
    ) -> Dict[int, str]:
        """
        Recommend OOM usage laps for a given strategy.

        Heuristics:
        - Last 10 laps of each stint where battery is expected to be healthy
        - Avoid final 2 laps of stint (save battery for first laps of new stint warmup)
        - Prioritise mid-stint where tyre pace is still strong

        Returns dict of {race_lap: recommendation_string}
        """
        from config import TEAMS
        team_cfg = TEAMS[team]
        recommendations: Dict[int, str] = {}

        for stint in strategy.stints:
            stint_mid = (stint.start_lap + stint.end_lap) // 2
            # Recommend OOM in the second half of each stint
            oom_start = max(stint.start_lap + 5, stint_mid - 2)
            oom_end = max(stint.start_lap, stint.end_lap - 3)

            for lap in range(oom_start, oom_end + 1):
                recommendations[lap] = (
                    f"OOM window: {stint.compound} stint, lap {lap} "
                    f"(~{lap - stint.start_lap + 1} laps old, "
                    f"{stint.end_lap - lap} to next stop/end)"
                )

        return recommendations
