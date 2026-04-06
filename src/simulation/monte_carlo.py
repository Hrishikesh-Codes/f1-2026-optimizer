"""
Monte Carlo Race Simulator — F1 2026

Runs 1000+ stochastic race simulations per strategy to determine:
  - Optimal strategy (lowest mean total race time)
  - Alternative strategies within 5-second window
  - Safety Car impact analysis
  - Undercut window identification
  - OOM usage recommendations
  - 95% confidence intervals on race time

Stochastic variables sampled per simulation:
  - SC deployment (timing, type, duration)
  - Tyre degradation variance (±10%)
  - Track temperature variation (±3°C)
  - Fuel burn rate variance (±4%)
  - Lap-time noise (σ = 0.05s)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import (
    CIRCUITS, TEAMS, FUEL_PARAMS, SIM_PARAMS,
    TYRE_COMPOUNDS, ERS_PARAMS,
    load_calibration_overrides,
)
from .tyre import TyreDegradationModel
from .ers import ERSModel, ERSState
from .safety_car import SafetyCarModel, SCEvent
from .laptime import LapTimeModel
from .strategy import Strategy, Stint, StrategyGenerator, StrategyResult


@dataclass
class MonteCarloConfig:
    """Configuration for the Monte Carlo simulation run."""
    num_simulations: int = SIM_PARAMS["default_simulations"]
    deg_variance_sigma: float = SIM_PARAMS["deg_variance_sigma"]
    include_sc: bool = True
    include_weather_variance: bool = True
    max_stops: int = 3
    parallel: bool = True
    random_seed: Optional[int] = None


@dataclass
class MonteCarloResult:
    """Aggregated results from a full Monte Carlo run."""
    circuit: str
    team: str
    driver: str
    optimal_strategy: Strategy
    optimal_strategy_mean_time: float
    optimal_strategy_std_time: float
    alternative_strategies: List[Tuple[Strategy, float, float]]   # (strategy, mean_s, std_s)
    sc_impact_analysis: Dict[int, float]                          # {sc_lap: time_delta_s}
    undercut_windows: List[Tuple[int, float]]                     # (lap, net_gain_s)
    oom_recommendations: Dict[int, str]
    confidence_interval_95: Tuple[float, float]
    simulation_count: int
    strategy_win_distribution: Dict[str, int]   # compound_sequence → wins
    all_strategy_results: List[Tuple[Strategy, float, float]]     # sorted by mean time


class MonteCarloSimulator:
    """
    Monte Carlo race strategy optimizer.

    Usage
    -----
    sim = MonteCarloSimulator(MonteCarloConfig(num_simulations=1000))
    result = sim.run(circuit="australia", team="Ferrari", driver="Charles Leclerc")
    """

    def __init__(self, config: Optional[MonteCarloConfig] = None) -> None:
        self.config = config or MonteCarloConfig()
        self.calibration = load_calibration_overrides()
        self.tyre_model = TyreDegradationModel(calibration=self.calibration)
        self.ers_model = ERSModel()
        self.sc_model = SafetyCarModel()
        self.laptime_model = LapTimeModel()
        self.strategy_gen = StrategyGenerator()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        circuit: str,
        team: str,
        driver: str,
        track_temp_c: float = SIM_PARAMS["baseline_track_temp_c"],
        starting_compound: Optional[str] = None,
    ) -> MonteCarloResult:
        """
        Run the full Monte Carlo optimization for a circuit/team combination.

        Parameters
        ----------
        circuit          : circuit key (e.g. "australia")
        team             : team name (e.g. "Ferrari")
        driver           : driver full name (e.g. "Charles Leclerc")
        track_temp_c     : baseline track temperature in °C
        starting_compound: force a starting compound (None = model selects best)

        Returns
        -------
        MonteCarloResult with optimal strategy, alternatives, and analytics
        """
        circuit_cfg = CIRCUITS[circuit]
        rng_master = np.random.default_rng(self.config.random_seed)

        # Generate all candidate strategies
        strategies = self.strategy_gen.generate_all_strategies(
            circuit_key=circuit,
            max_stops=self.config.max_stops,
        )

        if starting_compound:
            strategies = [
                s for s in strategies
                if s.stints[0].compound == starting_compound
            ]

        if not strategies:
            raise ValueError(
                f"No valid strategies found for {circuit} "
                f"with starting compound {starting_compound}"
            )

        # Cap to top N strategies to keep runtime manageable
        strategies = strategies[:40]

        # Run Monte Carlo for each strategy
        strategy_times: Dict[int, List[float]] = {i: [] for i in range(len(strategies))}
        win_counts: Dict[int, int] = {i: 0 for i in range(len(strategies))}

        seeds = rng_master.integers(0, 2**31, size=self.config.num_simulations)

        for sim_idx in range(self.config.num_simulations):
            sim_rng = np.random.default_rng(int(seeds[sim_idx]))

            # Sample stochastic variables for this simulation
            deg_mult = float(sim_rng.normal(1.0, self.config.deg_variance_sigma))
            deg_mult = np.clip(deg_mult, 0.75, 1.30)

            temp_c = float(sim_rng.normal(
                track_temp_c, SIM_PARAMS["track_temp_variation_sigma"]
            ))
            temp_c = np.clip(temp_c, 15.0, 60.0)

            sc_events = []
            if self.config.include_sc:
                sc_events = self.sc_model.sample_sc_events(
                    total_laps=circuit_cfg.total_laps,
                    sc_probability=circuit_cfg.sc_probability,
                    rng=sim_rng,
                )

            # Evaluate all strategies under these conditions
            best_time = float("inf")
            best_idx = 0
            sim_times: List[float] = []

            for strat_idx, strategy in enumerate(strategies):
                race_time = self._simulate_single_race(
                    circuit=circuit,
                    team=team,
                    strategy=strategy,
                    sc_events=sc_events,
                    deg_multiplier=deg_mult,
                    track_temp_c=temp_c,
                    rng=sim_rng,
                )
                strategy_times[strat_idx].append(race_time)
                sim_times.append(race_time)
                if race_time < best_time:
                    best_time = race_time
                    best_idx = strat_idx

            win_counts[best_idx] += 1

        # Aggregate results
        strategy_stats: List[Tuple[int, float, float]] = []
        for idx in range(len(strategies)):
            times = np.array(strategy_times[idx])
            strategy_stats.append((idx, float(np.mean(times)), float(np.std(times))))

        strategy_stats.sort(key=lambda x: x[1])  # sort by mean time

        optimal_idx, optimal_mean, optimal_std = strategy_stats[0]
        optimal_strategy = strategies[optimal_idx]

        # Alternatives: within SIM_PARAMS["strategy_window_s"] of optimal
        window = SIM_PARAMS["strategy_window_s"]
        alternatives = [
            (strategies[idx], mean_t, std_t)
            for idx, mean_t, std_t in strategy_stats[1:]
            if mean_t - optimal_mean <= window
        ][:6]

        # All results sorted
        all_results = [
            (strategies[idx], mean_t, std_t)
            for idx, mean_t, std_t in strategy_stats
        ]

        # 95% CI on optimal strategy
        opt_times = np.array(strategy_times[optimal_idx])
        ci_low = float(np.percentile(opt_times, 2.5))
        ci_high = float(np.percentile(opt_times, 97.5))

        # SC impact analysis
        sc_impact = self.compute_sc_impact_analysis(
            circuit=circuit,
            team=team,
            optimal_strategy=optimal_strategy,
            track_temp_c=track_temp_c,
        )

        # Undercut windows (first stint of optimal strategy)
        # Multiply net_gain by (1 - overtake_difficulty) as probability factor:
        # harder overtaking = undercut more valuable (higher success rate)
        undercut_windows: List[Tuple[int, float]] = []
        if optimal_strategy.stints and len(optimal_strategy.stints) > 1:
            first_stint = optimal_strategy.stints[0]
            raw_windows = self.strategy_gen.compute_undercut_windows(
                circuit_key=circuit,
                compound_old=first_stint.compound,
                compound_new=optimal_strategy.stints[1].compound,
                laps_on_old=first_stint.length - 5,
                current_lap=first_stint.end_lap - 8,
            )
            # Scale by undercut success probability
            overtake_diff = circuit_cfg.overtake_difficulty
            undercut_value_factor = 1.0 + overtake_diff  # harder to overtake = undercut more valuable
            undercut_windows = [
                (lap, gain * undercut_value_factor)
                for lap, gain in raw_windows
            ]

        # OOM recommendations
        oom_recs = self.strategy_gen.recommend_oom_laps(
            strategy=optimal_strategy,
            circuit_key=circuit,
            team=team,
        )

        # Win distribution
        win_dist = {
            strategies[idx].full_compound_sequence: count
            for idx, count in win_counts.items()
            if count > 0
        }

        return MonteCarloResult(
            circuit=circuit,
            team=team,
            driver=driver,
            optimal_strategy=optimal_strategy,
            optimal_strategy_mean_time=optimal_mean,
            optimal_strategy_std_time=optimal_std,
            alternative_strategies=alternatives,
            sc_impact_analysis=sc_impact,
            undercut_windows=undercut_windows,
            oom_recommendations=oom_recs,
            confidence_interval_95=(ci_low, ci_high),
            simulation_count=self.config.num_simulations,
            strategy_win_distribution=win_dist,
            all_strategy_results=all_results,
        )

    # ------------------------------------------------------------------
    # Core simulation
    # ------------------------------------------------------------------

    def _simulate_single_race(
        self,
        circuit: str,
        team: str,
        strategy: Strategy,
        sc_events: List[SCEvent],
        deg_multiplier: float,
        track_temp_c: float,
        rng: np.random.Generator,
    ) -> float:
        """
        Simulate a single race with given strategy and stochastic conditions.

        Returns total race time in seconds (excluding formation lap).
        """
        circuit_cfg = CIRCUITS[circuit]
        team_cfg = TEAMS[team]
        total_laps = circuit_cfg.total_laps

        fuel_kg = FUEL_PARAMS["fuel_load_kg"]
        fuel_per_lap = circuit_cfg.fuel_per_lap * rng.normal(
            1.0, FUEL_PARAMS["fuel_burn_variance"]
        )

        ers_state = ERSModel.fresh_state(0.80)
        total_time = 0.0

        from config import TYRE_COMPOUNDS as TC

        for stint_idx, stint in enumerate(strategy.stints):
            tyre_age = 0

            for race_lap in range(stint.start_lap, stint.end_lap + 1):
                if race_lap > total_laps:
                    break

                sc_mult = self.sc_model.get_lap_multiplier(race_lap, sc_events)
                under_sc = sc_mult > 1.0

                # Laps remaining in stint — conserve battery near pit stop
                laps_left_in_stint = stint.end_lap - race_lap
                near_pit = laps_left_in_stint <= 3

                # ERS boost: deploy on green-flag laps with healthy battery,
                # but hold off when close to pit (fresh tyres will benefit more)
                use_boost = (
                    not under_sc
                    and not near_pit
                    and ers_state.can_boost
                    and ers_state.state_of_charge > 0.50
                )
                # Super clip preferred (no aero penalty); lift-off when battery nearly full
                mode = "lift_off" if ers_state.state_of_charge > 0.88 else "super_clip"

                # Lap time components
                compound_cfg = TC[stint.compound]
                base = circuit_cfg.base_lap_time

                # Fuel
                base += max(0.0, fuel_kg) * FUEL_PARAMS["fuel_effect_per_kg_s"]

                # Compound offset
                base += compound_cfg.base_pace_offset

                # Tyre degradation
                deg = self.tyre_model.compute_lap_time_delta(
                    compound=stint.compound,
                    lap_in_stint=tyre_age + 1,
                    track_abrasiveness=circuit_cfg.track_abrasiveness,
                    track_temp_celsius=track_temp_c,
                    push_level=stint.push_level if not under_sc else 0.4,
                ) * deg_multiplier
                base += deg

                # ERS
                ers_delta, ers_state = self.ers_model.compute_lap_ers_delta(
                    ers_state=ers_state,
                    use_boost=use_boost,
                    recharge_mode=mode,
                    pu_supplier=team_cfg.pu_supplier,
                    circuit=circuit,
                )
                base += ers_delta

                # Team delta
                base += team_cfg.base_lap_delta

                # Track evolution
                evo = circuit_cfg.track_evolution_rate * min(race_lap, 25) * (-1.0)
                base += evo

                # Noise
                base += rng.normal(0.0, SIM_PARAMS["lap_time_noise_sigma"])

                # SC multiplier
                lap_time = base * sc_mult

                total_time += max(lap_time, circuit_cfg.base_lap_time * 0.88)
                tyre_age += 1
                fuel_kg -= fuel_per_lap

            # Pit stop time
            if stint_idx < len(strategy.stints) - 1:
                total_time += circuit_cfg.pit_loss_time
                # Partial battery recovery in pit lane
                ers_state = ERSState(
                    battery_level_mj=min(
                        ers_state.battery_level_mj + 0.25,
                        ERS_PARAMS["battery_capacity_mj"],
                    ),
                    battery_capacity_mj=ERS_PARAMS["battery_capacity_mj"],
                )

        return total_time

    # ------------------------------------------------------------------
    # SC Impact Analysis
    # ------------------------------------------------------------------

    def compute_sc_impact_analysis(
        self,
        circuit: str,
        team: str,
        optimal_strategy: Strategy,
        track_temp_c: float = SIM_PARAMS["baseline_track_temp_c"],
    ) -> Dict[int, float]:
        """
        For each test SC lap, return the STRATEGIC time delta vs the field.

        Under SC, all cars slow by the same amount — the absolute race-time
        change is irrelevant. What matters is whether this strategy's pit
        windows align with the SC free-pit window, giving a position/time
        advantage over rivals.

        Negative = SC helps (free pit window aligns with planned stop).
        Positive = SC hurts (rivals can take a free pit, you cannot).

        Returns {sc_lap: strategic_time_delta_seconds}
        """
        circuit_cfg = CIRCUITS[circuit]
        test_laps = [
            int(circuit_cfg.total_laps * f)
            for f in [0.17, 0.33, 0.50, 0.67]
            if int(circuit_cfg.total_laps * f) > 2
        ]

        impact: Dict[int, float] = {}
        for test_lap in test_laps:
            sc_event = SCEvent(
                deploy_lap=test_lap,
                end_lap=min(test_lap + 4, circuit_cfg.total_laps - 1),
                event_type="SC",
            )
            # Use strategic impact relative to field (not raw absolute time)
            delta = self.sc_model.compute_strategic_impact(
                strategy=optimal_strategy,
                sc_events=[sc_event],
                pit_loss_time=circuit_cfg.pit_loss_time,
            )
            impact[test_lap] = delta

        return impact

    # ------------------------------------------------------------------
    # Quick stint analysis (for Stint Calculator page)
    # ------------------------------------------------------------------

    def analyze_current_stint(
        self,
        circuit: str,
        team: str,
        current_lap: int,
        compound: str,
        tyre_age: int,
        battery_fraction: float,
        gap_ahead: float,
        track_temp_c: float = 35.0,
    ) -> Dict:
        """
        Given current race state, return:
          - Laps remaining before tyre cliff
          - Undercut window (next N laps)
          - OOM recommendation
          - Optimal pit lap

        Returns a dict suitable for the Stint Calculator UI.
        """
        circuit_cfg = CIRCUITS[circuit]
        tyre_model = TyreDegradationModel()
        ers_state = ERSModel.fresh_state(battery_fraction)

        # Cliff lap
        cliff_lap = tyre_model.compute_cliff_lap(
            compound, circuit_cfg.track_abrasiveness, track_temp_c
        )
        laps_to_cliff = max(0, cliff_lap - tyre_age)

        # Stint length recommendation
        min_l, opt_l, max_l = tyre_model.estimate_optimal_stint_length(
            compound, circuit_cfg.track_abrasiveness, track_temp_c
        )
        optimal_pit_lap = current_lap + max(0, opt_l - tyre_age)
        laps_remaining_race = circuit_cfg.total_laps - current_lap

        # Undercut window — check all available compounds
        undercut_info = {}
        for new_compound in circuit_cfg.compounds:
            if new_compound == compound:
                continue
            window = tyre_model.compute_undercut_window(
                compound_old=compound,
                compound_new=new_compound,
                laps_on_old=tyre_age,
                pit_loss_time=circuit_cfg.pit_loss_time,
                abrasiveness=circuit_cfg.track_abrasiveness,
                track_temp_c=track_temp_c,
                window_laps=min(12, laps_remaining_race),
            )
            positive = [(lap_offset + current_lap, gain) for lap_offset, gain in window if gain > 0]
            if positive:
                undercut_info[new_compound] = positive

        # OOM decision
        from .ers import ERSModel as EM
        erm = EM()
        oom_use, oom_reason = erm.compute_oom_decision(
            ers_state=ers_state,
            gap_to_car_ahead=gap_ahead,
            laps_remaining=laps_remaining_race,
            tyre_age=tyre_age,
            compound=compound,
        )

        return {
            "laps_to_cliff": laps_to_cliff,
            "cliff_lap_in_stint": cliff_lap,
            "optimal_pit_lap": min(optimal_pit_lap, circuit_cfg.total_laps - 5),
            "max_laps_on_compound": max_l,
            "undercut_windows": undercut_info,
            "oom_recommended": oom_use,
            "oom_reason": oom_reason,
            "battery_level_mj": ers_state.battery_level_mj,
            "laps_remaining": laps_remaining_race,
        }
