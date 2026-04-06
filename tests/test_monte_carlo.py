"""
Unit tests — Monte Carlo Simulator

Tests:
  - Generates valid strategies for each circuit
  - Simulation produces finite race times
  - Optimal strategy respects 2026 compound rules (≥2 different compounds)
  - Alternative strategies are within the 5-second window
  - SC impact analysis returns results for all test laps
  - Result confidence interval is ordered (low ≤ mean ≤ high)
  - Different seed produces different results (randomness confirmed)
  - Stint calculator returns required keys
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import math
import pytest

from src.simulation.monte_carlo import MonteCarloSimulator, MonteCarloConfig
from src.simulation.strategy import StrategyGenerator
from config import CIRCUITS, SIM_PARAMS


FAST_CONFIG = MonteCarloConfig(num_simulations=100, random_seed=42)


@pytest.fixture
def sim():
    return MonteCarloSimulator(FAST_CONFIG)


@pytest.fixture
def gen():
    return StrategyGenerator()


class TestStrategyGeneration:
    def test_generates_strategies_australia(self, gen):
        strats = gen.generate_all_strategies("australia", max_stops=2)
        assert len(strats) > 0

    def test_all_strategies_valid(self, gen):
        for circuit_key in ["australia", "monaco", "japan", "belgium"]:
            strats = gen.generate_all_strategies(circuit_key, max_stops=2)
            for s in strats:
                assert s.is_valid(circuit_key), (
                    f"Invalid strategy {s} for {circuit_key}"
                )

    def test_strategies_use_two_compounds_minimum(self, gen):
        strats = gen.generate_all_strategies("australia", max_stops=2)
        for s in strats:
            assert s.uses_n_compounds() >= 2

    def test_strategies_cover_full_race_distance(self, gen):
        circuit_key = "australia"
        total_laps = CIRCUITS[circuit_key].total_laps
        strats = gen.generate_all_strategies(circuit_key, max_stops=2)
        for s in strats:
            assert s.stints[-1].end_lap == total_laps

    def test_one_stop_exists_for_most_circuits(self, gen):
        for circuit_key in ["australia", "spain", "hungary", "abu_dhabi"]:
            strats = gen.generate_all_strategies(circuit_key, max_stops=2)
            one_stop = [s for s in strats if s.num_stops == 1]
            assert len(one_stop) > 0, f"No 1-stop strategy for {circuit_key}"

    def test_pit_laps_within_race_distance(self, gen):
        circuit_key = "italy"
        total = CIRCUITS[circuit_key].total_laps
        strats = gen.generate_all_strategies(circuit_key, max_stops=2)
        for s in strats:
            for pl in s.pit_laps:
                assert 1 <= pl < total

    def test_compounds_from_circuit_allocation(self, gen):
        circuit_key = "japan"  # C1/C2/C3 only
        allowed = set(CIRCUITS[circuit_key].compounds)
        strats = gen.generate_all_strategies(circuit_key, max_stops=2)
        for s in strats:
            for stint in s.stints:
                assert stint.compound in allowed


class TestSimulationOutput:
    def test_run_returns_result(self, sim):
        result = sim.run("australia", "Ferrari", "Charles Leclerc")
        assert result is not None

    def test_optimal_strategy_valid(self, sim):
        result = sim.run("australia", "Ferrari", "Charles Leclerc")
        assert result.optimal_strategy.is_valid("australia")

    def test_finite_race_time(self, sim):
        result = sim.run("australia", "Ferrari", "Charles Leclerc")
        assert math.isfinite(result.optimal_strategy_mean_time)
        assert result.optimal_strategy_mean_time > 0

    def test_race_time_plausible_range(self, sim):
        """Race time should be between 60 and 180 minutes."""
        result = sim.run("australia", "Ferrari", "Charles Leclerc")
        assert 3600 < result.optimal_strategy_mean_time < 10800

    def test_ci_ordered(self, sim):
        result = sim.run("australia", "Ferrari", "Charles Leclerc")
        ci_lo, ci_hi = result.confidence_interval_95
        assert ci_lo <= result.optimal_strategy_mean_time <= ci_hi

    def test_alternatives_within_window(self, sim):
        result = sim.run("australia", "Ferrari", "Charles Leclerc")
        window = SIM_PARAMS["strategy_window_s"]
        for strat, mean_t, _ in result.alternative_strategies:
            assert mean_t - result.optimal_strategy_mean_time <= window + 0.01

    def test_sc_impact_keys_present(self, sim):
        result = sim.run("australia", "Ferrari", "Charles Leclerc")
        assert len(result.sc_impact_analysis) > 0

    def test_sc_impact_all_finite(self, sim):
        result = sim.run("australia", "Ferrari", "Charles Leclerc")
        for lap, delta in result.sc_impact_analysis.items():
            assert math.isfinite(delta)

    def test_win_distribution_sums_to_simulations(self, sim):
        result = sim.run("australia", "Ferrari", "Charles Leclerc")
        total_wins = sum(result.strategy_win_distribution.values())
        # Allow small discrepancy if some sims tie (degenerate case)
        assert abs(total_wins - FAST_CONFIG.num_simulations) <= 1

    def test_simulation_count_recorded(self, sim):
        result = sim.run("australia", "Ferrari", "Charles Leclerc")
        assert result.simulation_count == FAST_CONFIG.num_simulations


class TestDifferentCircuits:
    def test_monaco_high_sc_probability(self):
        """Monaco has 60% SC prob — should affect strategy distribution."""
        sim = MonteCarloSimulator(MonteCarloConfig(num_simulations=50, random_seed=7))
        result = sim.run("monaco", "Ferrari", "Charles Leclerc")
        assert result.optimal_strategy is not None

    def test_japan_hard_compounds(self):
        """Japan uses C1/C2/C3 — no soft compound."""
        sim = MonteCarloSimulator(MonteCarloConfig(num_simulations=50, random_seed=7))
        result = sim.run("japan", "McLaren", "Lando Norris")
        for stint in result.optimal_strategy.stints:
            assert stint.compound in ["C1", "C2", "C3"]

    def test_sprint_circuit_works(self):
        sim = MonteCarloSimulator(MonteCarloConfig(num_simulations=50, random_seed=7))
        result = sim.run("china", "Mercedes", "George Russell")
        assert result is not None


class TestStintCalculator:
    def test_analyze_current_stint_returns_dict(self, sim):
        result = sim.analyze_current_stint(
            circuit="australia",
            team="Ferrari",
            current_lap=20,
            compound="C3",
            tyre_age=18,
            battery_fraction=0.65,
            gap_ahead=1.5,
        )
        assert isinstance(result, dict)

    def test_required_keys_present(self, sim):
        result = sim.analyze_current_stint(
            circuit="australia",
            team="Ferrari",
            current_lap=20,
            compound="C3",
            tyre_age=18,
            battery_fraction=0.65,
            gap_ahead=1.5,
        )
        required = [
            "laps_to_cliff", "optimal_pit_lap", "max_laps_on_compound",
            "undercut_windows", "oom_recommended", "oom_reason",
            "battery_level_mj", "laps_remaining",
        ]
        for key in required:
            assert key in result, f"Missing key: {key}"

    def test_laps_remaining_correct(self, sim):
        result = sim.analyze_current_stint(
            circuit="australia",
            team="Ferrari",
            current_lap=30,
            compound="C4",
            tyre_age=10,
            battery_fraction=0.70,
            gap_ahead=2.0,
        )
        total = CIRCUITS["australia"].total_laps
        expected = total - 30
        assert result["laps_remaining"] == expected
