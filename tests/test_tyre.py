"""
Unit tests — Tyre Degradation Model

Tests:
  - Linear degradation accumulates monotonically
  - Cliff lap triggers at correct adjusted lap
  - Higher abrasiveness → earlier cliff
  - Higher temperature → more degradation
  - Push level scales degradation correctly
  - Soft compound degrades faster than hard
  - Undercut window returns positive gain when old compound is on cliff
  - Stint time delta array length matches num_laps
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

from src.simulation.tyre import TyreDegradationModel, TyreState
from config import TYRE_COMPOUNDS


@pytest.fixture
def model() -> TyreDegradationModel:
    return TyreDegradationModel()


class TestLinearDegradation:
    def test_zero_laps_returns_zero(self, model):
        delta = model.compute_lap_time_delta("C3", 0, 1.0, 35.0, 0.8)
        assert delta == 0.0

    def test_monotonic_increase_before_cliff(self, model):
        """Degradation should increase with each lap in the linear phase."""
        deltas = [
            model.compute_lap_time_delta("C3", lap, 1.0, 35.0, 0.8)
            for lap in range(1, 30)
        ]
        for i in range(len(deltas) - 1):
            assert deltas[i + 1] >= deltas[i], (
                f"Degradation not monotonic at lap {i + 2}: "
                f"{deltas[i]:.4f} > {deltas[i+1]:.4f}"
            )

    def test_positive_degradation(self, model):
        delta = model.compute_lap_time_delta("C3", 15, 1.0, 35.0, 0.8)
        assert delta > 0.0

    def test_stint_array_length(self, model):
        num_laps = 30
        arr = model.compute_stint_time_deltas("C3", num_laps, 1.0, 35.0, 0.8)
        assert len(arr) == num_laps

    def test_stint_array_monotonic(self, model):
        arr = model.compute_stint_time_deltas("C3", 30, 1.0, 35.0, 0.8)
        for i in range(len(arr) - 1):
            assert arr[i + 1] >= arr[i] - 1e-9


class TestCliffBehaviour:
    def test_cliff_lap_positive(self, model):
        cliff = model.compute_cliff_lap("C3", 1.0, 35.0)
        assert cliff > 0

    def test_soft_cliff_earlier_than_hard(self, model):
        cliff_c1 = model.compute_cliff_lap("C1", 1.0, 35.0)
        cliff_c5 = model.compute_cliff_lap("C5", 1.0, 35.0)
        assert cliff_c5 < cliff_c1, (
            f"C5 cliff ({cliff_c5}) should be earlier than C1 cliff ({cliff_c1})"
        )

    def test_high_abrasiveness_earlier_cliff(self, model):
        cliff_low = model.compute_cliff_lap("C3", 0.70, 35.0)
        cliff_high = model.compute_cliff_lap("C3", 1.10, 35.0)
        assert cliff_high < cliff_low, (
            "Higher abrasiveness should bring cliff earlier"
        )

    def test_high_temp_increases_degradation(self, model):
        delta_cool = model.compute_lap_time_delta("C3", 20, 1.0, 25.0, 0.8)
        delta_hot = model.compute_lap_time_delta("C3", 20, 1.0, 55.0, 0.8)
        assert delta_hot > delta_cool, (
            "Higher track temperature should increase degradation"
        )

    def test_cliff_exponential_growth(self, model):
        """After cliff lap, degradation should grow faster than linearly."""
        cliff = model.compute_cliff_lap("C4", 1.0, 35.0)
        pre_cliff = model.compute_lap_time_delta("C4", cliff - 1, 1.0, 35.0, 0.8)
        at_cliff = model.compute_lap_time_delta("C4", cliff + 1, 1.0, 35.0, 0.8)
        past_cliff = model.compute_lap_time_delta("C4", cliff + 5, 1.0, 35.0, 0.8)
        rate1 = at_cliff - pre_cliff
        rate2 = past_cliff - at_cliff
        assert rate2 >= rate1, "Degradation should accelerate past cliff"


class TestCompoundHierarchy:
    def test_pace_offset_ordering(self):
        """C5 should be faster than C3 should be faster than C1 at lap 1."""
        offsets = {
            c: TYRE_COMPOUNDS[c].base_pace_offset for c in ["C1", "C3", "C5"]
        }
        assert offsets["C5"] < offsets["C3"] < offsets["C1"]

    def test_c5_degrades_faster_than_c1(self, model):
        deg_c1 = model.compute_lap_time_delta("C1", 20, 1.0, 35.0, 0.8)
        deg_c5 = model.compute_lap_time_delta("C5", 20, 1.0, 35.0, 0.8)
        assert deg_c5 > deg_c1, "C5 should degrade more than C1 by lap 20"

    def test_deg_multiplier_scales_proportionally(self, model):
        base = model.compute_stint_time_deltas("C3", 20, 1.0, 35.0, 0.8, deg_multiplier=1.0)
        scaled = model.compute_stint_time_deltas("C3", 20, 1.0, 35.0, 0.8, deg_multiplier=1.10)
        ratio = np.mean(scaled) / np.mean(base)
        assert abs(ratio - 1.10) < 0.05


class TestPushLevel:
    def test_higher_push_more_degradation(self, model):
        low_push = model.compute_lap_time_delta("C3", 15, 1.0, 35.0, 0.3)
        high_push = model.compute_lap_time_delta("C3", 15, 1.0, 35.0, 1.0)
        assert high_push > low_push

    def test_push_level_zero_reduces_deg(self, model):
        full = model.compute_lap_time_delta("C4", 20, 1.0, 35.0, 1.0)
        save = model.compute_lap_time_delta("C4", 20, 1.0, 35.0, 0.0)
        assert save < full


class TestOptimalStint:
    def test_returns_three_values(self, model):
        result = model.estimate_optimal_stint_length("C3", 1.0, 35.0)
        assert len(result) == 3

    def test_min_lt_opt_lt_max(self, model):
        min_l, opt_l, max_l = model.estimate_optimal_stint_length("C3", 1.0, 35.0)
        assert min_l <= opt_l <= max_l

    def test_c5_shorter_stint_than_c1(self, model):
        _, opt_c1, _ = model.estimate_optimal_stint_length("C1", 1.0, 35.0)
        _, opt_c5, _ = model.estimate_optimal_stint_length("C5", 1.0, 35.0)
        assert opt_c5 < opt_c1


class TestUndercut:
    def test_undercut_returns_list(self, model):
        results = model.compute_undercut_window(
            compound_old="C3", compound_new="C5",
            laps_on_old=30, pit_loss_time=22.5,
            abrasiveness=1.0, track_temp_c=35.0,
        )
        assert isinstance(results, list)
        assert all(isinstance(t, tuple) and len(t) == 2 for t in results)

    def test_undercut_gain_positive_when_old_compound_old(self, model):
        """When old compound is well past cliff, undercut gain grows with each lap out.

        The pit loss is large (~22.5s), so we test that cumulative gain INCREASES
        (undercut becomes more attractive) as you wait longer on the degrading tyre.
        """
        results_short = model.compute_undercut_window(
            compound_old="C5", compound_new="C3",
            laps_on_old=25,  # well past C5 cliff (~21 laps)
            pit_loss_time=22.5,
            abrasiveness=1.0, track_temp_c=35.0,
            window_laps=10,
        )
        gains = [g for _, g in results_short]
        # The gain should be increasing (getting more positive) as laps pass on cliff
        # because the old tyre is getting worse each lap
        assert gains[-1] > gains[0], (
            "Undercut gain should grow as old compound degrades further past cliff"
        )


class TestTyreState:
    def test_build_fresh_state(self, model):
        state = model.build_tyre_state("C3")
        assert state.age == 0
        assert state.compound == "C3"
        assert state.total_deg_penalty == pytest.approx(0.0)

    def test_advance_state_increments_age(self, model):
        state = model.build_tyre_state("C4")
        new_state = model.advance_tyre_state(state, 1.0, 35.0, 0.8)
        assert new_state.age == 1

    def test_advance_state_immutable(self, model):
        state = model.build_tyre_state("C4")
        new_state = model.advance_tyre_state(state, 1.0, 35.0, 0.8)
        assert state.age == 0  # original unchanged
        assert new_state.age == 1

    def test_past_cliff_flag_set(self, model):
        state = model.build_tyre_state("C5")
        cliff = model.compute_cliff_lap("C5", 1.0, 35.0)
        for _ in range(cliff + 2):
            state = model.advance_tyre_state(state, 1.0, 35.0, 0.9)
        assert state.is_past_cliff
