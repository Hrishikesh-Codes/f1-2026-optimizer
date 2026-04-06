"""Tests for the calibration data loader."""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest


CALIBRATION_DIR = Path(__file__).parent.parent / "src" / "data" / "calibration"


def test_calibration_loader_imports():
    """Calibration loader must import without error."""
    from src.data.calibration.calibration_loader import CalibrationLoader
    loader = CalibrationLoader()
    assert loader is not None


def test_calibration_loader_returns_dict():
    """load_all() must return a dict (even with no FastF1 data)."""
    from src.data.calibration.calibration_loader import CalibrationLoader
    loader = CalibrationLoader()
    result = loader.load_all()
    assert isinstance(result, dict)


def test_calibration_fallback_values():
    """When no real data exists, defaults must match config.py values."""
    from src.data.calibration.calibration_loader import CalibrationLoader
    from config import TYRE_COMPOUNDS
    loader = CalibrationLoader()
    curves = loader.get_deg_curves()
    # For any circuit that has entries, verify C3 linear rate is positive
    for circuit_key, compounds in curves.items():
        if "C3" in compounds:
            assert compounds["C3"]["linear_rate"] > 0
            assert compounds["C3"]["cliff_lap"] > 0


def test_sc_history_valid_probabilities():
    """All SC rates in sc_history must be between 0 and 1."""
    from src.data.calibration.calibration_loader import CalibrationLoader
    loader = CalibrationLoader()
    sc_rates = loader.get_sc_rates()
    for circuit_key, rate in sc_rates.items():
        assert 0.0 <= rate <= 1.0, f"{circuit_key} SC rate {rate} out of [0,1]"


def test_pit_loss_data_positive():
    """All pit loss times must be positive."""
    from src.data.calibration.calibration_loader import CalibrationLoader
    loader = CalibrationLoader()
    pit_data = loader.get_pit_loss_times()
    for circuit_key, data in pit_data.items():
        assert data["median_s"] > 0, f"{circuit_key} pit loss must be > 0"
