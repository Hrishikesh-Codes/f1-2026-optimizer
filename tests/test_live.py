"""Tests for the live dashboard page."""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest


def test_live_dashboard_imports():
    """live_dashboard module must be importable."""
    from src.frontend import live_dashboard  # noqa: F401


def test_race_weekend_detection_no_match():
    """Detection should return either None or a valid circuit key string."""
    from src.frontend.live_dashboard import _detect_race_weekend
    result = _detect_race_weekend()
    # Result is either None (no race weekend) or a valid circuit key string
    assert result is None or isinstance(result, str)


def test_race_weekend_detection_returns_valid_circuit():
    """If detection returns a key, it must be in CIRCUITS."""
    from src.frontend.live_dashboard import _detect_race_weekend
    from config import CIRCUITS
    result = _detect_race_weekend()
    if result is not None:
        assert result in CIRCUITS


def test_build_timing_df_shape():
    """Live timing DataFrame must have correct columns."""
    from src.frontend.live_dashboard import _build_live_timing_df
    from config import CIRCUIT_ORDER
    df = _build_live_timing_df(CIRCUIT_ORDER[0], "Max Verstappen")
    required_cols = {"Pos", "Driver", "Team", "Gap", "Tyre", "Age", "Pit Status"}
    assert required_cols.issubset(set(df.columns))
    assert len(df) > 0
