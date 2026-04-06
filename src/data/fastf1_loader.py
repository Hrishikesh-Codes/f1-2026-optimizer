"""
FastF1 Data Loader — F1 2026

Fetches, caches, and serves historical race data via FastF1.

Responsibilities:
  1. Fetch lap data, stint data, and telemetry for historical seasons (2022–2025)
     and 2026 completed rounds (Australia, China, Japan so far).
  2. Cache all data in SQLite (data/cache/f1_data.db) to avoid repeated API calls.
  3. Provide calibration data for the tyre degradation model.
  4. Fall back gracefully if FastF1 network access is unavailable.

FastF1 session types used:
  - "R"  → Race
  - "Q"  → Qualifying (for baseline lap times)
  - "FP1", "FP2", "FP3" → Practice (for tyre data)
"""

from __future__ import annotations

import logging
import sqlite3
import json
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple
import numpy as np
import pandas as pd

from config import (
    FASTF1_CACHE_DIR, FASTF1_DB_PATH,
    HISTORICAL_SEASONS, CURRENT_SEASON,
    COMPLETED_2026_ROUNDS, CIRCUITS,
)

logger = logging.getLogger(__name__)

# Map circuit keys → FastF1 event name fragments
CIRCUIT_TO_FF1_NAME: Dict[str, str] = {
    "australia": "Australian",
    "china": "Chinese",
    "japan": "Japanese",
    "miami": "Miami",
    "canada": "Canadian",
    "monaco": "Monaco",
    "spain": "Spanish",
    "austria": "Austrian",
    "great_britain": "British",
    "belgium": "Belgian",
    "hungary": "Hungarian",
    "netherlands": "Dutch",
    "italy": "Italian",
    "madrid": "Madrid",
    "azerbaijan": "Azerbaijan",
    "singapore": "Singapore",
    "usa": "United States",
    "mexico": "Mexico",
    "brazil": "São Paulo",
    "las_vegas": "Las Vegas",
    "qatar": "Qatar",
    "abu_dhabi": "Abu Dhabi",
}


def _ensure_cache_dir() -> None:
    Path(FASTF1_CACHE_DIR).mkdir(parents=True, exist_ok=True)


def _get_db_connection() -> sqlite3.Connection:
    _ensure_cache_dir()
    conn = sqlite3.connect(FASTF1_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _init_db(conn: sqlite3.Connection) -> None:
    """Create SQLite tables if they don't exist."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS race_laps (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            season      INTEGER NOT NULL,
            circuit     TEXT NOT NULL,
            driver      TEXT NOT NULL,
            lap_number  INTEGER NOT NULL,
            lap_time_s  REAL,
            compound    TEXT,
            tyre_age    INTEGER,
            stint_number INTEGER,
            is_valid    INTEGER,
            sector1_s   REAL,
            sector2_s   REAL,
            sector3_s   REAL,
            fuel_est_kg REAL,
            UNIQUE(season, circuit, driver, lap_number)
        );

        CREATE TABLE IF NOT EXISTS race_stints (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            season          INTEGER NOT NULL,
            circuit         TEXT NOT NULL,
            driver          TEXT NOT NULL,
            stint_number    INTEGER NOT NULL,
            compound        TEXT,
            start_lap       INTEGER,
            end_lap         INTEGER,
            stint_length    INTEGER,
            median_lap_s    REAL,
            deg_rate_s_lap  REAL,
            UNIQUE(season, circuit, driver, stint_number)
        );

        CREATE TABLE IF NOT EXISTS tyre_calibration (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            circuit         TEXT NOT NULL,
            compound        TEXT NOT NULL,
            season          INTEGER NOT NULL,
            cliff_lap_obs   INTEGER,
            linear_deg_obs  REAL,
            cliff_exp_obs   REAL,
            sample_count    INTEGER,
            UNIQUE(circuit, compound, season)
        );

        CREATE TABLE IF NOT EXISTS session_metadata (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            season      INTEGER NOT NULL,
            circuit     TEXT NOT NULL,
            session_type TEXT NOT NULL,
            fetched_at  TEXT,
            row_count   INTEGER,
            UNIQUE(season, circuit, session_type)
        );
    """)
    conn.commit()


class FastF1Loader:
    """
    Manages all FastF1 data access with SQLite caching.

    Usage
    -----
    loader = FastF1Loader()
    df = loader.get_race_laps("australia", season=2024)
    stints = loader.get_stint_data("australia", season=2024)
    cal = loader.get_tyre_calibration("australia", "C3")
    """

    def __init__(self) -> None:
        _ensure_cache_dir()
        self._ff1_available = self._check_fastf1()
        self.conn = _get_db_connection()
        _init_db(self.conn)

    def _check_fastf1(self) -> bool:
        """Check if fastf1 library is installed and functional."""
        try:
            import fastf1  # noqa: F401
            return True
        except ImportError:
            logger.warning("fastf1 not installed — using cached/synthetic data only.")
            return False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_race_laps(
        self,
        circuit_key: str,
        season: int = CURRENT_SEASON,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """
        Return a DataFrame of all laps for the race at a given circuit/season.

        Columns: driver, lap_number, lap_time_s, compound, tyre_age,
                 stint_number, is_valid, sector1_s, sector2_s, sector3_s

        Falls back to synthetic data if FastF1 unavailable or race not yet run.
        """
        # Check cache first
        if not force_refresh:
            cached = self._query_race_laps(circuit_key, season)
            if cached is not None and len(cached) > 0:
                return cached

        # Try to fetch from FastF1
        if self._ff1_available:
            df = self._fetch_race_laps_ff1(circuit_key, season)
            if df is not None and len(df) > 0:
                self._cache_race_laps(df, circuit_key, season)
                return df

        # Fall back to synthetic historical data
        logger.info(f"Using synthetic data for {circuit_key} {season}")
        return self._generate_synthetic_laps(circuit_key, season)

    def get_stint_data(
        self,
        circuit_key: str,
        season: int = CURRENT_SEASON,
    ) -> pd.DataFrame:
        """
        Return aggregated stint-level data for a circuit/season.

        Columns: driver, stint_number, compound, start_lap, end_lap,
                 stint_length, median_lap_s, deg_rate_s_lap
        """
        cached = self._query_stints(circuit_key, season)
        if cached is not None and len(cached) > 0:
            return cached

        # Compute from lap data
        laps_df = self.get_race_laps(circuit_key, season)
        if laps_df.empty:
            return pd.DataFrame()

        stints = self._compute_stints_from_laps(laps_df)
        self._cache_stints(stints, circuit_key, season)
        return stints

    def get_tyre_calibration(
        self,
        circuit_key: str,
        compound: str,
    ) -> Optional[Dict[str, float]]:
        """
        Return observed tyre degradation parameters calibrated from historical data.

        Used to adjust the model's default parameters for specific circuits.

        Returns dict with keys: cliff_lap_obs, linear_deg_obs, cliff_exp_obs
        or None if insufficient data.
        """
        row = self.conn.execute("""
            SELECT AVG(cliff_lap_obs) as cliff_lap,
                   AVG(linear_deg_obs) as linear_deg,
                   AVG(cliff_exp_obs) as cliff_exp,
                   SUM(sample_count) as samples
            FROM tyre_calibration
            WHERE circuit=? AND compound=?
        """, (circuit_key, compound)).fetchone()

        if row and row["samples"] and row["samples"] > 0:
            return {
                "cliff_lap_obs": row["cliff_lap"],
                "linear_deg_obs": row["linear_deg"],
                "cliff_exp_obs": row["cliff_exp"],
                "sample_count": row["samples"],
            }

        # If no DB data, return model defaults
        from config import TYRE_COMPOUNDS
        if compound in TYRE_COMPOUNDS:
            cfg = TYRE_COMPOUNDS[compound]
            return {
                "cliff_lap_obs": float(cfg.cliff_lap),
                "linear_deg_obs": cfg.linear_deg_rate,
                "cliff_exp_obs": cfg.cliff_exponent,
                "sample_count": 0,
            }
        return None

    def get_historical_deg_curves(
        self,
        circuit_key: str,
        compound: str,
        seasons: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """
        Return observed lap-time degradation curves for a compound at a circuit.

        Aggregates across multiple seasons and drivers to produce a
        representative deg curve (mean lap delta vs stint lap).

        Returns DataFrame: lap_in_stint (1-N), mean_delta_s, std_delta_s, sample_count
        """
        if seasons is None:
            seasons = HISTORICAL_SEASONS

        rows = []
        for season in seasons:
            df = self.get_race_laps(circuit_key, season)
            if df.empty:
                continue
            df_comp = df[df["compound"] == compound].copy()
            if df_comp.empty:
                continue
            df_comp = df_comp[df_comp["is_valid"] == 1]
            if df_comp.empty:
                continue

            # Normalise: subtract each driver's first-lap-of-stint time
            for driver in df_comp["driver"].unique():
                driver_df = df_comp[df_comp["driver"] == driver].copy()
                if driver_df.empty:
                    continue
                for stint_num in driver_df["stint_number"].unique():
                    stint_df = driver_df[driver_df["stint_number"] == stint_num].copy()
                    if len(stint_df) < 3:
                        continue
                    stint_df = stint_df.sort_values("tyre_age")
                    ref_time = stint_df.iloc[0]["lap_time_s"]
                    if pd.isna(ref_time) or ref_time <= 0:
                        continue
                    for _, row in stint_df.iterrows():
                        delta = row["lap_time_s"] - ref_time
                        rows.append({
                            "lap_in_stint": int(row["tyre_age"]),
                            "delta_s": float(delta),
                            "season": season,
                        })

        if not rows:
            return self._synthetic_deg_curve(circuit_key, compound)

        raw = pd.DataFrame(rows)
        raw = raw[raw["delta_s"].between(-2.0, 15.0)]  # filter outliers
        grouped = raw.groupby("lap_in_stint")["delta_s"].agg(
            mean_delta_s="mean", std_delta_s="std", sample_count="count"
        ).reset_index()
        return grouped

    def prefetch_circuit_data(
        self,
        circuit_key: str,
        seasons: Optional[List[int]] = None,
    ) -> Dict[str, int]:
        """
        Pre-fetch and cache all data for a circuit across specified seasons.
        Returns dict of {season: rows_cached}.
        """
        if seasons is None:
            seasons = HISTORICAL_SEASONS + [CURRENT_SEASON]

        results = {}
        for season in seasons:
            df = self.get_race_laps(circuit_key, season, force_refresh=True)
            results[season] = len(df)
            if len(df) > 0:
                _ = self.get_stint_data(circuit_key, season)
        return results

    # ------------------------------------------------------------------
    # FastF1 fetching
    # ------------------------------------------------------------------

    def _fetch_race_laps_ff1(
        self, circuit_key: str, season: int
    ) -> Optional[pd.DataFrame]:
        """Fetch race lap data from FastF1 API."""
        try:
            import fastf1
            fastf1.Cache.enable_cache(FASTF1_CACHE_DIR)

            event_name = CIRCUIT_TO_FF1_NAME.get(circuit_key, "")
            if not event_name:
                logger.warning(f"No FastF1 mapping for circuit: {circuit_key}")
                return None

            # Load race session
            session = fastf1.get_session(season, event_name, "R")
            session.load(laps=True, telemetry=False, weather=False, messages=False)

            laps = session.laps
            if laps is None or len(laps) == 0:
                return None

            # Normalise columns
            df = laps[["Driver", "LapNumber", "LapTime", "Compound",
                        "TyreLife", "Stint", "IsPersonalBest",
                        "Sector1Time", "Sector2Time", "Sector3Time"]].copy()

            df.columns = [
                "driver", "lap_number", "lap_time", "compound",
                "tyre_age", "stint_number", "is_valid",
                "sector1", "sector2", "sector3",
            ]

            # Convert timedelta to seconds
            for col in ["lap_time", "sector1", "sector2", "sector3"]:
                if col in df.columns:
                    df[f"{col}_s"] = df[col].dt.total_seconds()
            df.rename(columns={"lap_time_s": "lap_time_s"}, inplace=True)
            df = df.drop(columns=["lap_time", "sector1", "sector2", "sector3"], errors="ignore")

            df["circuit"] = circuit_key
            df["season"] = season
            df["is_valid"] = df["is_valid"].fillna(False).astype(int)
            df["tyre_age"] = df["tyre_age"].fillna(0).astype(int)
            df["lap_time_s"] = pd.to_numeric(df.get("lap_time_s"), errors="coerce")
            df = df.dropna(subset=["lap_time_s"])

            logger.info(f"FastF1 fetched {len(df)} laps for {circuit_key} {season}")
            return df

        except Exception as exc:
            logger.warning(f"FastF1 fetch failed for {circuit_key} {season}: {exc}")
            return None

    # ------------------------------------------------------------------
    # SQLite cache helpers
    # ------------------------------------------------------------------

    def _query_race_laps(self, circuit_key: str, season: int) -> Optional[pd.DataFrame]:
        rows = self.conn.execute("""
            SELECT driver, lap_number, lap_time_s, compound, tyre_age,
                   stint_number, is_valid, sector1_s, sector2_s, sector3_s
            FROM race_laps
            WHERE circuit=? AND season=?
            ORDER BY driver, lap_number
        """, (circuit_key, season)).fetchall()

        if not rows:
            return None
        return pd.DataFrame([dict(r) for r in rows])

    def _cache_race_laps(self, df: pd.DataFrame, circuit_key: str, season: int) -> None:
        rows = []
        for _, row in df.iterrows():
            rows.append((
                int(season), circuit_key,
                str(row.get("driver", "")),
                int(row.get("lap_number", 0)),
                float(row.get("lap_time_s", 0.0) or 0.0),
                str(row.get("compound", "UNKNOWN")),
                int(row.get("tyre_age", 0) or 0),
                int(row.get("stint_number", 1) or 1),
                int(row.get("is_valid", 1) or 1),
                float(row.get("sector1_s", 0.0) or 0.0),
                float(row.get("sector2_s", 0.0) or 0.0),
                float(row.get("sector3_s", 0.0) or 0.0),
                None,   # fuel_est_kg
            ))

        self.conn.executemany("""
            INSERT OR REPLACE INTO race_laps
            (season, circuit, driver, lap_number, lap_time_s, compound, tyre_age,
             stint_number, is_valid, sector1_s, sector2_s, sector3_s, fuel_est_kg)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, rows)
        self.conn.commit()

    def _query_stints(self, circuit_key: str, season: int) -> Optional[pd.DataFrame]:
        rows = self.conn.execute("""
            SELECT driver, stint_number, compound, start_lap, end_lap,
                   stint_length, median_lap_s, deg_rate_s_lap
            FROM race_stints
            WHERE circuit=? AND season=?
            ORDER BY driver, stint_number
        """, (circuit_key, season)).fetchall()

        if not rows:
            return None
        return pd.DataFrame([dict(r) for r in rows])

    def _cache_stints(self, df: pd.DataFrame, circuit_key: str, season: int) -> None:
        rows = []
        for _, row in df.iterrows():
            rows.append((
                int(season), circuit_key,
                str(row.get("driver", "")),
                int(row.get("stint_number", 1)),
                str(row.get("compound", "")),
                int(row.get("start_lap", 1)),
                int(row.get("end_lap", 1)),
                int(row.get("stint_length", 0)),
                float(row.get("median_lap_s", 0.0) or 0.0),
                float(row.get("deg_rate_s_lap", 0.0) or 0.0),
            ))
        self.conn.executemany("""
            INSERT OR REPLACE INTO race_stints
            (season, circuit, driver, stint_number, compound, start_lap, end_lap,
             stint_length, median_lap_s, deg_rate_s_lap)
            VALUES (?,?,?,?,?,?,?,?,?,?)
        """, rows)
        self.conn.commit()

    # ------------------------------------------------------------------
    # Synthetic / fallback data generation
    # ------------------------------------------------------------------

    def _generate_synthetic_laps(
        self, circuit_key: str, season: int
    ) -> pd.DataFrame:
        """
        Generate plausible synthetic lap data when FastF1 is unavailable.
        Uses the model's parameters to produce realistic-looking data.
        """
        from config import TYRE_COMPOUNDS, TEAMS
        from src.simulation.tyre import TyreDegradationModel

        circuit_cfg = CIRCUITS.get(circuit_key)
        if circuit_cfg is None:
            return pd.DataFrame()

        tyre_model = TyreDegradationModel()
        rng = np.random.default_rng(hash(f"{circuit_key}{season}") % (2**31))

        rows = []
        drivers = [d for team in TEAMS.values() for d in team.drivers[:1]]  # one per team

        for driver_idx, driver in enumerate(drivers):
            team_name = [t for t, tc in TEAMS.items() if driver in tc.drivers][0]
            team_cfg = TEAMS[team_name]

            # Simple 1-stop strategy for synthetic data
            total_laps = circuit_cfg.total_laps
            pit_lap = int(total_laps * 0.48 + rng.integers(-5, 6))
            compound1, compound2 = circuit_cfg.compounds[1], circuit_cfg.compounds[0]

            for lap in range(1, total_laps + 1):
                in_stint2 = lap > pit_lap
                compound = compound2 if in_stint2 else compound1
                tyre_age = (lap - pit_lap - 1) if in_stint2 else (lap - 1)

                base = circuit_cfg.base_lap_time
                base += team_cfg.base_lap_delta
                base += TYRE_COMPOUNDS[compound].base_pace_offset
                base += tyre_model.compute_lap_time_delta(
                    compound, tyre_age + 1,
                    circuit_cfg.track_abrasiveness, 35.0, 0.75,
                )
                base += rng.normal(0.0, 0.12)

                rows.append({
                    "driver": driver,
                    "lap_number": lap,
                    "lap_time_s": round(base, 3),
                    "compound": compound,
                    "tyre_age": tyre_age,
                    "stint_number": 2 if in_stint2 else 1,
                    "is_valid": 1,
                    "sector1_s": base * 0.28,
                    "sector2_s": base * 0.40,
                    "sector3_s": base * 0.32,
                })

        return pd.DataFrame(rows)

    def _compute_stints_from_laps(self, laps_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate lap-level data into stint-level summaries."""
        rows = []
        for driver in laps_df["driver"].unique():
            d_df = laps_df[laps_df["driver"] == driver].copy()
            for stint_num in d_df["stint_number"].unique():
                s_df = d_df[d_df["stint_number"] == stint_num].sort_values("lap_number")
                if s_df.empty:
                    continue
                compound = s_df.iloc[0]["compound"]
                start_lap = int(s_df["lap_number"].min())
                end_lap = int(s_df["lap_number"].max())
                stint_length = end_lap - start_lap + 1
                valid = s_df[s_df["is_valid"] == 1]["lap_time_s"].dropna()
                if len(valid) == 0:
                    continue
                median_time = float(valid.median())

                # Compute deg rate: linear regression on lap times vs tyre_age
                sub = s_df[s_df["is_valid"] == 1][["tyre_age", "lap_time_s"]].dropna()
                deg_rate = 0.0
                if len(sub) >= 3:
                    try:
                        coeffs = np.polyfit(sub["tyre_age"].values, sub["lap_time_s"].values, 1)
                        deg_rate = float(coeffs[0])
                    except Exception:
                        pass

                rows.append({
                    "driver": driver,
                    "stint_number": stint_num,
                    "compound": compound,
                    "start_lap": start_lap,
                    "end_lap": end_lap,
                    "stint_length": stint_length,
                    "median_lap_s": median_time,
                    "deg_rate_s_lap": deg_rate,
                })

        return pd.DataFrame(rows) if rows else pd.DataFrame()

    def _synthetic_deg_curve(self, circuit_key: str, compound: str) -> pd.DataFrame:
        """Generate a model-based degradation curve when no historical data exists."""
        from config import TYRE_COMPOUNDS
        from src.simulation.tyre import TyreDegradationModel

        circuit_cfg = CIRCUITS.get(circuit_key)
        if circuit_cfg is None:
            return pd.DataFrame()

        tyre_model = TyreDegradationModel()
        cfg = TYRE_COMPOUNDS.get(compound)
        if cfg is None:
            return pd.DataFrame()

        max_laps = cfg.max_viable_laps
        rows = []
        for lap in range(0, max_laps + 1):
            delta = tyre_model.compute_lap_time_delta(
                compound, lap,
                circuit_cfg.track_abrasiveness, 35.0, 0.80,
            )
            rows.append({
                "lap_in_stint": lap,
                "mean_delta_s": delta,
                "std_delta_s": delta * 0.15,
                "sample_count": 0,
            })

        return pd.DataFrame(rows)

    def close(self) -> None:
        """Close the SQLite connection."""
        if self.conn:
            self.conn.close()
