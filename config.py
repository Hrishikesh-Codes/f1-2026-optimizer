"""
F1 2026 Race Strategy Optimizer — Central Configuration
All constants, circuit data, tyre parameters, team data, and simulation settings live here.
No magic numbers anywhere else in the codebase.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class CircuitConfig:
    name: str
    circuit: str
    round_number: int
    total_laps: int
    pit_loss_time: float          # net time loss for a pit stop (seconds)
    compounds: List[str]          # [Hard_compound, Medium_compound, Soft_compound]
    circuit_type: str             # "permanent", "street", "hybrid"
    sc_probability: float         # probability of at least one SC/VSC in race
    is_sprint: bool
    base_lap_time: float          # representative race lap time in seconds (2026 est.)
    track_abrasiveness: float     # 0.65 (smooth) → 1.15 (very abrasive)
    fuel_per_lap: float           # kg burned per racing lap
    track_evolution_rate: float   # seconds of lap-time improvement per lap (rubber laid down)
    rain_probability: float       # probability of rain during race
    overtake_difficulty: float    # 0.0 (easy) → 1.0 (impossible)  Monaco=1.0
    location: str
    num_drs_zones: int            # Active-aero straight zones (replaces DRS in 2026)


@dataclass
class TyreCompoundConfig:
    name: str                     # "C1" … "C5"
    label: str                    # "Hard" / "Medium" / "Soft"
    color: str                    # hex
    base_pace_offset: float       # seconds relative to C3 (negative = faster)
    linear_deg_rate: float        # seconds of extra time added per lap (linear phase)
    cliff_lap: int                # stint lap at which cliff degradation starts (ref conditions)
    cliff_exponent: float         # coefficient for (lap - cliff_lap)^2 term
    thermal_sensitivity: float    # deg-rate multiplier per 10 °C above 35 °C baseline
    max_viable_laps: int          # absolute maximum stint length before tyre failure risk
    min_recommended_laps: int     # minimum sensible stint (get compounds up to temp)


@dataclass
class TeamConfig:
    name: str
    short_name: str               # e.g. "MER", "FER"
    drivers: List[str]
    pu_supplier: str
    performance_tier: int         # 1 = front-runner, 4 = backmarker
    base_lap_delta: float         # seconds vs fastest car (positive = slower)
    chassis_aero_efficiency: float  # 0-1; higher = better Active-Aero integration
    ers_deployment_efficiency: float  # 0-1; higher = more effective boost usage


@dataclass
class PowerUnitConfig:
    supplier: str
    ice_power_kw: float
    mguk_power_kw: float
    has_mguh: bool                # False for 2026 (eliminated)
    energy_recovery_efficiency: float   # 0-1
    top_speed_factor: float       # multiplier on straight speed delta vs reference
    ers_lap_time_bonus: float     # seconds gained per lap at max ERS vs min ERS
    boost_duration_seconds: float # how long a single boost lasts on-track


# ---------------------------------------------------------------------------
# Tyre Compound Definitions  (C1–C5, C6 dropped in 2026)
# ---------------------------------------------------------------------------

TYRE_COMPOUNDS: Dict[str, TyreCompoundConfig] = {
    "C1": TyreCompoundConfig(
        name="C1", label="Hard", color="#FFFFFF",
        base_pace_offset=+0.45,   # race pace gap vs C3 ~0.45s (quali gap ~0.9s)
        linear_deg_rate=0.014,
        cliff_lap=52,
        cliff_exponent=0.006,
        thermal_sensitivity=0.06,
        max_viable_laps=65,
        min_recommended_laps=15,  # needs 15 laps min to justify Hard compound
    ),
    "C2": TyreCompoundConfig(
        name="C2", label="Hard", color="#FFFFFF",
        base_pace_offset=+0.25,   # race pace gap vs C3 ~0.25s
        linear_deg_rate=0.020,
        cliff_lap=44,
        cliff_exponent=0.009,
        thermal_sensitivity=0.08,
        max_viable_laps=55,
        min_recommended_laps=12,
    ),
    "C3": TyreCompoundConfig(
        name="C3", label="Medium", color="#FFD700",
        base_pace_offset=0.00,    # baseline reference compound
        linear_deg_rate=0.028,
        cliff_lap=36,
        cliff_exponent=0.014,
        thermal_sensitivity=0.10,
        max_viable_laps=45,
        min_recommended_laps=10,
    ),
    "C4": TyreCompoundConfig(
        name="C4", label="Soft", color="#DC0000",
        base_pace_offset=-0.25,   # race pace gap vs C3 ~0.25s
        linear_deg_rate=0.040,
        cliff_lap=27,
        cliff_exponent=0.022,
        thermal_sensitivity=0.12, # reduced from 0.14 — softs not 4× more sensitive than Hard
        max_viable_laps=32,       # reduced from 34 based on actual stint data
        min_recommended_laps=8,
    ),
    "C5": TyreCompoundConfig(
        name="C5", label="Soft", color="#DC0000",
        base_pace_offset=-0.45,   # race pace gap vs C3 ~0.45s
        linear_deg_rate=0.056,
        cliff_lap=21,
        cliff_exponent=0.034,
        thermal_sensitivity=0.14, # reduced from 0.18
        max_viable_laps=25,       # reduced from 27
        min_recommended_laps=7,
    ),
}

# Compound role per circuit weekend: index 0=Hard, 1=Medium, 2=Soft
# When two identical compounds listed (e.g. C5/C5) the soft variant has +15% deg
COMPOUND_ROLE = {c: i for i, c in enumerate(["C1", "C2", "C3", "C4", "C5"])}

# Relative order of all compounds (lower = harder)
_COMPOUND_ORDER = ["C1", "C2", "C3", "C4", "C5"]


def compound_label(compound: str, circuit_compounds: list) -> str:
    """
    Return the contextual race-weekend label for a compound.

    Given the 3 (or 2) compounds nominated for a circuit weekend,
    label them Hard / Medium / Soft based on relative hardness.

    Examples
    --------
    circuit_compounds = ["C2", "C3", "C4"]  →  C2=Hard, C3=Medium, C4=Soft
    circuit_compounds = ["C1", "C2", "C3"]  →  C1=Hard, C2=Medium, C3=Soft
    circuit_compounds = ["C4", "C5", "C5"]  →  C4=Hard, first C5=Medium, second C5=Soft
    """
    sorted_cpds = sorted(set(circuit_compounds), key=lambda c: _COMPOUND_ORDER.index(c))
    labels = ["Hard", "Medium", "Soft"]
    label_map = {}
    for i, cpd in enumerate(sorted_cpds):
        label_map[cpd] = labels[min(i, len(labels) - 1)]
    return label_map.get(compound, compound)


def compound_label_with_code(compound: str, circuit_compounds: list) -> str:
    """Return label with code, e.g. 'Soft (C4)'."""
    return f"{compound_label(compound, circuit_compounds)} ({compound})"


# ---------------------------------------------------------------------------
# Circuit Definitions — all 22 rounds of 2026 season
# ---------------------------------------------------------------------------

CIRCUITS: Dict[str, CircuitConfig] = {
    "australia": CircuitConfig(
        name="Australian Grand Prix", circuit="Albert Park",
        round_number=1, total_laps=58, pit_loss_time=22.5,
        compounds=["C3", "C4", "C5"], circuit_type="permanent",
        sc_probability=0.75, is_sprint=False,
        base_lap_time=81.4, track_abrasiveness=0.88, fuel_per_lap=1.50,
        track_evolution_rate=0.045, rain_probability=0.15,
        overtake_difficulty=0.50, location="Melbourne, Australia", num_drs_zones=4,
    ),
    "china": CircuitConfig(
        name="Chinese Grand Prix", circuit="Shanghai International Circuit",
        round_number=2, total_laps=56, pit_loss_time=23.0,
        compounds=["C2", "C3", "C4"], circuit_type="permanent",
        sc_probability=0.30, is_sprint=True,
        base_lap_time=94.5, track_abrasiveness=0.90, fuel_per_lap=1.55,
        track_evolution_rate=0.055, rain_probability=0.20,
        overtake_difficulty=0.40, location="Shanghai, China", num_drs_zones=2,
    ),
    "japan": CircuitConfig(
        name="Japanese Grand Prix", circuit="Suzuka Circuit",
        round_number=3, total_laps=53, pit_loss_time=22.0,
        compounds=["C1", "C2", "C3"], circuit_type="permanent",
        sc_probability=0.15, is_sprint=False,
        base_lap_time=91.5, track_abrasiveness=1.00, fuel_per_lap=1.58,
        track_evolution_rate=0.035, rain_probability=0.25,
        overtake_difficulty=0.65, location="Suzuka, Japan", num_drs_zones=2,
    ),
    "miami": CircuitConfig(
        name="Miami Grand Prix", circuit="Hard Rock Stadium Circuit",
        round_number=4, total_laps=57, pit_loss_time=23.5,
        compounds=["C3", "C4", "C5"], circuit_type="street",
        sc_probability=0.40, is_sprint=True,
        base_lap_time=91.2, track_abrasiveness=0.87, fuel_per_lap=1.52,
        track_evolution_rate=0.060, rain_probability=0.20,
        overtake_difficulty=0.55, location="Miami, Florida, USA", num_drs_zones=3,
    ),
    "canada": CircuitConfig(
        name="Canadian Grand Prix", circuit="Circuit Gilles Villeneuve",
        round_number=5, total_laps=70, pit_loss_time=22.0,
        compounds=["C3", "C4", "C5"], circuit_type="street",
        sc_probability=0.50, is_sprint=True,
        base_lap_time=75.5, track_abrasiveness=0.82, fuel_per_lap=1.40,
        track_evolution_rate=0.050, rain_probability=0.30,
        overtake_difficulty=0.45, location="Montreal, Canada", num_drs_zones=3,
    ),
    "monaco": CircuitConfig(
        name="Monaco Grand Prix", circuit="Circuit de Monaco",
        round_number=6, total_laps=78, pit_loss_time=24.5,
        compounds=["C4", "C5", "C5"], circuit_type="street",
        sc_probability=0.60, is_sprint=False,
        base_lap_time=74.8, track_abrasiveness=0.65, fuel_per_lap=1.35,
        track_evolution_rate=0.030, rain_probability=0.20,
        overtake_difficulty=0.98, location="Monte Carlo, Monaco", num_drs_zones=1,
    ),
    "spain": CircuitConfig(
        name="Spanish Grand Prix", circuit="Circuit de Barcelona-Catalunya",
        round_number=7, total_laps=66, pit_loss_time=22.5,
        compounds=["C2", "C3", "C4"], circuit_type="permanent",
        sc_probability=0.20, is_sprint=False,
        base_lap_time=83.5, track_abrasiveness=1.05, fuel_per_lap=1.56,
        track_evolution_rate=0.040, rain_probability=0.10,
        overtake_difficulty=0.60, location="Barcelona, Spain", num_drs_zones=2,
    ),
    "austria": CircuitConfig(
        name="Austrian Grand Prix", circuit="Red Bull Ring",
        round_number=8, total_laps=71, pit_loss_time=21.5,
        compounds=["C3", "C4", "C5"], circuit_type="permanent",
        sc_probability=0.25, is_sprint=False,
        base_lap_time=66.8, track_abrasiveness=0.85, fuel_per_lap=1.38,
        track_evolution_rate=0.035, rain_probability=0.25,
        overtake_difficulty=0.40, location="Spielberg, Austria", num_drs_zones=3,
    ),
    "great_britain": CircuitConfig(
        name="British Grand Prix", circuit="Silverstone Circuit",
        round_number=9, total_laps=52, pit_loss_time=22.0,
        compounds=["C2", "C3", "C4"], circuit_type="permanent",
        sc_probability=0.20, is_sprint=True,
        base_lap_time=87.2, track_abrasiveness=0.90, fuel_per_lap=1.54,
        track_evolution_rate=0.040, rain_probability=0.35,
        overtake_difficulty=0.45, location="Silverstone, UK", num_drs_zones=2,
    ),
    "belgium": CircuitConfig(
        name="Belgian Grand Prix", circuit="Circuit de Spa-Francorchamps",
        round_number=10, total_laps=44, pit_loss_time=23.5,
        compounds=["C1", "C2", "C3"], circuit_type="permanent",
        sc_probability=0.30, is_sprint=False,
        base_lap_time=106.0, track_abrasiveness=0.90, fuel_per_lap=1.68,
        track_evolution_rate=0.030, rain_probability=0.40,
        overtake_difficulty=0.35, location="Stavelot, Belgium", num_drs_zones=2,
    ),
    "hungary": CircuitConfig(
        name="Hungarian Grand Prix", circuit="Hungaroring",
        round_number=11, total_laps=70, pit_loss_time=22.5,
        compounds=["C3", "C4", "C5"], circuit_type="permanent",
        sc_probability=0.20, is_sprint=False,
        base_lap_time=80.5, track_abrasiveness=0.95, fuel_per_lap=1.48,
        track_evolution_rate=0.045, rain_probability=0.20,
        overtake_difficulty=0.72, location="Budapest, Hungary", num_drs_zones=1,
    ),
    "netherlands": CircuitConfig(
        name="Dutch Grand Prix", circuit="Circuit Zandvoort",
        round_number=12, total_laps=72, pit_loss_time=22.5,
        compounds=["C3", "C4", "C5"], circuit_type="permanent",
        sc_probability=0.25, is_sprint=True,
        base_lap_time=73.0, track_abrasiveness=0.88, fuel_per_lap=1.42,
        track_evolution_rate=0.040, rain_probability=0.30,
        overtake_difficulty=0.70, location="Zandvoort, Netherlands", num_drs_zones=2,
    ),
    "italy": CircuitConfig(
        name="Italian Grand Prix", circuit="Autodromo Nazionale Monza",
        round_number=13, total_laps=53, pit_loss_time=23.0,
        compounds=["C1", "C2", "C3"], circuit_type="permanent",
        sc_probability=0.20, is_sprint=False,
        base_lap_time=83.0, track_abrasiveness=0.75, fuel_per_lap=1.42,
        track_evolution_rate=0.030, rain_probability=0.20,
        overtake_difficulty=0.35, location="Monza, Italy", num_drs_zones=2,
    ),
    "madrid": CircuitConfig(
        name="Madrid Grand Prix", circuit="Madring Street Circuit",
        round_number=14, total_laps=55, pit_loss_time=24.0,
        compounds=["C4", "C5", "C5"], circuit_type="street",
        sc_probability=0.55, is_sprint=False,
        base_lap_time=85.0, track_abrasiveness=0.80, fuel_per_lap=1.50,
        track_evolution_rate=0.065, rain_probability=0.10,
        overtake_difficulty=0.65, location="Madrid, Spain", num_drs_zones=3,
    ),
    "azerbaijan": CircuitConfig(
        name="Azerbaijan Grand Prix", circuit="Baku City Circuit",
        round_number=15, total_laps=51, pit_loss_time=24.0,
        compounds=["C4", "C5", "C5"], circuit_type="street",
        sc_probability=0.50, is_sprint=False,
        base_lap_time=103.0, track_abrasiveness=0.72, fuel_per_lap=1.55,
        track_evolution_rate=0.050, rain_probability=0.10,
        overtake_difficulty=0.40, location="Baku, Azerbaijan", num_drs_zones=2,
    ),
    "singapore": CircuitConfig(
        name="Singapore Grand Prix", circuit="Marina Bay Street Circuit",
        round_number=16, total_laps=62, pit_loss_time=25.0,
        compounds=["C4", "C5", "C5"], circuit_type="street",
        sc_probability=0.55, is_sprint=True,
        base_lap_time=104.5, track_abrasiveness=0.75, fuel_per_lap=1.52,
        track_evolution_rate=0.055, rain_probability=0.35,
        overtake_difficulty=0.85, location="Singapore", num_drs_zones=3,
    ),
    "usa": CircuitConfig(
        name="United States Grand Prix", circuit="Circuit of the Americas",
        round_number=17, total_laps=56, pit_loss_time=23.0,
        compounds=["C2", "C3", "C4"], circuit_type="permanent",
        sc_probability=0.25, is_sprint=False,
        base_lap_time=98.5, track_abrasiveness=1.05, fuel_per_lap=1.62,
        track_evolution_rate=0.045, rain_probability=0.20,
        overtake_difficulty=0.45, location="Austin, Texas, USA", num_drs_zones=2,
    ),
    "mexico": CircuitConfig(
        name="Mexico City Grand Prix", circuit="Autódromo Hermanos Rodríguez",
        round_number=18, total_laps=71, pit_loss_time=22.5,
        compounds=["C2", "C3", "C4"], circuit_type="permanent",
        sc_probability=0.20, is_sprint=False,
        base_lap_time=80.8, track_abrasiveness=1.00, fuel_per_lap=1.48,
        track_evolution_rate=0.040, rain_probability=0.15,
        overtake_difficulty=0.50, location="Mexico City, Mexico", num_drs_zones=3,
    ),
    "brazil": CircuitConfig(
        name="Brazilian Grand Prix", circuit="Autódromo José Carlos Pace (Interlagos)",
        round_number=19, total_laps=71, pit_loss_time=22.0,
        compounds=["C3", "C4", "C5"], circuit_type="permanent",
        sc_probability=0.45, is_sprint=False,
        base_lap_time=72.5, track_abrasiveness=1.05, fuel_per_lap=1.42,
        track_evolution_rate=0.045, rain_probability=0.40,
        overtake_difficulty=0.40, location="São Paulo, Brazil", num_drs_zones=2,
    ),
    "las_vegas": CircuitConfig(
        name="Las Vegas Grand Prix", circuit="Las Vegas Strip Circuit",
        round_number=20, total_laps=50, pit_loss_time=23.5,
        compounds=["C3", "C4", "C5"], circuit_type="street",
        sc_probability=0.40, is_sprint=False,
        base_lap_time=97.5, track_abrasiveness=0.70, fuel_per_lap=1.50,
        track_evolution_rate=0.060, rain_probability=0.05,
        overtake_difficulty=0.40, location="Las Vegas, Nevada, USA", num_drs_zones=3,
    ),
    "qatar": CircuitConfig(
        name="Qatar Grand Prix", circuit="Lusail International Circuit",
        round_number=21, total_laps=57, pit_loss_time=22.5,
        compounds=["C1", "C2", "C3"], circuit_type="permanent",
        sc_probability=0.20, is_sprint=False,
        base_lap_time=84.0, track_abrasiveness=1.10, fuel_per_lap=1.55,
        track_evolution_rate=0.035, rain_probability=0.05,
        overtake_difficulty=0.50, location="Lusail, Qatar", num_drs_zones=2,
    ),
    "abu_dhabi": CircuitConfig(
        name="Abu Dhabi Grand Prix", circuit="Yas Marina Circuit",
        round_number=22, total_laps=58, pit_loss_time=22.5,
        compounds=["C3", "C4", "C5"], circuit_type="permanent",
        sc_probability=0.15, is_sprint=False,
        base_lap_time=88.5, track_abrasiveness=0.85, fuel_per_lap=1.52,
        track_evolution_rate=0.040, rain_probability=0.05,
        overtake_difficulty=0.50, location="Abu Dhabi, UAE", num_drs_zones=2,
    ),
}

# Ordered list for season overview
CIRCUIT_ORDER: List[str] = [
    "australia", "china", "japan", "miami", "canada", "monaco", "spain",
    "austria", "great_britain", "belgium", "hungary", "netherlands", "italy",
    "madrid", "azerbaijan", "singapore", "usa", "mexico", "brazil",
    "las_vegas", "qatar", "abu_dhabi",
]


# ---------------------------------------------------------------------------
# Power Unit Configurations (2026 — no MGU-H)
# ---------------------------------------------------------------------------

POWER_UNITS: Dict[str, PowerUnitConfig] = {
    "Mercedes": PowerUnitConfig(
        supplier="Mercedes", ice_power_kw=402, mguk_power_kw=352,
        has_mguh=False, energy_recovery_efficiency=0.94,
        top_speed_factor=1.00, ers_lap_time_bonus=0.42, boost_duration_seconds=2.2,
    ),
    "Ferrari": PowerUnitConfig(
        supplier="Ferrari", ice_power_kw=398, mguk_power_kw=350,
        has_mguh=False, energy_recovery_efficiency=0.93,
        top_speed_factor=1.00, ers_lap_time_bonus=0.40, boost_duration_seconds=2.1,
    ),
    "Honda_RBPT": PowerUnitConfig(
        supplier="Honda_RBPT", ice_power_kw=395, mguk_power_kw=348,
        has_mguh=False, energy_recovery_efficiency=0.91,
        top_speed_factor=0.99, ers_lap_time_bonus=0.38, boost_duration_seconds=2.0,
    ),
    "Renault": PowerUnitConfig(
        supplier="Renault", ice_power_kw=390, mguk_power_kw=345,
        has_mguh=False, energy_recovery_efficiency=0.89,
        top_speed_factor=0.98, ers_lap_time_bonus=0.35, boost_duration_seconds=1.9,
    ),
    "Audi": PowerUnitConfig(
        supplier="Audi", ice_power_kw=388, mguk_power_kw=344,
        has_mguh=False, energy_recovery_efficiency=0.88,
        top_speed_factor=0.97, ers_lap_time_bonus=0.33, boost_duration_seconds=1.85,
    ),
}


# ---------------------------------------------------------------------------
# Team & Driver Grid (2026)
# ---------------------------------------------------------------------------

TEAMS: Dict[str, TeamConfig] = {
    "Mercedes": TeamConfig(
        name="Mercedes-AMG Petronas F1 Team", short_name="MER",
        drivers=["George Russell", "Kimi Antonelli"],
        pu_supplier="Mercedes", performance_tier=1,
        base_lap_delta=0.00, chassis_aero_efficiency=0.96,
        ers_deployment_efficiency=0.95,
    ),
    "Ferrari": TeamConfig(
        name="Scuderia Ferrari", short_name="FER",
        drivers=["Charles Leclerc", "Lewis Hamilton"],
        pu_supplier="Ferrari", performance_tier=1,
        base_lap_delta=0.05, chassis_aero_efficiency=0.95,
        ers_deployment_efficiency=0.94,
    ),
    "McLaren": TeamConfig(
        name="McLaren Formula 1 Team", short_name="MCL",
        drivers=["Lando Norris", "Oscar Piastri"],
        pu_supplier="Mercedes", performance_tier=1,
        base_lap_delta=0.08, chassis_aero_efficiency=0.97,
        ers_deployment_efficiency=0.96,
    ),
    "Red Bull": TeamConfig(
        name="Oracle Red Bull Racing", short_name="RBR",
        drivers=["Max Verstappen", "Yuki Tsunoda"],
        pu_supplier="Honda_RBPT", performance_tier=2,
        base_lap_delta=0.25, chassis_aero_efficiency=0.92,
        ers_deployment_efficiency=0.91,
    ),
    "Aston Martin": TeamConfig(
        name="Aston Martin Aramco F1 Team", short_name="AMR",
        drivers=["Fernando Alonso", "Lance Stroll"],
        pu_supplier="Mercedes", performance_tier=2,
        base_lap_delta=0.40, chassis_aero_efficiency=0.90,
        ers_deployment_efficiency=0.90,
    ),
    "Alpine": TeamConfig(
        name="BWT Alpine F1 Team", short_name="ALP",
        drivers=["Pierre Gasly", "Jack Doohan"],
        pu_supplier="Renault", performance_tier=2,
        base_lap_delta=0.55, chassis_aero_efficiency=0.88,
        ers_deployment_efficiency=0.87,
    ),
    "Williams": TeamConfig(
        name="Williams Racing", short_name="WIL",
        drivers=["Carlos Sainz", "Alexander Albon"],
        pu_supplier="Mercedes", performance_tier=3,
        base_lap_delta=0.70, chassis_aero_efficiency=0.87,
        ers_deployment_efficiency=0.88,
    ),
    "Racing Bulls": TeamConfig(
        name="Visa Cash App RB Formula One Team", short_name="RB",
        drivers=["Isack Hadjar", "Liam Lawson"],
        pu_supplier="Honda_RBPT", performance_tier=3,
        base_lap_delta=0.80, chassis_aero_efficiency=0.86,
        ers_deployment_efficiency=0.89,
    ),
    "Haas": TeamConfig(
        name="MoneyGram Haas F1 Team", short_name="HAS",
        drivers=["Oliver Bearman", "Esteban Ocon"],
        pu_supplier="Ferrari", performance_tier=3,
        base_lap_delta=0.90, chassis_aero_efficiency=0.85,
        ers_deployment_efficiency=0.86,
    ),
    "Audi": TeamConfig(
        name="Stake F1 Team Audi", short_name="AUD",
        drivers=["Nico Hulkenberg", "Gabriel Bortoleto"],
        pu_supplier="Audi", performance_tier=4,
        base_lap_delta=1.20, chassis_aero_efficiency=0.82,
        ers_deployment_efficiency=0.83,
    ),
    "Cadillac": TeamConfig(
        name="Cadillac Formula 1 Team", short_name="CAD",
        drivers=["Callum Ilott", "Marcus Armstrong"],
        pu_supplier="Ferrari", performance_tier=4,
        base_lap_delta=1.40, chassis_aero_efficiency=0.80,
        ers_deployment_efficiency=0.81,
    ),
}

# Driver → Team lookup
DRIVER_TEAM: Dict[str, str] = {
    driver: team_name
    for team_name, tc in TEAMS.items()
    for driver in tc.drivers
}

ALL_DRIVERS: List[str] = [d for tc in TEAMS.values() for d in tc.drivers]


# ---------------------------------------------------------------------------
# ERS & Active Aero Parameters (2026)
# ---------------------------------------------------------------------------

ERS_PARAMS = {
    # Battery
    "battery_capacity_mj": 4.0,          # ~2× 2025
    "battery_min_reserve_mj": 0.2,       # never drain below this
    # Recovery modes
    "super_clip_recovery_mj_per_lap": 1.0,   # full-throttle recovery, no aero penalty
    "lift_off_recovery_mj_per_lap": 0.80,    # lift-off recovery, disables active aero
    "coast_recovery_mj_per_lap": 0.40,       # minimal — coasting only
    # Lift-off recharge aero penalty
    "lift_off_aero_time_penalty_s": 0.18,    # per lap using lift-off mode
    # Boost deployment
    "boost_cost_mj": 0.50,
    "boost_lap_time_gain_s": 0.30,           # per lap with boost active
    # Overtake Override Mode (OOM)
    "oom_detection_gap_s": 1.0,             # must be within this gap to trigger
    "oom_extra_capacity_mj": 0.50,          # extra battery cap when OOM active
    "oom_enhanced_power_gain_s": 0.15,      # additional lap-time gain beyond boost
    "oom_battery_threshold_mj": 0.80,       # minimum battery to permit OOM
    # Active Aero
    "corner_mode_downforce_factor": 1.0,
    "straight_mode_drag_reduction_s": 0.35, # time gain vs corner mode on straights
}


# ---------------------------------------------------------------------------
# Safety Car / VSC Parameters
# ---------------------------------------------------------------------------

SC_PARAMS = {
    "sc_lap_time_multiplier": 1.42,         # cars follow SC at this fraction of race pace
    "vsc_lap_time_multiplier": 1.20,
    "sc_typical_duration_laps": (3, 6),     # (min, max) laps under SC
    "vsc_typical_duration_laps": (2, 4),
    "vsc_fraction_of_sc": 0.30,             # 30% of events are VSC not SC
    "sc_field_compression_s": 30.0,         # seconds by which gap is compressed under SC
    "free_pit_window_laps": 3,              # laps after SC deployment = free pit
    # Timing distribution (relative weights: early/mid/late)
    # Real F1 2022-2024 data: incidents cluster mid-to-late race (accumulated damage,
    # tyre failures, strategy battles). Early incidents are less common.
    "sc_timing_weights": [0.25, 0.45, 0.30],
    "sc_timing_boundaries": [0.28, 0.57],   # as fraction of total laps
}


# ---------------------------------------------------------------------------
# Fuel Parameters
# ---------------------------------------------------------------------------

FUEL_PARAMS = {
    "fuel_load_kg": 100.0,
    "fuel_effect_per_kg_s": 0.028,   # seconds of lap time per kg of fuel
                                      # (real F1 2024 data: ~0.028-0.030 s/kg)
    "fuel_burn_variance": 0.04,      # ±4% lap-to-lap variance
    "min_fuel_kg": 1.0,              # FIA minimum fuel at race end
}


# ---------------------------------------------------------------------------
# Simulation / Monte Carlo Settings
# ---------------------------------------------------------------------------

SIM_PARAMS = {
    "default_simulations": 2000,
    "deg_variance_sigma": 0.12,          # ±12% degradation rate variation (real F1 ~10-15%)
    "lap_time_noise_sigma": 0.08,        # ±0.08s gaussian lap-time noise (traffic/minor errors)
    "rival_strategy_options": 3,         # how many rival strategy variants to sample
    "strategy_window_s": 5.0,           # strategies within this of optimal = "alternative"
    "min_stop_spacing_laps": 10,         # minimum laps between pit stops
    "sc_impact_test_laps": [10, 20, 30, 40],  # lap numbers to test SC impact
    "track_temp_variation_sigma": 4.0,   # °C variation in track temp sampling (real ~3-5°C)
    "baseline_track_temp_c": 35.0,
}


# ---------------------------------------------------------------------------
# Championship Points
# ---------------------------------------------------------------------------

POINTS_SYSTEM: Dict[int, int] = {
    1: 25, 2: 18, 3: 15, 4: 12, 5: 10,
    6: 8, 7: 6, 8: 4, 9: 2, 10: 1,
}
FASTEST_LAP_POINT = 1      # awarded if driver finishes in top 10
SPRINT_POINTS: Dict[int, int] = {1: 8, 2: 7, 3: 6, 4: 5, 5: 4, 6: 3, 7: 2, 8: 1}


# ---------------------------------------------------------------------------
# 2026 Regulation Constants
# ---------------------------------------------------------------------------

REGULATIONS_2026 = {
    # Power unit
    "ice_power_kw": 400,
    "mguk_power_kw": 350,
    "has_mguh": False,
    "total_hybrid_fraction": 0.50,       # 50/50 ICE/ERS power split
    "fuel_type": "100% Sustainable Fuel",
    "fuel_energy_flow_regulation": True,  # energy content regulated, not just mass flow
    # Chassis
    "weight_reduction_vs_2025_kg": 30,
    "active_aero_positions": 2,          # exactly 2: corner (closed) and straight (open)
    "drs_abolished": True,
    "active_aero_all_cars_all_straights": True,
    # Tyres
    "front_tyre_width_reduction_mm": 25,
    "rear_tyre_width_reduction_mm": 30,
    "front_diameter_reduction_mm": 15,
    "rear_diameter_reduction_mm": 10,
    "c6_dropped": True,
    "compounds_available": ["C1", "C2", "C3", "C4", "C5"],
    "sets_per_weekend": 13,
    "sets_sprint_weekend": 12,
    "min_compounds_used": 2,
    "min_pit_stops": 1,
    # Cost cap
    "team_cost_cap_usd_m": 215,
    "pu_manufacturer_cap_usd_m": 130,
    # Refuelling
    "refuelling_allowed": False,
}


# ---------------------------------------------------------------------------
# FastF1 Cache Settings
# ---------------------------------------------------------------------------

FASTF1_CACHE_DIR = "data/cache"
FASTF1_DB_PATH = "data/cache/f1_data.db"
HISTORICAL_SEASONS = [2022, 2023, 2024, 2025]
CURRENT_SEASON = 2026
COMPLETED_2026_ROUNDS = ["australia", "china", "japan"]  # update as season progresses


# ---------------------------------------------------------------------------
# UI / Visualisation Settings
# ---------------------------------------------------------------------------

UI_COLORS = {
    "ferrari_red": "#DC0000",
    "mercedes_teal": "#00D2BE",
    "mclaren_papaya": "#FF8000",
    "redbull_blue": "#3671C6",
    "background": "#0F0F0F",
    "card_bg": "#1A1A1A",
    "text_primary": "#FFFFFF",
    "text_secondary": "#AAAAAA",
    "compound_hard": "#FFFFFF",
    "compound_medium": "#FFD700",
    "compound_soft": "#DC0000",
    "compound_inter": "#39B54A",
    "compound_wet": "#0067FF",
}

PLOTLY_TEMPLATE = "plotly_dark"


# ---------------------------------------------------------------------------
# Calibration Overrides
# ---------------------------------------------------------------------------

def load_calibration_overrides() -> dict:
    """
    Read calibration JSON files (deg_curves, pit_loss, sc_history).
    Returns a dict with keys 'deg_curves', 'pit_loss', 'sc_history'.
    Falls back to {} on any error.
    """
    try:
        from src.data.calibration.calibration_loader import CalibrationLoader
        loader = CalibrationLoader()
        return loader.load_all()
    except Exception:
        return {}
