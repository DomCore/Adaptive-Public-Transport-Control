"""
Central configuration for the adaptive public-transport control system.

This is the single source of truth for every tunable parameter and file path.
It lives inside the ``transit`` package so that the library, the command-line
tool (``transit-control``) and the Streamlit dashboard all share exactly the
same settings. The repo-root ``config.py`` is a thin backwards-compatible shim
that re-exports everything from here, so the numbered pipeline scripts
(``01_prepare_data.py`` … ``05_generate_tables.py``) keep working unchanged.

The system couples two independent, real-world data sources:
  * NY Bus Breakdown & Delays (NYC Open Data) - trains the Bayesian network.
  * Prague public-transport stops (GTFS feed, PID) - the routing graph.

See README.md for the mapping between these parameters and the dissertation
(sections 2.1, 3.1 and 4.4.5).
"""

import sys
from pathlib import Path

# Make stdout/stderr UTF-8 tolerant. The pipeline prints Ukrainian text and
# symbols like λ, Δ and ≠; on a legacy Windows console (cp1251) those would
# otherwise raise UnicodeEncodeError and abort the script before it writes its
# output. Importing config (which every entry point does) applies the guard.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):  # pragma: no cover - non-reconfigurable stream
        pass

# ---------- Paths ----------
# config.py lives in transit/, so the repository root is one level up. Resolving
# keeps the data/ and output/ directories anchored at the repo root regardless of
# the current working directory or whether the package is installed editable.
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# Input data (not committed to git; see data/README.md on how to obtain it).
INPUT_CSV = DATA_DIR / "data.csv"                       # NY Bus Breakdown CSV
INPUT_STOPS_GEOJSON = DATA_DIR / "prague_stops.geojson"  # Prague stops (GTFS->GeoJSON)

# Pipeline artefacts.
OUT_DATA_EXTENDED = OUTPUT_DIR / "data_extended.parquet"
OUT_BBN_MODEL = OUTPUT_DIR / "bbn_model.pkl"
OUT_BBN_META = OUTPUT_DIR / "bbn_meta.json"
OUT_STOPS_ENRICHED = OUTPUT_DIR / "stops_enriched.json"
OUT_RESULTS_RAW = OUTPUT_DIR / "results_raw.csv"
OUT_RESULTS_AGG = OUTPUT_DIR / "results_aggregated.csv"
OUT_STATS = OUTPUT_DIR / "statistical_test.csv"
OUT_PARETO_PNG = OUTPUT_DIR / "pareto_curve.png"
OUT_TABLES_MD = OUTPUT_DIR / "tables_for_dissertation.md"


# ---------- Preprocessing ----------
# Hour-of-day buckets used to derive the TimeOfDay node.
TIME_OF_DAY_BINS = [
    (0, 6, "night"),
    (6, 10, "peak_morning"),
    (10, 16, "midday"),
    (16, 19, "peak_evening"),
    (19, 24, "evening"),
]


# ---------- Bayesian Belief Network ----------
# Eight-node multi-parent network (dissertation sections 2.1.2 and 4.4.5).
#
# The "core" sub-network that drives routing is unchanged:
#     TimeOfDay, Reason -> overload      (with Reason <- TimeOfDay, DayOfWeek)
# Because the A* pipeline only ever queries P(overload | TimeOfDay, Reason) — i.e.
# conditions on BOTH parents of `overload` — that probability is a direct CPD
# lookup and is d-separated from everything else. The four extra nodes below are
# therefore added strictly as descendants / non-ancestors of `overload` and
# `Reason`, so every routing number and the marginal smoke-test are preserved
# exactly while the network matches the conceptual model in section 2.1.2.
# Every node has at most two parents, so the treewidth stays 2 (claim "w = 2").
BBN_NODES = [
    "TimeOfDay",
    "DayOfWeek",
    "Boro",
    "School_Age_or_PreK",
    "Reason",
    "overload",
    "How_Long_Delayed",
    "Has_Contractor_Notified_Schools",
]

BBN_EDGES = [
    # --- core (drives routing; do NOT change without re-validating results) ---
    ("TimeOfDay", "Reason"),    # time of day influences the type of failure
    ("DayOfWeek", "Reason"),    # day of week influences the type of failure
    ("TimeOfDay", "overload"),  # time of day directly influences overload
    ("Reason", "overload"),     # failure cause directly influences overload
    # --- consequence / context layer (descendants; result-preserving) ---
    ("Reason", "How_Long_Delayed"),  # cause shapes the delay duration
    ("Boro", "How_Long_Delayed"),    # borough (traffic, geography) shapes duration
    ("overload", "Has_Contractor_Notified_Schools"),       # severity -> notify
    ("School_Age_or_PreK", "Has_Contractor_Notified_Schools"),  # school type -> notify
]


# ---------- Geographic clustering ----------
# Number of K-means clusters used to localise the BBN signal onto target-city
# stops. 12 clusters give a smooth gradient of P(overload) across the city.
N_CLUSTERS = 12
KMEANS_RANDOM_STATE = 42


# ---------- Inference ----------
DEFAULT_INFERENCE_CONTEXT = {
    "TimeOfDay": "peak_morning",
    "DayOfWeek": "weekday",
}


# ---------- Transport graph ----------
# Two stops are connected by an edge if the geodesic distance between them is
# below this threshold (km). Smaller -> denser-looking graph but more components.
GRAPH_D_MAX_KM = 0.5


# ---------- Risk-aware A* experiment ----------
# Risk-sensitivity sweep for f*(n) = g(n) + h(n) + lambda * R(n).
LAMBDA_VALUES = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]
N_OD_PAIRS = 300
OD_PAIRS_RANDOM_STATE = 42

# Hard exclusion safety net: vertices above this P(overload) are never used,
# regardless of lambda. Set high so that lambda drives the trade-off.
HARD_EXCLUSION_THRESHOLD = 0.99


# ---------- Statistical test ----------
ALPHA = 0.05
