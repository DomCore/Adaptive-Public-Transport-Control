"""
run_all.py
==========

Orchestrator for the core pipeline (steps 00-05). Runs each numbered script in
order using the current Python interpreter and stops on the first failure.

Examples
--------
    # Full run, fetching Prague stops first
    python run_all.py --fetch

    # Skip the network fetch (data/prague_stops.geojson already present)
    python run_all.py

    # Re-run only the experiment + tables (BBN already built and enriched)
    python run_all.py --from 4

Step map
--------
    0  00_fetch_prague_stops.py   download GTFS feed -> data/prague_stops.geojson
    1  01_prepare_data.py         preprocess NY Bus CSV -> data_extended.parquet
    2  02_build_bbn_pgmpy.py      fit the Bayesian network -> bbn_model.pkl
    3  03_enrich_stops.py         cluster stops + BBN inference -> stops_enriched
    4  04_run_experiment.py       risk-aware A* sweep -> results + pareto curve
    5  05_generate_tables.py      format Markdown tables for the dissertation
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).parent

# (step index, script filename, one-line description)
STEPS = [
    (0, "00_fetch_prague_stops.py", "fetch Prague stops (network)"),
    (1, "01_prepare_data.py", "preprocess NY Bus CSV"),
    (2, "02_build_bbn_pgmpy.py", "build Bayesian network"),
    (3, "03_enrich_stops.py", "cluster stops + BBN inference"),
    (4, "04_run_experiment.py", "risk-aware A* sweep"),
    (5, "05_generate_tables.py", "generate dissertation tables"),
]


def run_step(script: str, description: str) -> None:
    """Run one pipeline script, raising CalledProcessError on failure."""
    print(f"\n{'=' * 70}\n>>> {script}  —  {description}\n{'=' * 70}")
    started = time.time()
    subprocess.run([sys.executable, script], cwd=BASE_DIR, check=True)
    print(f"--- {script} finished in {time.time() - started:.1f}s")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the core pipeline (steps 00-05).")
    parser.add_argument("--fetch", action="store_true",
                        help="include step 0 (download Prague stops). Off by default.")
    parser.add_argument("--from", dest="from_step", type=int, default=1,
                        help="first step index to run (default: 1).")
    parser.add_argument("--to", dest="to_step", type=int, default=5,
                        help="last step index to run (default: 5).")
    args = parser.parse_args()

    start = 0 if args.fetch else max(args.from_step, 1)
    # If --fetch is given we still honour an explicit --from above 0.
    if args.fetch and args.from_step > 1:
        start = args.from_step

    selected = [s for s in STEPS if start <= s[0] <= args.to_step]
    if args.fetch and not any(s[0] == 0 for s in selected):
        selected = [STEPS[0]] + selected

    if not selected:
        print("Nothing to run for the given --from/--to range.")
        return 1

    print("Pipeline plan:")
    for idx, script, desc in selected:
        print(f"  [{idx}] {script:28s} {desc}")

    overall = time.time()
    for _, script, description in selected:
        try:
            run_step(script, description)
        except subprocess.CalledProcessError as exc:
            print(f"\n!!! Step '{script}' failed (exit {exc.returncode}). Stopping.",
                  file=sys.stderr)
            return exc.returncode

    print(f"\nAll selected steps completed in {time.time() - overall:.1f}s.")
    print(f"See {BASE_DIR / 'output'} for results.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
