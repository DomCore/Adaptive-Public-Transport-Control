# Adaptive Public-Transport Control

Reference implementation for the dissertation *"Methods and information
technology for adaptive control of urban public transport"*
(V. Zetchenko, Chernihiv Polytechnic National University).

The system couples three components into a single, reproducible pipeline:

1. **Bayesian Belief Network (BBN)** — predicts the probability that a stop /
   route segment is *overloaded*, given a context vector (time of day, day of
   week, failure cause). Trained on the **NY Bus Breakdown & Delays** dataset.
2. **Risk-aware A\*** — a route planner whose cost function
   `f*(n) = g(n) + h(n) + λ·R(n)` adds a probabilistic risk term
   `R(n) = −ln(1 − P(overload | xₙ))` on top of geometric distance. Demonstrated
   on the **Prague** public-transport stop network.
3. **Adaptive decision module** — maps the overload probability onto six control
   actions (`a0`–`a5`) and closes the loop with an incremental, forgetting-factor
   Bayesian update.

A separate experiment compares tabular **data-augmentation** methods
(Random Oversampling, SMOTE, ADASYN, GAN) and motivates the *CPD-aware* targeted
augmentation proposed in the thesis.

> **Cross-domain transfer.** The BBN is trained on New York school-bus data and
> applied, without retraining, to Prague city transit. Only a lightweight
> geographic localisation (K-means clustering of stops + context mapping) is
> needed. This mirrors the realistic case where a target city has no historical
> failure database of its own.

---

## Repository layout

```
adaptive-transit-control/
├── config.py                  # single source of truth for paths & parameters
├── run_all.py                 # orchestrator for the core pipeline (steps 00-05)
│
├── 00_fetch_prague_stops.py   # GTFS feed  -> data/prague_stops.geojson
├── 01_prepare_data.py         # NY Bus CSV -> output/data_extended.parquet
├── 02_build_bbn_pgmpy.py      # multi-parent BBN (pgmpy) -> bbn_model.pkl
├── 03_enrich_stops.py         # cluster stops + BBN inference -> stops_enriched.json
├── 04_run_experiment.py       # risk-aware A* sweep over λ -> results + Pareto curve
├── 05_generate_tables.py      # Markdown tables for the dissertation
│
├── decision/                  # adaptive decision module (a0-a5) + CPD feedback
│   └── decision_engine.py
├── augmentation/              # SMOTE / ADASYN / GAN comparison (needs TensorFlow)
│   └── compare_augmentation.py
├── tests/                     # pytest suite (decision engine, A*, preprocessing)
├── data/                      # input data (not committed; see data/README.md)
└── output/                    # generated artefacts (git-ignored)
```

### Mapping to the dissertation

| Component                         | Code                                   | Section |
|-----------------------------------|----------------------------------------|---------|
| Transport graph + A\*             | `04_run_experiment.py`                 | 2.1     |
| Risk term `R(n)`, modified `f*(n)`| `04_run_experiment.py`                 | 2.1.3   |
| Bayesian Belief Network           | `02_build_bbn_pgmpy.py`                | 2.2     |
| Data augmentation (SMOTE/GAN)     | `augmentation/compare_augmentation.py` | 2.3, 3.4.2 |
| Adaptive decision-making (a0-a5)  | `decision/decision_engine.py`          | 2.4     |
| Cross-domain transfer (NY→Prague) | `03_enrich_stops.py`                   | 3.4.5   |
| End-to-end experiment             | `run_all.py` (steps 01-05)             | 3.4.5   |

---

## Installation

Python 3.9–3.11.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The GAN experiment needs TensorFlow on top of the core requirements:

```bash
pip install -r augmentation/requirements-gan.txt
```

## Data

Two openly licensed datasets are required but **not** committed (one is ~90 MB).
See [`data/README.md`](data/README.md). In short:

* `data/data.csv` — NY Bus Breakdown & Delays (NYC Open Data, `ez4e-fazm`).
* `data/prague_stops.geojson` — produced automatically by step 00 from the
  Prague PID GTFS feed.

## Running the pipeline

```bash
# Fetch Prague stops, then run the whole pipeline (steps 0-5)
python run_all.py --fetch

# data/prague_stops.geojson already present -> skip the network fetch
python run_all.py

# Re-run only the A* experiment + tables (steps 4-5)
python run_all.py --from 4
```

Or run the steps individually:

```bash
python 00_fetch_prague_stops.py
python 01_prepare_data.py
python 02_build_bbn_pgmpy.py
python 03_enrich_stops.py
python 04_run_experiment.py
python 05_generate_tables.py
```

Each step writes to `output/` and prints a summary. End-to-end runtime is
roughly 5–10 minutes on a mid-range laptop (the A* sweep dominates).

### Outputs

```
output/
├── data_extended.parquet       # preprocessed dataset with BBN nodes
├── bbn_model.pkl               # fitted pgmpy model
├── stops_enriched.json         # stops + p_overload + cluster assignment
├── results_raw.csv             # every A* run (OD pair × λ)
├── results_aggregated.csv      # per-λ aggregates (dissertation Table 3.9)
├── statistical_test.csv        # paired t-test vs λ=0 (Table 3.10)
├── pareto_curve.png            # length–risk trade-off (Figure 3.2)
└── tables_for_dissertation.md  # ready-to-paste Markdown tables
```

## The decision module

The decision engine is independent of the heavy pipeline and can be used on its
own:

```python
from decision import DecisionEngine, IncrementalCPDUpdater

engine = DecisionEngine()
decision = engine.decide(p_overload=0.78, free_units_available=True)
print(decision)            # -> a3 + a5 (activate reserve + inform passengers)
print(decision.action_codes, decision.total_cost)

# Forgetting-factor feedback (CPD_new = λ·CPD_old + (1-λ)·CPD_observed)
updater = IncrementalCPDUpdater(forgetting_factor=0.90)
updater.update([0.8, 0.2], [0.4, 0.6])   # -> [0.76, 0.24]
```

| `P(overload)`        | Primary action            | Notes |
|----------------------|---------------------------|-------|
| `< 0.30`             | `a0` do nothing           | `a5` held in reserve |
| `0.30 – 0.50`        | `a5` inform passengers    | escalate to `a1` after 15 min |
| `0.50 – 0.70`        | `a1` increase frequency   | `a2` reallocate if no free units |
| `0.70 – 0.85`        | `a3` activate reserve     | `+ a5` inform passengers |
| `≥ 0.85`             | `a4` reroute via A\*       | `+ a1 + a5` |

`python decision/decision_engine.py` prints a worked example over the whole
probability range.

## Augmentation experiment

```bash
pip install -r augmentation/requirements-gan.txt
python augmentation/compare_augmentation.py
```

Writes `augmentation/output/comparison_table.md`. The headline finding: for this
well-separated tabular task (ROC-AUC > 0.95 without augmentation) no global
augmentation method — from Random Oversampling to GAN — improves the model's
discriminative power, which motivates the CPD-aware targeted approach.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

The suite covers the decision engine (band selection, validation, CPD update),
the risk-aware A\* (distance-vs-risk trade-off) and the preprocessing helpers.
It does **not** require the input datasets.

## Reproducibility

Every stochastic step uses a fixed seed (`random_state = 42`): the K-means
clustering, the OD-pair sampling and the train/test split. Re-running the
pipeline on the same inputs reproduces the tables and figures referenced in the
dissertation.

## License

Code: [MIT](LICENSE). The input datasets keep their own licenses (NYC Open Data
terms of use; Prague PID open data, CC-BY).
