"""
04_run_experiment.py
====================

Risk-aware A* з f*(n) = g(n) + h(n) + λ·R(n).

Версія для Праги (4349 зупинок). Випадковий sampling OD-пар працює,
бо реальне місто має достатню географічну варіативність ризику.

Вхід:  output/stops_enriched.json
Вихід: output/results_raw.csv
       output/results_aggregated.csv
       output/statistical_test.csv
       output/pareto_curve.png
"""

import json
import numpy as np
import pandas as pd
from geopy.distance import geodesic
from scipy import stats
import matplotlib.pyplot as plt
from tqdm import tqdm

# Routing primitives live in the transit package (single source of truth). This
# script keeps the experiment orchestration: OD sweep, aggregation, stats, plot.
from transit.routing import (  # noqa: F401  (re-exported for the test-suite)
    build_graph,
    heuristic_km,
    risk_component,
    astar,
    path_metrics,
    sample_od_pairs,
)

from config import (
    OUT_STOPS_ENRICHED,
    OUT_RESULTS_RAW,
    OUT_RESULTS_AGG,
    OUT_STATS,
    OUT_PARETO_PNG,
    GRAPH_D_MAX_KM,
    LAMBDA_VALUES,
    N_OD_PAIRS,
    OD_PAIRS_RANDOM_STATE,
    HARD_EXCLUSION_THRESHOLD,
    ALPHA,
)


# ---------- Експеримент ----------

def run_experiment(graph: dict, od_pairs: list[tuple[str, str]],
                   lambdas: list[float]) -> pd.DataFrame:
    print(f"[04] Запуск експерименту: "
          f"{len(od_pairs)} OD-пар × {len(lambdas)} λ "
          f"= {len(od_pairs) * len(lambdas)} запусків A*")

    rows = []
    for lam in lambdas:
        for origin, dest in tqdm(od_pairs, desc=f"     λ={lam}"):
            path = astar(graph, origin, dest, lambda_risk=lam)
            if path is None:
                rows.append({
                    "lambda": lam, "origin": origin, "dest": dest,
                    "reachable": False, "length_km": np.nan,
                    "r_overload": np.nan, "total_risk": np.nan, "n_stops": 0,
                })
                continue
            m = path_metrics(path, graph)
            rows.append({"lambda": lam, "origin": origin, "dest": dest,
                         "reachable": True, **m})
    return pd.DataFrame(rows)


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    rb = df[df["reachable"]]
    if len(rb) == 0:
        # порожній DataFrame з потрібними колонками
        return pd.DataFrame(columns=[
            "lambda", "avg_length", "avg_r_overload", "avg_total_risk",
            "avg_n_stops", "coverage", "delta_length_pct"
        ])
    
    agg = rb.groupby("lambda").agg(
        avg_length=("length_km", "mean"),
        avg_r_overload=("r_overload", "mean"),
        avg_total_risk=("total_risk", "mean"),
        avg_n_stops=("n_stops", "mean"),
    ).reset_index()
    cov = df.groupby("lambda")["reachable"].mean().reset_index().rename(
        columns={"reachable": "coverage"})
    agg = agg.merge(cov, on="lambda")

    base = agg.loc[agg["lambda"] == 0.0, "avg_length"]
    if len(base) and base.iloc[0] > 0:
        baseline_length = base.iloc[0]
        agg["delta_length_pct"] = (
            (agg["avg_length"] - baseline_length) / baseline_length * 100
        )
    else:
        agg["delta_length_pct"] = np.nan
    return agg


def statistical_test(df: pd.DataFrame, lambdas: list[float]) -> pd.DataFrame:
    base = df[(df["lambda"] == 0.0) & df["reachable"]].set_index(["origin", "dest"])
    rows = []
    for lam in lambdas:
        if lam == 0.0:
            continue
        treat = df[(df["lambda"] == lam) & df["reachable"]].set_index(["origin", "dest"])
        common = base.index.intersection(treat.index)
        if len(common) < 30:
            rows.append({"lambda": lam, "n_pairs": len(common),
                         "warning": "too few common pairs"})
            continue
        b = base.loc[common, "r_overload"].values
        t = treat.loc[common, "r_overload"].values
        # Перевірка чи є взагалі різниця (інакше t-test = NaN)
        if np.allclose(b, t):
            rows.append({"lambda": lam, "n_pairs": len(common),
                         "mean_baseline": float(b.mean()),
                         "mean_treated": float(t.mean()),
                         "reduction_pct": 0.0,
                         "t_stat": np.nan, "p_value": np.nan,
                         "significant_at_alpha": False,
                         "warning": "identical results — λ has no effect"})
            continue
        stat, p = stats.ttest_rel(b, t)
        rows.append({
            "lambda": lam, "n_pairs": len(common),
            "mean_baseline": float(b.mean()), "mean_treated": float(t.mean()),
            "reduction_pct": float((b.mean() - t.mean()) / b.mean() * 100) if b.mean() > 0 else 0.0,
            "t_stat": float(stat), "p_value": float(p),
            "significant_at_alpha": p < ALPHA,
        })
    return pd.DataFrame(rows)


def plot_pareto(agg: pd.DataFrame, save_path):
    fig, ax = plt.subplots(figsize=(9, 6))
    valid = agg.dropna(subset=["delta_length_pct"])
    if len(valid) == 0:
        plt.close()
        print(f"[04] Графік пропущено: немає валідних даних")
        return
    ax.plot(valid["delta_length_pct"], valid["avg_r_overload"],
            "o-", linewidth=2, markersize=10, color="#2c3e50")
    for _, row in valid.iterrows():
        ax.annotate(f"λ={row['lambda']:.1f}",
                    (row["delta_length_pct"], row["avg_r_overload"]),
                    textcoords="offset points", xytext=(10, 8),
                    fontsize=11, fontweight="bold")
    ax.set_xlabel("Δ довжини маршруту, %", fontsize=13)
    ax.set_ylabel("Середня P(overload) маршруту", fontsize=13)
    ax.set_title("Крива Парето: компроміс «довжина — ризик»", fontsize=14)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[04] Графік збережено: {save_path}")


# ---------- Main ----------

def main():
    print(f"[04] Завантаження зупинок: {OUT_STOPS_ENRICHED}")
    with open(OUT_STOPS_ENRICHED, "r", encoding="utf-8") as f:
        data = json.load(f)
    stops = data["stops"]
    print(f"     Завантажено зупинок: {len(stops)}")

    p_list = [s["p_overload"] for s in stops]
    print(f"     P(overload) розкид: "
          f"min={min(p_list):.3f}, mean={np.mean(p_list):.3f}, "
          f"max={max(p_list):.3f}, std={np.std(p_list):.3f}")

    graph = build_graph(stops, GRAPH_D_MAX_KM)
    od_pairs = sample_od_pairs(stops, graph, N_OD_PAIRS, OD_PAIRS_RANDOM_STATE)

    df_raw = run_experiment(graph, od_pairs, LAMBDA_VALUES)
    df_raw.to_csv(OUT_RESULTS_RAW, index=False)
    print(f"[04] Сирі результати: {OUT_RESULTS_RAW}")

    df_agg = aggregate(df_raw)
    df_agg.to_csv(OUT_RESULTS_AGG, index=False)
    print(f"[04] Агрегати: {OUT_RESULTS_AGG}")

    df_stats = statistical_test(df_raw, LAMBDA_VALUES)
    df_stats.to_csv(OUT_STATS, index=False)
    print(f"[04] Стат-тест: {OUT_STATS}")

    plot_pareto(df_agg, OUT_PARETO_PNG)

    print("\n=== Агреговані результати ===")
    print(df_agg.to_string(index=False))
    print("\n=== Статистичний тест ===")
    print(df_stats.to_string(index=False))


if __name__ == "__main__":
    main()
