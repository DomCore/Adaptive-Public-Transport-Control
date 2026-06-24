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
from heapq import heappush, heappop
from geopy.distance import geodesic
from scipy import stats
import matplotlib.pyplot as plt
from tqdm import tqdm

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


# ---------- 1. Граф ----------

def build_graph(stops: list[dict], d_max_km: float) -> dict:
    """Граф зупинок: ребра між парами на відстані <= d_max."""
    print(f"[04] Побудова графа (d_max={d_max_km} км) ...")
    graph = {
        s["id"]: {
            "lat": s["lat"],
            "lon": s["lon"],
            "p_overload": s["p_overload"],
            "neighbors": {},
        }
        for s in stops
    }

    n = len(stops)
    edges_count = 0
    # Оптимізація: попередньо тригонометричний фільтр.
    # Якщо різниця широт > d_max/111, то і відстань точно > d_max.
    lats = np.array([s["lat"] for s in stops])
    lons = np.array([s["lon"] for s in stops])
    
    # Поріг у градусах: 1° lat ≈ 111 км; 1° lon ≈ 111 * cos(lat) км
    lat_threshold = d_max_km / 111.0
    
    for i in tqdm(range(n), desc="     edges"):
        si_id = stops[i]["id"]
        # Швидкий префільтр по широті
        nearby_idx = np.where(np.abs(lats[i+1:] - lats[i]) < lat_threshold)[0] + (i + 1)
        for j in nearby_idx:
            d = geodesic((lats[i], lons[i]), (lats[j], lons[j])).kilometers
            if d <= d_max_km:
                sj_id = stops[j]["id"]
                graph[si_id]["neighbors"][sj_id] = d
                graph[sj_id]["neighbors"][si_id] = d
                edges_count += 1

    avg_neighbors = np.mean([len(g["neighbors"]) for g in graph.values()])
    isolated = sum(1 for g in graph.values() if len(g["neighbors"]) == 0)
    print(f"     Ребер: {edges_count:,}")
    print(f"     Середня кількість сусідів: {avg_neighbors:.1f}")
    print(f"     Ізольованих зупинок: {isolated}")
    return graph


# ---------- 2. A* ----------

def heuristic_km(graph: dict, n1: str, n2: str) -> float:
    """Геодезична відстань між вершинами в км. Допустима евристика."""
    return geodesic(
        (graph[n1]["lat"], graph[n1]["lon"]),
        (graph[n2]["lat"], graph[n2]["lon"]),
    ).kilometers


def risk_component(p: float, eps: float = 1e-6) -> float:
    """R(n) = -ln(1 - P(overload)). eps щоб не було log(0)."""
    return -np.log(max(1 - p, eps))


def astar(
    graph: dict,
    start: str,
    goal: str,
    lambda_risk: float = 0.0,
    hard_threshold: float = HARD_EXCLUSION_THRESHOLD,
) -> list[str] | None:
    """
    Risk-aware A*. Мінімізує C(n) = g(n) + λ·sum_R(n), а не просто g(n).
    Допустима евристика h(n) — геодезична відстань. Relaxation за C.
    """
    if start not in graph or goal not in graph:
        return None
    if start == goal:
        return [start]

    open_set = []
    counter = 0
    initial_cost = lambda_risk * risk_component(graph[start]["p_overload"])
    heappush(open_set, (initial_cost + heuristic_km(graph, start, goal),
                        counter, start))

    came_from = {}
    cost_score = {start: initial_cost}
    closed = set()

    while open_set:
        _, _, current = heappop(open_set)
        if current in closed:
            continue
        closed.add(current)

        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path

        for neighbor, edge_dist in graph[current]["neighbors"].items():
            if neighbor in closed:
                continue
            if graph[neighbor]["p_overload"] > hard_threshold:
                continue

            tentative_cost = (
                cost_score[current]
                + edge_dist
                + lambda_risk * risk_component(graph[neighbor]["p_overload"])
            )

            if neighbor in cost_score and tentative_cost >= cost_score[neighbor]:
                continue

            came_from[neighbor] = current
            cost_score[neighbor] = tentative_cost

            h = heuristic_km(graph, neighbor, goal)
            f_star = tentative_cost + h

            counter += 1
            heappush(open_set, (f_star, counter, neighbor))

    return None


# ---------- 3. Метрики ----------

def path_metrics(path: list[str], graph: dict) -> dict:
    if not path or len(path) < 2:
        return {"length_km": 0.0, "r_overload": 0.0,
                "total_risk": 0.0, "n_stops": len(path) if path else 0}

    length = 0.0
    for i in range(len(path) - 1):
        length += geodesic(
            (graph[path[i]]["lat"], graph[path[i]]["lon"]),
            (graph[path[i + 1]]["lat"], graph[path[i + 1]]["lon"]),
        ).kilometers

    p_list = [graph[v]["p_overload"] for v in path]
    return {
        "length_km": float(length),
        "r_overload": float(np.mean(p_list)),
        "total_risk": float(sum(risk_component(p) for p in p_list)),
        "n_stops": len(path),
    }


# ---------- 4. Sampling OD-пар ----------

def sample_od_pairs(
    stops: list[dict],
    graph: dict,
    n_pairs: int,
    seed: int,
    min_distance_km: float = 2.0,
) -> list[tuple[str, str]]:
    """
    Випадковий sampling OD-пар з обмеженням мінімальної відстані.
    
    min_distance_km гарантує, що пара не сусідня — A* має реально
    щось вибрати, а не повернути 1-хоповий маршрут.
    """
    rng = np.random.default_rng(seed)
    eligible = [s for s in stops if len(graph[s["id"]]["neighbors"]) > 0]
    if len(eligible) < 2:
        raise RuntimeError("Замало неізольованих зупинок.")

    pairs = set()
    attempts = 0
    max_attempts = n_pairs * 30
    while len(pairs) < n_pairs and attempts < max_attempts:
        attempts += 1
        a, b = rng.choice(len(eligible), size=2, replace=False)
        sa, sb = eligible[a], eligible[b]
        d = geodesic((sa["lat"], sa["lon"]), (sb["lat"], sb["lon"])).kilometers
        if d < min_distance_km:
            continue
        pairs.add((sa["id"], sb["id"]))

    print(f"[04] Згенеровано {len(pairs)} OD-пар "
          f"(мін. відстань {min_distance_km} км, {attempts} спроб)")
    return list(pairs)


# ---------- 5. Експеримент ----------

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
