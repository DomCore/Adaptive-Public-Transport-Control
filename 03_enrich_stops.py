"""
03_enrich_stops.py
==================

Кластеризує зупинки цільового міста (Прага) і запитує BBN для кожного
кластера: P(overload | context).

N_CLUSTERS=12 дає реальний градієнт P(overload) — від низького до високого,
з проміжними значеннями.

Вхід:  output/bbn_model.pkl
       output/data_extended.parquet
       data/prague_stops.geojson
Вихід: output/stops_enriched.json
"""

import json
import pickle
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from pgmpy.inference import VariableElimination
from tqdm import tqdm

from config import (
    INPUT_STOPS_GEOJSON,
    OUT_BBN_MODEL,
    OUT_DATA_EXTENDED,
    OUT_STOPS_ENRICHED,
    N_CLUSTERS,
    KMEANS_RANDOM_STATE,
    DEFAULT_INFERENCE_CONTEXT,
)


def load_stops(path) -> list[dict]:
    print(f"[03] Завантаження зупинок: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    stops = []
    for i, feat in enumerate(data["features"]):
        coords = feat["geometry"]["coordinates"]
        if isinstance(coords[0], list):
            coords = coords[0]
        lon, lat = coords[0], coords[1]
        props = feat.get("properties", {})
        stop_id = (
            props.get("stop_id") or props.get("id")
            or props.get("atco_code") or props.get("naptan_code")
            or f"stop_{i}"
        )
        stops.append({
            "id": str(stop_id),
            "lat": float(lat),
            "lon": float(lon),
            "name": props.get("name", ""),
        })
    print(f"     Зупинок завантажено: {len(stops)}")
    return stops


def cluster_stops(stops: list[dict], k: int) -> tuple[np.ndarray, np.ndarray]:
    print(f"[03] K-means кластеризація на {k} районів ...")
    coords = np.array([[s["lat"], s["lon"]] for s in stops])
    km = KMeans(n_clusters=k, random_state=KMEANS_RANDOM_STATE, n_init=10)
    labels = km.fit_predict(coords)
    sizes = np.bincount(labels)
    print(f"     Розміри кластерів: {sizes.tolist()}")
    return labels, km.cluster_centers_


def build_cluster_to_context(df_ny: pd.DataFrame, n_clusters: int) -> dict:
    """
    Рівномірно розподіляємо рангові комбінації (TimeOfDay, Reason)
    за P(overload) серед n_clusters кластерів.
    
    Беремо тільки достатньо представлені комбінації (n>=50), щоб уникнути
    шумних оцінок. Викидаємо Reason='Unknown' (агрегує записи без причини).
    """
    print("[03] Побудова mapping cluster → context ...")
    
    combo = (
        df_ny.groupby(["TimeOfDay", "Reason"])
        .agg(p_overload=("overload", "mean"), n=("overload", "size"))
        .reset_index()
    )
    combo = combo[combo["Reason"] != "Unknown"]
    combo = combo[combo["n"] >= 50].sort_values("p_overload")
    
    if len(combo) < n_clusters:
        print(f"     [!] Замало надійних комбінацій ({len(combo)}). "
              f"Дублюємо крайні значення.")
        # Дублюємо найкращі/найгірші щоб заповнити n_clusters
        while len(combo) < n_clusters:
            combo = pd.concat([combo, combo.iloc[[-1]]], ignore_index=True)
    
    indices = np.linspace(0, len(combo) - 1, n_clusters).astype(int)
    selected = combo.iloc[indices].reset_index(drop=True)
    
    mapping = {}
    print("     Mapping (cluster → context, sort by P(overload)):")
    for cluster_id in range(n_clusters):
        row = selected.iloc[cluster_id]
        mapping[cluster_id] = {
            "TimeOfDay": row["TimeOfDay"],
            "Reason": row["Reason"],
            "DayOfWeek": "weekday",
        }
        print(f"       cluster {cluster_id:2d} → "
              f"TimeOfDay={row['TimeOfDay']:13s} "
              f"Reason={row['Reason']:30s} "
              f"P(overload)={row['p_overload']:.3f} (n={int(row['n'])})")
    return mapping


def query_overload(inference: VariableElimination, evidence: dict) -> float:
    try:
        result = inference.query(
            variables=["overload"],
            evidence=evidence,
            show_progress=False,
        )
        states = result.state_names["overload"]
        for i, s in enumerate(states):
            if str(s) == "1":
                return float(result.values[i])
        return float(result.values[-1])
    except Exception:
        return 0.5


def enrich_with_probabilities(
    stops: list[dict],
    labels: np.ndarray,
    cluster_to_context: dict,
    inference: VariableElimination,
) -> list[dict]:
    """Додає p_overload до кожної зупинки (єдина ймовірність на кластер)."""
    print("[03] Запит P(overload) для кожного кластера ...")
    cache = {}
    for cluster_id, evidence in cluster_to_context.items():
        key = tuple(sorted(evidence.items()))
        if key not in cache:
            p = query_overload(inference, evidence)
            cache[key] = p
            print(f"     cluster {cluster_id:2d}: P(overload | {evidence}) = {p:.4f}")
    
    for stop, cluster_id in zip(stops, labels):
        evidence = cluster_to_context[int(cluster_id)]
        key = tuple(sorted(evidence.items()))
        stop["cluster"] = int(cluster_id)
        stop["context"] = evidence
        stop["p_overload"] = float(cache[key])
    return stops


def main():
    print(f"[03] Завантаження BBN: {OUT_BBN_MODEL}")
    with open(OUT_BBN_MODEL, "rb") as f:
        model = pickle.load(f)
    inference = VariableElimination(model)

    print(f"[03] Завантаження референсних даних: {OUT_DATA_EXTENDED}")
    df_ny = pd.read_parquet(OUT_DATA_EXTENDED)

    stops = load_stops(INPUT_STOPS_GEOJSON)
    labels, _ = cluster_stops(stops, N_CLUSTERS)
    cluster_to_context = build_cluster_to_context(df_ny, N_CLUSTERS)
    stops = enrich_with_probabilities(stops, labels, cluster_to_context, inference)

    p_list = [s["p_overload"] for s in stops]
    print(f"\n[03] Розкид P(overload) по зупинках:")
    print(f"     min:    {min(p_list):.4f}")
    print(f"     mean:   {np.mean(p_list):.4f}")
    print(f"     median: {np.median(p_list):.4f}")
    print(f"     max:    {max(p_list):.4f}")
    print(f"     std:    {np.std(p_list):.4f}")

    with open(OUT_STOPS_ENRICHED, "w", encoding="utf-8") as f:
        json.dump({
            "stops": stops,
            "cluster_to_context": cluster_to_context,
            "default_context": DEFAULT_INFERENCE_CONTEXT,
        }, f, indent=2, ensure_ascii=False)
    print(f"[03] Збережено: {OUT_STOPS_ENRICHED}")


if __name__ == "__main__":
    main()
