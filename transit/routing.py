"""
transit.routing — risk-aware A* routing primitives (dissertation section 3.1.3).

This module is the single source of truth for the routing core:

    f*(n) = g(n) + h(n) + lambda * R(n),   R(n) = -ln(1 - P(overload | x_n))

It holds the graph builder, the admissible heuristic, the risk term and the A*
itself. The numbered experiment script (``04_run_experiment.py``) imports these
functions so there is exactly one implementation; the CLI and the dashboard use
the higher-level :class:`TransportGraph` wrapper below.
"""

from __future__ import annotations

import json
from heapq import heappush, heappop
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from geopy.distance import geodesic

from transit.config import GRAPH_D_MAX_KM, HARD_EXCLUSION_THRESHOLD


# --------------------------------------------------------------------------- #
# Graph construction
# --------------------------------------------------------------------------- #

def build_graph(stops: List[dict], d_max_km: float = GRAPH_D_MAX_KM,
                verbose: bool = True) -> dict:
    """Build a stop graph: an edge connects every pair within ``d_max_km``.

    A latitude pre-filter avoids the full O(n^2) geodesic computation: if two
    stops differ in latitude by more than ``d_max/111`` degrees they cannot be
    within ``d_max`` km, so the expensive geodesic call is skipped.
    """
    if verbose:
        print(f"[routing] Building graph (d_max={d_max_km} km) ...")
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
    lats = np.array([s["lat"] for s in stops])
    lons = np.array([s["lon"] for s in stops])

    # Threshold in degrees: 1° lat ≈ 111 km.
    lat_threshold = d_max_km / 111.0

    iterator = range(n)
    if verbose:
        try:
            from tqdm import tqdm
            iterator = tqdm(iterator, desc="     edges")
        except ImportError:  # pragma: no cover
            pass

    for i in iterator:
        si_id = stops[i]["id"]
        nearby_idx = np.where(np.abs(lats[i + 1:] - lats[i]) < lat_threshold)[0] + (i + 1)
        for j in nearby_idx:
            d = geodesic((lats[i], lons[i]), (lats[j], lons[j])).kilometers
            if d <= d_max_km:
                sj_id = stops[j]["id"]
                graph[si_id]["neighbors"][sj_id] = d
                graph[sj_id]["neighbors"][si_id] = d
                edges_count += 1

    if verbose:
        avg_neighbors = np.mean([len(g["neighbors"]) for g in graph.values()])
        isolated = sum(1 for g in graph.values() if len(g["neighbors"]) == 0)
        print(f"     Edges: {edges_count:,}")
        print(f"     Average neighbours: {avg_neighbors:.1f}")
        print(f"     Isolated stops: {isolated}")
    return graph


# --------------------------------------------------------------------------- #
# Risk-aware A*
# --------------------------------------------------------------------------- #

def heuristic_km(graph: dict, n1: str, n2: str) -> float:
    """Geodesic distance between two vertices, in km. Admissible heuristic."""
    return geodesic(
        (graph[n1]["lat"], graph[n1]["lon"]),
        (graph[n2]["lat"], graph[n2]["lon"]),
    ).kilometers


def risk_component(p: float, eps: float = 1e-6) -> float:
    """R(n) = -ln(1 - P(overload)). eps guards against log(0)."""
    return -np.log(max(1 - p, eps))


def astar(
    graph: dict,
    start: str,
    goal: str,
    lambda_risk: float = 0.0,
    hard_threshold: float = HARD_EXCLUSION_THRESHOLD,
) -> Optional[List[str]]:
    """Risk-aware A*.

    Minimises C(n) = g(n) + lambda * sum_R(n) rather than plain distance g(n).
    The heuristic h(n) is the geodesic distance (admissible); relaxation is by C.
    Returns the list of vertex ids, or ``None`` if the goal is unreachable.
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


def path_metrics(path: Optional[List[str]], graph: dict) -> dict:
    """Length (km), mean P(overload), total risk and stop count for a path."""
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


# --------------------------------------------------------------------------- #
# OD-pair sampling (used by the experiment)
# --------------------------------------------------------------------------- #

def sample_od_pairs(
    stops: List[dict],
    graph: dict,
    n_pairs: int,
    seed: int,
    min_distance_km: float = 2.0,
) -> List[Tuple[str, str]]:
    """Randomly sample origin-destination pairs at least ``min_distance_km`` apart.

    The minimum distance guarantees the pair is non-adjacent so A* has a real
    choice to make rather than returning a one-hop route.
    """
    rng = np.random.default_rng(seed)
    eligible = [s for s in stops if len(graph[s["id"]]["neighbors"]) > 0]
    if len(eligible) < 2:
        raise RuntimeError("Too few non-isolated stops.")

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

    print(f"[routing] Sampled {len(pairs)} OD pairs "
          f"(min distance {min_distance_km} km, {attempts} attempts)")
    return list(pairs)


# --------------------------------------------------------------------------- #
# Loading enriched stops
# --------------------------------------------------------------------------- #

def load_enriched_stops(path) -> Tuple[List[dict], dict]:
    """Load ``stops_enriched.json`` -> (stops, full payload)."""
    with open(Path(path), "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["stops"], data


# --------------------------------------------------------------------------- #
# High-level wrapper used by the CLI and dashboard
# --------------------------------------------------------------------------- #

class TransportGraph:
    """A loaded, queryable transport network.

    Wraps the plain ``graph`` dict with conveniences the CLI and dashboard need:
    name <-> id lookup, a one-call :meth:`route`, and tabular export for maps.
    """

    def __init__(self, graph: dict, stops: List[dict], meta: Optional[dict] = None):
        self.graph = graph
        self.stops = stops
        self.meta = meta or {}
        self._by_id = {s["id"]: s for s in stops}

    # -- construction ---------------------------------------------------- #

    @classmethod
    def from_enriched(cls, path, d_max_km: float = GRAPH_D_MAX_KM,
                      verbose: bool = False) -> "TransportGraph":
        """Build the graph from a ``stops_enriched.json`` file."""
        stops, payload = load_enriched_stops(path)
        graph = build_graph(stops, d_max_km=d_max_km, verbose=verbose)
        return cls(graph, stops, meta=payload)

    # -- lookups --------------------------------------------------------- #

    def has_stop(self, stop_id: str) -> bool:
        return stop_id in self.graph

    def stop(self, stop_id: str) -> dict:
        return self._by_id[stop_id]

    def named_stops(self, limit: Optional[int] = None,
                    only_connected: bool = True) -> List[dict]:
        """Stops that carry a non-empty name, optionally only connected ones.

        Returned newest-first by neighbour count so the dashboard's pickers
        default to well-connected, routable stops.
        """
        items = [
            s for s in self.stops
            if s.get("name")
            and (not only_connected or len(self.graph[s["id"]]["neighbors"]) > 0)
        ]
        items.sort(key=lambda s: len(self.graph[s["id"]]["neighbors"]), reverse=True)
        return items[:limit] if limit else items

    # -- routing --------------------------------------------------------- #

    def route(self, origin: str, dest: str, lambda_risk: float = 0.0) -> dict:
        """Plan a route and return path + metrics in one call."""
        path = astar(self.graph, origin, dest, lambda_risk=lambda_risk)
        metrics = path_metrics(path, self.graph)
        return {
            "origin": origin,
            "dest": dest,
            "lambda": lambda_risk,
            "reachable": path is not None,
            "path": path or [],
            **metrics,
        }

    def path_coords(self, path: Sequence[str]) -> List[List[float]]:
        """[[lon, lat], ...] for a path — the order pydeck PathLayer expects."""
        return [[self.graph[v]["lon"], self.graph[v]["lat"]] for v in path]

    # -- export ---------------------------------------------------------- #

    def stops_records(self) -> List[Dict[str, float]]:
        """Flat per-stop records (id, name, lat, lon, p_overload) for mapping."""
        return [
            {
                "id": s["id"],
                "name": s.get("name", ""),
                "lat": s["lat"],
                "lon": s["lon"],
                "p_overload": s["p_overload"],
                "cluster": s.get("cluster"),
            }
            for s in self.stops
        ]
