"""
Behavioural tests for the risk-aware A* (dissertation section 2.1.3).

The key property: with lambda = 0 the planner minimises distance only, while a
large lambda makes it prefer a longer but lower-overload-probability route.
"""

import pytest

from conftest import load_script

astar_mod = load_script("04_run_experiment.py")


def _toy_graph():
    """
    Two routes from S to G:
      * S-A-G : short, but A has a high overload probability (0.9)
      * S-B-G : a northern detour, but B is safe (0.0)
    """
    coords = {
        "S": (50.00, 14.00),
        "A": (50.00, 14.01),   # on the direct S->G line, risky
        "B": (50.01, 14.01),   # detour to the north, safe
        "G": (50.00, 14.02),
    }
    p = {"S": 0.0, "A": 0.9, "B": 0.0, "G": 0.0}

    def dist(u, v):
        return astar_mod.geodesic(coords[u], coords[v]).kilometers

    graph = {
        node: {"lat": lat, "lon": lon, "p_overload": p[node], "neighbors": {}}
        for node, (lat, lon) in coords.items()
    }
    for u, v in [("S", "A"), ("A", "G"), ("S", "B"), ("B", "G")]:
        d = dist(u, v)
        graph[u]["neighbors"][v] = d
        graph[v]["neighbors"][u] = d
    return graph


def test_lambda_zero_takes_short_risky_route():
    graph = _toy_graph()
    path = astar_mod.astar(graph, "S", "G", lambda_risk=0.0)
    assert path == ["S", "A", "G"]


def test_high_lambda_avoids_risky_vertex():
    graph = _toy_graph()
    path = astar_mod.astar(graph, "S", "G", lambda_risk=10.0)
    assert "A" not in path
    assert path == ["S", "B", "G"]


def test_trivial_and_unreachable_cases():
    graph = _toy_graph()
    assert astar_mod.astar(graph, "S", "S") == ["S"]
    assert astar_mod.astar(graph, "S", "missing") is None


def test_risk_component_is_additive_log():
    # R(n) = -ln(1 - p); two independent 0.5 risks combine to -ln(0.25).
    import math
    r = astar_mod.risk_component
    assert r(0.0) == pytest.approx(0.0)
    assert 2 * r(0.5) == pytest.approx(-math.log(0.25))


def test_path_metrics_reports_mean_overload():
    graph = _toy_graph()
    m = astar_mod.path_metrics(["S", "A", "G"], graph)
    assert m["n_stops"] == 3
    assert m["r_overload"] == pytest.approx((0.0 + 0.9 + 0.0) / 3)
    assert m["length_km"] > 0
