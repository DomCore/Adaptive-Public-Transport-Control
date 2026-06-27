"""
Tests for the public ``transit`` package API (the library face of the tool).

These exercise the parts that do not need the input datasets: the package
re-exports, the config shim, and the high-level TransportGraph wrapper around a
hand-built graph. The BBN wrapper is covered indirectly (it shares its inference
primitive with 03_enrich_stops.py) and is exercised live by the CLI smoke run.
"""

import importlib

import pytest

import transit
from transit import DecisionEngine, IncrementalCPDUpdater
from transit.routing import TransportGraph, risk_component, astar


def test_package_exposes_version_and_decision():
    assert isinstance(transit.__version__, str)
    # Decision engine is the same object whether imported from transit or decision.
    from decision import DecisionEngine as DE
    assert DecisionEngine is DE


def test_config_shim_matches_canonical():
    root_cfg = importlib.import_module("config")
    pkg_cfg = importlib.import_module("transit.config")
    assert root_cfg.BBN_NODES == pkg_cfg.BBN_NODES
    assert root_cfg.LAMBDA_VALUES == pkg_cfg.LAMBDA_VALUES
    # The shim and the canonical module must point at the same repo root.
    assert root_cfg.BASE_DIR == pkg_cfg.BASE_DIR


def _toy_transport_graph():
    """S->A->G (short, risky A) vs S->B->G (detour, safe B).

    The edge set is built explicitly (only S-A, A-G, S-B, B-G) so the planner
    has a genuine choice — mirroring tests/test_astar.py.
    """
    from geopy.distance import geodesic

    coords = {
        "S": (50.00, 14.00), "A": (50.00, 14.01),
        "B": (50.01, 14.01), "G": (50.00, 14.02),
    }
    p = {"S": 0.0, "A": 0.9, "B": 0.0, "G": 0.0}
    stops = [
        {"id": k, "lat": lat, "lon": lon, "p_overload": p[k], "name": k}
        for k, (lat, lon) in coords.items()
    ]
    graph = {
        k: {"lat": lat, "lon": lon, "p_overload": p[k], "neighbors": {}}
        for k, (lat, lon) in coords.items()
    }
    for u, v in [("S", "A"), ("A", "G"), ("S", "B"), ("B", "G")]:
        d = geodesic(coords[u], coords[v]).kilometers
        graph[u]["neighbors"][v] = d
        graph[v]["neighbors"][u] = d
    return TransportGraph(graph, stops)


def test_transport_graph_route_respects_lambda():
    tg = _toy_transport_graph()
    cheap = tg.route("S", "G", lambda_risk=0.0)
    safe = tg.route("S", "G", lambda_risk=10.0)
    assert cheap["reachable"] and safe["reachable"]
    assert cheap["path"] == ["S", "A", "G"]      # shortest, through risky A
    assert "A" not in safe["path"]                # high lambda avoids A
    # The risk-aware route trades length for a lower mean overload.
    assert safe["r_overload"] <= cheap["r_overload"]


def test_transport_graph_named_and_records():
    tg = _toy_transport_graph()
    assert {s["id"] for s in tg.named_stops()} == {"S", "A", "B", "G"}
    rec = tg.stops_records()
    assert all({"id", "lat", "lon", "p_overload"} <= set(r) for r in rec)


def test_risk_component_reexported_is_additive_log():
    import math
    assert risk_component(0.0) == pytest.approx(0.0)
    assert 2 * risk_component(0.5) == pytest.approx(-math.log(0.25))


def test_incremental_cpd_updater_from_transit():
    out = IncrementalCPDUpdater(0.9).update([0.8, 0.2], [0.4, 0.6])
    assert out == pytest.approx([0.76, 0.24])
