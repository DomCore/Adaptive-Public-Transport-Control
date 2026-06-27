"""
transit — adaptive public-transport control toolkit
===================================================

A small, installable library that turns the dissertation's pipeline from a set
of numbered scripts into a coherent instrument with three faces:

  * a Python API           (``import transit``)
  * a command-line tool     (``transit-control`` — see :mod:`transit.cli`)
  * an interactive dashboard (``transit-control serve`` — see :mod:`transit.app`)

Public API
----------
    from transit import (
        BBNModel,              # load the BBN, query P(overload | context)
        TransportGraph,        # build the routing graph, run risk-aware A*
        DecisionEngine,        # map P(overload) -> control action a0..a5
        IncrementalCPDUpdater, # forgetting-factor feedback
    )

The heavyweight scientific dependencies (pgmpy, scikit-learn) are imported
lazily, so ``import transit`` stays cheap until a model is actually loaded.
"""

from __future__ import annotations

__version__ = "0.2.0"

# Cheap to import (pure-Python) — expose eagerly.
from .decision import (  # noqa: F401
    Action,
    ACTIONS,
    Decision,
    DecisionEngine,
    IncrementalCPDUpdater,
)

__all__ = [
    "__version__",
    "Action",
    "ACTIONS",
    "Decision",
    "DecisionEngine",
    "IncrementalCPDUpdater",
    "BBNModel",
    "TransportGraph",
    "astar",
    "risk_component",
]


def __getattr__(name: str):
    """Lazily expose the heavier symbols (PEP 562).

    Keeps ``import transit`` fast: pgmpy / numpy / geopy are only pulled in when
    the BBN or the routing graph is first accessed.
    """
    if name in ("BBNModel", "load_bbn"):
        from . import bbn
        return getattr(bbn, name)
    if name in ("TransportGraph", "astar", "risk_component", "path_metrics",
                "build_graph", "load_enriched_stops"):
        from . import routing
        return getattr(routing, name)
    raise AttributeError(f"module 'transit' has no attribute {name!r}")
