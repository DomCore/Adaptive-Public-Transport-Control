"""
Adaptive decision-making module.

Turns the overload probability produced by the Bayesian Belief Network into a
concrete control action (a0-a5) and closes the loop with an incremental,
forgetting-factor Bayesian update. This package is the executable counterpart
of dissertation section 2.4 ("Method of adaptive decision-making").
"""

from .decision_engine import (
    Action,
    ACTIONS,
    Decision,
    DecisionEngine,
    IncrementalCPDUpdater,
)

__all__ = [
    "Action",
    "ACTIONS",
    "Decision",
    "DecisionEngine",
    "IncrementalCPDUpdater",
]
