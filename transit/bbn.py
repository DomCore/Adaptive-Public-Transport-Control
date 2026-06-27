"""
transit.bbn — Bayesian Belief Network: build, persist, query (section 2.2).

Single source of truth for the BBN. The numbered builder script
(``02_build_bbn_pgmpy.py``) imports :func:`build_model` and
:func:`repair_unobserved_cpds`; the enrichment script
(``03_enrich_stops.py``) imports :func:`query_overload`. The CLI and the
dashboard use the high-level :class:`BBNModel` wrapper, which loads the fitted
model and answers ``P(overload | context)`` with exact inference (Variable
Elimination).
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# Compatibility: ``BayesianNetwork`` is the name in pgmpy <= 0.1.25; newer
# versions (>= 0.1.26) renamed the class to ``DiscreteBayesianNetwork``.
try:
    from pgmpy.models import BayesianNetwork
except ImportError:  # pragma: no cover - depends on pgmpy version
    from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork

from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

from transit.config import BBN_NODES, BBN_EDGES


# --------------------------------------------------------------------------- #
# Building / fitting
# --------------------------------------------------------------------------- #

def build_model(df, verbose: bool = True) -> BayesianNetwork:
    """Build the BBN with the structure from config and fit CPDs on ``df``."""
    if verbose:
        print("[bbn] Creating BayesianNetwork from structure ...")
        print(f"     Nodes:  {BBN_NODES}")
        print(f"     Edges:  {BBN_EDGES}")

    missing = [n for n in BBN_NODES if n not in df.columns]
    if missing:
        raise ValueError(f"Nodes missing from data: {missing}")

    model = BayesianNetwork(BBN_EDGES)
    if verbose:
        print("[bbn] Fitting CPDs by Maximum Likelihood ...")
    model.fit(df[BBN_NODES], estimator=MaximumLikelihoodEstimator)
    repair_unobserved_cpds(model, verbose=verbose)
    return model


def repair_unobserved_cpds(model: BayesianNetwork, verbose: bool = True) -> None:
    """Replace NaN CPD columns with a uniform distribution.

    For multi-parent nodes (``How_Long_Delayed`` <- Reason, Boro;
    ``Has_Contractor_Notified_Schools`` <- overload, School_Age_or_PreK) some
    parent-value combinations may not occur in the data. MaximumLikelihood
    leaves NaN (0/0) for those; we assign them a uniform (non-informative) prior
    so ``check_model()`` passes.

    The core nodes (``overload``, ``Reason``) have no unobserved parent
    combinations, so their CPDs are untouched — routing results are preserved.
    """
    repaired = []
    for cpd in model.get_cpds():
        values_2d = cpd.get_values()              # shape (var_card, n_parent_configs)
        col_sums = values_2d.sum(axis=0)
        bad = np.isnan(col_sums) | (col_sums == 0)
        if bad.any():
            values_2d[:, bad] = 1.0 / values_2d.shape[0]
            cpd.values = values_2d.reshape(cpd.cardinality)
            repaired.append((cpd.variable, int(bad.sum())))
    if repaired and verbose:
        print("[bbn] Repaired unobserved CPD combinations (uniform prior):")
        for var, n in repaired:
            print(f"       {var}: {n} combinations")


# --------------------------------------------------------------------------- #
# Inference primitive (shared by 03_enrich_stops.py and BBNModel)
# --------------------------------------------------------------------------- #

def query_overload(inference: VariableElimination, evidence: dict) -> float:
    """P(overload = 1 | evidence) via Variable Elimination.

    Returns 0.5 (maximum entropy) if the query fails, e.g. because the evidence
    references a state the model never saw.
    """
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


# --------------------------------------------------------------------------- #
# High-level wrapper used by the CLI and dashboard
# --------------------------------------------------------------------------- #

class BBNModel:
    """A fitted BBN ready for ``P(overload | context)`` queries."""

    def __init__(self, model: BayesianNetwork):
        self.model = model
        self.inference = VariableElimination(model)

    # -- construction ---------------------------------------------------- #

    @classmethod
    def load(cls, path) -> "BBNModel":
        """Load a pickled pgmpy model produced by ``02_build_bbn_pgmpy.py``."""
        with open(Path(path), "rb") as f:
            model = pickle.load(f)
        return cls(model)

    @classmethod
    def fit(cls, df, verbose: bool = True) -> "BBNModel":
        """Build and fit a fresh model from a prepared dataframe."""
        return cls(build_model(df, verbose=verbose))

    def save(self, path) -> None:
        with open(Path(path), "wb") as f:
            pickle.dump(self.model, f)

    # -- introspection --------------------------------------------------- #

    @property
    def nodes(self) -> List[str]:
        return list(self.model.nodes())

    def states(self, node: str) -> List[str]:
        """Valid state labels for a node, as strings."""
        cpd = self.model.get_cpds(node)
        return [str(s) for s in cpd.state_names[node]]

    def context_options(
        self,
        nodes: Optional[List[str]] = None,
    ) -> Dict[str, List[str]]:
        """{node: [states]} for the evidence nodes a user can pick.

        Defaults to the nodes that actually shape the overload posterior:
        TimeOfDay and Reason (the parents of ``overload``) plus DayOfWeek
        (a parent of Reason).
        """
        nodes = nodes or ["TimeOfDay", "DayOfWeek", "Reason"]
        return {n: self.states(n) for n in nodes if n in self.model.nodes()}

    # -- queries --------------------------------------------------------- #

    def predict_overload(self, context: dict) -> float:
        """P(overload = 1 | context). The headline signal of the whole system."""
        # Keep only evidence the model actually knows about.
        evidence = {k: v for k, v in context.items() if k in self.model.nodes()}
        return query_overload(self.inference, evidence)

    def reason_given(self, context: dict) -> Dict[str, float]:
        """Posterior distribution over Reason given partial context.

        Handy for the dashboard: shows *why* overload is high by surfacing the
        most likely failure cause for the chosen time/day.
        """
        evidence = {
            k: v for k, v in context.items()
            if k in self.model.nodes() and k != "Reason"
        }
        try:
            result = self.inference.query(
                variables=["Reason"], evidence=evidence, show_progress=False,
            )
            return {
                str(s): float(p)
                for s, p in zip(result.state_names["Reason"], result.values)
            }
        except Exception:
            return {}
