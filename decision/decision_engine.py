"""
decision_engine.py
==================

Adaptive decision-making engine (dissertation section 2.4).

The Bayesian Belief Network (section 2.2) outputs a posterior overload
probability ``P(overload | context)`` for a stop / route segment. This module
maps that probability onto one of six management actions and provides the
feedback mechanism that keeps the system adaptive.

Pipeline position
-----------------
    BBN  --P(overload)-->  DecisionEngine.decide()  -->  control action
                                   ^                              |
                                   |        observed outcome      |
                          IncrementalCPDUpdater  <----------------+

Action set (section 2.4.2)
--------------------------
    a0  do nothing (maintain current operation)        cost 0.0
    a1  increase service frequency (+1..3 trips)        needs free units
    a2  reallocate vehicles from quieter routes         needs spare route capacity
    a3  activate reserve vehicles (~15-20 min lead)     needs reserve fleet
    a4  reroute via the risk-aware A* module            most expensive
    a5  inform passengers (push / e-paper signage)      cheap, preventive

Selection rules (section 2.4.3) are threshold-based on ``P(overload)`` and form
a cascade so that the strength of the response is proportional to the threat.

The thresholds and the action catalogue are configurable; the defaults
reproduce the values used in the dissertation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

# --------------------------------------------------------------------------- #
# Action catalogue
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Action:
    """A single management action with its operational metadata."""

    code: str               # canonical id, e.g. "a1"
    name: str               # short human-readable name
    description: str        # what the action does operationally
    cost: float             # relative implementation cost in [0, 1]
    needs_free_units: bool = False   # requires spare vehicles to be available


# Catalogue keyed by code. Costs are relative (a0 = free, a4 = most expensive),
# matching the priority ordering described in section 2.4.2.
ACTIONS: Dict[str, Action] = {
    "a0": Action("a0", "do nothing",
                 "Maintain the current operating mode.", 0.0),
    "a1": Action("a1", "increase frequency",
                 "Add 1-3 extra trips on the congested route.", 0.4,
                 needs_free_units=True),
    "a2": Action("a2", "reallocate vehicles",
                 "Move vehicles from less-loaded routes.", 0.5,
                 needs_free_units=True),
    "a3": Action("a3", "activate reserve",
                 "Bring reserve vehicles into service (~15-20 min lead).", 0.7,
                 needs_free_units=True),
    "a4": Action("a4", "reroute (A*)",
                 "Temporarily modify the route graph via the A* module.", 1.0),
    "a5": Action("a5", "inform passengers",
                 "Push notifications and electronic-signage updates.", 0.1),
}


# --------------------------------------------------------------------------- #
# Decision result
# --------------------------------------------------------------------------- #


@dataclass
class Decision:
    """The outcome of a single decision step."""

    p_overload: float
    primary: Action                       # main action to execute now
    supporting: List[Action] = field(default_factory=list)  # run in parallel
    reserve: Optional[Action] = None      # fallback if no improvement in time
    escalate_after_min: Optional[int] = None  # when to escalate to the reserve
    band: str = ""                        # threshold band label
    rationale: str = ""                   # human-readable explanation

    @property
    def action_codes(self) -> List[str]:
        """All action codes triggered by this decision, primary first."""
        codes = [self.primary.code] + [a.code for a in self.supporting]
        return codes

    @property
    def total_cost(self) -> float:
        """Combined relative cost of every action executed now."""
        return round(self.primary.cost + sum(a.cost for a in self.supporting), 4)

    def __str__(self) -> str:
        sup = f" + {', '.join(a.code for a in self.supporting)}" if self.supporting else ""
        return (f"P(overload)={self.p_overload:.3f} [{self.band}] "
                f"-> {self.primary.code}{sup} ({self.rationale})")


# --------------------------------------------------------------------------- #
# Decision engine
# --------------------------------------------------------------------------- #

# Default threshold edges (section 2.4.3). Five bands -> [0, .3, .5, .7, .85, 1].
DEFAULT_THRESHOLDS: Sequence[float] = (0.3, 0.5, 0.7, 0.85)


class DecisionEngine:
    """
    Threshold-based controller mapping P(overload) -> control action.

    Parameters
    ----------
    thresholds:
        Four ascending edges that split [0, 1] into five bands. Must be strictly
        increasing and lie in (0, 1).
    reroute_fn:
        Optional callback implementing action a4. Called as
        ``reroute_fn(context)`` when the highest band is reached; its return
        value is attached to the rationale. In the full system this is wired to
        the risk-aware A* planner (``04_run_experiment.astar``).
    """

    def __init__(
        self,
        thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
        reroute_fn: Optional[Callable[[dict], object]] = None,
    ) -> None:
        self.thresholds = self._validate_thresholds(thresholds)
        self.reroute_fn = reroute_fn

    @staticmethod
    def _validate_thresholds(thresholds: Sequence[float]) -> List[float]:
        t = list(thresholds)
        if len(t) != 4:
            raise ValueError("Expected exactly 4 threshold edges (5 bands).")
        if not all(0.0 < x < 1.0 for x in t):
            raise ValueError("Thresholds must lie strictly in (0, 1).")
        if any(t[i] >= t[i + 1] for i in range(len(t) - 1)):
            raise ValueError("Thresholds must be strictly increasing.")
        return t

    # -- core API -------------------------------------------------------- #

    def decide(
        self,
        p_overload: float,
        free_units_available: bool = True,
        context: Optional[dict] = None,
    ) -> Decision:
        """
        Choose a control action for the given overload probability.

        Parameters
        ----------
        p_overload:
            Posterior probability of overload from the BBN, in [0, 1].
        free_units_available:
            Whether spare vehicles exist. Governs the a1/a2 vs a3 choice in the
            mid/high bands, per section 2.4.3.
        context:
            Optional evidence vector passed to ``reroute_fn`` for action a4.
        """
        if not 0.0 <= p_overload <= 1.0:
            raise ValueError(f"p_overload must be in [0, 1], got {p_overload}")

        t0, t1, t2, t3 = self.thresholds

        if p_overload < t0:
            return self._band_low(p_overload, t0)
        if p_overload < t1:
            return self._band_elevated(p_overload, t0, t1)
        if p_overload < t2:
            return self._band_high(p_overload, t1, t2, free_units_available)
        if p_overload < t3:
            return self._band_critical(p_overload, t2, t3)
        return self._band_emergency(p_overload, t3, context)

    # -- per-band handlers ---------------------------------------------- #

    def _band_low(self, p: float, t0: float) -> Decision:
        # P < t0: routine operation, keep preventive info as a fallback.
        return Decision(
            p_overload=p, primary=ACTIONS["a0"], reserve=ACTIONS["a5"],
            band=f"P < {t0}",
            rationale="low risk: maintain operation, a5 held in reserve",
        )

    def _band_elevated(self, p: float, t0: float, t1: float) -> Decision:
        # t0 <= P < t1: inform passengers, escalate to a1 if no improvement.
        return Decision(
            p_overload=p, primary=ACTIONS["a5"], reserve=ACTIONS["a1"],
            escalate_after_min=15, band=f"{t0} <= P < {t1}",
            rationale="elevated risk: inform passengers, escalate to a1 in 15 min",
        )

    def _band_high(self, p: float, t1: float, t2: float,
                   free_units: bool) -> Decision:
        # t1 <= P < t2: add capacity. a1 if free units, otherwise reallocate.
        primary = ACTIONS["a1"] if free_units else ACTIONS["a2"]
        reason = ("increase frequency (free units available)" if free_units
                  else "reallocate vehicles (no free units)")
        return Decision(
            p_overload=p, primary=primary, band=f"{t1} <= P < {t2}",
            rationale=f"high risk: {reason}",
        )

    def _band_critical(self, p: float, t2: float, t3: float) -> Decision:
        # t2 <= P < t3: activate reserve and inform passengers in parallel.
        return Decision(
            p_overload=p, primary=ACTIONS["a3"], supporting=[ACTIONS["a5"]],
            band=f"{t2} <= P < {t3}",
            rationale="critical risk: activate reserve fleet + inform passengers",
        )

    def _band_emergency(self, p: float, t3: float,
                        context: Optional[dict]) -> Decision:
        # P >= t3: reroute via A* and throw all supporting resources at it.
        rationale = "emergency risk: reroute via A* + increase frequency + inform"
        if self.reroute_fn is not None:
            result = self.reroute_fn(context or {})
            rationale += f" (reroute result: {result})"
        return Decision(
            p_overload=p, primary=ACTIONS["a4"],
            supporting=[ACTIONS["a1"], ACTIONS["a5"]],
            band=f"P >= {t3}", rationale=rationale,
        )

    # -- batch helper ---------------------------------------------------- #

    def decide_batch(
        self,
        probabilities: Sequence[float],
        free_units_available: bool = True,
    ) -> List[Decision]:
        """Vectorised convenience wrapper over :meth:`decide`."""
        return [self.decide(p, free_units_available) for p in probabilities]


# --------------------------------------------------------------------------- #
# Feedback: incremental Bayesian CPD update (section 2.4.4)
# --------------------------------------------------------------------------- #


class IncrementalCPDUpdater:
    """
    Forgetting-factor update of a conditional probability vector.

    Implements ``CPD_new = lambda * CPD_old + (1 - lambda) * CPD_observed``
    (section 2.4.4), where ``lambda`` is the forgetting factor:

        * lambda = 0.85 -> faster adaptation, less stable
        * lambda = 0.95 -> slower adaptation, more stable
        * lambda = 0.90 -> tuned optimum used in the dissertation

    A smaller lambda puts more weight on the newest observation, so the system
    reacts faster to regime changes (e.g. seasonal demand shifts).
    """

    def __init__(self, forgetting_factor: float = 0.90) -> None:
        if not 0.0 < forgetting_factor < 1.0:
            raise ValueError("forgetting_factor (lambda) must lie in (0, 1).")
        self.lam = forgetting_factor

    def update(
        self,
        cpd_old: Sequence[float],
        cpd_observed: Sequence[float],
    ) -> List[float]:
        """
        Blend a prior CPD with a freshly observed one and renormalise.

        Both inputs must be probability vectors of equal length. The result is
        renormalised to sum to 1 to guard against floating-point drift.
        """
        if len(cpd_old) != len(cpd_observed):
            raise ValueError("CPD vectors must have the same length.")
        if not cpd_old:
            raise ValueError("CPD vectors must be non-empty.")

        blended = [
            self.lam * o + (1.0 - self.lam) * n
            for o, n in zip(cpd_old, cpd_observed)
        ]
        total = sum(blended)
        if total <= 0.0:
            raise ValueError("Blended CPD sums to zero; check the inputs.")
        return [x / total for x in blended]


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #


def _demo() -> None:
    """Print the decision for a sweep of overload probabilities."""
    engine = DecisionEngine()
    print("Adaptive decision engine — threshold sweep")
    print("-" * 64)
    cases = [
        (0.10, True), (0.35, True),
        (0.55, True),   # high band, free units -> a1
        (0.55, False),  # high band, no free units -> a2
        (0.78, True), (0.92, True),
    ]
    for p, free in cases:
        d = engine.decide(p, free_units_available=free)
        print(f"  free_units={str(free):5s}  {d}")
    print("-" * 64)

    updater = IncrementalCPDUpdater(forgetting_factor=0.90)
    prior = [0.8, 0.2]
    observed = [0.4, 0.6]
    print(f"CPD update (lambda=0.90): {prior} <- {observed} = "
          f"{[round(x, 3) for x in updater.update(prior, observed)]}")


if __name__ == "__main__":
    _demo()
