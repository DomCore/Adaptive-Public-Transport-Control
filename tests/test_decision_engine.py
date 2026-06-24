"""Tests for the adaptive decision module (dissertation section 2.4)."""

import pytest

from decision import DecisionEngine, IncrementalCPDUpdater, ACTIONS


@pytest.fixture
def engine():
    return DecisionEngine()


# --- band selection (section 2.4.3) ----------------------------------- #


@pytest.mark.parametrize("p, expected", [
    (0.00, "a0"),
    (0.29, "a0"),
    (0.30, "a5"),
    (0.49, "a5"),
    (0.50, "a1"),
    (0.69, "a1"),
    (0.70, "a3"),
    (0.84, "a3"),
    (0.85, "a4"),
    (1.00, "a4"),
])
def test_primary_action_per_band(engine, p, expected):
    assert engine.decide(p).primary.code == expected


def test_high_band_without_free_units_reallocates(engine):
    # 0.5 <= P < 0.7: a1 needs free units; without them fall back to a2.
    assert engine.decide(0.55, free_units_available=True).primary.code == "a1"
    assert engine.decide(0.55, free_units_available=False).primary.code == "a2"


def test_emergency_band_supporting_actions(engine):
    d = engine.decide(0.95)
    assert d.primary.code == "a4"
    assert d.action_codes == ["a4", "a1", "a5"]


def test_critical_band_informs_passengers(engine):
    d = engine.decide(0.80)
    assert d.primary.code == "a3"
    assert "a5" in d.action_codes


def test_reserve_and_escalation_in_elevated_band(engine):
    d = engine.decide(0.40)
    assert d.reserve is ACTIONS["a1"]
    assert d.escalate_after_min == 15


def test_total_cost_monotonic_trend(engine):
    # Emergency response must be at least as expensive as routine operation.
    assert engine.decide(0.95).total_cost > engine.decide(0.1).total_cost


# --- input validation ------------------------------------------------- #


@pytest.mark.parametrize("bad_p", [-0.01, 1.01, 2.0])
def test_decide_rejects_out_of_range(engine, bad_p):
    with pytest.raises(ValueError):
        engine.decide(bad_p)


@pytest.mark.parametrize("bad_thresholds", [
    (0.3, 0.5, 0.7),            # too few
    (0.3, 0.5, 0.5, 0.85),      # not strictly increasing
    (0.0, 0.5, 0.7, 0.85),      # edge at 0
    (0.3, 0.5, 0.7, 1.0),       # edge at 1
])
def test_invalid_thresholds_rejected(bad_thresholds):
    with pytest.raises(ValueError):
        DecisionEngine(thresholds=bad_thresholds)


def test_reroute_callback_invoked():
    calls = []
    engine = DecisionEngine(reroute_fn=lambda ctx: calls.append(ctx) or "rerouted")
    d = engine.decide(0.9, context={"stop": "X"})
    assert calls == [{"stop": "X"}]
    assert "rerouted" in d.rationale


# --- incremental CPD update (section 2.4.4) --------------------------- #


def test_cpd_update_blends_and_normalises():
    updater = IncrementalCPDUpdater(forgetting_factor=0.90)
    out = updater.update([0.8, 0.2], [0.4, 0.6])
    assert out == pytest.approx([0.76, 0.24])
    assert sum(out) == pytest.approx(1.0)


def test_smaller_lambda_reacts_faster():
    prior, observed = [1.0, 0.0], [0.0, 1.0]
    fast = IncrementalCPDUpdater(0.85).update(prior, observed)
    slow = IncrementalCPDUpdater(0.95).update(prior, observed)
    # Lower lambda moves further toward the new observation.
    assert fast[1] > slow[1]


def test_cpd_update_validates_inputs():
    updater = IncrementalCPDUpdater()
    with pytest.raises(ValueError):
        updater.update([0.5, 0.5], [1.0])     # length mismatch
    with pytest.raises(ValueError):
        updater.update([], [])                 # empty


@pytest.mark.parametrize("bad_lambda", [0.0, 1.0, -0.1, 1.5])
def test_forgetting_factor_range(bad_lambda):
    with pytest.raises(ValueError):
        IncrementalCPDUpdater(bad_lambda)
