"""Tests for the preprocessing helpers (01_prepare_data.py)."""

import pytest

from conftest import load_script

prep = load_script("01_prepare_data.py")


@pytest.mark.parametrize("raw, expected", [
    ("15 min", 15.0),
    ("30 min", 30.0),
    ("1 hour", 60.0),
    ("2 hours", 120.0),
    ("1 hour 30 min", 90.0),
    ("1.5 hour", 90.0),
    ("15-30", 22.5),
    ("15 to 30", 22.5),
    ("90", 90.0),
    ("90.0", 90.0),
])
def test_parse_delay_minutes(raw, expected):
    assert prep.parse_delay_minutes(raw) == pytest.approx(expected)


@pytest.mark.parametrize("empty", [None, "", "nan", "no delay", "0 min"])
def test_parse_delay_minutes_empty_is_zero(empty):
    assert prep.parse_delay_minutes(empty) == 0.0


@pytest.mark.parametrize("hour, label", [
    (0, "night"), (5, "night"),
    (6, "peak_morning"), (9, "peak_morning"),
    (10, "midday"), (15, "midday"),
    (16, "peak_evening"), (18, "peak_evening"),
    (19, "evening"), (23, "evening"),
])
def test_assign_time_of_day(hour, label):
    assert prep.assign_time_of_day(hour) == label
