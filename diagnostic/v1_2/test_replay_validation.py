# diagnostic/v1_2/test_replay_validation.py
# -*- coding: utf-8 -*-
import json
import os

import pytest

import replay_validation as V

E8 = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "E8_RESULT.json")


def test_rows_cover_every_scored_replay_prediction():
    rows = V.rows(E8)
    assert len(rows) == 15
    assert all(0.0 <= r["error_points"] <= 100.0 for r in rows)


def test_median_error_equals_the_frozen_summary():
    frozen = json.load(open(E8, encoding="utf-8"))["summary"]["reference"]["median_magnitude_error"]
    s = V.summary(V.rows(E8))
    assert s["median_error_points"] == pytest.approx(100.0 * frozen)
    assert s["n"] == 15 and 0 <= s["within_5_points"] <= 15
