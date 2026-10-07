# validation_v2/analysis/tests/test_intervals.py
# -*- coding: utf-8 -*-
import math
import os

import pytest

import intervals as I

V2 = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.path.dirname(V2)
A = os.path.join(V2, "results", "analysis_2026-09-29")


def test_one_and_two_allocations_get_no_interval():
    assert I.summarize([1.5])["ci"] is None
    two = I.summarize([1.5, 1.7])
    assert two["ci"] is None and (two["min"], two["max"]) == (1.5, 1.7)


def test_three_equal_allocations_give_a_zero_width_interval():
    s = I.summarize([2.0, 2.0, 2.0])
    assert s["center"] == pytest.approx(2.0) and s["ci"] == pytest.approx((2.0, 2.0))


def test_log_interval_matches_a_hand_computation():
    values = [1.0, math.e, math.e ** 2]            # logs 0, 1, 2: mean 1, sd 1
    s = I.summarize(values)
    half = 4.303 / math.sqrt(3)
    assert s["center"] == pytest.approx(math.e)
    assert s["ci"] == pytest.approx((math.exp(1 - half), math.exp(1 + half)))


def test_a_gain_uses_the_linear_scale_and_may_be_negative():
    s = I.summarize([-0.02, 0.0, 0.02], ratio=False)
    assert s["center"] == pytest.approx(0.0)
    with pytest.raises(ValueError):
        I.summarize([-0.02, 0.0, 0.02])


def test_validation_campaign_ranges_match_the_paper():
    gains = I.v2_gains(os.path.join(A, "v2_expA.csv"))
    e = I.summarize(gains[("g_matched", "E")], ratio=False)
    b = I.summarize(gains[("g_matched", "B")], ratio=False)
    assert e["n"] == 3 and (round(e["min"], 2), round(e["max"], 2)) == (0.50, 0.52)
    assert (round(b["min"], 2), round(b["max"], 2)) == (0.53, 0.56)
    hooks = I.v2_hook_ratios(os.path.join(A, "v2_expB.csv"))
    h = I.summarize(hooks[("ddp-int8-hook", 32)])
    assert (round(h["min"], 2), round(h["max"], 2)) == (1.74, 1.84) and h["ci"] is not None


def test_h100_overlap_interval_from_published_moments():
    t = I.ratio_from_moments(7.8589508, 0.17277219, 9.0187104, 0.02817137, 3)
    assert 100 * (t["center"] - 1) == pytest.approx(-12.86, abs=0.01)
    assert t["ci"][1] < 1.0
    r = I.ratio_from_moments(33.807416, 0.18960821, 33.117557, 0.19315716, 3)
    assert 100 * (r["center"] - 1) == pytest.approx(2.08, abs=0.01)


def test_intel_control_exchange_ratio_per_allocation():
    table = I.intel_control(os.path.join(ROOT, "processed", "control_campaign_summary.txt"))
    ratios = I.intel_exchange_ratio(table, "transformer", "gpu-quant", "native", 32)
    assert ratios["A"] == pytest.approx(528.95 / 63.10)
    assert ratios["B"] == pytest.approx(427.50 / 61.52)
