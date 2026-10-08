# diagnostic/v1_2/test_metrics.py
# -*- coding: utf-8 -*-
import json
import os

import pytest

import metrics as M

E7 = os.path.join(M.DIAG, "E7_RESULT.txt")
E8 = os.path.join(M.DIAG, "E8_RESULT.json")


def only(rows, method):
    return [r for r in rows if r["method"] == method]


def test_e7_totals_equal_the_frozen_summary():
    rows = M.e7_rows(E7)
    want = {"full": (2.5, 2, 0, 0.5, 2, 5.5), "A1": (0, 0, 0, 0, 2, 8),
            "A2": (9.5, 4, 3.5, 2, 0, 2), "A3": (2.5, 2, 0, 0.5, 2, 5.5)}
    for method, expected in want.items():
        c = M.tally(only(rows, method))
        got = (c["recommended"], c["cleared"], c["false_positive"], c["not_tested"],
               c["abstain_correct"], c["abstain_miss"])
        assert got == pytest.approx(expected), method


def test_e8_totals_equal_the_frozen_summary():
    data = json.load(open(E8, encoding="utf-8"))
    rows = M.e8_rows(E8)
    for method, summary in data["summary"].items():
        c = M.tally(only(rows, "replay" if method == "reference" else method))
        for key in ("recommended", "cleared", "false_positive", "not_tested",
                    "abstain_correct", "abstain_miss", "missing"):
            assert c[key] == pytest.approx(summary[key]), (method, key)


def test_e6_cleared_counts_match_table_x():
    want = {"as_scored": {"full": 5, "A1": 5, "A2": 7, "A3": 5},
            "extractor_as_frozen": {"full": 3, "A1": 3, "A2": 0, "A3": 5}}
    for reading, by_method in want.items():
        rows = M.e6_rows(reading)
        for method, n in by_method.items():
            assert M.tally(only(rows, method))["cleared"] == pytest.approx(n), (reading, method)


def test_rates_on_the_second_set():
    rows = M.e7_rows(E7)
    a2 = M.rates(M.tally(only(rows, "A2")))
    full = M.rates(M.tally(only(rows, "full")))
    assert a2["precision"] == pytest.approx(4 / 7.5)
    assert a2["recall"] == pytest.approx(4 / 8)
    assert a2["false_positive_rate"] == pytest.approx(3.5 / 4)
    assert a2["coverage"] == pytest.approx(9.5 / 12)
    assert full["precision"] == pytest.approx(1.0)
    assert full["recall"] == pytest.approx(2 / 8)
    assert full["false_positive_rate"] == pytest.approx(0.0)


def test_unknown_label_is_an_error():
    row = {"case": "x", "alloc": "1", "method": "full", "verdict": "none",
           "delta": None, "label": "abstain-mis"}
    with pytest.raises(ValueError):
        M.tally([row])


def test_bootstrap_is_reproducible_and_brackets_the_estimate():
    rows = only(M.e7_rows(E7), "A2")
    a = M.bootstrap(rows, draws=2000, seed=7)
    b = M.bootstrap(rows, draws=2000, seed=7)
    assert a == b
    lo, hi, dropped = a["precision"]
    assert lo <= 4 / 7.5 <= hi and dropped >= 0


def test_aggregation_rules_bracket_the_half_count():
    rows = only(M.e7_rows(E7), "full")
    strict = M.aggregate(rows, "all")["recommended"]
    loose = M.aggregate(rows, "any")["recommended"]
    assert strict <= 2.5 <= loose
    assert (strict, loose) == (2, 3)


def test_a_rate_of_zero_or_one_gets_its_count_and_no_interval():
    # A percentile bootstrap cannot return a value it has not drawn, so at 0
    # or 1 it prints a zero-width "interval" that looks like certainty. Such
    # a rate carries its numerator and denominator and no interval.
    out = M.build(draws=300)
    full = out["E7"]["full"]
    assert full["rates"]["precision"] == pytest.approx(1.0)
    assert full["intervals"]["precision"] is None
    assert list(full["support"]["precision"]) == pytest.approx([2.0, 2.0])
    assert full["intervals"]["false_positive_rate"] is None
    assert list(full["support"]["false_positive_rate"]) == pytest.approx([0.0, 4.0])
    lo, hi, dropped = out["E7"]["A2"]["intervals"]["precision"]
    assert lo < 4 / 7.5 < hi
    replay = out["E8"]["replay"]
    assert replay["intervals"]["recall"] is None
    assert list(replay["support"]["recall"]) == pytest.approx([7.5, 7.5])


def test_the_text_table_prints_the_count_where_no_interval_is_defined():
    text = M.render(M.build(draws=300))
    assert "1.00 (2 of 2)" in text and "1.00 (7.5 of 7.5)" in text
    assert "[1.00, 1.00]" not in text and "[0.00, 0.00]" not in text
