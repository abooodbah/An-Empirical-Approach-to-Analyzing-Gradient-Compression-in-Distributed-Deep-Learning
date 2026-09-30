# -*- coding: utf-8 -*-
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rescore as R  # noqa: E402

ROWS = R.load_rows()


def counts(m, **kw):
    return R.rescore(ROWS, R.METHODS[m], **kw)[0]


def lines(m, **kw):
    return R.rescore(ROWS, R.METHODS[m], **kw)[1]


def test_frozen_reading_reproduces_e6_final():
    c = counts("full")
    assert (c["correct"], c["short"], c["abstain-miss"], c["abstain-correct"]) == (4, 1, 4, 2)
    assert counts("A1")["correct"] == 5
    assert counts("A2")["correct"] == 7
    a3 = counts("A3")
    assert (a3["correct"], a3["short"], a3["null"]) == (4, 1, 1)


def test_hooks_read_insufficient_not_none():
    hooks = [l for l in lines("full", reading="described") if l["case"] in (7, 8)]
    assert hooks and all(l["verdict"] == "insufficient" for l in hooks)
    frozen = [l for l in lines("full") if l["case"] in (7, 8)]
    assert all(l["verdict"] == "none" for l in frozen)


def test_a1_as_described_fires_on_the_24_rank_hook_only():
    ls = lines("A1", reading="described", c5=True)
    case7 = [l for l in ls if l["case"] == 7][0]
    case8 = [l for l in ls if l["case"] == 8][0]
    assert (case7["verdict"], case7["outcome"]) == ("shape", "correct")
    assert case8["verdict"] == "none"          # r_V = 12/8 = 1.5, strict threshold
    assert counts("A1", reading="described", c5=True)["correct"] == 6


def test_nominal_recount_includes_case_12():
    c = counts("A1", reading="nominal_frozen_hooks")
    assert c["correct"] == 3 and R.cleared(c) == 3
    verdicts = {l["case"]: l["verdict"] for l in lines("A1", reading="nominal_frozen_hooks")}
    assert verdicts[3] == "none" and verdicts[12] == "none"
    assert counts("A1", reading="nominal", c5=True)["correct"] == 4


def test_same_encoding_effect_changes_no_outcome():
    frozen = lines("full")
    same = lines("full", e_same=True)
    assert [l["outcome"] for l in frozen] == [l["outcome"] for l in same]
    es = {l["case"]: l["e"] for l in same if l["verdict"] == "shape"}
    assert sorted(es) == [2, 3, 4]
    for case, value in ((2, 0.464), (3, 0.492), (4, 0.501)):
        assert es[case] == pytest.approx(value, abs=6e-4)


def test_c5_labels():
    a3 = lines("A3", reading="described", c5=True)
    assert [l["outcome"] for l in a3 if l["case"] == 9] == ["inapplicable"]
    a2 = lines("A2", reading="described", c5=True)
    assert [l["outcome"] for l in a2 if l["case"] == 11] == ["inapplicable"]


def test_extractor_as_frozen_counterfactual():
    full = counts("full", reading="as_frozen_extractor")
    assert (full["correct"], full["short"], full["abstain-miss"]) == (2, 1, 6)
    assert counts("A1", reading="as_frozen_extractor")["correct"] == 3
    assert counts("A2", reading="as_frozen_extractor").get("correct", 0) == 0
    a3 = counts("A3", reading="as_frozen_extractor")
    assert (a3["correct"], a3["short"], a3["null"]) == (4, 1, 1)


def test_frozen_rows_are_not_mutated():
    before = [dict(r) for r in ROWS]
    for reading in R.READINGS:
        R.rescore(ROWS, R.METHODS["full"], reading=reading, e_same=True, c5=True)
    assert ROWS == before


def test_unknown_reading_rejected():
    with pytest.raises(ValueError):
        R.transform(ROWS[0], "bogus")
