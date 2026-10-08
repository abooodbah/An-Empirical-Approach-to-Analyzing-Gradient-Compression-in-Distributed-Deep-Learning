# -*- coding: utf-8 -*-
import glob
import io
import json
import os
import shutil
import tarfile
import tempfile

import pytest

import e9_pass_spread_posthoc as S

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RESULT = os.path.join(ROOT, "validation_v2", "results", "rev4", "E9_PASS_SPREAD_POSTHOC.json")


def test_the_spread_is_the_difference_of_the_two_passes_over_their_mean():
    assert S.spread(54.0, 46.0) == pytest.approx(0.16)
    assert S.spread(46.0, 54.0) == pytest.approx(0.16)
    assert S.spread(10.0, 10.0) == 0.0


def test_a_condition_with_one_pass_has_no_spread():
    out = S.conditions({"transformer|native": {"p1": 10.0, "p2": 11.0}, "resnet50|native": {"p2": 50.0}})
    assert out["transformer|native"]["spread"] == pytest.approx(1.0 / 10.5)
    assert out["transformer|native"]["passes_ms"] == {"p1": 10.0, "p2": 11.0}
    assert "spread" not in out["resnet50|native"]


def allocation(world, name, spreads):
    conditions = {}
    for i, s in enumerate(spreads):
        low = 100.0 * (1 - s / 2.0)
        high = 100.0 * (1 + s / 2.0)
        conditions["transformer|mode%d" % i] = {"passes_ms": {"p1": low, "p2": high}, "spread": S.spread(low, high)}
    conditions["resnet50|one-pass"] = {"passes_ms": {"p1": 50.0}}
    return {"run_dir": name, "world": world, "conditions": conditions}


def test_the_summary_is_per_scale_over_its_allocations():
    out = S.summary([allocation(12, "a", [0.01, 0.02, 0.03]), allocation(12, "b", [0.04, 0.06]),
                     allocation(24, "c", [0.10, 0.20])])
    twelve = out["12"]
    assert twelve["two_pass_conditions"] == 5 and twelve["above_5_percent"] == 1
    assert twelve["median"] == pytest.approx(0.03)
    assert twelve["largest"]["run_dir"] == "b" and twelve["largest"]["condition"] == "transformer|mode1"
    assert twelve["largest"]["spread"] == pytest.approx(0.06)
    assert out["24"]["above_5_percent"] == 2 and out["24"]["largest"]["spread"] == pytest.approx(0.20)


def test_a_spread_of_exactly_5_percent_is_not_above_it():
    out = S.summary([allocation(12, "a", [0.05, 0.0500001])])
    assert out["12"]["above_5_percent"] == 1


def test_an_allocation_without_an_outcome_run_is_left_out_of_the_summary():
    out = S.summary([{"run_dir": "x", "world": None, "conditions": {}}, allocation(12, "a", [0.01])])
    assert sorted(out) == ["12"] and out["12"]["two_pass_conditions"] == 1


# ---- the committed record
def test_the_script_on_a_committed_archive_gives_the_committed_entry():
    archive = glob.glob(os.path.join(ROOT, "raw", "rev4", "e9-q12b-*.tgz"))
    assert len(archive) == 1
    work = tempfile.mkdtemp(prefix="e9s", dir=os.environ.get("REV4_SHORT_TMP"))
    try:
        with tarfile.open(archive[0]) as tf:
            try:
                tf.extractall(work, filter="data")
            except TypeError:
                tf.extractall(work)
        run = glob.glob(os.path.join(work, "e9-q12b-*"))[0].replace("\\", "/") + "/"
        got = json.loads(json.dumps(S.allocation(run)))
    finally:
        shutil.rmtree(work, ignore_errors=True)
    committed = json.load(io.open(RESULT, encoding="utf-8"))
    want = [a for a in committed["allocations"] if a["run_dir"] == got["run_dir"]]
    assert len(want) == 1 and got == want[0]


def test_what_the_paper_says_about_the_passes_is_in_the_committed_record():
    out = json.load(io.open(RESULT, encoding="utf-8"))
    assert "POST HOC" in out["label"]
    twelve, large = out["summary"]["12"], out["summary"]["24"]
    assert (twelve["two_pass_conditions"], twelve["above_5_percent"]) == (34, 2)
    assert (large["two_pass_conditions"], large["above_5_percent"]) == (35, 8)
    assert round(100 * twelve["median"], 1) == 0.8 and round(100 * large["median"], 1) == 0.9
    assert large["largest"]["condition"] == "transformer|gpu-quant-sr-fp16"
    assert large["largest"]["run_dir"] == "e9-q24b-8n-3495232"
    assert round(100 * large["largest"]["spread"], 1) == 14.8
    # each step of the scored result is the mean of the passes recorded here
    scored = json.load(io.open(os.path.join(os.path.dirname(RESULT), "E9_RESULT.json"), encoding="utf-8"))
    by_run = {a["run_dir"]: a for a in scored["allocations"]}
    for a in out["allocations"]:
        for key, c in a["conditions"].items():
            passes = list(c["passes_ms"].values())
            assert by_run[a["run_dir"]]["table"][key]["T_ms"] == pytest.approx(sum(passes) / len(passes), rel=1e-12)
            assert by_run[a["run_dir"]]["table"][key]["passes"] == len(passes)
