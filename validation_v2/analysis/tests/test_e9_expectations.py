# -*- coding: utf-8 -*-
import copy
import io
import json
import os

import e9_expectations as X

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RESULTS = os.path.join(ROOT, "validation_v2", "results", "rev4")
MODES = ("native", "fp16-native", "ddp-overlap", "ddp-fp16-hook", "ddp-int8-hook", "gpu-quant",
         "gpu-quant-reducible", "gpu-quant-sr-int8", "gpu-quant-sr-fp16")


def allocation(world, name, fidelity=False):
    table = {"%s|%s" % (model, mode): {"T_ms": 10.0, "passes": 2} for model in X.MODELS for mode in MODES}
    for model in X.MODELS:
        table["%s|ddp-int8-hook" % model]["T_ms"] = 10.0 * world / 6.0          # slower, more at 24 ranks
    fid = {}
    if fidelity:
        for model in X.MODELS:
            fid["%s|gpu-quant-sr-fp16" % model] = {"relative_l2_error": 0.4}
            fid["%s|gpu-quant-sr-int8" % model] = {"relative_l2_error": 2.0}
    probe = {"P1": {"slope_ratio": world / 2.0, "model": world / 2.0, "holds": True},
             "P2": {"default_over_ring": 1.0, "holds": True},
             "P3": {"errors": {"allreduce 256 MiB": 0.05}, "holds": True}}
    return {"run_dir": name, "world": world, "table": table, "fidelity": fid,
            "probe_checks": {"gdr": copy.deepcopy(probe), "host_staged": copy.deepcopy(probe) if fidelity else {}}}


def result(allocations=None):
    allocations = allocations or [allocation(12, "a", True), allocation(12, "b"),
                                  allocation(24, "c", True), allocation(24, "d")]
    summary = {}
    for world in (12, 24):
        cells = {}
        for model in X.MODELS:
            cells["%s|gpu-quant-sr-fp16|fp16-native" % model] = {"call": "within 5%", "values": [1.0, 1.0]}
            cells["%s|ddp-fp16-hook|ddp-overlap" % model] = {"call": "within 5%", "values": [1.0, 1.0]}
            cells["%s|ddp-int8-hook|ddp-overlap" % model] = {"call": "slower", "values": [world / 6.0] * 2}
        summary[str(world)] = cells
    return {"allocations": allocations, "summary": summary}


def test_every_expectation_holds_on_a_result_built_to_meet_them():
    out = X.outcomes(result())
    assert [k for k in ("X1", "X2", "X3", "X4", "X5") if out[k]["held"]] == ["X1", "X2", "X3", "X4", "X5"]
    assert out["X1"]["values"]["12|transformer"] == [1.0, 1.0]
    assert out["P1"]["gdr"] == {"12": [True, True], "24": [True, True]}
    assert out["P1"]["host_staged"] == {"12": [True], "24": [True]}


def test_x1_fails_when_one_allocation_is_more_than_10_percent_apart():
    r = result()
    r["allocations"][1]["table"]["resnet50|gpu-quant-sr-int8"]["T_ms"] = 11.2
    out = X.outcomes(r)["X1"]
    assert out["held"] is False and out["failed"] == ["12|resnet50"]
    r["allocations"][1]["table"]["resnet50|gpu-quant-sr-int8"]["T_ms"] = 10.9
    assert X.outcomes(r)["X1"]["held"] is True


def test_x2_fails_only_on_the_protocols_call_faster():
    r = result()
    r["summary"]["24"]["transformer|gpu-quant-sr-fp16|fp16-native"] = {"call": "allocation-dependent",
                                                                       "values": [0.99, 1.05]}
    assert X.outcomes(r)["X2"]["held"] is True
    r["summary"]["24"]["transformer|gpu-quant-sr-fp16|fp16-native"] = {"call": "faster", "values": [0.9, 0.9]}
    out = X.outcomes(r)["X2"]
    assert out["held"] is False and out["failed"] == ["24|transformer"]


def test_x3_needs_slower_everywhere_and_a_larger_ratio_at_24_ranks():
    r = result()
    assert X.outcomes(r)["X3"]["held"] is True
    r["summary"]["24"]["resnet50|ddp-int8-hook|ddp-overlap"]["values"] = [1.5, 1.5]      # below the 12-rank 2.0
    out = X.outcomes(r)["X3"]
    assert out["held"] is False and out["failed"] == ["resnet50: not larger at 24 ranks"]
    r = result()
    r["summary"]["12"]["transformer|ddp-int8-hook|ddp-overlap"]["call"] = "within 5%"
    assert X.outcomes(r)["X3"]["failed"] == ["12|transformer"]


def test_x4_names_the_cell_where_the_16_bit_hook_is_slower():
    r = result()
    r["summary"]["24"]["resnet50|ddp-fp16-hook|ddp-overlap"] = {"call": "slower", "values": [1.06, 1.06]}
    out = X.outcomes(r)["X4"]
    assert out["held"] is False and out["failed"] == ["24|resnet50"]
    assert out["calls"]["24|resnet50"] == "slower" and out["calls"]["12|transformer"] == "within 5%"


def test_x5_compares_the_two_errors_wherever_both_were_measured():
    r = result()
    r["allocations"][2]["fidelity"]["resnet50|gpu-quant-sr-fp16"]["relative_l2_error"] = 2.5
    out = X.outcomes(r)["X5"]
    assert out["held"] is False and out["failed"] == ["24|resnet50"]
    assert out["values"]["12|transformer"] == [0.4, 2.0]


def test_a_probe_check_that_fails_in_one_allocation_is_shown_per_allocation():
    r = result()
    r["allocations"][3]["probe_checks"]["gdr"]["P2"]["holds"] = False
    assert X.outcomes(r)["P2"]["gdr"] == {"12": [True, True], "24": [True, False]}


def test_an_allocation_without_an_outcome_run_is_left_out():
    dead = {"run_dir": "x", "world": None, "table": {}, "fidelity": {}, "probe_checks": {"gdr": {}, "host_staged": {}}}
    r = result()
    r["allocations"].insert(0, dead)
    assert X.outcomes(r)["X1"]["values"]["12|transformer"] == [1.0, 1.0]


# ---- the committed result
def load(name):
    return json.load(io.open(os.path.join(RESULTS, name), encoding="utf-8"))


def test_outcomes_of_the_scored_lonestar6_jobs():
    out = X.outcomes(load("E9_RESULT.json"))
    assert {k: out[k]["held"] for k in ("X1", "X2", "X3", "X4", "X5")} == {
        "X1": True, "X2": True, "X3": True, "X4": False, "X5": True}
    assert out["X4"]["failed"] == ["24|resnet50"]
    assert out["X2"]["calls"]["24|transformer"] == "allocation-dependent"
    for check in ("P1", "P2", "P3"):
        assert out[check]["gdr"] == {"12": [True, True], "24": [False, False]}
        assert out[check]["host_staged"] == {"12": [True], "24": [True]}


def test_the_committed_files_are_what_the_script_writes():
    out = X.outcomes(load("E9_RESULT.json"))
    assert load("E9_EXPECTATIONS.json") == json.loads(json.dumps(out))
    text = io.open(os.path.join(RESULTS, "E9_EXPECTATIONS.md"), encoding="utf-8").read()
    assert text == X.render(out, load("E9_PROBES_POSTHOC.json"))
    assert "| X4 |" in text and "not held" in text and "POST HOC" in text
