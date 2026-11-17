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


def test_every_row_of_a_rendered_table_has_as_many_columns_as_its_header():
    # a cell name such as 24|resnet50 must not reach a table cell: the bar is the column separator
    r = result()
    r["summary"]["24"]["resnet50|ddp-fp16-hook|ddp-overlap"] = {"call": "slower", "values": [1.06, 1.06]}
    posthoc = {"allocations": []}
    lines = X.render(X.outcomes(r), posthoc).split("\n")
    width = None
    for line in lines:
        if not line.startswith("|"):
            width = None
            continue
        width = width or line.count("|")
        assert line.count("|") == width, line
    assert "not held (resnet50 at 24 ranks)" in "\n".join(lines)


def test_the_numbers_the_readme_states_about_e9_are_the_committed_records():
    readme = io.open(os.path.join(ROOT, "RECORDS.md"), encoding="utf-8").read()
    out, probes = load("E9_EXPECTATIONS.json"), load("E9_PROBES_POSTHOC.json")
    gdr = {w: [a["gdr"] for a in probes["allocations"] if a["world"] == w] for w in (12, 24)}
    stated = [
        "X1, X2, X3 and X5 held",
        "(%s in both allocations)" % " and ".join(sorted({"%.2f" % v for v in out["X4"]["values"]["24|resnet50"]})),
        "slope ratio %s against p/2 = 12" % " and ".join("%.1f" % v for v in out["P1"]["values"]["gdr"]["24"]),
        "took %s ms" % " and ".join("%.2f" % g["points_ms"]["ring_allreduce"]["1"] for g in gdr[24]),
        "(%s ms)" % " and ".join("%.2f" % g["points_ms"]["ring_allreduce"]["16"] for g in gdr[24]),
        "is %s at 24 ranks" % " and ".join("%.2f" % g["from_16_to_64_mib"]["gather_over_ring"] for g in gdr[24]),
        "%s at 12" % " and ".join("%.2f" % g["from_16_to_64_mib"]["gather_over_ring"] for g in gdr[12]),
    ]
    for text in stated:
        assert text in readme, text
    times = [g["tree_over_ring_time_at_1024_mib"] for w in (12, 24) for g in gdr[w]]
    assert "took %.1f to %.1f times the ring's time" % (min(times), max(times)) in readme
    assert "e9_tree_posthoc.py` of commit 5573a1d" in readme and "E9_EXPECTATIONS.md" in readme


# ---- added after the second review (2026-10-08): the boundaries of each reading
def with_step(r, key, allocation_index, t_ms):
    r["allocations"][allocation_index]["table"][key]["T_ms"] = t_ms
    return r


def test_x1_is_two_sided_and_its_bound_is_ten_percent_inclusive():
    for t_ms, held in ((9.0, True), (11.0, True), (8.9, False), (11.1, False), (8.0, False)):
        out = X.outcomes(with_step(result(), "transformer|gpu-quant-sr-int8", 0, t_ms))["X1"]
        assert out["held"] is held, t_ms


def test_x2_fails_when_every_allocation_that_ran_is_at_or_below_095():
    r = result()
    r["summary"]["24"]["transformer|gpu-quant-sr-fp16|fp16-native"] = {"call": "single allocation", "values": [0.80]}
    assert X.outcomes(r)["X2"]["failed"] == ["24|transformer"]
    r["summary"]["24"]["transformer|gpu-quant-sr-fp16|fp16-native"] = {"call": "allocation-dependent",
                                                                       "values": [0.95, 0.96]}
    assert X.outcomes(r)["X2"]["held"] is True
    r["summary"]["24"]["transformer|gpu-quant-sr-fp16|fp16-native"] = {"call": "faster", "values": [0.95, 0.95]}
    assert X.outcomes(r)["X2"]["held"] is False


def test_x3_needs_every_24_rank_value_strictly_above_every_12_rank_value():
    r = result()
    r["summary"]["24"]["resnet50|ddp-int8-hook|ddp-overlap"]["values"] = [2.0, 2.5]      # 12 ranks: 2.0 and 2.0
    assert X.outcomes(r)["X3"]["failed"] == ["resnet50: not larger at 24 ranks"]
    r["summary"]["24"]["resnet50|ddp-int8-hook|ddp-overlap"]["values"] = [2.01, 2.5]
    assert X.outcomes(r)["X3"]["held"] is True
    r["summary"]["12"]["resnet50|ddp-int8-hook|ddp-overlap"]["values"] = [1.5, 2.2]      # one 12-rank value above
    assert X.outcomes(r)["X3"]["failed"] == ["resnet50: not larger at 24 ranks"]


def test_x4_is_read_from_the_values_and_holds_when_every_allocation_is_below_105():
    r = result()
    # faster in one allocation and within 5% in the other: the call is allocation-dependent, the expectation holds
    r["summary"]["12"]["transformer|ddp-fp16-hook|ddp-overlap"] = {"call": "allocation-dependent",
                                                                   "values": [0.94, 0.96]}
    assert X.outcomes(r)["X4"]["held"] is True
    r["summary"]["12"]["transformer|ddp-fp16-hook|ddp-overlap"] = {"call": "allocation-dependent",
                                                                   "values": [1.049, 1.05]}
    assert X.outcomes(r)["X4"]["failed"] == ["12|transformer"]
    r["summary"]["12"]["transformer|ddp-fp16-hook|ddp-overlap"] = {"call": "within 5%", "values": [1.049, 1.049]}
    assert X.outcomes(r)["X4"]["held"] is True


def test_x5_needs_the_wider_quantizers_error_strictly_below():
    r = result()
    r["allocations"][0]["fidelity"]["transformer|gpu-quant-sr-fp16"]["relative_l2_error"] = 2.0     # equal to int8's
    assert X.outcomes(r)["X5"]["failed"] == ["12|transformer"]


def test_a_probe_cell_reads_held_only_if_every_allocation_held():
    r = result()
    r["allocations"][3]["probe_checks"]["gdr"]["P2"]["holds"] = False
    out = X.outcomes(r)
    assert X.probe_cell(out["P2"], "P2", "gdr", "24").startswith("not held")
    assert X.probe_cell(out["P2"], "P2", "gdr", "12").startswith("held")


def test_a_result_without_any_outcome_run_gives_no_held_expectation_and_no_error():
    dead = {"run_dir": "x", "world": None, "table": {}, "fidelity": {}, "probe_checks": {"gdr": {}, "host_staged": {}}}
    out = X.outcomes({"allocations": [dead], "summary": {}})
    assert [out[k]["held"] for k in ("X1", "X2", "X3", "X4", "X5")] == [False] * 5
    assert "| X1 |" in X.render(out, {"allocations": []})


def test_a_probe_without_a_point_above_64_mib_is_said_to_have_none():
    r = result()
    for a in r["allocations"]:
        a["probe_checks"]["gdr"]["P3"] = {"errors": {}, "holds": False}
    text = X.render(X.outcomes(r), {"allocations": []})
    assert "not held: no point above 64 MiB" in text


def posthoc_ranges():
    def reading(ratio, default, error):
        return {"gather_over_ring": ratio, "default_over_ring": default, "prediction_errors": {"allreduce 1024 MiB": error}}
    def transport(a, b, c):
        return {"rule_1_to_64_mib": reading(30.0, 2.2, 0.24), "from_16_to_64_mib": reading(*a),
                "from_16_to_256_mib": reading(*b), "from_64_to_256_mib": reading(*c),
                "points_ms": {"ring_allreduce": {"1": 10.3, "16": 3.8}}, "tree_over_ring_time_at_1024_mib": 11.8}
    return {"allocations": [
        {"run_dir": "c", "world": 24, "gdr": transport((11.75, 0.977, 0.134), (10.63, 1.042, 0.211), (10.46, 1.052, 0.235))},
        {"run_dir": "d", "world": 24, "gdr": transport((12.12, 1.058, 0.103), (11.07, 1.099, 0.216), (10.91, 1.106, 0.234))}]}


def test_the_second_reading_is_printed_for_every_range_with_a_verdict_for_each_check():
    text = X.render(X.outcomes(result()), posthoc_ranges())
    rows = [line for line in text.split("\n") if line.startswith("| 16 to 64 MiB") or line.startswith("| 16 to 256 MiB")
            or line.startswith("| 64 to 256 MiB")]
    assert len(rows) == 3
    assert "held: 11.75, 12.12" in rows[0] and "not held: 0.98, 1.06" in rows[0] and "held: 13.4%, 10.3%" in rows[0]
    assert "held: 10.63, 11.07" in rows[1] and "not held: 1.04, 1.10" in rows[1] and "not held: 21.1%, 21.6%" in rows[1]
    assert "not held: 1.05, 1.11" in rows[2]            # 1.052 is above the 5% bound although it prints as 1.05
    assert "Readings of the expectations" in text and "after the records were read" in text


def test_the_readme_states_what_the_second_review_added_from_the_records():
    # the README is hard-wrapped: compare with its line breaks folded into spaces
    readme = " ".join(io.open(os.path.join(ROOT, "RECORDS.md"), encoding="utf-8").read().split())
    probes = load("E9_PROBES_POSTHOC.json")
    spread = load("E9_PASS_SPREAD_POSTHOC.json")["summary"]
    nodes = json.load(io.open(os.path.join(ROOT, "environment", "E9_NODE_FACTS.json"), encoding="utf-8"))
    ranges = ("from_16_to_64_mib", "from_16_to_256_mib", "from_64_to_256_mib")
    gdr = {w: [a["gdr"] for a in probes["allocations"] if a["world"] == w] for w in (12, 24)}
    ratios = [g[r]["gather_over_ring"] for g in gdr[24] for r in ranges]
    defaults = [g[r]["default_over_ring"] for g in gdr[24] for r in ranges]
    longer = [max(g[r]["prediction_errors"].values()) for w in (12, 24) for g in gdr[w] for r in ranges[1:]]
    both = [a for a in probes["allocations"] if "host_staged" in a]
    faster = [a["host_staged"]["points_ms"][c][k] / a["gdr"]["points_ms"][c][k]
              for a in both for c in ("ring_allreduce", "ring_allgather")
              for k in a["gdr"]["points_ms"][c] if int(k) >= 16 and k in a["host_staged"]["points_ms"][c]]
    stated = [
        "it is %.2f to %.2f at 24 ranks" % (min(ratios), max(ratios)),
        "(%s with the line through 16 and 64 MiB, up to %.2f with the others)"
        % (" and ".join("%.2f" % g["from_16_to_64_mib"]["default_over_ring"] for g in gdr[24]), max(defaults)),
        "(%s)" % " and ".join("%.1f%%" % (100 * max(g["from_16_to_64_mib"]["prediction_errors"].values()))
                             for g in gdr[24]),
        "(%.1f%% to %.1f%%)" % (100 * min(longer), 100 * max(longer)),
        "%d node records describe %d nodes" % (nodes["records"], nodes["nodes"]),
        "by a median of %.1f%% at 12 ranks and %.1f%% at 24" % (100 * spread["12"]["median"], 100 * spread["24"]["median"]),
        "by more than 5%% in %d of %d conditions at 12 ranks and %d of %d at 24"
        % (spread["12"]["above_5_percent"], spread["12"]["two_pass_conditions"],
           spread["24"]["above_5_percent"], spread["24"]["two_pass_conditions"]),
        "by %.1f%% at most" % (100 * max(spread[w]["largest"]["spread"] for w in ("12", "24"))),
        "by %.1f to %.1f times" % (min(faster), max(faster)),
    ]
    for text in stated:
        assert text in readme, text
    odd = [h for h, f in nodes["by_node"].items() if "BlueField" in f["adapter"]]
    assert len(odd) == 1 and all(odd[0] in hosts for run, hosts in nodes["runs"].items() if "q24" in run)
    assert not any(odd[0] in hosts for run, hosts in nodes["runs"].items() if "q12" in run)
    assert "present in both 24-rank jobs and in neither 12-rank job" in readme
    assert "will record the same" not in readme
