# -*- coding: utf-8 -*-
"""Tests for the E12 scorer (diagnostic/E12_PROTOCOL.md sections 4 to 7) on
synthetic run directories laid out as the E12 jobs write them."""
import csv
import io
import json
import math
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ART, "validation_v2", "harness"))
import score_e12 as S  # noqa: E402

M1_FIELDS = ("run_id", "profile", "timing_mode", "sync_every", "tau_barrier", "model", "comm_mode", "world_size",
             "rank", "iteration", "observed_step_s", "forward_s", "barrier_wait_s", "backward_compute_s",
             "serialize_s", "network_s", "deserialize_s", "optimizer_step_s", "network_s_is_estimate",
             "collective_input_bytes_per_rank", "bytes_uncompressed", "gradient_elements",
             "alpha_allreduce_s", "beta_allreduce_s_per_byte", "alpha_allgather_s", "beta_allgather_s_per_byte",
             "equivalence_ok")
CAL = {"alpha_allreduce_s": 3e-5, "beta_allreduce_s_per_byte": 1e-10,
       "alpha_allgather_s": 3e-5, "beta_allgather_s_per_byte": 6e-10}
N = {"transformer": 11_430_000, "resnet50": 25_557_032}


def results(d):
    path = os.path.join(d, "results")
    os.makedirs(path, exist_ok=True)
    return path


def write_csv(d, run_id, rows, cond="x"):
    for rank in sorted({r["rank"] for r in rows}):
        with io.open(os.path.join(results(d), "%s__%s__rank%04d.csv" % (run_id, cond, rank)), "w", newline="",
                     encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=M1_FIELDS)
            w.writeheader()
            for r in rows:
                if r["rank"] == rank:
                    w.writerow({k: r.get(k, "") for k in M1_FIELDS})


def write_block(d, model, tag, p, T_s, ranks=None, status="PASS", hashes=True, loss=1.0, syncs=0, prov=None):
    run_id = S.rid(model, tag, "block", p)
    for r in (range(S.P) if ranks is None else ranks):
        rec = {"run_id": run_id, "rank": r, "status": status, "final_param_hashes_equal": hashes, "loss": loss,
               "block_step_equiv_s": T_s, "iteration": 0}
        with io.open(os.path.join(results(d), "%s__x__rank%04d.jsonl" % (run_id, r)), "w", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
    mode, dtype = S.split_tag(tag)
    ddp = {}
    if mode == "ddp-int8-hook":
        ddp = {"hook_state": {"mode": "sync"}}
    elif mode == "ddp-fp16-hook":
        ddp = {"dtype": "bfloat16" if dtype == "bf16" else "float16"}
    p_rec = {"run_id": run_id, "block": {"runtime_synchronizations_inside_block": syncs},
             "condition": {"compute_dtype": dtype}, "ddp": ddp}
    p_rec.update(prov or {})
    with io.open(os.path.join(results(d), "%s__x__rank0000.provenance.json" % run_id), "w", encoding="utf-8") as fh:
        json.dump(p_rec, fh)


def write_m1(d, model, tag, step_s, network_s=None, serialize_s=0.0, deserialize_s=0.0, barrier_s=0.0,
             coll_bytes=None):
    mode, _ = S.split_tag(tag)
    rows = []
    for rank in range(S.P):
        for it in range(4):
            rows.append({"run_id": S.rid(model, tag, "phase"), "profile": "extension", "timing_mode": "phase",
                         "sync_every": 1, "tau_barrier": "every_iteration", "model": model, "comm_mode": mode,
                         "world_size": S.P, "rank": rank, "iteration": it, "observed_step_s": step_s,
                         "barrier_wait_s": barrier_s, "serialize_s": serialize_s, "deserialize_s": deserialize_s,
                         "network_s": "" if network_s is None else network_s, "network_s_is_estimate": "False",
                         "bytes_uncompressed": 4 * N[model], "gradient_elements": N[model],
                         "collective_input_bytes_per_rank": "" if coll_bytes is None else coll_bytes})
    write_csv(d, S.rid(model, tag, "phase"), rows)


def write_calibration(d):
    row = dict(CAL, run_id=S.CALIBRATION_RUN, profile="paper", timing_mode="phase", sync_every=1,
               tau_barrier="every_iteration", model="resnet50", comm_mode="native", world_size=S.P, rank=0,
               iteration=0, observed_step_s=0.05)
    write_csv(d, S.CALIBRATION_RUN, [row])


def write_trace(d, model, tag):
    path = os.path.join(d, "traces")
    os.makedirs(path, exist_ok=True)
    for r in (0, 3):
        io.open(os.path.join(path, "%s__x__rank%04d.trace.json" % (S.rid(model, tag, "trace"), r)), "w").write("{}")


def write_fid(q, model, tag, errs):
    with io.open(os.path.join(results(q), "%s__x__rank0000.v52.json" % S.rid(model, tag, "fid")), "w",
                 encoding="utf-8") as fh:
        json.dump({"fidelity": [{"iteration": i, "relative_l2_error": e, "cosine": 1.0} for i, e in enumerate(errs)]}, fh)


def write_equiv(d, model, tag, ok=True):
    rows = [{"run_id": S.rid(model, tag, "equiv"), "rank": r, "model": model, "equivalence_ok": str(ok)}
            for r in range(S.P)]
    write_csv(d, S.rid(model, tag, "equiv"), rows)


# Synthetic step times (s): the Transformer pays for the matched substitution
# and the 16-bit hooks; ResNet-50 is compute-bound.
T = {"transformer": {"ddp-overlap": 0.020, "ddp-fp16-hook": 0.017, "ddp-int8-hook": 0.025, "gpu-quant": 0.024,
                     "cpu-quant": 0.170, "gpu-quant-gathersum": 0.023, "gpu-quant-reducible": 0.018,
                     "ddp-overlap-bf16": 0.014, "ddp-fp16-hook-bf16": 0.011, "ddp-int8-hook-bf16": 0.020},
     "resnet50": {"ddp-overlap": 0.085, "ddp-fp16-hook": 0.0845, "ddp-int8-hook": 0.100, "gpu-quant": 0.100,
                  "cpu-quant": 0.380, "gpu-quant-gathersum": 0.099, "gpu-quant-reducible": 0.090,
                  "ddp-overlap-bf16": 0.055, "ddp-fp16-hook-bf16": 0.054, "ddp-int8-hook-bf16": 0.070}}
ERR = {"gpu-quant": 0.010, "cpu-quant": 0.010, "gpu-quant-gathersum": 0.060, "gpu-quant-reducible": 0.060,
       "ddp-fp16-hook": 0.0004, "ddp-int8-hook": 0.012, "ddp-fp16-hook-bf16": 0.003, "ddp-int8-hook-bf16": 0.012}


def allocation_dir(tmp_path, name):
    d = str(tmp_path / name)
    write_calibration(d)
    for model in S.MODELS:
        for tag in S.TAGS:
            for p in (1, 2):
                write_block(d, model, tag, p, T[model][tag])
            write_trace(d, model, tag)
        step = T[model]
        write_m1(d, model, "gpu-quant-gathersum", step["gpu-quant-gathersum"] * 1.2, network_s=0.6 * step["gpu-quant-gathersum"],
                 coll_bytes=N[model] + 4)
        write_m1(d, model, "gpu-quant-reducible", step["gpu-quant-reducible"] * 1.1, network_s=0.3 * step["gpu-quant-reducible"],
                 coll_bytes=N[model] + 4)
        write_m1(d, model, "gpu-quant", step["gpu-quant"] * 1.2, network_s=0.3 * step["gpu-quant"],
                 serialize_s=0.15 * step["gpu-quant"], deserialize_s=0.1 * step["gpu-quant"], coll_bytes=N[model] + 4)
        write_m1(d, model, "cpu-quant", step["cpu-quant"], network_s=0.1 * step["cpu-quant"],
                 serialize_s=0.5 * step["cpu-quant"], deserialize_s=0.2 * step["cpu-quant"], coll_bytes=N[model] + 4)
        for tag in ("ddp-int8-hook", "ddp-overlap", "ddp-overlap-bf16", "ddp-int8-hook-bf16", "ddp-fp16-hook",
                    "ddp-fp16-hook-bf16"):
            write_m1(d, model, tag, step[tag] * 1.05)
        for tag in ("gpu-quant-gathersum", "gpu-quant-reducible"):
            write_equiv(d, model, tag)
    return d


def qual_dir(tmp_path):
    q = str(tmp_path / "e12-q-2n-100")
    for model in S.MODELS:
        for tag, e in ERR.items():
            write_fid(q, model, tag, [e] * S.FIDELITY_ITERS)
    return q


def fake_replay(paths, family, alpha, beta):
    """The frozen forward replay: families G and H only, all-reduce calibration."""
    assert family in ("G", "H")
    assert len(paths) == 2 and alpha == CAL["alpha_allreduce_s"] and beta == CAL["beta_allreduce_s_per_byte"]
    return {"G": 0.20, "H": 0.30}[family], {"paths": [os.path.basename(p) for p in paths]}


def fake_reverse(paths, family, cal):
    """The reverse substitutions: the mirror families, the full calibration."""
    assert family in ("G-rev", "H-rev") and len(paths) == 2 and cal == pytest.approx(CAL)
    return {"G-rev": -0.15, "H-rev": -0.25}[family], {"paths": [os.path.basename(p) for p in paths]}


def score(d, err, **kw):
    return S.allocation(d, err, predictor=fake_replay, rev_predictor=fake_reverse, **kw)


# ------------------------------------------------------------------ the case table
def test_case_table_is_the_protocols():
    ids = [c[0] for c in S.CASES]
    assert ids == ["%02d" % i for i in range(1, 13)] + ["M01", "M04", "M12"]
    assert all(c[1] in S.TAGS and c[2] in S.TAGS and c[1] != c[2] for c in S.CASES)
    assert {c[0]: c[4] for c in S.CASES if c[4]} == {"01": "G", "04": "H", "12": "H",
                                                     "M01": "G-rev", "M04": "H-rev", "M12": "H-rev"}
    assert [c[0] for c in S.CASES if c[5] == "production"] == ["10", "11", "12", "M12"]
    expect = [e for c in S.CASES for e in c[6]]
    assert (expect.count("help"), expect.count("harm"), expect.count("neutral")) == (12, 14, 4)
    assert len({(c[1], c[2]) for c in S.CASES}) == 15


def test_each_mirror_reverses_its_forward_case():
    case = {c[0]: c for c in S.CASES}
    for m in ("M01", "M04", "M12"):
        fwd, mir = case[m[1:]], case[m]
        assert (mir[1], mir[2]) == (fwd[2], fwd[1]) and mir[4] == fwd[4] + "-rev"
        assert mir[3] == set() and mir[6] == ("harm", "harm") and mir[5] == fwd[5]


def test_every_original_has_an_m1_run_in_the_jobs():
    sets = open(os.path.join(ART, "harness", "launch", "e12_sets.sh"), encoding="utf-8").read()
    phase = sets.split("E12_PHASE=(", 1)[1].split(")", 1)[0].split()
    assert set(S.ORIGINALS) <= set(phase) and len(phase) == 10


def test_thresholds_are_the_protocols():
    assert S.THRESHOLD == 0.05 and S.WEIGHT == 0.5 and S.QUALITY_TOLERANCE == 0.01 and S.BATCH == 64
    assert S.pays(0.05) is True and S.pays(0.0499) is False and S.pays(None) is None


@pytest.mark.parametrize("tag", S.TAGS)
def test_run_ids_round_trip_for_every_tag(tag):
    for pat, rid in ((S.BLOCK, S.rid("resnet50", tag, "block", 2)), (S.PHASE, S.rid("resnet50", tag, "phase")),
                     (S.TRACE, S.rid("resnet50", tag, "trace")), (S.FID, S.rid("resnet50", tag, "fid")),
                     (S.EQUIV, S.rid("resnet50", tag, "equiv"))):
        m = pat.match(rid)
        assert m and m.group("tag") == tag and m.group("b") == "64" and m.group("ws") == "6"


def test_split_tag():
    assert S.split_tag("ddp-overlap-bf16") == ("ddp-overlap", "bf16")
    assert S.split_tag("ddp-overlap") == ("ddp-overlap", "fp32")


# ------------------------------------------------------------------ outcomes
def test_block_value_is_the_mean_of_two_passes(tmp_path):
    d = str(tmp_path / "e12-e12a-2n-1")
    write_block(d, "resnet50", "ddp-overlap", 1, 0.080)
    write_block(d, "resnet50", "ddp-overlap", 2, 0.090)
    v = S.block_values(d)[("resnet50", "ddp-overlap")]
    assert v["T_ms"] == pytest.approx(85.0) and v["passes"] == 2 and v["pass_gap"] == pytest.approx(10 / 85)


@pytest.mark.parametrize("tag,kw", [
    ("ddp-overlap", {"ranks": range(5)}),
    ("ddp-overlap", {"status": "QUALIFICATION_ONLY"}),
    ("ddp-overlap", {"hashes": False}),
    ("ddp-overlap", {"loss": float("nan")}),
    ("ddp-overlap", {"syncs": 2}),
    ("ddp-overlap-bf16", {"prov": {"condition": {"compute_dtype": "fp32"}}}),
    ("ddp-overlap", {"prov": {"condition": {"compute_dtype": "bf16"}}}),
    ("ddp-int8-hook", {"prov": {"ddp": {"hook_state": {"mode": "async"}}}}),
    ("ddp-int8-hook-bf16", {"prov": {"ddp": {"hook_state": {}}}}),
    ("ddp-fp16-hook-bf16", {"prov": {"ddp": {"dtype": "float16"}}}),
    ("ddp-fp16-hook", {"prov": {"ddp": {"dtype": "bfloat16"}}}),
])
def test_an_invalid_pass_is_dropped(tmp_path, tag, kw):
    d = str(tmp_path / "e12-e12a-2n-1")
    write_block(d, "transformer", tag, 1, 0.020, **kw)
    write_block(d, "transformer", tag, 2, 0.030)
    v = S.block_values(d)[("transformer", tag)]
    assert v["passes"] == 1 and v["T_ms"] == pytest.approx(30.0)


def test_no_valid_pass_leaves_the_case_missing(tmp_path, monkeypatch):
    d = allocation_dir(tmp_path, "e12-e12a-2n-1")
    write_block(d, "transformer", "cpu-quant", 1, 0.17, status="FAIL")
    write_block(d, "transformer", "cpu-quant", 2, 0.17, status="FAIL")
    rows = score(d, S.fidelity_errors(None))
    r = [x for x in rows if x["case"] == "E12-T03" and x["method"] == "always"][0]
    assert r["g"] is None and r["pays"] is None
    m = S.metrics([x for x in rows if x["case"] == "E12-T03"], "always")
    assert m["cases_weight"] == 0 and m["missing_weight"] == pytest.approx(0.5)


# ------------------------------------------------------------------ inputs
def test_m1_rows_of_the_fp32_and_bf16_forms_stay_apart(tmp_path):
    d = str(tmp_path / "e12-e12a-2n-7")
    write_calibration(d)
    write_m1(d, "transformer", "ddp-overlap", 0.020)
    write_m1(d, "transformer", "ddp-overlap-bf16", 0.014)
    a = S.m1_row(d, "transformer", "ddp-overlap")
    b = S.m1_row(d, "transformer", "ddp-overlap-bf16")
    assert a["step_ms"] == pytest.approx(20.0) and b["step_ms"] == pytest.approx(14.0)
    assert a["alloc"] == b["alloc"] == "7" and a["gradient_elements"] == N["transformer"]
    assert S.m1_row(d, "resnet50", "ddp-overlap") is None


def test_m1_row_carries_the_calibration_into_the_shape_ratio(tmp_path):
    d = str(tmp_path / "e12-e12a-2n-7")
    write_calibration(d)
    write_m1(d, "transformer", "gpu-quant-gathersum", 0.03, network_s=0.018, coll_bytes=N["transformer"] + 4)
    f = S.m1_row(d, "transformer", "gpu-quant-gathersum")
    assert f["kind"] == "gather" and f["exchange_share"] == pytest.approx(0.6)
    calibrated = CAL["alpha_allreduce_s"] + CAL["beta_allreduce_s_per_byte"] * (N["transformer"] + 4)
    assert f["shape_ratio"] == pytest.approx(0.018 / calibrated)


def test_calibration_reads_both_collectives(tmp_path):
    d = str(tmp_path / "e12-e12a-2n-1")
    write_calibration(d)
    assert S.calibration(d) == pytest.approx(CAL)
    assert S.calibration(str(tmp_path / "none")) is None


# ------------------------------------------------------------------ cost model
def test_line_undoes_the_harness_fit():
    import control_workload_benchmark_v5_5 as H
    sizes = [1024, 1 << 20, 16 << 20, 64 << 20]
    for coll, intercept, slope in (("allreduce", 2e-4, 1.7e-10), ("allgather", 1.5e-4, 5e-10)):
        times = [intercept + slope * s for s in sizes]
        a, b, _ = H.fit_alpha_beta(sizes, times, S.P, coll)
        key = "allreduce" if coll == "allreduce" else "allgather"
        cal = {"alpha_%s_s" % key: a, "beta_%s_s_per_byte" % key: b}
        cal.update({k: 0.0 for k in CAL if k not in cal})
        assert S.line(cal, coll, 32 << 20) == pytest.approx(intercept + slope * (32 << 20), rel=1e-9)


def test_exchange_time_per_mode():
    n = 1000
    ar = lambda b: S.line(CAL, "allreduce", b)  # noqa: E731
    ag = lambda b: S.line(CAL, "allgather", b)  # noqa: E731
    assert S.exchange_time(CAL, "ddp-overlap-bf16", n) == pytest.approx(ar(4 * n))
    assert S.exchange_time(CAL, "ddp-fp16-hook", n) == pytest.approx(ar(2 * n))
    assert S.exchange_time(CAL, "gpu-quant-reducible", n) == pytest.approx(ar(4) + ar(n))
    assert S.exchange_time(CAL, "gpu-quant-gathersum", n) == pytest.approx(ar(4) + ag(n))
    assert S.exchange_time(CAL, "cpu-quant", n) == pytest.approx(ag(4) + ag(n)) == S.exchange_time(CAL, "gpu-quant", n)
    assert S.exchange_time(CAL, "ddp-int8-hook", n) == pytest.approx(ag(8) + ag(n))


def test_cost_model_gain():
    m1 = {"gradient_elements": N["transformer"], "step_ms": 20.0}
    g = S.cost_model_gain(CAL, m1, "ddp-overlap", "ddp-int8-hook")
    want = (S.exchange_time(CAL, "ddp-overlap", N["transformer"]) - S.exchange_time(CAL, "ddp-int8-hook", N["transformer"])) / 0.020
    assert g == pytest.approx(want) and g > S.THRESHOLD   # fewer bytes: the cost model recommends the int8 hook
    assert S.cost_model_gain(CAL, {"gradient_elements": None, "step_ms": 20.0}, "ddp-overlap", "gpu-quant") is None


# ------------------------------------------------------------------ quality
def test_fidelity_errors_and_quality(tmp_path):
    q = qual_dir(tmp_path)
    write_fid(q, "resnet50", "ddp-int8-hook", [0.01] * 7)            # one iteration short: missing
    err = S.fidelity_errors(q)
    assert err[("transformer", "ddp-overlap")] == 0.0 and err[("transformer", "ddp-overlap-bf16")] == 0.0
    assert err[("transformer", "ddp-int8-hook")] == pytest.approx(0.012)
    assert ("resnet50", "ddp-int8-hook") not in err
    assert S.quality_ok(err, "transformer", "ddp-int8-hook", "ddp-fp16-hook") is True
    assert S.quality_ok(err, "transformer", "ddp-overlap", "ddp-fp16-hook") is False   # any error exceeds zero
    assert S.quality_ok(err, "transformer", "gpu-quant-gathersum", "gpu-quant-reducible") is True  # equal
    assert S.quality_ok(err, "transformer", "gpu-quant", "gpu-quant-reducible") is False
    assert S.quality_ok(err, "resnet50", "ddp-int8-hook", "ddp-fp16-hook") is None


def test_quality_tolerance_boundary():
    err = {("transformer", "a"): 0.010, ("transformer", "b"): 0.0101, ("transformer", "c"): 0.010101}
    assert S.quality_ok(err, "transformer", "a", "b") is True       # exactly 1% above
    assert S.quality_ok(err, "transformer", "a", "c") is False


# ------------------------------------------------------------------ metrics
def row(method, recommends, pays_, g=0.1, e=None, verdict=None, label="", case="c", pays_q=None):
    return {"method": method, "recommends": recommends, "pays": pays_, "pays_q": pays_q, "g": g, "e": e,
            "verdict": verdict or ("recommend" if recommends else "none"), "label": label, "case": case}


def test_wilson_matches_the_published_values():
    lo, hi = S.wilson(1, 2)
    assert lo == pytest.approx(0.0945, abs=5e-4) and hi == pytest.approx(0.9055, abs=5e-4)
    lo, hi = S.wilson(0, 10)
    assert lo == 0.0 and hi == pytest.approx(0.2775, abs=5e-4)
    assert S.wilson(0, 0) is None


def test_metrics_counts_and_rates():
    rows = [row("m", True, True), row("m", True, False), row("m", False, True), row("m", False, False),
            row("m", False, False, verdict="placement", label="not tested"), row("m", None, True),
            row("m", True, None)]
    m = S.metrics(rows, "m")
    assert m["cases_weight"] == pytest.approx(2.5) and m["missing_weight"] == pytest.approx(1.0)
    assert m["recommended"] == pytest.approx(1.0) and m["true_pos"] == pytest.approx(0.5)
    assert m["precision"] == pytest.approx(0.5) and m["recall"] == pytest.approx(0.5)
    assert m["fpr"] == pytest.approx(0.5 / 1.5) and m["abstention"] == pytest.approx(1.0 / 2.5)
    assert m["not_tested"] == pytest.approx(0.5)
    assert m["precision_ci"] == pytest.approx(S.wilson(0.5, 1.0))


def test_quality_truth_changes_what_counts_as_a_fix():
    rows = [row("m", True, True, pays_q=False), row("m", True, True, pays_q=True)]
    assert S.metrics(rows, "m")["precision"] == pytest.approx(1.0)
    assert S.metrics(rows, "m", "pays_q")["precision"] == pytest.approx(0.5)


def test_gain_error_over_recommendations_and_over_all_predictions():
    rows = [row("cm", True, True, g=0.30, e=0.20, case="a"), row("cm", False, False, g=0.0, e=0.04, case="b"),
            row("cm", True, False, g=-0.1, e=0.10, case="c")]
    m = S.metrics(rows, "cm")
    assert m["gain_error_median_rec"] == pytest.approx(0.15) and m["gain_error_n_rec"] == 2
    assert m["gain_error_median_all"] == pytest.approx(0.10) and m["gain_error_n_all"] == 3
    assert m["gain_error_ci_all"] == S.bootstrap_median({"a": [0.1], "b": [0.04], "c": [0.2]})


def test_bootstrap_is_deterministic_and_brackets_the_median():
    data = {str(i): [i / 100.0] for i in range(20)}
    a, b = S.bootstrap_median(data), S.bootstrap_median(data)
    assert a == b and a[0] <= 0.095 <= a[1]
    assert S.bootstrap_median({"x": [0.1]}) is None


# ------------------------------------------------------------------ end to end
def test_allocation_rows_and_methods(tmp_path):
    d = allocation_dir(tmp_path, "e12-e12a-2n-11")
    rows = score(d, S.fidelity_errors(qual_dir(tmp_path)))
    assert len(rows) == 30 * len(S.METHODS) and {r["alloc"] for r in rows} == {"11"}
    get = lambda case, m: [r for r in rows if r["case"] == case and r["method"] == m][0]  # noqa: E731
    # measured gains
    assert get("E12-T01", "always")["g"] == pytest.approx(1 - 0.018 / 0.023)
    assert get("E12-R08", "always")["pays"] is False and get("E12-T08", "always")["pays"] is True
    # replay: families G and H only
    assert get("E12-T01", "replay")["e"] == pytest.approx(0.20) and get("E12-T01", "replay")["recommends"]
    assert get("E12-R12", "replay")["e"] == pytest.approx(0.30)
    assert get("E12-T05", "replay")["verdict"] == "no substitution model" and not get("E12-T05", "replay")["recommends"]
    # rules: placement on cpu-quant recommends case 03 only; DDP originals give none
    assert get("E12-T03", "full")["verdict"] == "placement" and get("E12-T03", "full")["recommends"]
    assert get("E12-T05", "full")["verdict"] == "none" and get("E12-T05", "full")["recommends"] is False
    assert get("E12-T02", "full")["verdict"] == "placement" and get("E12-T02", "full")["label"] == "not tested"
    assert not get("E12-T02", "full")["recommends"]
    # benchmark check: shape on the gathers with a large exchange-to-calibration ratio
    assert get("E12-T01", "A2")["verdict"] == "shape" and get("E12-T01", "A2")["recommends"]
    assert get("E12-T04", "A2")["verdict"] == "insufficient"
    # always and the cost model
    assert all(r["recommends"] for r in rows if r["method"] == "always")
    assert get("E12-T05", "cm")["recommends"] is True                 # fewer bytes, a harmful change
    # quality
    assert get("E12-T04", "always")["pays_q"] is True and get("E12-T08", "always")["pays_q"] is False
    # mirror cases: the reverse substitutions, never the frozen forward ones
    assert get("E12-TM01", "always")["g"] == pytest.approx(1 - 0.023 / 0.018) and get("E12-TM01", "always")["pays"] is False
    assert get("E12-TM01", "replay")["e"] == pytest.approx(-0.15) and get("E12-TM01", "replay")["recommends"] is False
    assert get("E12-RM04", "replay")["e"] == pytest.approx(-0.25) and get("E12-RM12", "replay")["verdict"] == "none"
    assert get("E12-TM04", "replay")["family"] == "H-rev" and get("E12-T04", "replay")["family"] == "H"
    # the rules have no verdict that recommends a mirror's intervention
    assert not any(get("E12-%sM%s" % (m, c), k)["recommends"] for m in "TR" for c in ("01", "04", "12")
                   for k in ("full", "A1", "A2", "A3"))
    # M1 of the 16-bit hooks feeds the cost model: the int8 hook's gathers send fewer bytes
    assert get("E12-TM04", "cm")["e"] is not None and get("E12-RM12", "cm")["e"] is not None
    # quality: the matched control's arms have the same error; the int8 hook's is larger
    assert get("E12-TM01", "always")["quality_ok"] is True and get("E12-TM04", "always")["quality_ok"] is False


def test_a_mirror_whose_reverse_replay_fails_is_missing(tmp_path):
    d = allocation_dir(tmp_path, "e12-e12a-2n-11")

    def broken(paths, family, cal):
        raise ValueError("fp16 hook pattern (cast, div_, all-reduce) not found")
    rows = S.allocation(d, S.fidelity_errors(None), predictor=fake_replay, rev_predictor=broken)
    r = [x for x in rows if x["case"] == "E12-RM04" and x["method"] == "replay"][0]
    assert r["recommends"] is None and r["label"] == "missing (replay failed)"
    assert [x for x in rows if x["case"] == "E12-R04" and x["method"] == "replay"][0]["recommends"] is True


def test_summary_over_two_allocations(tmp_path):
    q = S.fidelity_errors(qual_dir(tmp_path))
    rows = (score(allocation_dir(tmp_path, "e12-e12a-2n-11"), q)
            + score(allocation_dir(tmp_path, "e12-e12b-2n-12"), q))
    s = S.summary(rows)
    always = s["pays"]["always"]
    paying = sum(1 for r in rows if r["method"] == "always" and r["pays"]) * 0.5
    assert always["recommended"] == pytest.approx(30.0) and always["cases_weight"] == pytest.approx(30.0)
    assert always["precision"] == pytest.approx(paying / 30.0) and always["recall"] == pytest.approx(1.0)
    assert always["fpr"] == pytest.approx(1.0) and always["abstention"] == pytest.approx(0.0)
    replay = s["pays"]["replay"]
    # 18 rows with no model and 6 mirror rows predicted to lose
    assert replay["recommended"] == pytest.approx(6.0) and replay["abstained"] == pytest.approx(24.0)
    assert s["pays_q"]["always"]["paying"] < always["paying"]
    # the replay on the case types it covers: 6 help (forward) and 6 harm (mirror) rows per allocation pair
    cov = s["replay_covered"]["pays"]
    assert cov["cases_weight"] == pytest.approx(12.0) and cov["recommended"] == pytest.approx(6.0)
    assert cov["false_pos"] == pytest.approx(sum(0.5 for r in rows if r["method"] == "replay" and r["family"]
                                                 and r["recommends"] and not r["pays"]))
    assert cov["fpr"] == pytest.approx(0.0)


def test_rho_and_equivalence(tmp_path):
    d = allocation_dir(tmp_path, "e12-e12a-2n-11")
    r = S.rho(d)["transformer"]
    assert r["rho"]["gpu-quant-gathersum"] == pytest.approx(1.2) and r["rho"]["gpu-quant-reducible"] == pytest.approx(1.1)
    assert r["g_block"] == pytest.approx(1 - 18 / 23.0)
    assert r["g_phase"] == pytest.approx(1 - (18 * 1.1) / (23 * 1.2))
    assert r["bias"] == pytest.approx(r["g_phase"] - r["g_block"]) and r["equal"] is True
    write_equiv(d, "transformer", "gpu-quant-reducible", ok=False)
    assert S.rho(d)["transformer"]["equal"] is False


def test_main_writes_the_json(tmp_path):
    q = qual_dir(tmp_path)
    a = allocation_dir(tmp_path, "e12-e12a-2n-11")
    out = str(tmp_path / "E12_RESULT.json")
    rows, s = S.main(["--qual", q, a, "--json", out], predictor=fake_replay, rev_predictor=fake_reverse)
    data = json.load(io.open(out, encoding="utf-8"))
    assert set(data["summary"]) == {"pays", "pays_q", "replay_covered"} and len(data["rows"]) == len(rows)
    assert "transformer" in data["allocations"]["e12-e12a-2n-11"]["rho"]
    assert data["expectations"]["rows"] == 30
    assert not math.isnan(data["summary"]["pays"]["cm"]["recommended"])
