# -*- coding: utf-8 -*-
"""Tests for the E8 scorer (diagnostic/E8_PROTOCOL.md sections 4 to 6) on
synthetic run directories."""
import io
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import score_e8 as S  # noqa: E402


def write_block(run_dir, world, model, mode, batch, p, T, status="PASS", ranks=None, syncs=0, loss=1.0):
    rid = "redE8-ws%d-%s-%s-b%d-block-p%d" % (world, model, mode, batch, p)
    res = os.path.join(run_dir, "results")
    os.makedirs(res, exist_ok=True)
    for r in (range(world) if ranks is None else ranks):
        rec = {"run_id": rid, "rank": r, "world_size": world, "status": status, "timing_mode": "block",
               "block_step_equiv_s": T, "loss": loss, "iteration": 0}
        with io.open(os.path.join(res, "%s__x__rank%04d.jsonl" % (rid, r)), "w", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
    with io.open(os.path.join(res, "%s__x__rank0000.provenance.json" % rid), "w", encoding="utf-8") as fh:
        json.dump({"run_id": rid, "block": {"runtime_synchronizations_inside_block": syncs}}, fh)


def test_block_value_is_the_mean_of_two_passes(tmp_path):
    d = str(tmp_path / "e8-e8s-2n-1")
    write_block(d, 6, "resnet50", "gpu-quant-gathersum", 32, 1, 0.100)
    write_block(d, 6, "resnet50", "gpu-quant-gathersum", 32, 2, 0.110)
    v = S.block_values(d)[("resnet50", "gpu-quant-gathersum")]
    assert v["T_ms"] == pytest.approx(105.0) and v["passes"] == 2
    assert v["pass_gap"] == pytest.approx(10.0 / 105.0)


def test_single_pass_is_used_and_marked(tmp_path):
    d = str(tmp_path / "e8-e8s-2n-1")
    write_block(d, 6, "resnet50", "ddp-fp16-hook", 32, 1, 0.050)
    write_block(d, 6, "resnet50", "ddp-fp16-hook", 32, 2, 0.060, status="QUALIFICATION_ONLY")
    v = S.block_values(d)[("resnet50", "ddp-fp16-hook")]
    assert v["T_ms"] == pytest.approx(50.0) and v["passes"] == 1


@pytest.mark.parametrize("kw", [{"ranks": [0, 1, 2, 3, 4]}, {"syncs": 3}, {"loss": float("nan")},
                                {"status": "FAIL"}])
def test_invalid_runs_are_missing(tmp_path, kw):
    d = str(tmp_path / "e8-e8s-2n-1")
    write_block(d, 6, "transformer", "ddp-int8-hook", 32, 1, 0.050, **kw)
    assert ("transformer", "ddp-int8-hook") not in S.block_values(d)


def test_measured_gain():
    assert S.measured_gain({"T_ms": 100.0}, {"T_ms": 60.0}) == pytest.approx(0.40)
    assert S.measured_gain(None, {"T_ms": 60.0}) is None


@pytest.mark.parametrize("verdict,e,g,expected", [
    ("shape", 0.40, 0.50, "correct"),
    ("shape", 0.40, 0.10, "short"),
    ("shape", 0.40, 0.01, "null"),
    ("shape", 0.40, -0.10, "harmful"),
    ("none", 0.0, 0.30, "abstain-miss"),
    ("insufficient", 0.0, 0.01, "abstain-correct"),
    ("placement", 0.30, 0.50, "not tested"),
    ("shape", 0.40, None, "missing (run failed)"),
])
def test_rule_labels(verdict, e, g, expected):
    assert S.rule_label(verdict, e, g, "gather") == expected


def test_reference_recommends_at_five_percent():
    assert S.reference_verdict(0.05) == "shape" and S.reference_verdict(0.0499) == "none"
    assert S.rule_label(S.reference_verdict(0.30), 0.30, 0.20, "gather") == "correct"
    assert S.rule_label(S.reference_verdict(0.30), 0.30, 0.10, "gather") == "short"


def test_summary_weights_and_magnitude_error():
    rows = [
        {"case": "c1", "alloc": "a", "method": "reference", "verdict": "shape", "e": 0.30, "g": 0.20, "label": "correct"},
        {"case": "c1", "alloc": "b", "method": "reference", "verdict": "shape", "e": 0.30, "g": 0.25, "label": "correct"},
        {"case": "c2", "alloc": "a", "method": "reference", "verdict": "none", "e": 0.01, "g": 0.00, "label": "abstain-correct"},
        {"case": "c2", "alloc": "b", "method": "reference", "verdict": "none", "e": 0.01, "g": None, "label": "missing (run failed)"},
    ]
    s = S.summary(rows)["reference"]
    assert s["recommended"] == pytest.approx(1.0) and s["cleared"] == pytest.approx(1.0)
    assert s["abstain_correct"] == pytest.approx(0.5) and s["missing"] == pytest.approx(0.5)
    assert s["median_magnitude_error"] == pytest.approx(0.075)


def test_calibration_reads_quoted_csv_fields(tmp_path):
    res = tmp_path / "e8-e8s-2n-1" / "results"
    res.mkdir(parents=True)
    (res / "redE8-ws6-resnet50-native__x__rank0000.csv").write_text(
        'run_id,note,alpha_allreduce_s,beta_allreduce_s_per_byte\n'
        'redE8-ws6-resnet50-native,"a, quoted, field",1.5e-05,9.5e-11\n', encoding="utf-8")
    assert S.calibration(str(tmp_path / "e8-e8s-2n-1")) == (1.5e-05, 9.5e-11)


def test_allocation_scores_the_reference_against_block_outcomes(tmp_path):
    d = tmp_path / "e8-e8l-8n-42"
    res = d / "results"
    res.mkdir(parents=True)
    (res / "redE8-ws24-resnet50-native__x__rank0000.csv").write_text(
        "run_id,alpha_allreduce_s,beta_allreduce_s_per_byte\nredE8-ws24-resnet50-native,1e-05,1e-10\n", encoding="utf-8")
    for model in ("transformer", "resnet50"):
        for mode, T in (("gpu-quant-gathersum", 0.100), ("gpu-quant-reducible", 0.050),
                        ("ddp-int8-hook", 0.080), ("ddp-fp16-hook", 0.078)):
            for p in (1, 2):
                write_block(str(d), 24, model, mode, 128, p, T)
    (d / "traces").mkdir()
    for mode in ("gpu-quant-gathersum", "ddp-int8-hook"):
        for r in (0, 3):
            (d / "traces" / ("redE8-ws24-trace-transformer-%s-b128__t__rank%04d.trace.json" % (mode, r))).write_text("{}")
    calls = []

    def stub(paths, fam, a, b):
        calls.append((len(paths), fam, a, b))
        return (0.40, {}) if fam == "G" else (0.20, {})

    rows = S.allocation(str(d), predictor=stub)
    ref = {r["case"]: r for r in rows if r["method"] == "reference"}
    assert ref["E8-e8l-TG"]["label"] == "correct"          # g = 0.5 >= max(0.05, 0.2)
    assert ref["E8-e8l-TH"]["label"] == "null"             # g = 0.025 < 0.05
    assert ref["E8-e8l-RG"]["label"].startswith("missing")  # no resnet50 traces
    assert calls == [(2, "G", 1e-05, 1e-10), (2, "H", 1e-05, 1e-10)]
    assert all(r["label"] == "missing (M1 failed)" for r in rows if r["method"] == "full")
