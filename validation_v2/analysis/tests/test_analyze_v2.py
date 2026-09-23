# -*- coding: utf-8 -*-
"""Synthetic-record tests of analyze_v2.py: known inputs, known verdicts."""
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import analyze_v2 as A  # noqa: E402
import make_manifest_v2 as M  # noqa: E402

# block step times (ms) by (model, mode, timing, batch); the same in every job
TIMES = {}
for model, scale in (("transformer", 1.0), ("resnet50", 3.0)):
    TIMES[(model, "gpu-quant-reducible", "block", 32)] = 10 * scale
    TIMES[(model, "gpu-quant-gathersum", "block", 32)] = 20 * scale
    TIMES[(model, "gpu-topk", "block", 32)] = 12 * scale
    TIMES[(model, "gpu-quant-reducible", "phase", 32)] = 11 * scale   # g_E 0.5417, gap +4.2 points
    TIMES[(model, "gpu-quant-gathersum", "phase", 32)] = 24 * scale
    TIMES[(model, "gpu-topk", "phase", 32)] = 18 * scale              # h_E 0.389, gap +22 points
    TIMES[(model, "gpu-quant-reducible", "wall", 32)] = 10.2 * scale
    TIMES[(model, "gpu-quant-gathersum", "wall", 32)] = 20.3 * scale
    TIMES[(model, "gpu-topk", "wall", 32)] = 12.1 * scale
    for batch, dense, fp16, int8 in ((32, 10, 8, 15), (128, 30, 27, 45)):
        TIMES[(model, "ddp-overlap", "block", batch)] = dense * scale
        TIMES[(model, "ddp-fp16-hook", "block", batch)] = fp16 * scale  # R 0.8 -> 0.9
        TIMES[(model, "ddp-int8-hook", "block", batch)] = int8 * scale  # R 1.5 -> 1.5


def write_job(root, job, drift=0.0, drop=None):
    manifest = M.build(job)
    jdir = os.path.join(root, f"v2-{job}-4n-1000")
    os.makedirs(os.path.join(jdir, "results"))
    os.makedirs(os.path.join(jdir, "provenance"))
    json.dump(manifest, open(os.path.join(jdir, "provenance", f"manifest_{job}.json"), "w"))
    for entry in manifest["runs"]:
        if entry["id"] == drop:
            continue
        args = entry["args"]
        key = (A._arg(args, "--model"), A._arg(args, "--comm-mode"), A._arg(args, "--timing-mode"),
               int(A._arg(args, "--batch-size", 0)))
        t_ms = TIMES.get(key, 5.0)
        if entry["id"] == "anchor-end":
            t_ms *= 1.0 + drift
        run_id = f"v2-{job}-{entry['id']}"
        for rank in range(2):
            rows = []
            for it in range(3):
                rows.append({
                    "run_id": run_id, "rank": rank, "world_size": 2, "iteration": it,
                    "status": "PASS" if entry["role"] in ("timed", "anchor") else "QUALIFICATION_ONLY",
                    "final_hashes_equal": True, "trace_capture": entry["role"] == "trace",
                    "block_step_equiv_s": t_ms / 1e3,
                    "observed_step_s": None if key[2] == "block" else t_ms / 1e3,
                    "network_s": 0.001 if key[2] == "phase" else None,
                })
            with open(os.path.join(jdir, "results", f"{run_id}__rank{rank:04d}.jsonl"), "w") as fh:
                fh.write("".join(json.dumps(r) + "\n" for r in rows))
        json.dump({"run_id": run_id, "block": {"runtime_synchronizations_inside_block": 0 if key[2] == "block" else 7}},
                  open(os.path.join(jdir, "results", f"{run_id}__rank0000.provenance.json"), "w"))


@pytest.fixture()
def campaign(tmp_path):
    for job in ("V1", "V2", "V3"):
        write_job(str(tmp_path), job, drift=0.08 if job == "V2" else 0.0)
    return str(tmp_path)


def test_rho_and_gains(campaign):
    runs = A.load_runs(campaign)
    conds = A.conditions(runs)
    rows = A.experiment_a(conds)
    e = next(r for r in rows if r["job"] == "V1" and r["model"] == "transformer" and r["convention"] == "E")
    assert e["rho_gpu-quant-reducible"] == pytest.approx(1.1)
    assert e["g_matched"] == pytest.approx(1 - 11 / 24)
    assert e["h_topk"] == pytest.approx(1 - 11 / 18)
    assert e["ranking"] == "reduce<topk<gather"


def test_verdicts_a(campaign):
    rows = A.experiment_a(A.conditions(A.load_runs(campaign)))
    text = "\n".join(A.verdicts_a(rows))
    assert "transformer g_matched E vs B: direction same practical direction; magnitude preserved" in text
    assert "transformer h_topk E vs B: direction same practical direction; magnitude not preserved" in text
    assert "perturbation rho_E differs between paths" in text


@pytest.mark.parametrize("q,b,expected", [
    (0.50, 0.40, "same"),               # both practical, same sign
    (0.30, -0.20, "reversed"),          # both practical, opposite signs
    (0.04, 0.06, "margin crossed"),     # same faster method, one gain under the margin
    (-0.03, 0.10, "margin crossed"),    # opposite signs, but one gain is not practical
    (0.02, -0.03, "both below margin"),
])
def test_direction_categories(q, b, expected):
    assert A.direction_category(q, b) == expected


def test_verdicts_b(campaign):
    rows = A.experiment_b(A.conditions(A.load_runs(campaign)))
    text = "\n".join(A.verdicts_b(rows))
    assert "transformer ddp-fp16-hook: relative cost changes with computation" in text
    assert "transformer ddp-int8-hook: relative cost stable with computation" in text
    assert "small batch faster than dense DDP" in text
    assert "large batch slower than dense DDP" in text


def test_checks_flag_drift_and_exclusions(tmp_path):
    write_job(str(tmp_path), "V1", drift=0.08, drop="A-transformer-gpu-topk-E-p2")
    runs = A.load_runs(str(tmp_path))
    rows = A.checks(runs, A.conditions(runs))
    drift = next(r for r in rows if r["check"] == "anchor drift")
    assert drift["flag"] == "FLAG"
    assert any(r["flag"] == "EXCLUDED" and "A-transformer-gpu-topk-E-p2" in r["check"] for r in rows)
    cond = next(c for c in A.conditions(runs) if c["condition"] == "A-transformer-gpu-topk-E")
    assert cond["passes"] == 1


def test_no_pairing_across_allocations(tmp_path):
    write_job(str(tmp_path), "V1")
    write_job(str(tmp_path), "V2", drop=None)
    # remove both passes of one V2 condition: V2 must not borrow V1's value
    for p in ("p1", "p2"):
        run_id = f"v2-V2-A-resnet50-gpu-quant-gathersum-S-{p}"
        for f in os.listdir(os.path.join(str(tmp_path), "v2-V2-4n-1000", "results")):
            if f.startswith(run_id):
                os.remove(os.path.join(str(tmp_path), "v2-V2-4n-1000", "results", f))
    rows = A.experiment_a(A.conditions(A.load_runs(str(tmp_path))))
    assert not any(r["job"] == "V2" and r["model"] == "resnet50" and r["convention"] == "S" for r in rows)
    assert any(r["job"] == "V1" and r["model"] == "resnet50" and r["convention"] == "S" for r in rows)
    text = "\n".join(A.verdicts_a(rows))
    assert "resnet50 g_matched S vs B: incomplete" in text


def test_block_run_with_synchronization_is_invalid(tmp_path):
    write_job(str(tmp_path), "V1")
    jdir = os.path.join(str(tmp_path), "v2-V1-4n-1000", "results")
    run_id = "v2-V1-A-resnet50-gpu-topk-B-p1"
    json.dump({"run_id": run_id, "block": {"runtime_synchronizations_inside_block": 3}},
              open(os.path.join(jdir, f"{run_id}__rank0000.provenance.json"), "w"))
    runs = A.load_runs(str(tmp_path))
    bad = next(r for r in runs if r["entry"] == "A-resnet50-gpu-topk-B-p1")
    assert not bad["valid"] and "synchronization" in bad["reason"]
