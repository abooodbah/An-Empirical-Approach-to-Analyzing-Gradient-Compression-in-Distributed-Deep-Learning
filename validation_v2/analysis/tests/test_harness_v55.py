# validation_v2/analysis/tests/test_harness_v55.py
# -*- coding: utf-8 -*-
"""Harness v5.5 (2026-10-18, E12): --dtype bf16 (bfloat16 autocast compute;
the 16-bit hook then sends bfloat16) and --verify-fidelity for the two DDP
hooks. The local runs use three CPU processes over gloo, as test_v52_local."""
import glob
import hashlib
import io
import json
import os
import socket
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
HARNESS_DIR = os.path.join(os.path.dirname(os.path.dirname(HERE)), "harness")
sys.path.insert(0, HARNESS_DIR)
import control_workload_benchmark_v5_5 as H  # noqa: E402

HARNESS = os.path.join(HARNESS_DIR, "control_workload_benchmark_v5_5.py")
RUNNER = os.path.join(HARNESS_DIR, "run_local_cpu.py")
BASE = ["--campaign-id", "local", "--run-id", "x", "--output-dir", "o", "--accelerator", "cpu", "--backend", "gloo"]


def source(name):
    return io.open(os.path.join(HARNESS_DIR, name), encoding="utf-8").read()


def parse(*extra):
    return H.parse_args(BASE + list(extra))


# ------------------------------------------------------------------ static
def test_v54_is_untouched():
    raw = open(os.path.join(HARNESS_DIR, "control_workload_benchmark_v5_4.py"), "rb").read()
    assert hashlib.sha256(raw).hexdigest().startswith("d8d0862e11eecce7")


def test_recorded_hash_matches_the_generated_file():
    raw = open(HARNESS, "rb").read()
    recorded = io.open(os.path.join(os.path.dirname(HARNESS_DIR), "CODE_v55.sha256"), encoding="utf-8").read().split()[0]
    assert hashlib.sha256(raw).hexdigest() == recorded


def test_build_is_reproducible(tmp_path, monkeypatch):
    import build_v55
    monkeypatch.setattr(build_v55, "DST", str(tmp_path / "v55.py"))
    monkeypatch.setattr(build_v55, "HERE", str(tmp_path / "harness"))
    os.makedirs(str(tmp_path / "harness"))
    assert build_v55.build() == hashlib.sha256(open(HARNESS, "rb").read()).hexdigest()


def test_default_dtype_is_fp32_and_adds_no_deviation():
    a = parse("--profile", "qualification")
    assert a.dtype == "fp32"
    assert not any(d.startswith("compute_dtype") for d in H.condition_deviations(a))


def test_bf16_is_recorded_as_a_deviation():
    a = parse("--profile", "qualification", "--dtype", "bf16")
    assert "compute_dtype=bf16_autocast" in H.condition_deviations(a)


def test_unknown_dtype_is_refused():
    with pytest.raises(SystemExit):
        parse("--dtype", "fp16")


def test_hook16_wire_type(monkeypatch):
    monkeypatch.delenv("RED_HOOK16_DTYPE", raising=False)
    assert H.hook16_dtype("nccl") == "float16"              # v5.4 behaviour
    assert H.hook16_dtype("ccl") == "bfloat16"              # v5.4 behaviour
    assert H.hook16_dtype("nccl", "bf16") == "bfloat16"     # v5.5
    monkeypatch.setenv("RED_HOOK16_DTYPE", "float16")
    assert H.hook16_dtype("nccl", "bf16") == "float16"      # the override still wins


def test_autocast_context():
    import contextlib
    import torch

    class R:
        kind = "cpu"
    assert isinstance(H.compute_autocast(R(), parse("--profile", "qualification")), contextlib.nullcontext)
    layer = torch.nn.Linear(4, 4)
    with H.compute_autocast(R(), parse("--profile", "qualification", "--dtype", "bf16")):
        out = layer(torch.ones(2, 4))
    assert out.dtype == torch.bfloat16 and layer.weight.dtype == torch.float32


@pytest.mark.parametrize("mode", ["ddp-fp16-hook", "ddp-int8-hook"])
def test_hook_modes_accept_fidelity_in_qualification(mode):
    assert parse("--profile", "qualification", "--comm-mode", mode, "--verify-fidelity").verify_fidelity


@pytest.mark.parametrize("mode", ["ddp-overlap", "powersgd"])
def test_other_ddp_modes_still_refuse_fidelity(mode):
    with pytest.raises(SystemExit):
        parse("--profile", "qualification", "--comm-mode", mode, "--verify-fidelity")


def test_fidelity_stays_out_of_validation_runs():
    with pytest.raises(SystemExit):
        parse("--profile", "validation", "--comm-mode", "ddp-fp16-hook", "--verify-fidelity",
              "--iters", "100", "--warmup-iters", "20", "--input-pool", "4")


def test_without_the_flag_the_hook_is_not_wrapped():
    a = parse("--profile", "qualification", "--comm-mode", "ddp-fp16-hook")
    assert H._with_fidelity(H._half_compress_hook, a) is H._half_compress_hook


def test_finish_hook_fidelity_from_bucket_sums():
    import torch
    H.HOOK_STATE.pop("fidelity", None)
    ref = torch.tensor([3.0, 4.0])
    out = torch.tensor([3.0, 4.5])
    H.HOOK_TENSORS["fidelity_acc"] = [torch.stack([(out - ref).pow(2).sum(), ref.pow(2).sum(),
                                                   out.pow(2).sum(), (out * ref).sum()])]
    e = H.finish_hook_fidelity(7)
    assert e["iteration"] == 7 and e["buckets"] == 1
    assert e["relative_l2_error"] == pytest.approx(0.5 / 5.0)
    assert e["cosine"] == pytest.approx(float((out * ref).sum() / (out.norm() * ref.norm())))
    assert "fidelity_acc" not in H.HOOK_TENSORS and H.HOOK_STATE["fidelity"][-1] is e
    assert H.finish_hook_fidelity(8) is None
    H.HOOK_STATE.pop("fidelity", None)


# ------------------------------------------------------------------ local runs
def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return str(s.getsockname()[1])


def run(tmp_path, name, mode, extra, env=None, world=3):
    out = tmp_path / name
    args = ["--campaign-id", "local", "--run-id", "local-" + name, "--output-dir", str(out),
            "--model", "transformer", "--comm-mode", mode, "--profile", "qualification",
            "--accelerator", "cpu", "--backend", "gloo", "--batch-size", "4", "--iters", "3",
            "--warmup-iters", "1", "--skip-calibration"] + extra
    done = subprocess.run([sys.executable, RUNNER, HARNESS, str(world), free_port()] + args,
                          capture_output=True, text=True, timeout=900, env=dict(os.environ, **(env or {})))
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-2000:]
    return out


def provenance(out, rank=0):
    path = glob.glob(os.path.join(str(out), "*rank%04d.provenance.json" % rank))[0]
    return json.load(io.open(path, encoding="utf-8"))


def sidecar(out, rank=0):
    path = glob.glob(os.path.join(str(out), "*rank%04d.v52.json" % rank))[0]
    return json.load(io.open(path, encoding="utf-8"))


def losses(out):
    rows = []
    for path in glob.glob(os.path.join(str(out), "*rank0000.jsonl")):
        rows += [json.loads(line) for line in io.open(path, encoding="utf-8") if line.strip()]
    return [r["loss"] for r in sorted(rows, key=lambda r: r["iteration"])]


def test_every_forward_pass_runs_under_the_compute_context():
    src = source("control_workload_benchmark_v5_5.py")
    assert src.count("with compute_autocast(runtime, args):") == 3  # two warm-up forms, the measured loop
    assert src.count("output = run_model(inputs)") == 3


def test_bf16_dense_ddp_runs_records_its_precision_and_changes_the_compute(tmp_path):
    # No warm-up: both runs start the measured loop from the same parameters,
    # so the losses differ only through the measured forward's precision.
    fp32 = run(tmp_path, "dense-fp32", "ddp-overlap", ["--warmup-iters", "0"])
    bf16 = run(tmp_path, "dense-bf16", "ddp-overlap", ["--dtype", "bf16", "--warmup-iters", "0"])
    p = provenance(bf16)
    assert p["condition"]["compute_dtype"] == "bf16"
    assert p["condition_id"].endswith(".bf16")
    assert "compute_dtype=bf16_autocast" in p["condition"]["deviations"]
    assert p["parameter_hashes"]["final_params_equal"] is True
    assert provenance(fp32)["condition"]["compute_dtype"] == "fp32"
    # same seed, same inputs: only the forward precision differs
    a, b = losses(fp32), losses(bf16)
    assert len(a) == len(b) == 3 and a != b


def test_bf16_sends_the_16bit_hook_in_bfloat16_and_measures_its_error(tmp_path):
    out = run(tmp_path, "hook16-bf16", "ddp-fp16-hook", ["--dtype", "bf16", "--verify-fidelity"])
    p = provenance(out)
    assert p["ddp"]["dtype"] == "bfloat16"
    fid = sidecar(out)["fidelity"]
    assert [e["iteration"] for e in fid] == [0, 1, 2]
    assert all(e["buckets"] >= 1 for e in fid)
    assert all(0.0 < e["relative_l2_error"] < 0.05 and e["cosine"] > 0.99 for e in fid)


def test_hook_fidelity_orders_float16_below_bfloat16_below_int8(tmp_path):
    fp16 = sidecar(run(tmp_path, "hook16", "ddp-fp16-hook", ["--verify-fidelity"]))["fidelity"]
    bf16 = sidecar(run(tmp_path, "hookbf16", "ddp-fp16-hook", ["--verify-fidelity"],
                       env={"RED_HOOK16_DTYPE": "bfloat16"}))["fidelity"]
    int8 = sidecar(run(tmp_path, "hook8", "ddp-int8-hook", ["--verify-fidelity"],
                       env={"RED_HOOK_SYNC": "1"}))["fidelity"]
    worst = lambda f: max(e["relative_l2_error"] for e in f)  # noqa: E731
    best = lambda f: min(e["relative_l2_error"] for e in f)  # noqa: E731
    assert len(fp16) == len(bf16) == len(int8) == 3
    # float16 keeps 10 mantissa bits, bfloat16 7, the per-bucket int8 code 8 bits over the full range
    assert worst(fp16) < best(bf16) and worst(bf16) < best(int8)
