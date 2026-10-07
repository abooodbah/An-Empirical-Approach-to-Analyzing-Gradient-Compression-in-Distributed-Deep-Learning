# validation_v2/analysis/tests/test_v52_local.py
# -*- coding: utf-8 -*-
"""Local three-process runs of harness v5.2 on CPU with gloo. They check the
arithmetic of the new modes inside the run, where both forms see the same
gradients."""
import csv
import glob
import io
import json
import os
import socket
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
HARNESS_DIR = os.path.join(os.path.dirname(os.path.dirname(HERE)), "harness")
HARNESS = os.path.join(HARNESS_DIR, "control_workload_benchmark_v5_2.py")
RUNNER = os.path.join(HARNESS_DIR, "run_local_cpu.py")
PROBE = os.path.join(HARNESS_DIR, "collective_probe.py")
BUCKET = "1000003"          # does not divide the Transformer's 11.43M gradient elements


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return str(s.getsockname()[1])


def run(tmp_path, mode, extra, script=HARNESS, world=3):
    out = tmp_path / mode
    args = ["--campaign-id", "local", "--run-id", "local-" + mode, "--output-dir", str(out),
            "--model", "transformer", "--comm-mode", mode, "--profile", "qualification",
            "--accelerator", "cpu", "--backend", "gloo", "--batch-size", "4", "--iters", "3",
            "--warmup-iters", "1", "--skip-calibration"] + extra
    done = subprocess.run([sys.executable, RUNNER, script, str(world), free_port()] + args,
                          capture_output=True, text=True, timeout=900)
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-2000:]
    return out


def column(out, name):
    values = set()
    for path in glob.glob(os.path.join(str(out), "*.csv")):
        with io.open(path, newline="", encoding="utf-8") as fh:
            values |= {row[name] for row in csv.DictReader(fh)}
    return values


def sidecar(out, rank=0):
    path = glob.glob(os.path.join(str(out), "*rank%04d.v52.json" % rank))[0]
    return json.load(io.open(path, encoding="utf-8"))


@pytest.mark.parametrize("mode", ["gpu-quant-bucketed", "gpu-quant-reducible-bucketed"])
def test_bucketed_integer_modes_equal_the_whole_exchange(tmp_path, mode):
    out = run(tmp_path, mode, ["--exchange-bucket-elems", BUCKET, "--verify-equivalence"])
    assert column(out, "equivalence_ok") == {"True"}
    assert column(out, "final_param_hashes_equal") == {"True"}
    s = sidecar(out)
    assert s["bucket_count"] == -(-s["gradient_elements"] // int(BUCKET)) and s["bucket_count"] > 1


def test_native_bucketed_runs_and_ranks_agree(tmp_path):
    out = run(tmp_path, "native-bucketed", ["--exchange-bucket-elems", BUCKET])
    assert column(out, "final_param_hashes_equal") == {"True"}


@pytest.mark.parametrize("mode", ["gpu-quant-sr-int8", "gpu-quant-sr-fp16"])
def test_stochastic_wire_sum_is_exact(tmp_path, mode):
    out = run(tmp_path, mode, ["--verify-equivalence"])
    assert column(out, "equivalence_ok") == {"True"}
    assert sidecar(out)["levels_per_side"] == (682 if mode.endswith("fp16") else 42)


def test_fidelity_orders_the_stochastic_quantizers_by_level_count(tmp_path):
    # Both arms round stochastically, so more levels must mean less error.
    # A deterministic arm is not compared here: on a heavy-tailed gradient
    # its L2 error can be below an unbiased quantizer's, which is why the
    # paper reports the error beside the statement about bias.
    coarse = sidecar(run(tmp_path, "gpu-quant-sr-int8", ["--verify-fidelity"]))["fidelity"]
    fine = sidecar(run(tmp_path, "gpu-quant-sr-fp16", ["--verify-fidelity"]))["fidelity"]
    exact = sidecar(run(tmp_path, "native", ["--verify-fidelity"]))["fidelity"]
    assert len(coarse) == len(fine) == len(exact) == 3
    assert all(e["relative_l2_error"] < 1e-5 for e in exact)
    assert max(e["relative_l2_error"] for e in fine) < min(e["relative_l2_error"] for e in coarse)


def test_progress_file_has_one_line_per_measured_iteration(tmp_path):
    progress = tmp_path / "progress"
    run(tmp_path, "gpu-quant-bucketed", ["--exchange-bucket-elems", BUCKET, "--progress-dir", str(progress)])
    files = glob.glob(os.path.join(str(progress), "*.progress.jsonl"))
    assert len(files) == 3
    rows = [json.loads(line) for line in io.open(files[0], encoding="utf-8")]
    assert [r["iteration"] for r in rows] == [0, 1, 2] and all(r["network_s"] >= 0 for r in rows)


def test_collective_probe_times_both_collectives(tmp_path):
    target = tmp_path / "probe.json"
    done = subprocess.run([sys.executable, RUNNER, PROBE, "2", free_port(), "--label", "local",
                           "--output", str(target), "--sizes", "1024,4096,65536",
                           "--accelerator", "cpu", "--backend", "gloo", "--dtype-check"],
                          capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stdout[-2000:]
    probe = json.load(io.open(str(target), encoding="utf-8"))
    assert probe["world_size"] == 2
    assert probe["allreduce"]["sizes_bytes"] == [1024, 4096, 65536] == probe["allgather"]["sizes_bytes"]
    assert probe["dtype_check"]["fp16_integer_sum_exact"] is True
    assert "int16_allreduce" in probe["dtype_check"]
