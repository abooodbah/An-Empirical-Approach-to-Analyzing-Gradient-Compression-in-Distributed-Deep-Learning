# -*- coding: utf-8 -*-
"""Local CPU/gloo qualification matrix for the v5 harness: every exchange
path used by the TPDS v2 campaign under every timing convention, plus a
trace run and a calibration run at named sizes. Prints one line per run
and exits nonzero on any failure or contract violation."""
import glob
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
HARNESS = os.path.join(HERE, "control_workload_benchmark_v5.py")
OUT = sys.argv[1]
shutil.rmtree(OUT, ignore_errors=True)
base = ["--accelerator", "cpu", "--backend", "gloo", "--profile", "qualification", "--model", "transformer",
        "--batch-size", "2", "--iters", "4", "--warmup-iters", "1", "--input-pool", "2",
        "--campaign-id", "cpu-matrix"]
runs = []
for mode in ("native", "gpu-quant-reducible", "gpu-quant-gathersum", "gpu-topk"):
    for timing in ("phase", "wall", "block"):
        runs.append((f"{mode}-{timing}", ["--comm-mode", mode, "--timing-mode", timing, "--skip-calibration"]))
for mode in ("ddp-overlap", "ddp-int8-hook", "ddp-fp16-hook"):
    runs.append((f"{mode}-block", ["--comm-mode", mode, "--timing-mode", "block", "--skip-calibration"]))
# The v4 int8 hook's asynchronous path reads NCCL's future-result layout,
# which gloo does not share; on CPU it runs through its synchronous path
# (RED_HOOK_SYNC=1, set below). The trace check uses the 16-bit hook.
runs.append(("trace-fp16hook", ["--comm-mode", "ddp-fp16-hook", "--timing-mode", "wall", "--skip-calibration",
                                "--trace-dir", os.path.join(OUT, "traces"), "--trace-ranks", "0,1"]))
runs.append(("calibration", ["--comm-mode", "native", "--timing-mode", "wall",
                             "--calibration-sizes", "1024,65536,1048576"]))
runs.append(("verify-reducible", ["--comm-mode", "gpu-quant-reducible", "--timing-mode", "phase",
                                  "--verify-equivalence", "--skip-calibration"]))
bad = 0
port = 29600
for name, extra in runs:
    port += 1
    out = os.path.join(OUT, name)
    cmd = [sys.executable, os.path.join(HERE, "run_local_cpu.py"), HARNESS, "2", str(port)] + base + extra + [
        "--run-id", name, "--output-dir", out]
    env = dict(os.environ, RED_HOOK_SYNC="1") if "int8-hook" in name else None
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=900, env=env)
    if p.returncode != 0:
        bad += 1
        print("FAIL", name, p.stdout[-1500:], p.stderr[-500:])
        continue
    rows = [json.loads(line) for line in open(sorted(glob.glob(out + "/*rank0000.jsonl"))[0])]
    prov = json.load(open(sorted(glob.glob(out + "/*rank0000.provenance.json"))[0]))
    r = rows[0]
    problems = []
    if r["status"] != "QUALIFICATION_ONLY":
        problems.append("status " + r["status"])
    if r["timing_mode"] == "block" and (r["observed_step_s"] is not None or prov["block"]["runtime_synchronizations_inside_block"] != 0):
        problems.append("block mode synchronized inside the block")
    if r["timing_mode"] == "phase" and any(x["forward_s"] is None for x in rows):
        problems.append("phase fields missing")
    if r["block_step_equiv_s"] <= 0 or r["block_iterations"] != 4:
        problems.append("block fields")
    if name == "trace-fp16hook" and len(glob.glob(os.path.join(OUT, "traces", "*.trace.json"))) != 2:
        problems.append("expected two traces")
    if name == "calibration" and prov["calibration"]["allreduce"]["sizes_bytes"] != [1024, 65536, 1048576]:
        problems.append("calibration sizes")
    if name.startswith("ddp-") and "logging_data_buckets" not in prov["ddp"]:
        problems.append("ddp bucket provenance")
    bad += bool(problems)
    print("%-26s %-5s syncs_in_block=%-3s T=%.4fs %s" % (
        name, r["timing_mode"], prov["block"]["runtime_synchronizations_inside_block"],
        r["block_step_equiv_s"], "OK" if not problems else "PROBLEM " + "; ".join(problems)))
print("RESULT", "all passed" if not bad else "%d failed" % bad)
sys.exit(1 if bad else 0)
