# -*- coding: utf-8 -*-
"""Ordered run list of one TPDS v2 validation job (2026-09-23).

Jobs V1-V3 (4 nodes x 3 A100 = 12 ranks) run Experiments A and B; L1 (2
nodes x 3 A100 = 6 ranks) runs the optional 1.05B bridge; SMOKE runs the
feasibility checks. Every timed condition runs in two passes: pass 1 is a
shuffle seeded by the job, pass 2 is pass 1 reversed. A calibration run
and a dense anchor open and close each job, the equivalence checks run
first, and (V1 only) four profiler-trace runs come last before the closing
anchor. Nothing about the order depends on any result.

usage: python make_manifest_v2.py JOB --out manifest.json --bash run_list.sh
                                  [--big-batch 128] [--b-passes 2]
"""
import argparse
import json
import random
import shlex
import sys

SEEDS = {"V1": 2026092301, "V2": 2026092302, "V3": 2026092303, "L1": 2026092311, "SMOKE": 2026092399}
MODELS = ("transformer", "resnet50")
A_PATHS = ("gpu-quant-reducible", "gpu-quant-gathersum", "gpu-topk")
CONVENTIONS = (("E", "phase"), ("S", "wall"), ("B", "block"))
B_METHODS = ("ddp-overlap", "ddp-fp16-hook", "ddp-int8-hook")
# Payloads the conditions submit (bytes): int8 (+4-byte scale), FP16, FP32
# gradients of each workload, and DDP's 25 MiB bucket cap. Calibrating at
# these sizes replaces the generic 4 B to 64 MiB sweep.
PAYLOADS = {
    "transformer": (11_425_794, 11_425_798, 22_851_588, 45_703_176),
    "resnet50": (25_557_032, 25_557_036, 51_114_064, 102_228_128),
}
BUCKET = 26_214_400
CAL_SIZES = sorted({1_048_576, BUCKET} | set(PAYLOADS["transformer"]) | set(PAYLOADS["resnet50"]))
CAL_SIZES_1B = (1_048_576, BUCKET, 104_857_600, 268_435_456)
CAL_GUARD = 1_610_612_736  # all-gather receive buffers up to 1.5 GiB per rank

VALIDATION = ["--profile", "validation", "--sync-every", "1", "--iters", "100", "--warmup-iters", "20",
              "--input-pool", "4", "--skip-calibration"]


def entry(eid, role, args, cap_min=6, experiment="", pass_no=0):
    return {"id": eid, "role": role, "experiment": experiment, "pass": pass_no,
            "step_time_cap_min": cap_min, "args": args}


def timed_conditions(job, big_batch):
    out = []
    if job in ("V1", "V2", "V3"):
        for model in MODELS:
            for path in A_PATHS:
                for conv, timing in CONVENTIONS:
                    out.append(("A", f"A-{model}-{path}-{conv}",
                                ["--model", model, "--comm-mode", path, "--timing-mode", timing,
                                 "--batch-size", "32"]))
        for model in MODELS:
            for method in B_METHODS:
                for batch in (32, big_batch):
                    out.append(("B", f"B-{model}-{method}-bs{batch}",
                                ["--model", model, "--comm-mode", method, "--timing-mode", "block",
                                 "--batch-size", str(batch)]))
    elif job == "L1":
        for method in B_METHODS:
            for conv, timing in (("S", "wall"), ("B", "block")):
                out.append(("L", f"L-llama1b-{method}-{conv}",
                            ["--model", "llama1b", "--seq-len", "512", "--comm-mode", method,
                             "--timing-mode", timing, "--batch-size", "1"]))
    return out


def build(job, big_batch=128, b_passes=2):
    if job not in SEEDS:
        raise SystemExit(f"unknown job {job}")
    rng = random.Random(SEEDS[job])
    cal_ref = f"v2-{job}-cal-start"
    ref = ["--calibration-reference", cal_ref]
    runs = []
    if job == "L1":
        cal = ["--profile", "qualification", "--model", "llama1b", "--seq-len", "512", "--comm-mode",
               "ddp-overlap", "--timing-mode", "wall", "--batch-size", "1", "--iters", "4", "--warmup-iters",
               "2", "--calibration-sizes", ",".join(map(str, CAL_SIZES_1B)),
               "--calibration-max-rank-bytes", str(CAL_GUARD)]
        anchor = ["--model", "llama1b", "--seq-len", "512", "--comm-mode", "ddp-overlap",
                  "--timing-mode", "block", "--batch-size", "1"]
        cap = 10
    else:
        cal = ["--profile", "qualification", "--model", "transformer", "--comm-mode", "native",
               "--timing-mode", "wall", "--batch-size", "32", "--iters", "4", "--warmup-iters", "2",
               "--calibration-sizes", ",".join(map(str, CAL_SIZES)),
               "--calibration-max-rank-bytes", str(CAL_GUARD)]
        anchor = ["--model", "transformer", "--comm-mode", "ddp-overlap", "--timing-mode", "block",
                  "--batch-size", "32"]
        cap = 6
    runs.append(entry("cal-start", "calibration", cal, cap_min=cap))
    runs.append(entry("anchor-start", "anchor", anchor + VALIDATION + ref, cap_min=cap))
    if job in ("V1", "V2", "V3", "SMOKE"):
        for model in MODELS:
            for path in ("gpu-quant-reducible", "gpu-quant-gathersum"):
                runs.append(entry(f"verify-{model}-{path}", "verify",
                                  ["--profile", "qualification", "--model", model, "--comm-mode", path,
                                   "--timing-mode", "phase", "--verify-equivalence", "--batch-size", "32",
                                   "--iters", "5", "--warmup-iters", "2", "--skip-calibration"] + ref))
    if job == "SMOKE":
        runs.extend(smoke_entries(ref, big_batch))
    else:
        conditions = timed_conditions(job, big_batch)
        pass1 = conditions[:]
        rng.shuffle(pass1)
        orders = [pass1, list(reversed(pass1))]
        for pass_no, order in enumerate(orders, start=1):
            for experiment, cid, args in order:
                if experiment == "B" and pass_no > b_passes:
                    continue
                runs.append(entry(f"{cid}-p{pass_no}", "timed", args + VALIDATION + ref,
                                  cap_min=cap, experiment=experiment, pass_no=pass_no))
    if job == "V1":
        for method in ("ddp-int8-hook", "ddp-fp16-hook"):
            for batch in (32, big_batch):
                runs.append(entry(f"trace-transformer-{method}-bs{batch}", "trace",
                                  ["--profile", "qualification", "--model", "transformer", "--comm-mode", method,
                                   "--timing-mode", "wall", "--batch-size", str(batch), "--iters", "8",
                                   "--warmup-iters", "20", "--input-pool", "4", "--trace-dir", "__TRACE_DIR__",
                                   "--trace-ranks", "0,3", "--skip-calibration"] + ref))
    runs.append(entry("anchor-end", "anchor", anchor + VALIDATION + ref, cap_min=cap))
    end_cal = [a for a in cal]
    runs.append(entry("cal-end", "calibration", end_cal, cap_min=cap))
    return {"job": job, "seed": SEEDS[job], "big_batch": big_batch, "b_passes": b_passes,
            "calibration_sizes": list(CAL_SIZES_1B if job == "L1" else CAL_SIZES), "runs": runs}


def smoke_entries(ref, big_batch):
    """Feasibility checks, not evidence: implicit synchronizations inside
    the block, memory at the large batch, warm-up adequacy, the 1B model at
    sequence 512, and one trace."""
    out = []
    for model in MODELS:
        for path in A_PATHS:
            out.append(entry(f"syncdebug-{model}-{path}", "smoke",
                             ["--profile", "qualification", "--model", model, "--comm-mode", path,
                              "--timing-mode", "block", "--sync-debug", "warn", "--batch-size", "32",
                              "--iters", "5", "--warmup-iters", "2", "--input-pool", "4",
                              "--skip-calibration"] + ref))
        for method in B_METHODS:
            out.append(entry(f"bigbatch-{model}-{method}-bs{big_batch}", "smoke",
                             ["--profile", "qualification", "--model", model, "--comm-mode", method,
                              "--timing-mode", "block", "--batch-size", str(big_batch), "--iters", "5",
                              "--warmup-iters", "2", "--input-pool", "4", "--skip-calibration"] + ref))
    for model, path in (("transformer", "gpu-quant-gathersum"), ("resnet50", "gpu-quant-reducible")):
        out.append(entry(f"warmup-{model}-{path}", "smoke",
                         ["--profile", "qualification", "--model", model, "--comm-mode", path,
                          "--timing-mode", "wall", "--batch-size", "32", "--iters", "100",
                          "--warmup-iters", "20", "--input-pool", "4", "--skip-calibration"] + ref))
    for method in B_METHODS:
        out.append(entry(f"llama1b-{method}", "smoke",
                         ["--profile", "qualification", "--model", "llama1b", "--seq-len", "512",
                          "--comm-mode", method, "--timing-mode", "block", "--batch-size", "1",
                          "--iters", "5", "--warmup-iters", "2", "--input-pool", "2",
                          "--skip-calibration"] + ref, cap_min=10))
    out.append(entry("trace-transformer-ddp-int8-hook-bs32", "smoke",
                     ["--profile", "qualification", "--model", "transformer", "--comm-mode", "ddp-int8-hook",
                      "--timing-mode", "wall", "--batch-size", "32", "--iters", "8", "--warmup-iters", "20",
                      "--input-pool", "4", "--trace-dir", "__TRACE_DIR__", "--trace-ranks", "0,3",
                      "--skip-calibration"] + ref))
    return out


def bash_lines(manifest):
    lines = ["# generated by make_manifest_v2.py; one harness run per line"]
    for run in manifest["runs"]:
        args = " ".join(shlex.quote(a) for a in run["args"])
        lines.append(f"run_condition {shlex.quote(run['id'])} {run['step_time_cap_min']} {args}")
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("job")
    ap.add_argument("--big-batch", type=int, default=128, choices=(64, 128))
    ap.add_argument("--b-passes", type=int, default=2, choices=(1, 2))
    ap.add_argument("--out", required=True)
    ap.add_argument("--bash", required=True)
    a = ap.parse_args(argv)
    manifest = build(a.job, a.big_batch, a.b_passes)
    with open(a.out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(manifest, fh, indent=1)
    with open(a.bash, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(bash_lines(manifest))
    roles = {}
    for run in manifest["runs"]:
        roles[run["role"]] = roles.get(run["role"], 0) + 1
    print(a.job, "runs:", len(manifest["runs"]), roles)
    return 0


if __name__ == "__main__":
    sys.exit(main())
