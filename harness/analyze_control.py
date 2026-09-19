# -*- coding: utf-8 -*-
"""Summarize the reducible-control campaign (redctl 2026-09-18).

Input: retrieved run directories (results/*.csv per rank). Conventions match
the paper: step time is the per-iteration maximum across ranks, then the mean
over iterations; phase quantities pool rank-iterations. Wall-mode rows carry
only observed_step_s.
"""
import csv
import glob
import io
import os
import statistics as st
import sys
from collections import defaultdict

ROOT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "retrieved")


def openable(path):
    if os.name == "nt":
        path = os.path.abspath(path).replace("/", "\\")
        if not path.startswith("\\\\?\\"):
            path = "\\\\?\\" + path
    return path

# condition key -> {rank: [(iteration, observed, network)]}
runs = defaultdict(lambda: defaultdict(list))
meta = {}

for path in glob.glob(os.path.join(ROOT, "**", "results", "*.csv"), recursive=True):
    tag = ("A" if "-A-" in path else "B" if "-B-" in path
           else "C" if "-C-" in path else "H")
    for row in csv.DictReader(io.open(openable(path), newline="", encoding="utf-8")):
        if row.get("profile") != "paper":
            continue
        key = (tag, row["model"], row["comm_mode"], int(row["world_size"]),
               row.get("timing_mode", "phase"))
        rank = int(row["rank"])
        net = row.get("network_s", "")
        runs[key][rank].append((
            int(row["iteration"]),
            float(row["observed_step_s"]),
            float(net) if net not in ("", None) else None,
        ))
        meta[key] = row["status"]

print("tag model comm_mode ws timing  step_ms  net_ms  n_ranks status")
summary = {}
for key in sorted(runs):
    ranks = runs[key]
    n_iters = min(len(v) for v in ranks.values())
    by_iter = defaultdict(list)
    nets = []
    for rank, samples in ranks.items():
        for iteration, obs, net in samples:
            by_iter[iteration].append(obs)
            if net is not None:
                nets.append(net)
    rank_max = [max(v) for v in by_iter.values()]
    step_ms = st.mean(rank_max) * 1e3
    step_p50_ms = st.median(rank_max) * 1e3
    net_ms = st.mean(nets) * 1e3 if nets else float("nan")
    summary[key] = (step_ms, net_ms, step_p50_ms)
    tag, model, mode, ws, timing = key
    print(f"{tag} {model:11s} {mode:20s} {ws:2d} {timing:5s} {step_ms:9.2f} {net_ms:8.2f} p50={step_p50_ms:9.2f} {len(ranks):3d} {meta[key]}")

print()
print("== collective-shape control (phase mode, network_s means, ms) ==")
for tag in ("A", "B"):
    for model in ("resnet50", "transformer"):
        for ws in (8, 16, 32):
            row = []
            for mode in ("native", "gpu-quant", "gpu-quant-reducible"):
                v = summary.get((tag, model, mode, ws, "phase"))
                row.append(f"{mode}={v[1]:.2f}" if v else f"{mode}=-")
            if any("=-" not in r for r in row):
                print(f"{tag} {model:11s} ws{ws:2d}: " + "  ".join(row))

print()
print("== instrumentation overhead: phase step vs wall step (ms, ratio) ==")
for tag in ("A", "C", "H"):
    for model in ("resnet50", "transformer"):
        for ws in (8, 32):
            for mode in ("native", "gpu-quant", "gpu-topk", "powersgd"):
                p = summary.get((tag, model, mode, ws, "phase"))
                w = summary.get((tag, model, mode, ws, "wall"))
                if p and w:
                    print(f"{tag} {model:11s} ws{ws:2d} {mode:9s} phase={p[0]:9.2f} wall={w[0]:9.2f} ratio={p[0]/w[0]:5.2f}")

print()
print("== local-EF vs legacy Top-k (phase, step ms) ==")
for tag in ("A", "C", "B"):
    for model in ("resnet50", "transformer"):
        for ws in (8, 32):
            a = summary.get((tag, model, "gpu-topk", ws, "phase"))
            b = summary.get((tag, model, "gpu-topk-localef", ws, "phase"))
            if a and b:
                print(f"{tag} {model:11s} ws{ws:2d} legacy={a[0]:9.2f} localEF={b[0]:9.2f} delta={100*(b[0]-a[0])/a[0]:+6.1f}%")

print()
print("== vs July campaign (p50 of rank-max, ms; July = intel_step_scaling.csv) ==")
TAGS_J = ("A", "B", "C")
JULY = {
    ("resnet50", "native", 8): 902.5, ("resnet50", "gpu-quant", 8): 910.5,
    ("resnet50", "gpu-topk", 8): 1177.6, ("resnet50", "powersgd", 8): 857.0,
    ("resnet50", "ddp-overlap", 8): 905.0,
    ("resnet50", "native", 32): 1026.3, ("resnet50", "gpu-quant", 32): 2041.9,
    ("resnet50", "gpu-topk", 32): 1321.4, ("resnet50", "powersgd", 32): 884.8,
    ("resnet50", "ddp-overlap", 32): 986.3,
    ("transformer", "native", 8): 45.8, ("transformer", "gpu-quant", 8): 48.3,
    ("transformer", "gpu-topk", 8): 226.6, ("transformer", "powersgd", 8): 15.8,
    ("transformer", "ddp-overlap", 8): 45.7,
    ("transformer", "native", 32): 107.4, ("transformer", "gpu-quant", 32): 536.0,
    ("transformer", "gpu-topk", 32): 291.5, ("transformer", "powersgd", 32): 21.6,
    ("transformer", "ddp-overlap", 32): 80.2,
}
for tag in TAGS_J:
    for (model, mode, ws), july in sorted(JULY.items()):
        v = summary.get((tag, model, mode, ws, "phase"))
        if v:
            print(f"{tag} {model:11s} ws{ws:2d} {mode:12s} july={july:9.2f} {tag}={v[2]:9.2f} delta={100*(v[2]-july)/july:+6.1f}%")

print()
print("== allocation repeat A vs B (phase, step ms) ==")
for model in ("resnet50", "transformer"):
    for ws in (8, 32):
        for mode in ("native", "gpu-quant", "gpu-topk", "powersgd", "ddp-overlap", "gpu-quant-reducible"):
            a = summary.get(("A", model, mode, ws, "phase"))
            b = summary.get(("B", model, mode, ws, "phase"))
            if a and b:
                print(f"{model:11s} ws{ws:2d} {mode:20s} A={a[0]:9.2f} B={b[0]:9.2f} delta={100*(b[0]-a[0])/a[0]:+6.1f}%")
