# -*- coding: utf-8 -*-
"""Diagnostic features from instrumented (Tier 1) records.

One row per (tag, model, comm_mode, world_size). Features are the inputs the
frozen rules read; nothing here looks at intervention outcomes.

  exchange_share      network_s / observed_step_s              (pooled rank-iteration means)
  codec_share         (serialize_s + deserialize_s) / observed_step_s
  barrier_share       barrier_wait_s / observed_step_s
  shape_ratio         measured exchange time / calibrated all-reduce time for the
                      same submitted bytes: alpha_AR + beta_AR * bytes. Values well
                      above 1 mean the exchange costs more than a reduce-shaped
                      exchange of the same payload would (gather amplification,
                      contention, or serialization inside the exchange).
  volume_ratio        (p-1)/c : 2(p-1)/p, the Eq. 1 received-traffic ratio for a
                      gather-shaped exchange at this p and compression ratio c
                      (1.0 for reduce-shaped modes).

Records with network_s_is_estimate or without a separable network phase
(DDP hooks, wall mode) produce no shape features and are marked.
"""
import csv
import glob
import io
import json
import os
import statistics as st
import sys
from collections import defaultdict

GATHER_MODES = {"gpu-quant", "cpu-quant", "gpu-topk", "gpu-topk-localef", "ddp-int8-hook", "gpu-quant-gathersum"}
REDUCE_MODES = {"native", "fp16-native", "gpu-quant-reducible", "ddp-fp16-hook", "powersgd", "ddp-overlap"}


def openable(path):
    if os.name == "nt":
        path = os.path.abspath(path).replace("/", "\\")
        if not path.startswith("\\\\?\\"):
            path = "\\\\?\\" + path
    return path


def tag_of(path):
    # R = round-4 held-out runs (Lonestar6, r4-* run directories). Checked
    # first because their paths also contain "ls6".
    if "r4-" in path:
        return "R"
    for t in ("-A-", "-B-", "-C-", "-V-"):
        if t in path:
            return t.strip("-")
    if "ls6-" in path:
        return "L"
    return "H"


def load(root, exclude_tags=()):
    acc = defaultdict(lambda: defaultdict(list))
    run_calibration = {}
    paths = glob.glob(os.path.join(root, "**", "results", "*.csv"), recursive=True)
    # pass 1: the calibration source of each run directory (the only condition
    # whose rows carry alpha/beta) supplies the allocation's calibration
    for path in paths:
        run_dir = os.path.dirname(os.path.dirname(path))
        if run_dir in run_calibration:
            continue
        for row in csv.DictReader(io.open(openable(path), newline="", encoding="utf-8")):
            a, b = row.get("alpha_allreduce_s", ""), row.get("beta_allreduce_s_per_byte", "")
            if a not in ("", None) and b not in ("", None):
                run_calibration[run_dir] = (float(a), float(b))
            break
    for path in paths:
        tag = tag_of(path)
        if tag in exclude_tags:
            continue
        run_dir = os.path.dirname(os.path.dirname(path))
        cal = run_calibration.get(run_dir)
        for row in csv.DictReader(io.open(openable(path), newline="", encoding="utf-8")):
            if row.get("profile") not in ("paper", "extension"):
                continue
            timing = row.get("timing_mode", "phase")
            tau = int(row.get("sync_every", 1))
            barrier = row.get("tau_barrier", "every_iteration")
            alloc = os.path.basename(run_dir).rsplit("-", 1)[-1]
            key = (tag, row["model"], row["comm_mode"], int(row["world_size"]), timing, tau, barrier, alloc)
            d = acc[key]
            for f in ("forward_s", "barrier_wait_s", "backward_compute_s", "serialize_s",
                      "network_s", "deserialize_s", "optimizer_step_s", "observed_step_s",
                      "alpha_allreduce_s", "beta_allreduce_s_per_byte",
                      "collective_input_bytes_per_rank", "bytes_uncompressed"):
                v = row.get(f, "")
                if v not in ("", None):
                    d[f].append(float(v))
            d["network_is_estimate"].append(row.get("network_s_is_estimate") == "True")
            if cal and not d["alpha_allreduce_s"]:
                d["alpha_allreduce_s"].append(cal[0]); d["beta_allreduce_s_per_byte"].append(cal[1])
            d["_iters"].append((int(row["rank"]), int(row["iteration"]), float(row["observed_step_s"])))
    return acc


def features(acc):
    out = []
    for key in sorted(acc):
        d = acc[key]
        tag, model, mode, ws, timing, tau, barrier, alloc = key
        by_iter = defaultdict(list)
        for r, i, obs in d["_iters"]:
            by_iter[i].append(obs)
        step = st.mean(max(v) for v in by_iter.values())
        mean = lambda f: st.mean(d[f]) if d[f] else float("nan")
        net = mean("network_s")
        codec = (mean("serialize_s") if d["serialize_s"] else 0.0) + (mean("deserialize_s") if d["deserialize_s"] else 0.0)
        bar = mean("barrier_wait_s") if d["barrier_wait_s"] else float("nan")
        estimate = any(d["network_is_estimate"])
        separable = bool(d["network_s"]) and not estimate
        row = {
            "tag": tag, "model": model, "mode": mode, "p": ws,
            "timing": timing, "tau": tau, "barrier": barrier, "alloc": alloc,
            "step_ms": step * 1e3,
            "exchange_share": (net / step) if separable else None,
            "codec_share": codec / step,
            "barrier_share": (bar / step) if d["barrier_wait_s"] else None,
            "shape_ratio": None, "volume_ratio": None,
            "kind": "gather" if mode in GATHER_MODES else "reduce",
        }
        if separable and d["alpha_allreduce_s"] and d["collective_input_bytes_per_rank"]:
            a, b = mean("alpha_allreduce_s"), mean("beta_allreduce_s_per_byte")
            payload = mean("collective_input_bytes_per_rank")
            calibrated = a + b * payload
            row["shape_ratio"] = net / calibrated if calibrated > 0 else None
        if row["kind"] == "gather" and d["bytes_uncompressed"] and d["collective_input_bytes_per_rank"]:
            c = mean("bytes_uncompressed") / max(1.0, mean("collective_input_bytes_per_rank"))
            row["volume_ratio"] = ((ws - 1) / c) / (2.0 * (ws - 1) / ws)
        elif row["kind"] == "reduce":
            row["volume_ratio"] = 1.0
        out.append(row)
    return out


if __name__ == "__main__":
    root = sys.argv[1]
    excl = tuple(sys.argv[2].split(",")) if len(sys.argv) > 2 else ()
    rows = features(load(root, excl))
    fmt = lambda v: "   -  " if v is None else f"{v:6.2f}"
    print("tag model       mode                  p  tim  tau bar step_ms  exch  codec  barr  shape  vol   kind")
    for r in rows:
        print(f"{r['tag']} {r['model']:11s} {r['mode']:20s} {r['p']:2d} {r['timing'][:4]:4s} {r['tau']:2d} {r['barrier'][:3]:3s} {r['step_ms']:9.1f} "
              f"{fmt(r['exchange_share'])} {fmt(r['codec_share'])} {fmt(r['barrier_share'])} "
              f"{fmt(r['shape_ratio'])} {fmt(r['volume_ratio'])} {r['kind']}")
    out_name = sys.argv[3] if len(sys.argv) > 3 else "features_dev.json"
    with io.open(os.path.join(os.path.dirname(os.path.abspath(__file__)), out_name), "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=1)
