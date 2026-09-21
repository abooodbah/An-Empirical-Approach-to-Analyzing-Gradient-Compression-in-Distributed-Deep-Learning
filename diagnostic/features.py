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

GATHER_MODES = {"gpu-quant", "cpu-quant", "gpu-topk", "gpu-topk-localef", "ddp-int8-hook"}
REDUCE_MODES = {"native", "gpu-quant-reducible", "ddp-fp16-hook", "powersgd", "ddp-overlap"}


def openable(path):
    if os.name == "nt":
        path = os.path.abspath(path).replace("/", "\\")
        if not path.startswith("\\\\?\\"):
            path = "\\\\?\\" + path
    return path


def tag_of(path):
    for t in ("-A-", "-B-", "-C-", "-V-"):
        if t in path:
            return t.strip("-")
    if "ls6-" in path:
        return "L"
    return "H"


def load(root, exclude_tags=()):
    acc = defaultdict(lambda: defaultdict(list))
    for path in glob.glob(os.path.join(root, "**", "results", "*.csv"), recursive=True):
        tag = tag_of(path)
        if tag in exclude_tags:
            continue
        for row in csv.DictReader(io.open(openable(path), newline="", encoding="utf-8")):
            if row.get("profile") not in ("paper", "extension"):
                continue
            if row.get("timing_mode", "phase") != "phase":
                continue
            key = (tag, row["model"], row["comm_mode"], int(row["world_size"]))
            d = acc[key]
            for f in ("forward_s", "barrier_wait_s", "backward_compute_s", "serialize_s",
                      "network_s", "deserialize_s", "optimizer_step_s", "observed_step_s",
                      "alpha_allreduce_s", "beta_allreduce_s_per_byte",
                      "collective_input_bytes_per_rank", "bytes_uncompressed"):
                v = row.get(f, "")
                if v not in ("", None):
                    d[f].append(float(v))
            d["network_is_estimate"].append(row.get("network_s_is_estimate") == "True")
            d["_iters"].append((int(row["rank"]), int(row["iteration"]), float(row["observed_step_s"])))
    return acc


def features(acc):
    out = []
    for key in sorted(acc):
        d = acc[key]
        tag, model, mode, ws = key
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
    print("tag model       mode                  p   step_ms  exch  codec  barr  shape  vol   kind")
    for r in rows:
        print(f"{r['tag']} {r['model']:11s} {r['mode']:20s} {r['p']:2d} {r['step_ms']:9.1f} "
              f"{fmt(r['exchange_share'])} {fmt(r['codec_share'])} {fmt(r['barrier_share'])} "
              f"{fmt(r['shape_ratio'])} {fmt(r['volume_ratio'])} {r['kind']}")
    with io.open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "features_dev.json"), "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=1)
