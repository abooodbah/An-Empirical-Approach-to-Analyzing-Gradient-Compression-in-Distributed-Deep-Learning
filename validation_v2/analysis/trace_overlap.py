# -*- coding: utf-8 -*-
"""Overlap of NCCL kernels with compute kernels in a PyTorch (Kineto) trace:
union of NCCL kernel intervals, union of other kernel intervals, and their
intersection. Traces are qualification runs, never timing evidence.
Usage: python trace_overlap.py <out.csv> <trace.json>..."""
import csv
import io
import json
import os
import sys


def _union(iv):
    iv = sorted(iv)
    out = []
    for a, b in iv:
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def _measure(iv):
    return sum(b - a for a, b in iv)


def _intersect(x, y):
    i = j = 0
    out = []
    while i < len(x) and j < len(y):
        a, b = max(x[i][0], y[j][0]), min(x[i][1], y[j][1])
        if a < b:
            out.append([a, b])
        if x[i][1] < y[j][1]:
            i += 1
        else:
            j += 1
    return out


def overlap(path):
    events = json.load(io.open(path, encoding="utf-8")).get("traceEvents", [])
    kern = [e for e in events if e.get("ph") == "X" and e.get("cat") == "kernel"]
    nccl = _union([(e["ts"], e["ts"] + e["dur"]) for e in kern if "nccl" in e["name"].lower()])
    comp = _union([(e["ts"], e["ts"] + e["dur"]) for e in kern if "nccl" not in e["name"].lower()])
    both = _measure(_intersect(nccl, comp))
    n = _measure(nccl)
    return {"nccl_ms": n / 1000.0, "compute_ms": _measure(comp) / 1000.0,
            "overlapped_ms": both / 1000.0, "exposed_ms": (n - both) / 1000.0,
            "overlap_frac": (both / n) if n else 0.0}


if __name__ == "__main__":
    with open(sys.argv[1], "w", newline="") as fh:
        w = None
        for path in sys.argv[2:]:
            o = dict(file=os.path.basename(path), **overlap(path))
            if w is None:
                w = csv.DictWriter(fh, fieldnames=list(o))
                w.writeheader()
            w.writerow(o)
    print("wrote", sys.argv[1])
