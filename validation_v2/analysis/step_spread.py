# -*- coding: utf-8 -*-
"""Job-to-job spread of the observed step behind the phase-attribution
figure of the H100 2x4 system. RETROSPECTIVE (2026-10-08): the figure draws
the mean of three jobs; this states how far the three jobs are apart.

For each condition, a job's value is the mean observed_step_s over its ranks
and measured iterations; the spread is the range of the three job values as
a percentage of their mean.

usage: python step_spread.py [folder holding h100-3328658, h100-3328659, h100-3328660]
       (default: $REV4_JULY_ROOT, then C:/r5w/july; extract raw/july/*.tar.gz there)
writes validation_v2/results/rev4/H100_STEP_SPREAD.json
"""
import csv
import glob
import io
import json
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
JOBS = (3328658, 3328659, 3328660)
FIGURE_CONDITIONS = [(model, mode, 1) for model in ("resnet50", "transformer")
                     for mode in ("native", "gpu-quant", "gpu-topk", "cpu-quant", "ddp-overlap")]


def job_means(job_dir):
    """Mean observed step in seconds per (model, comm_mode, sync_every)."""
    values = {}
    for path in glob.glob(os.path.join(job_dir, "results", "*.csv")):
        with io.open(path, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                key = (r["model"], r["comm_mode"], int(r["sync_every"]))
                values.setdefault(key, []).append(float(r["observed_step_s"]))
    return {key: st.mean(v) for key, v in values.items()}


def spread(job_dirs, conditions):
    per_job = [job_means(d) for d in job_dirs]
    out = {"jobs": [os.path.basename(os.path.normpath(d)) for d in job_dirs], "conditions": {}}
    for key in conditions:
        ms = [1000.0 * means[key] for means in per_job]      # KeyError if a job lacks the condition
        mean = st.mean(ms)
        out["conditions"]["%s|%s|%d" % key] = {
            "job_means_ms": ms, "mean_ms": mean, "min_ms": min(ms), "max_ms": max(ms),
            "spread_pct": 100.0 * (max(ms) - min(ms)) / mean}
    name, row = max(out["conditions"].items(), key=lambda kv: kv[1]["spread_pct"])
    out["largest"] = {"condition": name, "spread_pct": row["spread_pct"]}
    return out


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("REV4_JULY_ROOT", "C:/r5w/july")
    result = spread([os.path.join(root, "h100-%d" % j) for j in JOBS], FIGURE_CONDITIONS)
    target = os.path.join(os.path.dirname(HERE), "results", "rev4", "H100_STEP_SPREAD.json")
    json.dump(result, io.open(target, "w", encoding="utf-8"), indent=1, sort_keys=True)
    for name, row in result["conditions"].items():
        print("%-28s %8.2f ms, jobs %8.2f to %8.2f, spread %4.1f%%"
              % (name, row["mean_ms"], row["min_ms"], row["max_ms"], row["spread_pct"]))
    print("largest: %s, %.1f%%" % (result["largest"]["condition"], result["largest"]["spread_pct"]))
