# validation_v2/analysis/rev4_records.py
# -*- coding: utf-8 -*-
"""Reads the run directories of the Revision 4 jobs: per-rank records, v5.2
sidecars, progress files. Frozen with the E9 and E10 scorers before any job
was submitted."""
import csv
import glob
import io
import json
import os
import re
from collections import defaultdict

RANK = re.compile(r"__rank(\d+)\.")


def num(value):
    return None if value in (None, "", "None") else float(value)


def read_dir(run_dir):
    runs = defaultdict(lambda: {"rows": [], "sidecar": {}, "progress": {}})
    for path in sorted(glob.glob(os.path.join(run_dir, "results", "*.csv"))):
        with io.open(path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                runs[row["run_id"]]["rows"].append(row)
    for path in sorted(glob.glob(os.path.join(run_dir, "results", "*.v52.json"))):
        data = json.load(io.open(path, encoding="utf-8"))
        runs[data["run_id"]]["sidecar"][int(data["rank"])] = data
    for path in sorted(glob.glob(os.path.join(run_dir, "progress", "*.progress.jsonl"))):
        name = os.path.basename(path)
        rank = int(RANK.search(name).group(1))
        runs[name.split("__")[0]]["progress"][rank] = [
            json.loads(line) for line in io.open(path, encoding="utf-8") if line.strip()]
    return dict(runs)


def valid(rows, statuses=("PASS",)):
    return bool(rows) and all(r["final_param_hashes_equal"] == "True" and r["status"] in statuses
                              for r in rows)


def block_step_ms(rows):
    values = {num(r["block_step_equiv_s"]) for r in rows} - {None}
    if len(values) != 1:
        raise ValueError("expected one block_step_equiv_s, found %d" % len(values))
    return 1000.0 * values.pop()


def phase_step_ms(rows):
    by_iteration = defaultdict(list)
    for r in rows:
        by_iteration[int(r["iteration"])].append(num(r["observed_step_s"]))
    return 1000.0 * sum(max(v) for v in by_iteration.values()) / len(by_iteration)


def exchange_ms(rows):
    values = [num(r["network_s"]) for r in rows]
    return 1000.0 * sum(values) / len(values)


def peak_gb(rows):
    return max(num(r["peak_memory_mb"]) for r in rows) / 1000.0


def slope(sizes, times, low=1 << 20, high=64 << 20):
    """Least-squares line through the points with low <= size <= high.
    Returns (seconds per byte, intercept in seconds)."""
    points = [(float(s), float(t)) for s, t in zip(sizes, times) if low <= s <= high]
    if len(points) < 2:
        raise ValueError("need two calibrated points between %d and %d bytes" % (low, high))
    mean_s = sum(s for s, _ in points) / len(points)
    mean_t = sum(t for _, t in points) / len(points)
    b = sum((s - mean_s) * (t - mean_t) for s, t in points) / sum((s - mean_s) ** 2 for s, _ in points)
    return b, mean_t - b * mean_s
