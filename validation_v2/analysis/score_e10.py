# validation_v2/analysis/score_e10.py
# -*- coding: utf-8 -*-
"""E10 scorer (diagnostic/E10_PROTOCOL.md). Frozen before submission.

usage: python score_e10.py <run_dir> [<run_dir> ...] --json OUT.json
"""
import glob
import io
import json
import os
import re
import sys

import rev4_records as R

RUN = re.compile(r"^redE10-ws(\d+)-llama1b-(.+)$")
BUCKET = 16777216
WIRE = {"native-bucketed": ("allreduce", 4), "gpu-quant-reducible-bucketed": ("allreduce", 1),
        "gpu-quant-bucketed": ("allgather", 1)}


def condition(run, world):
    rows = run["rows"]
    if rows:
        ok = R.valid(rows, statuses=("PASS", "QUALIFICATION_ONLY"))
        return {"state": "complete" if ok else "invalid", "iterations": len({r["iteration"] for r in rows}),
                "step_ms": R.phase_step_ms(rows), "exchange_ms": R.exchange_ms(rows), "peak_gb": R.peak_gb(rows)}
    progress = run["progress"]
    if len(progress) == world and min(len(v) for v in progress.values()) >= 5:
        n = min(len(v) for v in progress.values())
        steps = [max(progress[r][i]["observed_step_s"] for r in progress) for i in range(n)]
        nets = [progress[r][i]["network_s"] for r in progress for i in range(n)]
        peaks = [progress[r][i]["peak_memory_mb"] for r in progress for i in range(n)
                 if progress[r][i]["peak_memory_mb"] is not None]
        return {"state": "incomplete", "iterations": n, "step_ms": 1000.0 * sum(steps) / n,
                "exchange_ms": 1000.0 * sum(nets) / len(nets),
                "peak_gb": max(peaks) / 1000.0 if peaks else None}
    return {"state": "missing"}


def predicted_exchange_ms(intercept_s, slope_s_per_byte, elements, wire_bytes, bucket=BUCKET):
    total = 0.0
    for start in range(0, elements, bucket):
        total += intercept_s + slope_s_per_byte * min(bucket, elements - start) * wire_bytes
    return 1000.0 * total


def first_error(run_dir, condition_name):
    for path in sorted(glob.glob(os.path.join(run_dir, "logs", condition_name + ".*.err"))):
        for line in io.open(path, encoding="utf-8", errors="replace"):
            if any(word in line for word in ("Error", "error:", "fault", "CANCELLED", "Traceback")):
                if "hmem" not in line and "ZE driver" not in line:
                    return "%s: %s" % (os.path.basename(path), line.strip()[:200])
    return None


def allocation(run_dir):
    runs = R.read_dir(run_dir)
    probe_path = os.path.join(run_dir, "probes", "default.json")
    probe = json.load(io.open(probe_path, encoding="utf-8")) if os.path.exists(probe_path) else None
    out = {"run_dir": os.path.basename(os.path.normpath(run_dir)), "conditions": {}, "ratios": {}, "checks": {}}
    for run_id, run in sorted(runs.items()):
        m = RUN.match(run_id)
        if not m or "smoke" in run_id:
            continue
        world, mode = int(m.group(1)), m.group(2)
        c = condition(run, world)
        side = run["sidecar"].get(0) or {}
        c["exchange_buffer_gb"] = (side.get("exchange_buffer_bytes") or 0) / 1e9 if side else None
        c["bucket_count"] = side.get("bucket_count")
        elements = side.get("gradient_elements")
        c["gradient_elements"] = elements
        if c["state"] == "missing":
            c["first_error"] = first_error(run_dir, "llama-ws%d-%s" % (world, mode))
        if probe and elements and mode in WIRE and c["state"] in ("complete", "incomplete"):
            collective, wire = WIRE[mode]
            b, a = R.slope(probe[collective]["sizes_bytes"], probe[collective]["times_s"])
            c["predicted_exchange_ms"] = predicted_exchange_ms(a, b, elements, wire)
            c["prediction_error"] = abs(c["predicted_exchange_ms"] - c["exchange_ms"]) / c["exchange_ms"]
        out["conditions"]["%d|%s" % (world, mode)] = c
    usable = {k: v for k, v in out["conditions"].items() if v["state"] in ("complete", "incomplete")}
    for key, gather in usable.items():
        world, mode = key.split("|")
        if mode != "gpu-quant-bucketed":
            continue
        p = int(world)
        for other, model in (("native-bucketed", p / 8.0), ("gpu-quant-reducible-bucketed", p / 2.0)):
            ref = usable.get("%s|%s" % (world, other))
            if ref:
                out["ratios"]["%s|gather over %s" % (world, other)] = {
                    "exchange": gather["exchange_ms"] / ref["exchange_ms"],
                    "step": gather["step_ms"] / ref["step_ms"], "traffic_model": model}
        whole = usable.get("%s|gpu-quant" % world)
        if whole:
            out["checks"]["%s|bucketing cost, gather" % world] = gather["exchange_ms"] / whole["exchange_ms"]
        dense = usable.get("%s|native-bucketed" % world)
        if dense and gather["peak_gb"] is not None and dense["peak_gb"] is not None:
            # beyond the dense arm the gather holds its encoded copy (1 byte per
            # element), its FP32 result (4 bytes), and the bucket buffers
            account = None
            if gather.get("gradient_elements") and gather["exchange_buffer_gb"] is not None:
                account = 5.0 * gather["gradient_elements"] / 1e9 + gather["exchange_buffer_gb"]
            out["checks"]["%s|memory" % world] = {
                "extra_gb": gather["peak_gb"] - dense["peak_gb"], "peak_gb": gather["peak_gb"],
                "account_gb": account, "below_45_gb": gather["peak_gb"] < 45.0}
    for world in ("16",):
        flat, bucketed = usable.get("%s|native" % world), usable.get("%s|native-bucketed" % world)
        if flat and bucketed:
            out["checks"]["%s|bucketing cost, dense" % world] = bucketed["exchange_ms"] / flat["exchange_ms"]
    return out


def score(run_dirs):
    return {"allocations": [allocation(d) for d in run_dirs]}


def main(argv):
    out = argv[argv.index("--json") + 1] if "--json" in argv else None
    dirs = [a for a in argv if not a.startswith("--") and a != out]
    result = score(dirs)
    if out:
        json.dump(result, io.open(out, "w", encoding="utf-8"), indent=1, sort_keys=True)
    for a in result["allocations"]:
        print("== %s" % a["run_dir"])
        for key, c in a["conditions"].items():
            print("%-40s %s" % (key, json.dumps({k: (round(v, 3) if isinstance(v, float) else v)
                                                 for k, v in c.items()}, sort_keys=True)))
        for key, r in a["ratios"].items():
            print("%-40s %s" % (key, json.dumps({k: round(v, 3) for k, v in r.items()}, sort_keys=True)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
