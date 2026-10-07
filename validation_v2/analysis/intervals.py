# validation_v2/analysis/intervals.py
# -*- coding: utf-8 -*-
"""Allocation-level summaries of the paper's primary ratios and gains.
RETROSPECTIVE (2026-10-07). The allocation, one Slurm job, is the unit.
Rule: one allocation, the value; two, both values; three or more, the mean
with a 95% t interval (on the log scale for ratios) and the range."""
import csv
import io
import json
import math
import os
import re
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
V2 = os.path.dirname(HERE)
ROOT = os.path.dirname(V2)
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
       8: 2.306, 9: 2.262, 10: 2.228}
SUMMARY_LINE = re.compile(r"^([ABCV])\s+(\S+)\s+(\S+)\s+(\d+)\s+(phase|wall)\s+(\S+)\s+(\S+)"
                          r"\s+p50=\s*(\S+)\s+(\d+)\s+(\S+)\s*$")


def summarize(values, ratio=True):
    values = [float(v) for v in values]
    n = len(values)
    if n == 0:
        raise ValueError("no allocations")
    out = {"n": n, "values": values, "min": min(values), "max": max(values), "ci": None}
    if ratio:
        if min(values) <= 0:
            raise ValueError("a ratio must be positive")
        xs, back = [math.log(v) for v in values], math.exp
    else:
        xs, back = values, float
    mean = sum(xs) / n
    out["center"] = back(mean)
    if n >= 3:
        sd = math.sqrt(sum((x - mean) ** 2 for x in xs) / (n - 1))
        half = T95[n - 1] * sd / math.sqrt(n)
        out["ci"] = (back(mean - half), back(mean + half))
    return out


def ratio_from_moments(m1, s1, m2, s2, n):
    """Ratio m1/m2 of two arms that each have a mean and a sample standard
    deviation over n allocations and no stored per-allocation values. Delta
    method on the log scale, arms treated as unpaired, n - 1 degrees of
    freedom. Pairing would narrow the interval, so this one is conservative
    when the arms share allocations."""
    center = m1 / m2
    se = math.sqrt(((s1 / m1) ** 2 + (s2 / m2) ** 2) / n)
    half = T95[n - 1] * se
    return {"n": n, "center": center, "ci": (center * math.exp(-half), center * math.exp(half))}


def v2_gains(path, model="transformer"):
    out = defaultdict(list)
    for r in csv.DictReader(io.open(path, newline="", encoding="utf-8")):
        if r["model"] == model:
            out[("g_matched", r["convention"])].append(float(r["g_matched"]))
            out[("h_topk", r["convention"])].append(float(r["h_topk"]))
    return dict(out)


def v2_hook_ratios(path, model="transformer"):
    out = defaultdict(list)
    for r in csv.DictReader(io.open(path, newline="", encoding="utf-8")):
        if r["model"] == model:
            for hook in ("ddp-int8-hook", "ddp-fp16-hook"):
                out[(hook, int(r["batch"]))].append(float(r["R_%s" % hook]))
    return dict(out)


def intel_control(path):
    out = {}
    for line in io.open(path, encoding="utf-8"):
        m = SUMMARY_LINE.match(line.rstrip("\n"))
        if not m or m.group(10) != "PASS":
            continue
        tag, model, mode, ws, timing, step, net = m.group(1, 2, 3, 4, 5, 6, 7)
        out[(tag, model, mode, int(ws), timing)] = {
            "step_ms": float(step), "net_ms": None if net == "nan" else float(net)}
    return out


def intel_exchange_ratio(table, model, numerator, denominator, ranks):
    values = {}
    for (tag, m, mode, ws, timing), row in table.items():
        if (m, mode, ws, timing) != (model, numerator, ranks, "phase"):
            continue
        other = table.get((tag, model, denominator, ranks, "phase"))
        if other and row["net_ms"] and other["net_ms"]:
            values[tag] = row["net_ms"] / other["net_ms"]
    return values


def build():
    a = os.path.join(V2, "results", "analysis_2026-09-29")
    out = {"validation campaign, Transformer, 12 ranks": {}, "H100 overlap control": {},
           "Intel control, exchange ratio": {}}
    for (gain, conv), values in sorted(v2_gains(os.path.join(a, "v2_expA.csv")).items()):
        out["validation campaign, Transformer, 12 ranks"]["%s under %s" % (gain, conv)] = \
            summarize(values, ratio=False)
    for (hook, batch), values in sorted(v2_hook_ratios(os.path.join(a, "v2_expB.csv")).items()):
        out["validation campaign, Transformer, 12 ranks"]["%s over dense DDP, batch %d" % (hook, batch)] = \
            summarize(values)
    for r in csv.DictReader(io.open(os.path.join(ROOT, "processed", "h100_overlap_control.csv"),
                                    newline="", encoding="utf-8")):
        out["H100 overlap control"][r["model"]] = ratio_from_moments(
            float(r["ddp_overlap_step_ms_mean"]), float(r["ddp_overlap_step_ms_sample_sd"]),
            float(r["native_no_overlap_step_ms_mean"]), float(r["native_no_overlap_step_ms_sample_sd"]),
            int(r["independent_job_count"]))
    table = intel_control(os.path.join(ROOT, "processed", "control_campaign_summary.txt"))
    for model in ("transformer", "resnet50"):
        for ranks in (8, 16, 32):
            for num in ("gpu-quant", "gpu-quant-reducible"):
                values = intel_exchange_ratio(table, model, num, "native", ranks)
                if values:
                    out["Intel control, exchange ratio"]["%s %s over native, %d ranks" % (model, num, ranks)] = \
                        dict(summarize(list(values.values())), allocations=sorted(values))
    return out


def render(result):
    lines = ["Allocation-level summaries (RETROSPECTIVE, 2026-10-07).",
             "n = allocations. Interval: 95% t, shown for n >= 3.", ""]
    for group, items in result.items():
        lines.append("## %s" % group)
        lines.append("")
        lines.append("| quantity | n | center | range | 95% interval |")
        lines.append("|---|---|---|---|---|")
        for name, s in items.items():
            rng = "" if "min" not in s else "%.3f to %.3f" % (s["min"], s["max"])
            ci = "" if not s["ci"] else "%.3f to %.3f" % tuple(s["ci"])
            lines.append("| %s | %d | %.3f | %s | %s |" % (name, s["n"], s["center"], rng, ci))
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    res = build()
    target = os.path.join(V2, "results", "rev4")
    os.makedirs(target, exist_ok=True)
    json.dump(res, io.open(os.path.join(target, "INTERVALS.json"), "w", encoding="utf-8"),
              indent=1, sort_keys=True)
    io.open(os.path.join(target, "INTERVALS.md"), "w", encoding="utf-8", newline="\n").write(render(res))
    print(render(res))
