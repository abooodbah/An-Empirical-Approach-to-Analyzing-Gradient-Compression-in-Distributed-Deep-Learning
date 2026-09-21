# -*- coding: utf-8 -*-
"""E6 scorer. Applies the FROZEN rules (rules.py v1) to the held-out feature
table and scores each manifest case against its paired uninstrumented
measurements, exactly as E6_PROTOCOL.md specifies. Prints the per-case
table first; counts second. Also scores the three ablations identically.

Usage: python score_e6.py features_r4.json

The manifest below binds each case to (original condition, intervention
condition). Both must exist in wall mode in the same run directory family
(same job) for the case to be scored; otherwise it is reported unscored.
Nothing in this file adjusts a threshold.
"""
import io
import json
import sys

from rules import (RULES_VERSION, diagnose, ablation_payload_only,
                   ablation_calibration_only, ablation_timeline_only)

# (case, model, p, original mode, intervention mode, tau_orig, barrier_orig, tau_int, barrier_int, family)
MANIFEST = [
    (1, "transformer", 24, "gpu-quant", "gpu-quant-reducible", 1, "every_iteration", 1, "every_iteration", "shape, rank-local substitution"),
    (2, "resnet50", 24, "gpu-quant", "gpu-quant-reducible", 1, "every_iteration", 1, "every_iteration", "shape, rank-local substitution"),
    (3, "transformer", 12, "gpu-quant-gathersum", "gpu-quant-reducible", 1, "every_iteration", 1, "every_iteration", "isolated shape"),
    (4, "resnet50", 24, "gpu-quant-gathersum", "gpu-quant-reducible", 1, "every_iteration", 1, "every_iteration", "isolated shape"),
    (5, "transformer", 12, "gpu-topk", "gpu-quant-reducible", 1, "every_iteration", 1, "every_iteration", "bundled placement"),
    (6, "resnet50", 24, "gpu-topk", "gpu-quant-reducible", 1, "every_iteration", 1, "every_iteration", "bundled placement"),
    (7, "transformer", 24, "ddp-int8-hook", "ddp-fp16-hook", 1, "every_iteration", 1, "every_iteration", "shape under overlap"),
    (8, "resnet50", 12, "ddp-int8-hook", "ddp-fp16-hook", 1, "every_iteration", 1, "every_iteration", "shape under overlap"),
    (9, "transformer", 12, "native", "native", 4, "every_iteration", 4, "none_on_local_iterations", "coordination"),
    (10, "resnet50", 12, "native", "native", 4, "every_iteration", 4, "none_on_local_iterations", "coordination"),
    (11, "transformer", 24, "fp16-native", None, 1, "every_iteration", None, None, "abstention expected"),
    (12, "resnet50", 12, "cpu-quant", "gpu-quant-reducible", 1, "every_iteration", 1, "every_iteration", "bundled placement"),
]
# Note on cases 5, 6, 12: the manifest in E6_PROTOCOL.md names a TopKC-style
# port as the intervention for Top-k; that port is not implemented in v4, so
# the intervention actually applied is the reduce-shaped GPU int8 exchange
# (a cheaper encoder AND a reduce-shaped collective: bundled, labeled so).
# Note on cases 1, 2: gpu-quant encodes with a per-rank scale and dequantizes
# p payloads; gpu-quant-reducible encodes with one shared scale and
# dequantizes once. The substitution therefore changes the scale sharing and
# the decode count together with the collective; only cases 3 and 4
# (gpu-quant-gathersum, identical encoding, arms verified bit-equal) isolate
# the collective. The family label records this; scoring is unchanged.


def find(rows, model, p, mode, timing, tau, barrier, alloc=None):
    for r in rows:
        if (r["model"], r["p"], r["mode"], r["timing"], r["tau"], r["barrier"]) == (model, p, mode, timing, tau, barrier)                 and (alloc is None or r.get("alloc") == alloc):
            return r
    return None


def allocs_with_pair(rows, model, p, orig, intv, tau_o, bar_o, tau_i, bar_i):
    """Allocations (job ids) that contain the M2 and, if named, the M3 run."""
    out = []
    for al in sorted({r.get("alloc") for r in rows}):
        m2 = find(rows, model, p, orig, "wall", tau_o, bar_o, al)
        m3 = find(rows, model, p, intv, "wall", tau_i, bar_i, al) if intv else True
        if m2 is not None and m3:
            out.append(al)
    return out


def trend_for(rows, r):
    """exchange-share growth from the smallest p measured for this mode/model (phase mode)."""
    cands = [x for x in rows if x["model"] == r["model"] and x["mode"] == r["mode"] and x["timing"] == "phase"
             and x["tau"] == r["tau"] and x["barrier"] == r["barrier"] and x["exchange_share"] is not None]
    if not cands:
        return None
    base = min(cands, key=lambda x: x["p"])
    if base["p"] == r["p"] or r["exchange_share"] is None:
        return None
    return r["exchange_share"] - base["exchange_share"]


def outcome(verdict, expected, delta, intervention_named):
    if verdict in ("none", "mixed", "insufficient"):
        # abstention: correct if no tried intervention paid
        if delta is None:
            return "abstain (no intervention run)"
        return "abstain-correct" if delta < 0.05 else "abstain-miss"
    if delta is None:
        return "unscored (no paired wall run)"
    if delta <= -0.05:
        return "harmful"
    if delta >= max(0.05, 0.5 * expected):
        return "correct"
    if delta >= 0.05:
        # the named intervention paid, but by less than half the predicted
        # effect: a magnitude shortfall, distinct from a null effect
        return "short"
    return "null"


def score(rows, method_name, method):
    print(f"\n=== {method_name} ===")
    print("case model       p  original             -> intervention        verdict       exp    T_orig^U  T_int^U   delta   outcome")
    counts = {}
    for (c, model, p, orig, intv, tau_o, bar_o, tau_i, bar_i, fam) in MANIFEST:
        m1 = find(rows, model, p, orig, "phase", tau_o, bar_o)
        if m1 is None:
            print(f"{c:4d} {model:11s} {p:2d} {orig:20s} -> {str(intv):20s} (M1 missing)")
            counts["unscored"] = counts.get("unscored", 0) + 1
            continue
        if method is diagnose:
            v, named, e = method(m1, trend_for(rows, m1))
        else:
            v, named, e = method(m1)
        # one line per allocation that holds the paired M2/M3 runs
        allocs = allocs_with_pair(rows, model, p, orig, intv, tau_o, bar_o, tau_i, bar_i) or [None]
        for al in allocs:
            m2 = find(rows, model, p, orig, "wall", tau_o, bar_o, al)
            m3 = find(rows, model, p, intv, "wall", tau_i, bar_i, al) if intv else None
            delta = None
            if m2 is not None and m3 is not None and m2["step_ms"] > 0:
                delta = (m2["step_ms"] - m3["step_ms"]) / m2["step_ms"]
            oc = outcome(v, e, delta, named)
            key = oc if len(allocs) == 1 else oc + " (per-alloc)"
            counts[oc] = counts.get(oc, 0) + (1.0 / len(allocs))
            t2 = f"{m2['step_ms']:8.1f}" if m2 else "    -   "
            t3 = f"{m3['step_ms']:8.1f}" if m3 else "    -   "
            dl = f"{delta:+6.2f}" if delta is not None else "   -  "
            print(f"{c:4d} {model:11s} {p:2d} {orig:20s} -> {str(intv):20s} {v:13s} {e:5.2f} {t2} {t3} {dl}   {oc}  [{fam}] alloc={al}")
    print("counts (cases split equally across their allocations):", json.dumps({k: round(v, 2) for k, v in counts.items()}))
    return counts


if __name__ == "__main__":
    rows = json.load(io.open(sys.argv[1], encoding="utf-8"))
    rows = [r for r in rows if r["tag"] == "R"]
    print(f"rules {RULES_VERSION}; held-out rows: {len(rows)}")
    score(rows, "full procedure (rules.py)", diagnose)
    score(rows, "ablation A1 payload-only", ablation_payload_only)
    score(rows, "ablation A2 calibration-only", ablation_calibration_only)
    score(rows, "ablation A3 timeline-only", ablation_timeline_only)
