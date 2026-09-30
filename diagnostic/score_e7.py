# -*- coding: utf-8 -*-
"""E7 scorer, pre-registered: committed and pushed before any E7 job was
submitted (push time in provenance/PROVENANCE.md).

Applies the FROZEN rules (rules.py 71329c33) to features extracted by the
E6-scoring extractor (features.py 95e240d8) from the E7 run directories
only, and scores each case against its same-allocation uninstrumented
pair as E7_PROTOCOL.md specifies. Fixed before any E7 measurement:
(1) each allocation's verdict comes from that allocation's own M1;
(2) trend is None (p = 6 is the only scale);
(3) 'inapplicable', 'not tested', 'missing (run failed)' and
    'missing (M1 failed)' labels (E7_PROTOCOL.md section 5).
Usage: python score_e7.py <features.json> <alloc_a,alloc_b> [frozen|described]
"""
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rules  # noqa: E402  frozen
from score_e6 import outcome as e6_outcome  # noqa: E402  frozen outcome rules

P = 6
MODELS = ("transformer", "resnet50")
CASES = [  # suffix, original, intervention, family, mechanisms the intervention tests
    ("1", "gpu-quant-gathersum", "gpu-quant-reducible", "isolated shape", {"shape"}),
    ("2", "gpu-quant", "gpu-quant-reducible", "shape, rank-local substitution", {"shape"}),
    ("3", "gpu-topk", "gpu-quant-reducible", "bundled placement", {"placement", "shape"}),
    ("4", "cpu-quant", "gpu-quant", "isolated placement", {"placement"}),
    ("5", "ddp-int8-hook", "ddp-fp16-hook", "shape under overlap", {"shape"}),
    ("6", "fp16-native", None, "abstention expected", set()),
]
METHODS = [("full", rules.diagnose), ("A1", rules.ablation_payload_only),
           ("A2", rules.ablation_calibration_only), ("A3", rules.ablation_timeline_only)]


def find(rows, model, mode, timing, alloc):
    for r in rows:
        if (r["model"], r["p"], r["mode"], r["timing"], r["tau"], r["barrier"], r["alloc"]) == \
                (model, P, mode, timing, 1, "every_iteration", alloc):
            return r
    return None


def label(verdict, e, delta, kind, tested):
    if verdict in ("none", "mixed", "insufficient"):
        return e6_outcome(verdict, e, delta, None)
    if verdict == "shape" and kind == "reduce":
        return "inapplicable"
    if verdict not in tested:
        return "not tested"
    return e6_outcome(verdict, e, delta, None)


def score(rows, allocs, reading="frozen"):
    transform = None
    if reading == "described":
        sys.path.insert(0, os.path.join(HERE, "v1_1"))
        from rescore import transform  # noqa: E402
    out = []
    for model in MODELS:
        for sfx, orig, intv, fam, tested in CASES:
            case = "E7-%s%s" % (model[0].upper(), sfx)
            for al in allocs:
                m1 = find(rows, model, orig, "phase", al)
                m2 = find(rows, model, orig, "wall", al)
                m3 = find(rows, model, intv, "wall", al) if intv else None
                d = None
                if m2 is not None and m3 is not None and m2["step_ms"] > 0:
                    d = (m2["step_ms"] - m3["step_ms"]) / m2["step_ms"]
                for name, method in METHODS:
                    if m1 is None:
                        out.append((case, al, name, "missing", 0.0, d, "missing (M1 failed)", fam))
                        continue
                    f = transform(m1, reading) if transform else m1
                    v, _named, e = method(f, None) if method is rules.diagnose else method(f)
                    if intv and (m2 is None or m3 is None):
                        lab = "missing (run failed)"
                    else:
                        lab = label(v, e, d, f["kind"], tested)
                    out.append((case, al, name, v, e, d, lab, fam))
    return out


def summary(results):
    allocs = sorted({r[1] for r in results})
    w = 1.0 / max(1, len(allocs))
    keys = ("recommended", "cleared", "false_positive", "not_tested",
            "abstain_correct", "abstain_miss", "missing")
    s = {name: dict.fromkeys(keys, 0.0) for name, _ in METHODS}
    for case, al, name, v, e, d, lab, fam in results:
        t = s[name]
        if lab.startswith("missing"):
            t["missing"] += w
            continue
        if v not in ("none", "mixed", "insufficient"):
            t["recommended"] += w
        if lab in ("correct", "short"):
            t["cleared"] += w
        elif lab in ("null", "harmful", "inapplicable"):
            t["false_positive"] += w
        elif lab == "not tested":
            t["not_tested"] += w
        elif lab == "abstain-correct":
            t["abstain_correct"] += w
        elif lab == "abstain-miss":
            t["abstain_miss"] += w
    return s


if __name__ == "__main__":
    rows = json.load(io.open(sys.argv[1], encoding="utf-8"))
    allocs = sys.argv[2].split(",")
    reading = sys.argv[3] if len(sys.argv) > 3 else "frozen"
    res = score(rows, allocs, reading)
    print("E7 scoring, reading %s, allocations %s" % (reading, ",".join(allocs)))
    for r in res:
        d = "   -  " if r[5] is None else "%+6.3f" % r[5]
        print("%-6s %-10s %-4s %-13s e=%5.3f delta=%s %s [%s]" % (r[0], r[1], r[2], r[3], r[4], d, r[6], r[7]))
    for name, t in summary(res).items():
        print("%-4s " % name + ", ".join("%s %g" % kv for kv in t.items()))
