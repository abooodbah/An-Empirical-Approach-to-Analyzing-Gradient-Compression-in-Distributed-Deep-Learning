# -*- coding: utf-8 -*-
"""Rescoring of the held-out E6 evaluation under documented readings.

RETROSPECTIVE (written 2026-09-29, after the E6 outcomes were known).
The frozen files (../rules.py, ../features.py at the E6 version,
../score_e6.py) are imported unchanged; each reading transforms a copy
of the frozen feature rows or relabels an outcome, and says why.

Readings of the feature rows:
  frozen                 the rows as scored in E6_FINAL.txt
  described              C1 + C2. C1: hooks without exchange or codec phases
                         get codec_share None (feature unavailable), so the
                         full procedure returns 'insufficient'. C2: int8
                         gathers without recorded payload bytes get
                         r_V = p / (2c), c = 4, as Section VII-A words A1.
  nominal_frozen_hooks   C3 as the paper applied it: r_V = p / 8 for the int8
                         gathers with recorded bytes; hooks as frozen
  nominal                C1 + C2 + C3: r_V = p / 8 for every int8 gather
  as_frozen_extractor    the classification and calibration of the extractor
                         hashed in RULES_FROZEN.sha256: the matched gather arm
                         reads reduce-shaped (r_V 1.0) and no allocation
                         calibration reaches the rows (shape_ratio None)
Options:
  e_same   C4: shape effect against the named intervention (a reduce of the
           same encoding, byte ratio p / 2): exchange_share * (1 - 2 / p)
  c5       C5: 'inapplicable' for a shape verdict on an exchange that is
           already reduce-shaped; 'not tested' when the verdict's mechanism is
           not among those the case's pre-registered intervention tests
"""
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DIAG = os.path.dirname(HERE)
if DIAG not in sys.path:
    sys.path.insert(0, DIAG)
import rules  # noqa: E402  frozen
import score_e6 as S  # noqa: E402  frozen

INT8_GATHERS = ("gpu-quant", "cpu-quant", "gpu-quant-gathersum", "ddp-int8-hook")
HOOKS = ("ddp-int8-hook", "ddp-fp16-hook")
# Mechanisms each case's pre-registered intervention tests (E6_PROTOCOL.md).
TESTS = {1: {"shape"}, 2: {"shape"}, 3: {"shape"}, 4: {"shape"},
         5: {"placement", "shape"}, 6: {"placement", "shape"},
         7: {"shape"}, 8: {"shape"}, 9: {"coordination"},
         10: {"coordination"}, 11: set(), 12: {"placement", "shape"}}
READINGS = ("frozen", "described", "nominal_frozen_hooks", "nominal",
            "as_frozen_extractor")
METHODS = {"full": rules.diagnose, "A1": rules.ablation_payload_only,
           "A2": rules.ablation_calibration_only,
           "A3": rules.ablation_timeline_only}
ORDER = ("correct", "short", "null", "harmful", "inapplicable", "not tested",
         "abstain-miss", "abstain-correct", "abstain (no intervention run)",
         "unscored (no paired wall run)")


def load_rows(path=None):
    path = path or os.path.join(DIAG, "features_r4.json")
    with io.open(path, encoding="utf-8") as fh:
        return [r for r in json.load(fh) if r["tag"] == "R"]


def transform(row, reading):
    if reading not in READINGS:
        raise ValueError("unknown reading: %s" % reading)
    r = dict(row)
    if reading in ("described", "nominal") and r["mode"] in HOOKS \
            and r["exchange_share"] is None:
        r["codec_share"] = None
    if r["kind"] == "gather" and r["mode"] in INT8_GATHERS:
        if reading == "nominal":
            r["volume_ratio"] = r["p"] / 8.0
        elif reading == "described" and r["volume_ratio"] is None:
            r["volume_ratio"] = r["p"] / 8.0
        elif reading == "nominal_frozen_hooks" and r["mode"] not in HOOKS:
            r["volume_ratio"] = r["p"] / 8.0
    if reading == "as_frozen_extractor":
        if r["mode"] == "gpu-quant-gathersum":
            r["kind"], r["volume_ratio"] = "reduce", 1.0
        r["shape_ratio"] = None
    return r


def label(case, f, verdict, e, delta, c5):
    if verdict in ("none", "mixed", "insufficient"):
        return S.outcome(verdict, e, delta, None)
    if c5 and verdict == "shape" and f["kind"] == "reduce":
        return "inapplicable"
    if c5 and verdict not in TESTS[case]:
        return "not tested"
    return S.outcome(verdict, e, delta, None)


def rescore(rows, method, reading="frozen", e_same=False, c5=False):
    counts, lines = {}, []
    for (c, model, p, orig, intv, to, bo, ti, bi, fam) in S.MANIFEST:
        m1 = S.find(rows, model, p, orig, "phase", to, bo)
        if m1 is None:
            continue
        f = transform(m1, reading)
        if method is rules.diagnose:
            v, _named, e = method(f, S.trend_for(rows, m1))
        else:
            v, _named, e = method(f)
        if e_same and v == "shape" and f.get("exchange_share") is not None:
            e = f["exchange_share"] * (1.0 - 2.0 / f["p"])
        allocs = S.allocs_with_pair(rows, model, p, orig, intv, to, bo, ti, bi) or [None]
        for al in allocs:
            m2 = S.find(rows, model, p, orig, "wall", to, bo, al)
            m3 = S.find(rows, model, p, intv, "wall", ti, bi, al) if intv else None
            d = None
            if m2 is not None and m3 is not None and m2["step_ms"] > 0:
                d = (m2["step_ms"] - m3["step_ms"]) / m2["step_ms"]
            oc = label(c, f, v, e, d, c5)
            counts[oc] = counts.get(oc, 0.0) + 1.0 / len(allocs)
            lines.append({"case": c, "alloc": al, "verdict": v, "e": e,
                          "delta": d, "outcome": oc, "family": fam})
    return counts, lines


def cleared(counts):
    """Recommendations whose measured change was at least 5% (correct or short)."""
    return counts.get("correct", 0.0) + counts.get("short", 0.0)


def fmt(counts):
    return ", ".join("%s %g" % (k, counts[k]) for k in ORDER if counts.get(k))
