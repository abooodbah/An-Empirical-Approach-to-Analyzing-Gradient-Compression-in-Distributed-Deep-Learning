# -*- coding: utf-8 -*-
"""E8 scorer (diagnostic/E8_PROTOCOL.md sections 4 to 6), frozen with the
reference before any E8 record is read.

For each E8 allocation (run directory e8-<set>-<N>n-<job>): the frozen rules
(rules.py 71329c33) read the original's M1 record through the frozen
extractor (features.py 95e240d8) with the allocation's calibration, trend
None, as in E7; the reference (replay_reference.py) predicts from the
original's traces on ranks 0 and 3 and the allocation's all-reduce
calibration. Each prediction is labeled against the block-timed gain of the
same allocation with E6's outcome rule and E7's labels.

Usage: python score_e8.py <run_dir> [<run_dir> ...] [--reading frozen|described] [--json out.json]
"""
import csv
import glob
import io
import json
import math
import os
import re
import statistics as st
import sys
import time
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
DIAG = os.path.dirname(HERE)
sys.path.insert(0, DIAG)
sys.path.insert(0, HERE)
import features as F  # noqa: E402  frozen extractor
import rules  # noqa: E402  frozen rules
from score_e6 import outcome  # noqa: E402  frozen outcome rule
import replay_reference as REF  # noqa: E402

MODELS = ("transformer", "resnet50")
FAMILIES = (("G", "gpu-quant-gathersum", "gpu-quant-reducible"),
            ("H", "ddp-int8-hook", "ddp-fp16-hook"))
METHODS = [("full", rules.diagnose), ("A1", rules.ablation_payload_only),
           ("A2", rules.ablation_calibration_only), ("A3", rules.ablation_timeline_only)]
ABSTAIN = ("none", "mixed", "insufficient")
TESTED = {"shape"}
BLOCK_RUN = re.compile(r"^redE8-ws(\d+)-(transformer|resnet50)-(.+)-b(\d+)-block-p(\d)$")
MAGNITUDE_METHODS = ("full", "reference")


def _open(path):
    return io.open(F.openable(path), encoding="utf-8")


def block_values(run_dir):
    """{(model, mode): {T_ms, passes, pass_gap}} from valid block runs only."""
    runs = defaultdict(list)
    for path in glob.glob(os.path.join(run_dir, "results", "*.jsonl")):
        with _open(path) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    if BLOCK_RUN.match(str(rec.get("run_id", ""))):
                        runs[rec["run_id"]].append(rec)
    syncs = {}
    for path in glob.glob(os.path.join(run_dir, "results", "*rank0000.provenance.json")):
        with _open(path) as fh:
            prov = json.load(fh)
        syncs[prov.get("run_id")] = (prov.get("block") or {}).get("runtime_synchronizations_inside_block")
    passes = defaultdict(list)
    for rid, recs in runs.items():
        m = BLOCK_RUN.match(rid)
        world, model, mode = int(m.group(1)), m.group(2), m.group(3)
        ranks = {r["rank"] for r in recs}
        valid = (ranks == set(range(world))
                 and all(r.get("status") == "PASS" for r in recs)
                 and all(isinstance(r.get("loss"), (int, float)) and math.isfinite(r["loss"]) for r in recs)
                 and len({round(float(r["block_step_equiv_s"]), 12) for r in recs}) == 1
                 and syncs.get(rid) in (0, None))
        if valid:
            passes[(model, mode)].append(1e3 * float(recs[0]["block_step_equiv_s"]))
    out = {}
    for key, ts in passes.items():
        mean = st.mean(ts)
        out[key] = {"T_ms": mean, "passes": len(ts),
                    "pass_gap": (max(ts) - min(ts)) / mean if len(ts) > 1 else None}
    return out


def measured_gain(orig, cand):
    if orig is None or cand is None:
        return None
    return 1.0 - cand["T_ms"] / orig["T_ms"]


def rule_label(verdict, e, g, kind):
    if g is None:
        return "missing (run failed)"
    if verdict in ABSTAIN:
        return outcome(verdict, e, g, None)
    if verdict == "shape" and kind == "reduce":
        return "inapplicable"
    if verdict not in TESTED:
        return "not tested"
    return outcome(verdict, e, g, None)


def reference_verdict(g_pred):
    return "shape" if REF.recommends(g_pred) else "none"


def calibration(run_dir):
    """The allocation's all-reduce alpha and beta, from its calibration run."""
    for path in sorted(glob.glob(os.path.join(run_dir, "results", "*.csv"))):
        with io.open(F.openable(path), newline="", encoding="utf-8") as fh:
            row = next(csv.DictReader(fh), None) or {}
        a, b = row.get("alpha_allreduce_s", ""), row.get("beta_allreduce_s_per_byte", "")
        if a not in ("", None) and b not in ("", None):
            return float(a), float(b)
    return None


def timeline(run_dir):
    starts, secs = {}, {}
    path = os.path.join(run_dir, "provenance", "timeline.txt")
    try:
        fh = _open(path)
    except OSError:
        return secs
    with fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 3:
                continue
            t = time.mktime(time.strptime(parts[0], "%Y-%m-%dT%H:%M:%SZ"))
            if parts[1] == "start":
                starts[parts[2]] = t
            elif parts[1] == "done" and parts[2] in starts:
                secs[parts[2]] = t - starts[parts[2]]
    return secs


def trace_paths(run_dir, model, mode):
    pats = [os.path.join(run_dir, "traces", "*-trace-%s-%s-b*__*rank%04d.trace.json" % (model, mode, r)) for r in (0, 3)]
    return sorted(p for pat in pats for p in glob.glob(pat))


def allocation(run_dir, reading="frozen", predictor=REF.predict):
    base = os.path.basename(os.path.normpath(run_dir))
    regime = base.split("-")[1]
    alloc = base.rsplit("-", 1)[-1]
    transform = None
    if reading == "described":
        sys.path.insert(0, os.path.join(DIAG, "v1_1"))
        from rescore import transform  # noqa: E402
    rows = F.features(F.load(run_dir))
    blocks = block_values(run_dir)
    cal = calibration(run_dir)
    secs = timeline(run_dir)
    out = []
    for model in MODELS:
        for fam, orig, cand in FAMILIES:
            case = "E8-%s-%s%s" % (regime, model[0].upper(), fam)
            g = measured_gain(blocks.get((model, orig)), blocks.get((model, cand)))
            common = {"case": case, "regime": regime, "model": model, "family": fam, "alloc": alloc, "g": g,
                      "T_orig": (blocks.get((model, orig)) or {}).get("T_ms"),
                      "T_cand": (blocks.get((model, cand)) or {}).get("T_ms"),
                      "passes": ((blocks.get((model, orig)) or {}).get("passes"), (blocks.get((model, cand)) or {}).get("passes"))}
            m1 = next((r for r in rows if (r["model"], r["mode"], r["timing"], r["tau"]) == (model, orig, "phase", 1)), None)
            m1_s = secs.get("m1-%s-%s" % (model, orig))
            cal_s = secs.get("paper-resnet50-native")
            for name, method in METHODS:
                row = dict(common, method=name)
                if m1 is None:
                    row.update(verdict="missing", e=0.0, label="missing (M1 failed)", seconds=None)
                else:
                    f = transform(m1, reading) if transform else m1
                    v, _named, e = method(f, None) if method is rules.diagnose else method(f)
                    row.update(verdict=v, e=e, label=rule_label(v, e, g, f["kind"]),
                               seconds=(m1_s or 0.0) + ((cal_s or 0.0) if name in ("full", "A2") else 0.0))
                out.append(row)
            row = dict(common, method="reference")
            paths = trace_paths(run_dir, model, orig)
            if not paths or cal is None:
                row.update(verdict="missing", e=0.0, label="missing (trace or calibration failed)", seconds=None)
            else:
                t0 = time.perf_counter()
                try:
                    g_pred, detail = predictor(paths, fam, cal[0], cal[1])
                except Exception as exc:  # the replay could not identify the substitution
                    row.update(verdict="missing", e=0.0, label="missing (replay failed)", seconds=None,
                               detail=repr(exc)[:200])
                else:
                    replay_s = time.perf_counter() - t0
                    v = reference_verdict(g_pred)
                    row.update(verdict=v, e=g_pred, label=rule_label(v, g_pred, g, "gather"), detail=detail,
                               seconds=(secs.get("trace-%s-%s" % (model, orig)) or 0.0) + (cal_s or 0.0) + replay_s)
            out.append(row)
    return out


def summary(rows):
    per_case_allocs = defaultdict(set)
    for r in rows:
        per_case_allocs[(r["case"], r["method"])].add(r["alloc"])
    keys = ("recommended", "cleared", "false_positive", "not_tested", "abstain_correct", "abstain_miss", "missing")
    s = defaultdict(lambda: dict.fromkeys(keys, 0.0))
    mag = defaultdict(list)
    secs = defaultdict(list)
    for r in rows:
        w = 1.0 / len(per_case_allocs[(r["case"], r["method"])])
        t = s[r["method"]]
        lab = r["label"]
        if lab.startswith("missing"):
            t["missing"] += w
            continue
        if r["verdict"] not in ABSTAIN:
            t["recommended"] += w
            if r["method"] in MAGNITUDE_METHODS and r["g"] is not None:
                mag[r["method"]].append(abs(r["e"] - r["g"]))
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
        if r.get("seconds") is not None:
            secs[r["method"]].append(r["seconds"])
    for m, t in s.items():
        t["median_magnitude_error"] = st.median(mag[m]) if mag[m] else None
        t["median_measurement_s"] = st.median(secs[m]) if secs[m] else None
    return dict(s)


def main(argv):
    reading, out_json, dirs = "frozen", None, []
    i = 0
    while i < len(argv):
        if argv[i] == "--reading":
            reading = argv[i + 1]
            i += 2
        elif argv[i] == "--json":
            out_json = argv[i + 1]
            i += 2
        else:
            dirs.append(argv[i])
            i += 1
    rows = []
    for d in dirs:
        rows.extend(allocation(d, reading))
    print("E8 scoring, reading %s" % reading)
    for r in rows:
        g = "   -  " if r["g"] is None else "%+6.3f" % r["g"]
        print("%-12s %-8s %-9s %-13s e=%6.3f g=%s %s" % (r["case"], r["alloc"], r["method"], r["verdict"], r["e"], g, r["label"]))
    s = summary(rows)
    for m in ("full", "A1", "A2", "A3", "reference"):
        if m in s:
            print("%-9s " % m + ", ".join("%s %s" % (k, ("%.3g" % v) if isinstance(v, float) else v) for k, v in s[m].items()))
    if out_json:
        with io.open(out_json, "w", encoding="utf-8") as fh:
            json.dump({"reading": reading, "rows": rows, "summary": s}, fh, indent=1, default=str)
    return rows


if __name__ == "__main__":
    main(sys.argv[1:])
