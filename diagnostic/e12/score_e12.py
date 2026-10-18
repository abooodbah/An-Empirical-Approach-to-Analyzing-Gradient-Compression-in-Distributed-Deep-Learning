# -*- coding: utf-8 -*-
"""E12 (A-plus) scorer, diagnostic/E12_PROTOCOL.md sections 4 to 7. Written
and frozen before any E12 job is submitted; the hashes are in
diagnostic/E12_FREEZE.sha256.

Inputs: the two scored run directories (e12-e12a-2n-<job>, e12-e12b-2n-<job>)
and the qualification run directory (e12-q-2n-<job>), whose fidelity runs
give each tag's per-exchange error.

Methods, each scored per case and allocation (an allocation counts 1/2):
  full, A1, A3   the frozen rules (rules.py 71329c33: full procedure,
                 traffic-ratio check, phase-share check) on the original's M1
                 record through the frozen extractor (features.py 95e240d8)
  A2             the benchmark check (rules.ablation_calibration_only)
  replay         the frozen E8 reference (diagnostic/e8/replay_reference.py)
                 on the original's traces, families G and H only
  always         recommends every intervention
  cm             a calibration-only cost model: the alpha-beta time of each
                 mode's collectives over the original's M1 step
Truth: the intervention pays if the block-timed step falls by at least 5%.
Quality-constrained truth: it pays and its per-exchange error does not
exceed the original's.

Usage: python score_e12.py --qual <q_dir> <e12a_dir> <e12b_dir> [--reading frozen|described] [--json out.json]
"""
import csv
import glob
import io
import json
import math
import os
import random
import re
import shutil
import statistics as st
import sys
import tempfile
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
DIAG = os.path.dirname(HERE)
for _p in (DIAG, os.path.join(DIAG, "e8")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import features as F  # noqa: E402  frozen extractor
import rules  # noqa: E402  frozen rules
from score_e6 import outcome  # noqa: E402  frozen outcome rule (continuity labels only)
import replay_reference as REF  # noqa: E402  frozen E8 reference

P = 6
BATCH = 64
THRESHOLD = 0.05            # a change pays if the block-timed step falls by at least 5%
QUALITY_TOLERANCE = 0.01    # the intervention's error may exceed the original's by at most 1% of it
WEIGHT = 0.5                # each allocation counts one half
FIDELITY_ITERS = 8
BOOTSTRAP = 2000
SEED = 20261018
Z = 1.959963984540054       # two-sided 95%
MODELS = ("transformer", "resnet50")
TAGS = ("ddp-overlap", "ddp-fp16-hook", "ddp-int8-hook", "gpu-quant", "cpu-quant", "gpu-quant-gathersum",
        "gpu-quant-reducible", "ddp-overlap-bf16", "ddp-fp16-hook-bf16", "ddp-int8-hook-bf16")
DENSE_TAGS = ("ddp-overlap", "ddp-overlap-bf16")   # send the dense average: per-exchange error 0
ABSTAIN = ("none", "mixed", "insufficient")
RULE_METHODS = [("full", rules.diagnose), ("A1", rules.ablation_payload_only),
                ("A2", rules.ablation_calibration_only), ("A3", rules.ablation_timeline_only)]
METHODS = ("full", "A1", "A2", "A3", "replay", "always", "cm")
MAGNITUDE = ("full", "replay", "cm")

# Section 2 of the protocol. id, original, intervention, mechanisms the
# intervention tests (the rules recommend it only with one of these verdicts),
# replay family (None: the frozen replay has no substitution for it), group,
# expectation per model (stated in advance; never used in scoring).
CASES = [
    ("01", "gpu-quant-gathersum", "gpu-quant-reducible", {"shape"}, "G", "held-out", ("help", "help")),
    ("02", "gpu-quant", "gpu-quant-reducible", {"shape"}, None, "held-out", ("help", "help")),
    ("03", "cpu-quant", "gpu-quant", {"placement"}, None, "held-out", ("help", "help")),
    ("04", "ddp-int8-hook", "ddp-fp16-hook", {"shape"}, "H", "held-out", ("help", "help")),
    ("05", "ddp-overlap", "ddp-int8-hook", set(), None, "held-out", ("harm", "harm")),
    ("06", "ddp-overlap", "gpu-quant", set(), None, "held-out", ("harm", "harm")),
    ("07", "ddp-overlap", "cpu-quant", set(), None, "held-out", ("harm", "harm")),
    ("08", "ddp-overlap", "ddp-fp16-hook", set(), None, "held-out", ("help", "neutral")),
    ("09", "gpu-quant", "gpu-quant-gathersum", set(), None, "held-out", ("neutral", "neutral")),
    ("10", "ddp-overlap-bf16", "ddp-fp16-hook-bf16", set(), None, "production", ("help", "neutral")),
    ("11", "ddp-overlap-bf16", "ddp-int8-hook-bf16", set(), None, "production", ("harm", "harm")),
    ("12", "ddp-int8-hook-bf16", "ddp-fp16-hook-bf16", {"shape"}, "H", "production", ("help", "help")),
]
ORIGINALS = sorted({c[1] for c in CASES})

_TAG = "(?P<tag>[a-z0-9-]+?)"
BLOCK = re.compile(r"^redE12-ws(?P<ws>\d+)-(?P<model>transformer|resnet50)-%s-b(?P<b>\d+)-block-p(?P<p>[12])$" % _TAG)
PHASE = re.compile(r"^redE12-ws(?P<ws>\d+)-(?P<model>transformer|resnet50)-%s-b(?P<b>\d+)-phase$" % _TAG)
TRACE = re.compile(r"^redE12-ws(?P<ws>\d+)-trace-(?P<model>transformer|resnet50)-%s-b(?P<b>\d+)$" % _TAG)
FID = re.compile(r"^redE12-ws(?P<ws>\d+)-(?P<model>transformer|resnet50)-%s-b(?P<b>\d+)-fid$" % _TAG)
EQUIV = re.compile(r"^redE12-ws(?P<ws>\d+)-(?P<model>transformer|resnet50)-%s-b(?P<b>\d+)-equiv$" % _TAG)
CALIBRATION_RUN = "redE12-ws%d-resnet50-native" % P


def rid(model, tag, kind, p=None):
    if kind == "block":
        return "redE12-ws%d-%s-%s-b%d-block-p%d" % (P, model, tag, BATCH, p)
    if kind == "trace":
        return "redE12-ws%d-trace-%s-%s-b%d" % (P, model, tag, BATCH)
    return "redE12-ws%d-%s-%s-b%d-%s" % (P, model, tag, BATCH, kind)


def split_tag(tag):
    """(harness mode, compute dtype) of a tag."""
    return (tag[:-5], "bf16") if tag.endswith("-bf16") else (tag, "fp32")


def _open(path, **kw):
    return io.open(F.openable(path), encoding="utf-8", **kw)


def _json(path):
    with _open(path) as fh:
        return json.load(fh)


def _results(run_dir, run_id, suffix):
    return sorted(glob.glob(os.path.join(run_dir, "results", glob.escape(run_id) + "__*" + suffix)))


# ------------------------------------------------------------------ outcomes
def provenance_ok(prov, tag):
    """The rank-0 provenance must record the tag's precision, its 16-bit wire
    type and the int8 hook's synchronous form (section 4)."""
    mode, dtype = split_tag(tag)
    if (prov.get("condition") or {}).get("compute_dtype", "fp32") != dtype:
        return False
    ddp = prov.get("ddp") or {}
    if mode == "ddp-int8-hook" and (ddp.get("hook_state") or {}).get("mode") != "sync":
        return False
    if mode == "ddp-fp16-hook" and ddp.get("dtype") != ("bfloat16" if dtype == "bf16" else "float16"):
        return False
    return True


def block_pass(run_dir, run_id, tag):
    """Block-timed step (ms) of one pass, or None if the run fails the
    harness's own validation or the provenance check."""
    recs = []
    for path in _results(run_dir, run_id, ".jsonl"):
        with _open(path) as fh:
            recs.extend(json.loads(line) for line in fh if line.strip())
    if not recs:
        return None
    ranks = {r.get("rank") for r in recs}
    ok = (ranks == set(range(P))
          and all(r.get("status") == "PASS" for r in recs)
          and all(r.get("final_param_hashes_equal") in (True, "True") for r in recs)
          and all(isinstance(r.get("loss"), (int, float)) and math.isfinite(r["loss"]) for r in recs)
          and len({round(float(r["block_step_equiv_s"]), 12) for r in recs}) == 1)
    prov_paths = _results(run_dir, run_id, "rank0000.provenance.json")
    if not ok or not prov_paths:
        return None
    prov = _json(prov_paths[0])
    if (prov.get("block") or {}).get("runtime_synchronizations_inside_block") not in (0, None):
        return None
    if not provenance_ok(prov, tag):
        return None
    return 1e3 * float(recs[0]["block_step_equiv_s"])


def block_values(run_dir):
    """{(model, tag): {T_ms, passes, pass_gap}}; T is the mean of the valid passes."""
    out = {}
    for model in MODELS:
        for tag in TAGS:
            ts = [t for t in (block_pass(run_dir, rid(model, tag, "block", p), tag) for p in (1, 2)) if t is not None]
            if ts:
                mean = st.mean(ts)
                out[(model, tag)] = {"T_ms": mean, "passes": len(ts),
                                     "pass_gap": (max(ts) - min(ts)) / mean if len(ts) > 1 else None}
    return out


def measured_gain(orig, cand):
    if orig is None or cand is None:
        return None
    return 1.0 - cand["T_ms"] / orig["T_ms"]


def pays(g):
    return None if g is None else g >= THRESHOLD


# ------------------------------------------------------------------ method inputs
def calibration(run_dir):
    """alpha and beta of both collectives from the allocation's calibration run."""
    for path in _results(run_dir, CALIBRATION_RUN, ".csv"):
        with _open(path, newline="") as fh:
            row = next(csv.DictReader(fh), None) or {}
        try:
            return {k: float(row[k]) for k in ("alpha_allreduce_s", "beta_allreduce_s_per_byte",
                                               "alpha_allgather_s", "beta_allgather_s_per_byte")}
        except (KeyError, TypeError, ValueError):
            continue
    return None


def m1_row(run_dir, model, tag, reading="frozen"):
    """The frozen extractor's row for one phase-timed run. The run's CSVs and
    the calibration run's are copied into a scratch run directory of the same
    name, so that rows of the fp32 and bf16 forms of one harness mode, which
    the extractor would merge, stay apart."""
    paths = _results(run_dir, rid(model, tag, "phase"), ".csv")
    if not paths:
        return None
    mode, _dtype = split_tag(tag)
    scratch = tempfile.mkdtemp(prefix="e12m1-")
    try:
        res = os.path.join(scratch, os.path.basename(os.path.normpath(run_dir)), "results")
        os.makedirs(res)
        for p in paths + _results(run_dir, CALIBRATION_RUN, ".csv"):
            shutil.copyfile(F.openable(p), F.openable(os.path.join(res, os.path.basename(p))))
        rows = F.features(F.load(scratch))
        grads = []
        for p in paths:
            with _open(p, newline="") as fh:
                grads.extend(float(r["gradient_elements"]) for r in csv.DictReader(fh)
                             if r.get("gradient_elements") not in ("", None))
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    row = next((r for r in rows if (r["model"], r["mode"], r["timing"], r["tau"]) == (model, mode, "phase", 1)), None)
    if row is None:
        return None
    if reading == "described":
        sys.path.insert(0, os.path.join(DIAG, "v1_1"))
        from rescore import transform  # noqa: E402
        row = transform(row, reading)
    row = dict(row)
    row["gradient_elements"] = st.mean(grads) if grads else None
    return row


def trace_paths(run_dir, model, tag):
    pats = [os.path.join(run_dir, "traces", glob.escape(rid(model, tag, "trace")) + "__*rank%04d.trace.json" % r)
            for r in (0, 3)]
    return sorted(p for pat in pats for p in glob.glob(pat))


# ------------------------------------------------------------------ cost model
def line(cal, collective, nbytes):
    """The calibration's fitted time (s) for one collective of nbytes per rank:
    intercept + slope x bytes, with the harness's own alpha = intercept /
    steps and beta = slope / volume factor undone."""
    if collective == "allreduce":
        steps, vf = 2 * (P - 1), 2.0 * (P - 1) / P
        a, b = cal["alpha_allreduce_s"], cal["beta_allreduce_s_per_byte"]
    else:
        steps, vf = P - 1, (P - 1.0) / P
        a, b = cal["alpha_allgather_s"], cal["beta_allgather_s_per_byte"]
    return a * steps + b * vf * nbytes


def exchange_time(cal, tag, n):
    """Section 5: the collectives each mode issues per step, n gradient elements."""
    mode, _ = split_tag(tag)
    if mode == "ddp-overlap":
        return line(cal, "allreduce", 4 * n)
    if mode == "ddp-fp16-hook":
        return line(cal, "allreduce", 2 * n)
    if mode == "gpu-quant-reducible":
        return line(cal, "allreduce", 4) + line(cal, "allreduce", n)
    if mode == "gpu-quant-gathersum":
        return line(cal, "allreduce", 4) + line(cal, "allgather", n)
    if mode in ("gpu-quant", "cpu-quant"):
        return line(cal, "allgather", 4) + line(cal, "allgather", n)
    if mode == "ddp-int8-hook":
        return line(cal, "allgather", 8) + line(cal, "allgather", n)
    raise ValueError(tag)


def cost_model_gain(cal, m1, orig, cand):
    n, step_ms = m1.get("gradient_elements"), m1.get("step_ms")
    if not n or not step_ms:
        return None
    return (exchange_time(cal, orig, n) - exchange_time(cal, cand, n)) / (step_ms / 1e3)


# ------------------------------------------------------------------ quality
def fidelity_errors(qual_dir):
    """{(model, tag): mean relative L2 error} from the qualification job's
    rank-0 fidelity sidecars; a run with fewer than 8 finite entries is missing.
    Dense DDP sends the dense average: 0 by definition."""
    out = {(m, t): 0.0 for m in MODELS for t in DENSE_TAGS}
    if qual_dir is None:
        return out
    for model in MODELS:
        for tag in TAGS:
            if tag in DENSE_TAGS:
                continue
            paths = _results(qual_dir, rid(model, tag, "fid"), "rank0000.v52.json")
            if not paths:
                continue
            errs = [e.get("relative_l2_error") for e in _json(paths[0]).get("fidelity", [])]
            if len(errs) == FIDELITY_ITERS and all(isinstance(v, (int, float)) and math.isfinite(v) for v in errs):
                out[(model, tag)] = st.mean(errs)
    return out


def quality_ok(err, model, orig, cand):
    eo, ec = err.get((model, orig)), err.get((model, cand))
    if eo is None or ec is None:
        return None
    return ec <= eo * (1.0 + QUALITY_TOLERANCE)


# ------------------------------------------------------------------ scoring
def rule_label(verdict, e, g, kind, tested):
    """E7/E8 continuity label (reported beside the E12 metrics)."""
    if g is None:
        return "missing (run failed)"
    if verdict in ABSTAIN:
        return outcome(verdict, e, g, None)
    if verdict == "shape" and kind == "reduce":
        return "inapplicable"
    if verdict not in tested:
        return "not tested"
    return outcome(verdict, e, g, None)


def allocation(run_dir, err, reading="frozen", predictor=REF.predict):
    """One row per case, model and method for one scored allocation."""
    alloc = os.path.basename(os.path.normpath(run_dir)).rsplit("-", 1)[-1]
    blocks = block_values(run_dir)
    cal = calibration(run_dir)
    m1 = {(m, t): m1_row(run_dir, m, t, reading) for m in MODELS for t in ORIGINALS}
    rows = []
    for mi, model in enumerate(MODELS):
        for cid, orig, cand, tested, family, group, expect in CASES:
            g = measured_gain(blocks.get((model, orig)), blocks.get((model, cand)))
            common = {"case": "E12-%s%s" % (model[0].upper(), cid), "model": model, "original": orig,
                      "intervention": cand, "group": group, "expectation": expect[mi], "alloc": alloc,
                      "g": g, "pays": pays(g), "quality_ok": quality_ok(err, model, orig, cand),
                      "T_orig": (blocks.get((model, orig)) or {}).get("T_ms"),
                      "T_cand": (blocks.get((model, cand)) or {}).get("T_ms"),
                      "passes": ((blocks.get((model, orig)) or {}).get("passes"),
                                 (blocks.get((model, cand)) or {}).get("passes"))}
            common["pays_q"] = (None if common["pays"] is None or common["quality_ok"] is None
                                else common["pays"] and common["quality_ok"])
            f = m1[(model, orig)]
            for name, method in RULE_METHODS:
                row = dict(common, method=name)
                if f is None:
                    row.update(verdict="missing", e=None, recommends=None, label="missing (M1 failed)")
                else:
                    v, _named, e = method(f, None) if method is rules.diagnose else method(f)
                    rec = v in tested and not (v == "shape" and f["kind"] == "reduce")
                    row.update(verdict=v, e=e if rec else None, recommends=rec,
                               label=rule_label(v, e, g, f["kind"], tested))
                rows.append(row)
            row = dict(common, method="replay")
            if family is None:
                row.update(verdict="no substitution model", e=None, recommends=False, label="abstain (no model)")
            else:
                paths = trace_paths(run_dir, model, orig)
                if not paths or cal is None:
                    row.update(verdict="missing", e=None, recommends=None, label="missing (trace or calibration)")
                else:
                    try:
                        gp, detail = predictor(paths, family, cal["alpha_allreduce_s"], cal["beta_allreduce_s_per_byte"])
                    except Exception as exc:  # the frozen replay could not find the substitution
                        row.update(verdict="missing", e=None, recommends=None, label="missing (replay failed)",
                                   detail=repr(exc)[:200])
                    else:
                        rec = REF.recommends(gp)
                        row.update(verdict="recommend" if rec else "none", e=gp, recommends=rec,
                                   label="recommend" if rec else "abstain", detail=detail)
            rows.append(row)
            rows.append(dict(common, method="always", verdict="recommend", e=None, recommends=True, label="recommend"))
            row = dict(common, method="cm")
            gp = None if (f is None or cal is None) else cost_model_gain(cal, f, orig, cand)
            if gp is None:
                row.update(verdict="missing", e=None, recommends=None, label="missing (M1 or calibration)")
            else:
                rec = gp >= THRESHOLD
                row.update(verdict="recommend" if rec else "none", e=gp, recommends=rec,
                           label="recommend" if rec else "abstain")
            rows.append(row)
    return rows


def wilson(k, n):
    """95% Wilson score interval; fractional counts allowed (half-weighted rows)."""
    if n <= 0:
        return None
    p = k / n
    den = 1.0 + Z * Z / n
    center = (p + Z * Z / (2 * n)) / den
    half = Z * math.sqrt(max(0.0, p * (1 - p) / n + Z * Z / (4 * n * n))) / den
    return (max(0.0, center - half), min(1.0, center + half))


def _ratio(k, n):
    return k / n if n > 0 else None


def bootstrap_median(errors_by_case, seed=SEED, reps=BOOTSTRAP):
    """Percentile 95% interval of the median, resampling cases (a case keeps
    both allocations' rows)."""
    cases = [v for v in errors_by_case.values() if v]
    if len(cases) < 2:
        return None
    rng = random.Random(seed)
    meds = []
    for _ in range(reps):
        pool = [e for _ in cases for e in rng.choice(cases)]
        meds.append(st.median(pool))
    meds.sort()
    return (meds[int(0.025 * reps)], meds[int(0.975 * reps) - 1])


def metrics(rows, method, truth="pays"):
    """Precision, recall, false-positive rate, abstention and gain error of
    one method against one truth ("pays" or "pays_q")."""
    rs = [r for r in rows if r["method"] == method]
    scored = [r for r in rs if r["recommends"] is not None and r[truth] is not None]
    w = WEIGHT
    pos = sum(w for r in scored if r[truth])
    neg = sum(w for r in scored if not r[truth])
    rec = sum(w for r in scored if r["recommends"])
    tp = sum(w for r in scored if r["recommends"] and r[truth])
    fp = rec - tp
    abst = sum(w for r in scored if not r["recommends"] and (r["verdict"] in ABSTAIN or r["label"].startswith("abstain")))
    not_tested = sum(w for r in scored if not r["recommends"] and r["label"] in ("not tested", "inapplicable"))
    total = pos + neg
    out = {"cases_weight": total, "missing_weight": sum(w for r in rs) - total, "paying": pos, "recommended": rec,
           "true_pos": tp, "false_pos": fp, "abstained": abst, "not_tested": not_tested,
           "precision": _ratio(tp, rec), "precision_ci": wilson(tp, rec),
           "recall": _ratio(tp, pos), "recall_ci": wilson(tp, pos),
           "fpr": _ratio(fp, neg), "fpr_ci": wilson(fp, neg),
           "abstention": _ratio(abst, total), "abstention_ci": wilson(abst, total)}
    if method in MAGNITUDE and truth == "pays":
        for key, keep in (("rec", lambda r: r["recommends"]), ("all", lambda r: True)):
            by_case = defaultdict(list)
            for r in scored:
                if keep(r) and r["e"] is not None and r["g"] is not None:
                    by_case[r["case"]].append(abs(r["e"] - r["g"]))
            errs = [e for v in by_case.values() for e in v]
            out["gain_error_median_" + key] = st.median(errs) if errs else None
            out["gain_error_ci_" + key] = bootstrap_median(by_case)
            out["gain_error_n_" + key] = len(errs)
    return out


def summary(rows):
    return {truth: {m: metrics(rows, m, truth) for m in METHODS} for truth in ("pays", "pays_q")}


def rho(run_dir, blocks=None, reading="frozen"):
    """Matched-encoding control: phase-timing bias per model (section 6)."""
    blocks = blocks if blocks is not None else block_values(run_dir)
    out = {}
    for model in MODELS:
        te = {t: (m1_row(run_dir, model, t, reading) or {}).get("step_ms")
              for t in ("gpu-quant-gathersum", "gpu-quant-reducible")}
        tb = {t: (blocks.get((model, t)) or {}).get("T_ms") for t in te}
        r = {t: (te[t] / tb[t] if te[t] and tb[t] else None) for t in te}
        g_e = 1 - te["gpu-quant-reducible"] / te["gpu-quant-gathersum"] if all(te.values()) else None
        g_b = 1 - tb["gpu-quant-reducible"] / tb["gpu-quant-gathersum"] if all(tb.values()) else None
        out[model] = {"T_phase_ms": te, "T_block_ms": tb, "rho": r, "g_phase": g_e, "g_block": g_b,
                      "bias": (g_e - g_b) if g_e is not None and g_b is not None else None,
                      "equal": equivalence(run_dir, model)}
    return out


def equivalence(run_dir, model):
    """True if both arms' equivalence runs record equivalence_ok on every rank."""
    oks = []
    for tag in ("gpu-quant-gathersum", "gpu-quant-reducible"):
        vals = []
        for path in _results(run_dir, rid(model, tag, "equiv"), ".csv"):
            with _open(path, newline="") as fh:
                vals.extend(r.get("equivalence_ok") for r in csv.DictReader(fh))
        oks.append(bool(vals) and all(v == "True" for v in vals))
    return all(oks)


def replay_fidelity(run_dir, blocks=None):
    """Secondary: the frozen replay of each tag's own trace (no substitution)
    against that tag's block-timed step. Never used to score a method."""
    blocks = blocks if blocks is not None else block_values(run_dir)
    out = {}
    for model in MODELS:
        for tag in TAGS:
            paths = trace_paths(run_dir, model, tag)
            t = (blocks.get((model, tag)) or {}).get("T_ms")
            if not paths or t is None:
                continue
            try:
                step = max(REF.replay(REF.load(p)).step_ms for p in paths)
            except Exception as exc:  # noqa: BLE001
                out["%s/%s" % (model, tag)] = {"error": repr(exc)[:200]}
                continue
            out["%s/%s" % (model, tag)] = {"replay_ms": step, "block_ms": t, "ratio": step / t}
    return out


def expectation_check(rows):
    """How many cases went the way section 2 expected (report only)."""
    seen, hits = 0, 0
    for r in rows:
        if r["method"] != "always" or r["g"] is None:
            continue
        g = r["g"]
        got = "help" if g >= THRESHOLD else "harm" if g <= -THRESHOLD else "neutral"
        seen += 1
        hits += got == r["expectation"]
    return {"rows": seen, "as_expected": hits}


def main(argv, predictor=REF.predict):
    reading, out_json, qual, dirs = "frozen", None, None, []
    i = 0
    while i < len(argv):
        if argv[i] in ("--reading", "--json", "--qual"):
            val = argv[i + 1]
            reading, out_json, qual = {"--reading": (val, out_json, qual), "--json": (reading, val, qual),
                                       "--qual": (reading, out_json, val)}[argv[i]]
            i += 2
        else:
            dirs.append(argv[i])
            i += 1
    err = fidelity_errors(qual)
    rows, extra = [], {}
    for d in dirs:
        rows.extend(allocation(d, err, reading, predictor))
        blocks = block_values(d)
        extra[os.path.basename(os.path.normpath(d))] = {"rho": rho(d, blocks, reading),
                                                        "replay_fidelity": replay_fidelity(d, blocks)}
    s = summary(rows)
    print("E12 scoring, reading %s, allocations %s" % (reading, ", ".join(os.path.basename(os.path.normpath(d)) for d in dirs)))
    for r in rows:
        g = "   -  " if r["g"] is None else "%+6.3f" % r["g"]
        e = "   -  " if r["e"] is None else "%+6.3f" % r["e"]
        print("%-9s %-8s %-7s %-22s e=%s g=%s %s" % (r["case"], r["alloc"], r["method"], r["verdict"], e, g, r["label"]))
    for truth in ("pays", "pays_q"):
        print("\ntruth: %s" % truth)
        for m in METHODS:
            t = s[truth][m]
            fmt = lambda v: "  -  " if v is None else "%.2f" % v  # noqa: E731
            print("%-7s rec %4.1f tp %4.1f fp %4.1f  precision %s recall %s fpr %s abstention %s" % (
                m, t["recommended"], t["true_pos"], t["false_pos"], fmt(t["precision"]), fmt(t["recall"]),
                fmt(t["fpr"]), fmt(t["abstention"])))
    print("\nexpectations:", expectation_check(rows))
    if out_json:
        with io.open(out_json, "w", encoding="utf-8") as fh:
            json.dump({"reading": reading, "rows": rows, "summary": s, "allocations": extra,
                       "fidelity_errors": {"%s/%s" % k: v for k, v in err.items()},
                       "expectations": expectation_check(rows)}, fh, indent=1, default=str)
    return rows, s


if __name__ == "__main__":
    main(sys.argv[1:])
