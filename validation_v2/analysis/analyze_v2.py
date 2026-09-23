# -*- coding: utf-8 -*-
"""Analysis of the TPDS v2 validation campaign (2026-09-23) under the rules
frozen in ANALYSIS_RULES_v2.md before the first timed job.

Input: a directory holding the retrieved job directories
runs/v2-<JOB>-<N>n-<SLURM_JOB_ID>/ (results/*.jsonl, results/*provenance.json,
provenance/manifest_<JOB>.json). Output directory receives:
  v2_runs.csv         one row per harness run (value, validity, auxiliaries)
  v2_conditions.csv   per job and condition: the two passes and their mean
  v2_expA.csv         per job, workload, convention: T of the three paths,
                      rho against B, matched gain g, bundled gain h, ranking
  v2_expB.csv         per job, workload, batch: dense and hook T, hook ratios
  v2_checks.csv       anchor drift, pass gaps, exclusions, calibration
  VERDICTS_v2.md      the pre-registered verdicts, one line each
  fig_v2_gain.pdf, fig_v2_hooks.pdf (with --figures)

usage: python analyze_v2.py <runs dir> <out dir> [--figures]
"""
import argparse
import csv
import glob
import json
import os
import statistics as st
import sys
from collections import defaultdict

MARGIN = 0.05       # practical margin (5%)
TIE = 0.02          # relative difference below which two paths tie in a ranking
PATHS = ("gpu-quant-reducible", "gpu-quant-gathersum", "gpu-topk")
CONVENTIONS = {"phase": "E", "wall": "S", "block": "B"}
HOOKS = ("ddp-fp16-hook", "ddp-int8-hook")
VJOBS = ("V1", "V2", "V3")


# ------------------------------------------------------------ loading
def _arg(args, flag, default=None):
    return args[args.index(flag) + 1] if flag in args else default


def load_runs(runs_dir):
    runs = []
    for job_dir in sorted(glob.glob(os.path.join(runs_dir, "v2-*"))):
        manifests = glob.glob(os.path.join(job_dir, "provenance", "manifest_*.json"))
        if not manifests:
            continue
        manifest = json.load(open(manifests[0], encoding="utf-8"))
        job = manifest["job"]
        rows_by_run = defaultdict(list)
        for path in glob.glob(os.path.join(job_dir, "results", "*.jsonl")):
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    record = json.loads(line)
                    rows_by_run[record["run_id"]].append(record)
        prov_by_run = {}
        for path in glob.glob(os.path.join(job_dir, "results", "*rank0000.provenance.json")):
            prov = json.load(open(path, encoding="utf-8"))
            prov_by_run[prov["run_id"]] = prov
        for entry in manifest["runs"]:
            run_id = f"v2-{job}-{entry['id']}"
            runs.append(summarize_run(job, entry, rows_by_run.get(run_id, []), prov_by_run.get(run_id)))
    return runs


def summarize_run(job, entry, rows, prov):
    args = entry["args"]
    run = {
        "job": job, "entry": entry["id"], "role": entry["role"], "experiment": entry["experiment"],
        "pass": entry["pass"], "condition": entry["id"].rsplit("-p", 1)[0] if entry["pass"] else entry["id"],
        "model": _arg(args, "--model"), "mode": _arg(args, "--comm-mode"),
        "timing": _arg(args, "--timing-mode"), "batch": int(_arg(args, "--batch-size", 0)),
        "valid": False, "reason": "", "T_ms": None, "step_p50_ms": None, "exchange_mean_ms": None,
        "syncs_in_block": None, "ranks": 0, "late_vs_early": None,
    }
    if not rows:
        run["reason"] = "no records"
        return run
    ranks = sorted({r["rank"] for r in rows})
    world = rows[0]["world_size"]
    run["ranks"] = len(ranks)
    by_rank = defaultdict(list)
    for r in rows:
        by_rank[r["rank"]].append(r)
    problems = []
    if ranks != list(range(world)):
        problems.append(f"ranks {len(ranks)}/{world}")
    if entry["role"] in ("timed", "anchor") and any(r["status"] != "PASS" for r in rows):
        problems.append("status not PASS")
    if any(not r["final_hashes_equal"] for r in rows):
        problems.append("final parameter hashes differ")
    if any(r["trace_capture"] for r in rows) and entry["role"] in ("timed", "anchor"):
        problems.append("trace capture in a timed run")
    equivs = {round(r["block_step_equiv_s"], 12) for r in rows}
    if len(equivs) != 1:
        problems.append("ranks disagree on the gathered block time")
    run["T_ms"] = 1e3 * rows[0]["block_step_equiv_s"]
    if run["timing"] in ("phase", "wall"):
        iters = sorted({r["iteration"] for r in rows})
        per_iter_max = []
        for i in iters:
            values = [r["observed_step_s"] for r in rows if r["iteration"] == i and r["observed_step_s"] is not None]
            if values:
                per_iter_max.append(max(values))
        if per_iter_max:
            run["step_p50_ms"] = 1e3 * st.median(per_iter_max)
        if len(per_iter_max) >= 40:
            # warm-up adequacy: last 20 measured steps against the first 20
            run["late_vs_early"] = st.mean(per_iter_max[-20:]) / st.mean(per_iter_max[:20]) - 1.0
    if run["timing"] == "phase":
        nets = [r["network_s"] for r in rows if r["network_s"] is not None]
        if nets:
            run["exchange_mean_ms"] = 1e3 * st.mean(nets)
    if prov is not None:
        run["syncs_in_block"] = prov.get("block", {}).get("runtime_synchronizations_inside_block")
        if run["timing"] == "block" and run["syncs_in_block"] not in (0, None):
            problems.append("measurement synchronization inside the block")
    run["valid"] = not problems
    run["reason"] = "; ".join(problems)
    return run


# ------------------------------------------------------------ aggregation
def conditions(runs):
    """Per job and timed condition: the two passes and their mean."""
    groups = defaultdict(dict)
    meta = {}
    for run in runs:
        if run["role"] != "timed" or not run["valid"]:
            continue
        key = (run["job"], run["condition"])
        groups[key][run["pass"]] = run["T_ms"]
        meta[key] = run
    out = []
    for key, passes in sorted(groups.items()):
        run = meta[key]
        values = [passes[p] for p in sorted(passes)]
        mean = st.mean(values)
        out.append({
            "job": key[0], "condition": key[1], "experiment": run["experiment"], "model": run["model"],
            "mode": run["mode"], "timing": run["timing"], "batch": run["batch"],
            "T_p1_ms": passes.get(1), "T_p2_ms": passes.get(2), "T_ms": mean, "passes": len(values),
            "pass_gap": (abs(values[0] - values[1]) / mean) if len(values) == 2 else None,
        })
    return out


def _lookup(conds):
    return {(c["job"], c["model"], c["mode"], c["timing"], c["batch"]): c["T_ms"] for c in conds}


def ranking(values):
    """Paths ordered by time; neighbors within TIE are joined with '='."""
    items = sorted(values.items(), key=lambda kv: kv[1])
    text = items[0][0]
    for (prev, pv), (name, v) in zip(items, items[1:]):
        text += ("=" if (v - pv) / pv < TIE else "<") + name
    return text


def experiment_a(conds):
    T = _lookup(conds)
    out = []
    jobs = sorted({c["job"] for c in conds})
    for job in jobs:
        for model in ("transformer", "resnet50"):
            for timing, conv in CONVENTIONS.items():
                vals = {p: T.get((job, model, p, timing, 32)) for p in PATHS}
                ref = {p: T.get((job, model, p, "block", 32)) for p in PATHS}
                if any(v is None for v in vals.values()):
                    continue
                row = {"job": job, "model": model, "convention": conv,
                       **{f"T_{p}_ms": vals[p] for p in PATHS}}
                for p in PATHS:
                    row[f"rho_{p}"] = vals[p] / ref[p] if ref[p] else None
                row["g_matched"] = 1.0 - vals["gpu-quant-reducible"] / vals["gpu-quant-gathersum"]
                row["h_topk"] = 1.0 - vals["gpu-quant-reducible"] / vals["gpu-topk"]
                row["ranking"] = ranking({"reduce": vals["gpu-quant-reducible"],
                                          "gather": vals["gpu-quant-gathersum"], "topk": vals["gpu-topk"]})
                out.append(row)
    return out


def experiment_b(conds):
    T = _lookup(conds)
    out = []
    for job in sorted({c["job"] for c in conds}):
        for model in ("transformer", "resnet50"):
            batches = sorted({c["batch"] for c in conds if c["job"] == job and c["model"] == model
                              and c["experiment"] == "B"})
            for batch in batches:
                dense = T.get((job, model, "ddp-overlap", "block", batch))
                row = {"job": job, "model": model, "batch": batch, "T_dense_ms": dense}
                for h in HOOKS:
                    th = T.get((job, model, h, "block", batch))
                    row[f"T_{h}_ms"] = th
                    row[f"R_{h}"] = th / dense if (th and dense) else None
                out.append(row)
    return out


# ------------------------------------------------------------ verdicts (frozen rules)
DIRECTION_WORDS = {
    "same": "same practical direction",
    "reversed": "reversed",
    "margin crossed": "crosses the 5% margin",
    "both below margin": "no practical difference under either",
}


def direction_category(q, b):
    """How convention q's practical conclusion about one gain compares with
    the block reference b. A reversal needs opposite signs with both gains at
    least the margin; when exactly one reaches the margin the conventions
    disagree about whether there is a practical difference, which is kept
    apart from a reversal; when neither does, no practical distinction is
    supported by either."""
    big_q, big_b = abs(q) >= MARGIN, abs(b) >= MARGIN
    if big_q and big_b:
        return "same" if (q > 0) == (b > 0) else "reversed"
    if big_q or big_b:
        return "margin crossed"
    return "both below margin"


def verdicts_a(rows):
    lines = []
    for model in ("transformer", "resnet50"):
        for gain in ("g_matched", "h_topk"):
            for conv in ("E", "S"):
                per_job = []
                for job in VJOBS:
                    q = next((r for r in rows if r["job"] == job and r["model"] == model and r["convention"] == conv), None)
                    b = next((r for r in rows if r["job"] == job and r["model"] == model and r["convention"] == "B"), None)
                    if q is None or b is None:
                        per_job.append(None)
                        continue
                    per_job.append({"direction": direction_category(q[gain], b[gain]),
                                    "magnitude": abs(q[gain] - b[gain]) <= MARGIN,
                                    "gap_pp": 100 * (q[gain] - b[gain]),
                                    "ranking": q["ranking"] == b["ranking"]})
                complete = [x for x in per_job if x is not None]
                if len(complete) < 3:
                    lines.append(f"{model} {gain} {conv} vs B: incomplete ({len(complete)} of 3 jobs)")
                    continue
                def word(key):
                    vals = {x[key] for x in complete}
                    return "preserved" if vals == {True} else "not preserved" if vals == {False} else "allocation-dependent"
                cats = [x["direction"] for x in complete]
                direction = DIRECTION_WORDS[cats[0]] if len(set(cats)) == 1 else \
                    "allocation-dependent (" + ", ".join(DIRECTION_WORDS[c] for c in cats) + ")"
                gaps = ", ".join("%+.1f" % x["gap_pp"] for x in complete)
                lines.append(f"{model} {gain} {conv} vs B: direction {direction}; magnitude "
                             f"{word('magnitude')} (gap in points by job: {gaps}); ranking {word('ranking')}")
    for model in ("transformer", "resnet50"):
        spreads = []
        for job in VJOBS:
            e = next((r for r in rows if r["job"] == job and r["model"] == model and r["convention"] == "E"), None)
            if e is not None:
                rhos = [e[f"rho_{p}"] for p in PATHS]
                spreads.append(max(rhos) - min(rhos))
        if len(spreads) == 3:
            word = "differs between paths" if all(s > MARGIN for s in spreads) else \
                   "similar across paths" if all(s <= MARGIN for s in spreads) else "allocation-dependent"
            lines.append(f"{model} perturbation rho_E {word} (spread by job: "
                         + ", ".join("%.3f" % s for s in spreads) + ")")
    return lines


def verdicts_b(rows):
    lines = []
    for model in ("transformer", "resnet50"):
        for h in HOOKS:
            deltas, lo_ratio, hi_ratio = [], [], []
            for job in VJOBS:
                mine = sorted((r for r in rows if r["job"] == job and r["model"] == model), key=lambda r: r["batch"])
                if len(mine) != 2 or mine[0][f"R_{h}"] is None or mine[1][f"R_{h}"] is None:
                    continue
                lo_ratio.append(mine[0][f"R_{h}"])
                hi_ratio.append(mine[1][f"R_{h}"])
                deltas.append(mine[1][f"R_{h}"] - mine[0][f"R_{h}"])
            if len(deltas) < 3:
                lines.append(f"{model} {h}: incomplete ({len(deltas)} of 3 jobs)")
                continue
            if all(abs(d) < MARGIN for d in deltas):
                word = "stable with computation"
            elif all(d >= MARGIN for d in deltas) or all(d <= -MARGIN for d in deltas):
                word = "changes with computation"
            else:
                word = "inconsistent across allocations"
            def side(values):
                if all(v <= 1 - MARGIN for v in values):
                    return "faster than dense DDP"
                if all(v >= 1 + MARGIN for v in values):
                    return "slower than dense DDP"
                if all(abs(v - 1) < MARGIN for v in values):
                    return "within 5% of dense DDP"
                return "inconsistent against dense DDP"
            lines.append(f"{model} {h}: relative cost {word} (change by job: "
                         + ", ".join("%+.3f" % d for d in deltas) + f"); small batch {side(lo_ratio)}; "
                         f"large batch {side(hi_ratio)}")
    return lines


def checks(runs, conds):
    out = []
    for job in sorted({r["job"] for r in runs}):
        anchors = {r["entry"]: r for r in runs if r["job"] == job and r["role"] == "anchor"}
        start, end = anchors.get("anchor-start"), anchors.get("anchor-end")
        if start and end and start["valid"] and end["valid"]:
            drift = end["T_ms"] / start["T_ms"] - 1.0
            out.append({"job": job, "check": "anchor drift", "value": "%.4f" % drift,
                        "flag": "FLAG" if abs(drift) > MARGIN else ""})
        else:
            out.append({"job": job, "check": "anchor drift", "value": "missing", "flag": "FLAG"})
        for c in conds:
            if c["job"] == job and c["pass_gap"] is not None and c["pass_gap"] > MARGIN:
                out.append({"job": job, "check": "pass gap " + c["condition"], "value": "%.4f" % c["pass_gap"],
                            "flag": "FLAG"})
        for c in conds:
            if c["job"] == job and c["passes"] == 1:
                out.append({"job": job, "check": "single pass " + c["condition"], "value": "1", "flag": "FLAG"})
        for r in runs:
            if r["job"] == job and r["role"] in ("timed", "anchor") and not r["valid"]:
                out.append({"job": job, "check": "excluded " + r["entry"], "value": r["reason"], "flag": "EXCLUDED"})
            if (r["job"] == job and r["role"] == "timed" and r["late_vs_early"] is not None
                    and abs(r["late_vs_early"]) > MARGIN):
                out.append({"job": job, "check": "warm-up trend " + r["entry"], "value": "%.4f" % r["late_vs_early"],
                            "flag": "FLAG"})
    return out


# ------------------------------------------------------------ output
def write_csv(path, rows):
    if not rows:
        open(path, "w").close()
        return
    fields = list(rows[0].keys())
    for r in rows[1:]:
        for k in r:
            if k not in fields:
                fields.append(k)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n", restval="")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("%.6g" % v if isinstance(v, float) else v) for k, v in r.items()})


def figures(out_dir, rows_a, rows_b):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    meta = {"CreationDate": None, "ModDate": None}
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.6), sharey=True)
    for ax, model in zip(axes, ("transformer", "resnet50")):
        for i, conv in enumerate(("E", "S", "B")):
            for gain, marker, off in (("g_matched", "o", -0.12), ("h_topk", "s", 0.12)):
                ys = [r[gain] for r in rows_a if r["model"] == model and r["convention"] == conv]
                ax.scatter([i + off] * len(ys), ys, marker=marker, s=18,
                           color="#0072B2" if gain == "g_matched" else "#D55E00",
                           label=("matched gather to reduce" if gain == "g_matched" else "Top-k to reduce")
                           if (i == 0 and model == "transformer") else None)
        ax.set_xticks([0, 1, 2], ["E (Tier 1)", "S (Tier 2)", "B (block)"])
        ax.set_title(model, fontsize=9)
        ax.axhline(0, color="#999999", lw=0.6)
    axes[0].set_ylabel("gain of the intervention")
    axes[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_v2_gain.pdf"), metadata=meta)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.6), sharey=True)
    for ax, model in zip(axes, ("transformer", "resnet50")):
        for h, color in (("ddp-fp16-hook", "#009E73"), ("ddp-int8-hook", "#D55E00")):
            for job in VJOBS:
                pts = sorted((r["batch"], r[f"R_{h}"]) for r in rows_b if r["model"] == model and r["job"] == job
                             and r[f"R_{h}"] is not None)
                if pts:
                    ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", ms=3, lw=0.9, color=color,
                            label=(h.replace("ddp-", "").replace("-", " ") if job == "V1" else None))
        ax.axhline(1.0, color="#999999", lw=0.6)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("local batch")
        ax.set_title(model, fontsize=9)
    axes[0].set_ylabel("hook step / dense DDP step")
    axes[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_v2_hooks.pdf"), metadata=meta)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("runs_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--figures", action="store_true")
    a = ap.parse_args(argv)
    os.makedirs(a.out_dir, exist_ok=True)
    runs = load_runs(a.runs_dir)
    conds = conditions(runs)
    rows_a = experiment_a(conds)
    rows_b = experiment_b(conds)
    write_csv(os.path.join(a.out_dir, "v2_runs.csv"), runs_table(runs))
    write_csv(os.path.join(a.out_dir, "v2_conditions.csv"), conds)
    write_csv(os.path.join(a.out_dir, "v2_expA.csv"), rows_a)
    write_csv(os.path.join(a.out_dir, "v2_expB.csv"), rows_b)
    write_csv(os.path.join(a.out_dir, "v2_checks.csv"), checks(runs, conds))
    lines = ["# TPDS v2 verdicts (rules frozen in ANALYSIS_RULES_v2.md)", "", "## Experiment A", ""]
    lines += ["- " + x for x in verdicts_a(rows_a)]
    lines += ["", "## Experiment B", ""] + ["- " + x for x in verdicts_b(rows_b)]
    with open(os.path.join(a.out_dir, "VERDICTS_v2.md"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")
    if a.figures:
        figures(a.out_dir, rows_a, rows_b)
    print("\n".join(lines))
    return 0


def runs_table(runs):
    return [{k: v for k, v in r.items() if k != "args"} for r in runs]


if __name__ == "__main__":
    sys.exit(main())
