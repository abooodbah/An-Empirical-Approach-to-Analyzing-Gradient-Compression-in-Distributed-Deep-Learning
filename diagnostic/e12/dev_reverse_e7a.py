# -*- coding: utf-8 -*-
"""Development check of the E12 reverse substitutions (replay_reverse.py) on
the e7a traces, the same way diagnostic/e8/dev_e7a.py checked the frozen
forward direction: both reverse families, both models, p = 6 at batch 128.
Ground truth: E7's wall-timed pairs (the original and the candidate) from the
frozen extractor, in the traced allocation (e7a) and in the other E7
allocation (e7b). No E12 record exists or is read. Writes
diagnostic/e12/DEV_REPORT_REVERSE.md.
Usage: python dev_reverse_e7a.py <rev-e7a run dir> <rev-e7b run dir>"""
import csv
import glob
import io
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DIAG = os.path.dirname(HERE)
for _p in (DIAG, os.path.join(DIAG, "e8"), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import features as F  # noqa: E402  frozen extractor
import replay_reverse as RV  # noqa: E402

FAMILIES = (("G-rev", "gpu-quant-reducible", "gpu-quant-gathersum"),
            ("H-rev", "ddp-fp16-hook", "ddp-int8-hook"))
KEYS = ("alpha_allreduce_s", "beta_allreduce_s_per_byte", "alpha_allgather_s", "beta_allgather_s_per_byte")


def calibration(run_dir):
    for path in sorted(glob.glob(os.path.join(run_dir, "results", "*.csv"))):
        with io.open(F.openable(path), newline="", encoding="utf-8") as fh:
            row = next(csv.DictReader(fh), None) or {}
        if all(row.get(k) not in ("", None) for k in KEYS):
            return {k: float(row[k]) for k in KEYS}
    return None


def wall_steps(run_dir):
    rows = F.features(F.load(run_dir))
    return {(r["model"], r["mode"]): r["step_ms"] for r in rows if r["timing"] == "wall" and r["tau"] == 1}


def traces(run_dir, model, mode):
    return sorted(p for r in (0, 3) for p in glob.glob(os.path.join(
        run_dir, "traces", "*-trace-%s-%s__*rank%04d.trace.json" % (model, mode, r))))


def main(e7a, e7b, out=os.path.join(HERE, "DEV_REPORT_REVERSE.md")):
    cal = calibration(e7a)
    steps = {os.path.basename(os.path.normpath(d)): wall_steps(d) for d in (e7a, e7b)}
    lines = ["# E12 reverse substitutions: development report on the e7a traces", "",
             "diagnostic/e12/replay_reverse.py applied to the e7a traces (p = 6, batch 128),",
             "before any E12 job exists. Calibration: e7a, all-gather alpha %.3g s, beta %.4g s/byte." % (
                 cal["alpha_allgather_s"], cal["beta_allgather_s_per_byte"]),
             "Ground truth: E7's wall-timed pairs. Error = (predicted - measured) / measured candidate step.", "",
             "| family | model | g_pred | allocation | T original (ms) | T candidate measured (ms) | measured g | predicted (ms) | error |",
             "|---|---|---|---|---|---|---|---|---|"]
    errors = {"G-rev": [], "H-rev": []}
    signs = {"G-rev": [], "H-rev": []}
    for fam, orig, cand in FAMILIES:
        for model in ("transformer", "resnet50"):
            paths = traces(e7a, model, orig)
            if not paths:
                lines.append("| %s | %s | no trace | | | | | | |" % (fam, model))
                continue
            try:
                g_pred, _ = RV.predict(paths, fam, cal)
            except Exception as exc:
                lines.append("| %s | %s | replay failed: %s | | | | | | |" % (fam, model, repr(exc)[:80]))
                continue
            for name, st_ in steps.items():
                to, tc = st_.get((model, orig)), st_.get((model, cand))
                if to is None or tc is None:
                    lines.append("| %s | %s | %.3f | %s | missing | | | | |" % (fam, model, g_pred, name))
                    continue
                pred = to * (1.0 - g_pred)
                err = (pred - tc) / tc
                errors[fam].append(abs(err))
                g = 1.0 - tc / to
                signs[fam].append((g_pred >= 0.05) == (g >= 0.05))
                lines.append("| %s | %s | %.3f | %s | %.2f | %.2f | %.3f | %.2f | %+.1f%% |" % (
                    fam, model, g_pred, name, to, tc, g, pred, 100 * err))
    lines += ["", "## Summary", ""]
    for fam in ("G-rev", "H-rev"):
        e = errors[fam]
        if e:
            lines.append("%s: median absolute error %.1f%% over %d points (largest %.1f%%); the replay's "
                         "recommendation agrees with the measured outcome on %d of %d points." % (
                             fam, 100 * st.median(e), len(e), 100 * max(e), sum(signs[fam]), len(signs[fam])))
        else:
            lines.append("%s: no development point." % fam)
    io.open(out, "w", encoding="utf-8", newline="\n").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
