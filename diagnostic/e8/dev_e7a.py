# -*- coding: utf-8 -*-
"""Development check of the FROZEN E8 reference on the e7a traces
(E8_PROTOCOL.md section 8), run before any E8 record is read. Both families,
both models, p = 6 at batch 128. Ground truth: E7's wall-timed pairs (M2 of
the original, M3 of the candidate) from the frozen extractor, in the traced
allocation (e7a) and in the other E7 allocation (e7b). Writes
diagnostic/e8/DEV_REPORT_E7A.md.
Usage: python dev_e7a.py <rev-e7a run dir> <rev-e7b run dir>"""
import glob
import io
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DIAG = os.path.dirname(HERE)
sys.path.insert(0, DIAG)
sys.path.insert(0, HERE)
import features as F  # noqa: E402  frozen extractor
import replay_reference as R  # noqa: E402  frozen reference
import score_e8 as S  # noqa: E402  frozen scorer (calibration reader)

FAMILIES = (("G", "gpu-quant-gathersum", "gpu-quant-reducible"), ("H", "ddp-int8-hook", "ddp-fp16-hook"))
G2_LIMIT = 0.15


def wall_steps(run_dir):
    rows = F.features(F.load(run_dir))
    return {(r["model"], r["mode"]): r["step_ms"] for r in rows if r["timing"] == "wall" and r["tau"] == 1}


def traces(run_dir, model, mode):
    return sorted(p for r in (0, 3) for p in glob.glob(os.path.join(
        run_dir, "traces", "*-trace-%s-%s__*rank%04d.trace.json" % (model, mode, r))))


def main(e7a, e7b):
    cal = S.calibration(e7a)
    steps = {os.path.basename(os.path.normpath(d)): wall_steps(d) for d in (e7a, e7b)}
    lines = ["# E8 reference: development report on the e7a traces", "",
             "The frozen reference (diagnostic/e8/E8_FREEZE.md) applied, before any E8 record",
             "was read, to the e7a traces (p = 6, batch 128). Calibration: e7a, alpha %.3g s," % cal[0],
             "beta %.4g s/byte. Ground truth: E7's wall-timed pairs." % cal[1], "",
             "| family | model | g_pred | allocation | T original (ms) | T candidate measured (ms) | predicted (ms) | error |",
             "|---|---|---|---|---|---|---|---|"]
    errors = {"G": [], "H": []}
    for fam, orig, cand in FAMILIES:
        for model in ("transformer", "resnet50"):
            paths = traces(e7a, model, orig)
            if not paths:
                lines.append("| %s | %s | no trace | | | | | |" % (fam, model))
                continue
            try:
                g_pred, _ = R.predict(paths, fam, cal[0], cal[1])
            except Exception as exc:
                lines.append("| %s | %s | replay failed: %s | | | | | |" % (fam, model, repr(exc)[:80]))
                continue
            for name, st_ in steps.items():
                to, tc = st_.get((model, orig)), st_.get((model, cand))
                if to is None or tc is None:
                    lines.append("| %s | %s | %.3f | %s | missing | | | |" % (fam, model, g_pred, name))
                    continue
                pred = to * (1.0 - g_pred)
                err = (pred - tc) / tc
                errors[fam].append(abs(err))
                lines.append("| %s | %s | %.3f | %s | %.2f | %.2f | %.2f | %+.1f%% |" % (
                    fam, model, g_pred, name, to, tc, pred, 100 * err))
    lines += ["", "## Gate G2", ""]
    for fam, label in (("G", "Gather family (G)"), ("H", "Hook family (H)")):
        e = errors[fam]
        if e:
            med = st.median(e)
            lines.append("%s: median absolute error %.1f%% over %d points; G2 %s on these points." % (
                label, 100 * med, len(e), "passes" if med <= G2_LIMIT else "fails"))
        else:
            lines.append("%s: no development point." % label)
    out = os.path.join(HERE, "DEV_REPORT_E7A.md")
    io.open(out, "w", encoding="utf-8", newline="\n").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
