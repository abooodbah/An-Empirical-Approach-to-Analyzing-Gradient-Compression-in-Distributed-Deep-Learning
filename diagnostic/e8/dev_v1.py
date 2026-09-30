# -*- coding: utf-8 -*-
"""Development check of the E8 reference on V1 (E8_PROTOCOL.md section 8):
the hook family, Transformer, 12 ranks, batch 32 and 128. Reads only the
artifact: raw/v2/v2-V1-traces-3465526.tgz (the traces), raw/v2/
v2-V1-4n-3465526.tgz (the calibration record), and the block-timed ground
truth validation_v2/results/analysis_2026-09-29/v2_expB.csv. Writes
diagnostic/e8/DEV_REPORT.md."""
import csv
import io
import json
import os
import statistics as st
import sys
import tarfile
import tempfile
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import replay_reference as R  # noqa: E402

RAW = os.path.join(ART, "raw", "v2")
TRUTH = os.path.join(ART, "validation_v2", "results", "analysis_2026-09-29", "v2_expB.csv")
G2_LIMIT = 0.15


def extract(tmp):
    with tarfile.open(os.path.join(RAW, "v2-V1-traces-3465526.tgz")) as tf:
        tf.extractall(tmp)
    with tarfile.open(os.path.join(RAW, "v2-V1-4n-3465526.tgz")) as tf:
        member = next(m for m in tf.getmembers() if "v2-V1-cal-start__" in m.name and m.name.endswith("rank0000.jsonl"))
        rec = json.loads(tf.extractfile(member).readline())
    return os.path.join(tmp, "v2-V1-4n-3465526", "traces"), rec["alpha_allreduce_s"], rec["beta_allreduce_s_per_byte"]


def trace(tdir, mode, bs, rank):
    name = next(f for f in os.listdir(tdir) if "-%s-bs%d__" % (mode, bs) in f and f.endswith("rank%04d.trace.json" % rank))
    return os.path.join(tdir, name)


def main():
    truth = defaultdict(dict)
    with io.open(TRUTH, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["model"] == "transformer":
                truth[int(r["batch"])][r["job"]] = (float(r["T_ddp-int8-hook_ms"]), float(r["T_ddp-fp16-hook_ms"]))
    lines = []
    with tempfile.TemporaryDirectory() as tmp:
        tdir, alpha, beta = extract(tmp)
        lines += ["# E8 reference: development report (V1, hook family)", "",
                  "Frozen with the reference. Written by `diagnostic/e8/dev_v1.py` from the",
                  "artifact's archives only. Calibration: V1 cal-start, alpha %.3g s, beta %.4g s/byte." % (alpha, beta), "",
                  "## Replay of the traced originals and candidates", "",
                  "| trace | replayed step (ms) | traced step (ms) | ratio |", "|---|---|---|---|"]
        for bs in (32, 128):
            for mode in ("ddp-int8-hook", "ddp-fp16-hook"):
                for rank in (0, 3):
                    res = R.replay(R.load(trace(tdir, mode, bs, rank)))
                    traced = st.median(res.traced_intervals_ms)
                    lines.append("| %s batch %d rank %d | %.2f | %.2f | %.3f |" % (mode, bs, rank, res.step_ms, traced, res.step_ms / traced))
        lines += ["", "## Buckets found in the int8-hook traces", ""]
        for bs in (32, 128):
            bk = R.hook_buckets(R.load(trace(tdir, "ddp-int8-hook", bs, 0)))
            ns = sorted({b["n"] for b in bk})
            lines.append("- batch %d, rank 0: %d hook regions in 8 iterations, bucket sizes %s elements" % (bs, len(bk), ", ".join(str(n) for n in ns)))
        lines += ["", "## Predictions against the block-timed ground truth", "",
                  "Predicted candidate step = measured original step x (1 - g_pred), per allocation.", "",
                  "| batch | g_pred | allocation | T int8 (ms) | T fp16 measured (ms) | T fp16 predicted (ms) | error |",
                  "|---|---|---|---|---|---|---|"]
        errors = []
        for bs in (32, 128):
            g_pred, det = R.predict([trace(tdir, "ddp-int8-hook", bs, r) for r in (0, 3)], "H", alpha, beta)
            for job in ("V1", "V2", "V3"):
                t8, t16 = truth[bs][job]
                pred = t8 * (1.0 - g_pred)
                err = (pred - t16) / t16
                errors.append(abs(err))
                lines.append("| %d | %.3f | %s | %.2f | %.2f | %.2f | %+.1f%% |" % (bs, g_pred, job, t8, t16, pred, 100 * err))
        med = st.median(errors)
        verdict = "passes" if med <= G2_LIMIT else "fails"
        lines += ["", "## Gate G2", "",
                  "Hook family (H): median absolute error %.1f%% over %d development points; G2 %s (limit 15%%)." % (100 * med, len(errors), verdict),
                  "",
                  "Gather family (G): no development trace exists at the freeze. The reference is",
                  "reported for G as \"not validated on development data\" unless the e7a traces",
                  "arrive before any E8 record is read; the frozen code is then applied to them and",
                  "the errors go to DEV_REPORT_E7A.md.", "",
                  "## Known bias (reported, not corrected)", "",
                  "The substituted all-reduce is costed from the isolated calibration, as section 7",
                  "of the protocol fixes. The V1 fp16 traces, used here only as a diagnostic, show",
                  "the in-training all-reduce kernels running longer than the calibration predicts:", "",
                  "| batch | bucket elements | real all-reduce (us) | calibrated (us) | ratio |", "|---|---|---|---|---|"]
        for bs in (32, 128):
            t16 = R.load(trace(tdir, "ddp-fp16-hook", bs, 0))
            by_n = defaultdict(list)
            for g in t16.gpu.values():
                if "AllReduce" in g["name"]:
                    by_n[int(g["args"].get("In msg nelems", 0))].append(g["dur"])
            for n, v in sorted(by_n.items()):
                cal = 1e6 * (alpha + beta * 2 * n)
                lines.append("| %d | %d | %.0f | %.0f | %.2f |" % (bs, n, st.median(v), cal, st.median(v) / cal))
        lines += ["", "The reference therefore predicts a cheaper candidate than the one measured, and",
                  "its gains are optimistic; the error table above includes this bias.", ""]
    out = os.path.join(HERE, "DEV_REPORT.md")
    with io.open(out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
