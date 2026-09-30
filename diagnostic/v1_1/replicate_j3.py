# -*- coding: utf-8 -*-
"""24-rank replication of the E6 scored pairs (cases 1, 2, 4, 6, 7) and the
hook-to-dense-DDP ratio at 24 ranks. For each allocation: delta, the frozen
verdict's outcome under that delta, and the verdict of the allocation's own
M1 (trend against the E6 12-rank baseline, as E6 computed it).
Usage: python replicate_j3.py <features_r4.json> <features_rev.json> <rep1,rep2>"""
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DIAG = os.path.dirname(HERE)
sys.path.insert(0, DIAG)
import rules  # noqa: E402
import score_e6 as S  # noqa: E402

ORIGINAL = "3459984"
CASES = [c for c in S.MANIFEST if c[0] in (1, 2, 4, 6, 7)]


def find(rows, model, mode, timing, alloc):
    for x in rows:
        if (x["model"], x["p"], x["mode"], x["timing"], x["tau"], x["alloc"]) == \
                (model, 24, mode, timing, 1, alloc):
            return x
    return None


def pair_delta(rows, model, orig, intv, alloc):
    m2, m3 = find(rows, model, orig, "wall", alloc), find(rows, model, intv, "wall", alloc)
    if m2 is None or m3 is None or m2["step_ms"] <= 0:
        return None
    return (m2["step_ms"] - m3["step_ms"]) / m2["step_ms"]


def report(features_r4, features_rev, rep_allocs):
    r4 = [x for x in json.load(io.open(features_r4, encoding="utf-8")) if x["tag"] == "R"]
    rev = json.load(io.open(features_rev, encoding="utf-8"))
    out = ["24-rank replication of the E6 scored pairs", ""]
    for (c, model, p, orig, intv, to, bo, ti, bi, fam) in CASES:
        m1 = S.find(r4, model, p, orig, "phase", to, bo)
        v, _n, e = rules.diagnose(m1, S.trend_for(r4, m1))
        for al in [ORIGINAL] + rep_allocs:
            rows = r4 if al == ORIGINAL else rev
            d = pair_delta(rows, model, orig, intv, al)
            own = find(rows, model, orig, "phase", al)
            own_v = rules.diagnose(own, S.trend_for(r4 + [own], own))[0] if own else "missing"
            oc = S.outcome(v, e, d, None) if d is not None else "missing"
            out.append("case %d %-11s alloc=%-8s delta=%s frozen-verdict=%s -> %s | own M1 verdict=%s" % (
                c, model, al, "  -  " if d is None else "%+.3f" % d, v, oc, own_v))
    out += ["", "hook / dense DDP step at 24 ranks"]
    for model in ("transformer", "resnet50"):
        for al in rep_allocs:
            dense = find(rev, model, "ddp-overlap", "wall", al)
            for hook in ("ddp-fp16-hook", "ddp-int8-hook"):
                t = find(rev, model, hook, "wall", al)
                ratio = "missing" if not (dense and t) else "%.3f" % (t["step_ms"] / dense["step_ms"])
                out.append("%-11s alloc=%-8s %-14s %s" % (model, al, hook, ratio))
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    text = report(sys.argv[1], sys.argv[2], sys.argv[3].split(","))
    io.open(os.path.join(HERE, "REPLICATION_24.txt"), "w", encoding="utf-8", newline="\n").write(text)
    print(text)
