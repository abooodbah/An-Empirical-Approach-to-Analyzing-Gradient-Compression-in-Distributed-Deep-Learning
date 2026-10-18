# -*- coding: utf-8 -*-
"""Go or no-go after the E12 qualification job (diagnostic/E12_PROTOCOL.md
section 8). Reads only the qualification run directory: return codes,
fidelity sidecars, the two BF16 int8-hook traces (parsed by the frozen
replay) and the two BF16 16-bit-hook traces (parsed by the reverse
substitution of mirror M12), never replayed for a prediction, the two timed
block runs and the timeline. No scored outcome exists at this point.

Decision, fixed in advance:
  stop    a check failed, or the projected main job exceeds 43 minutes
  go-40   every check passed and the projection is at most 36 minutes:
          submit ls6_e12a.slurm and ls6_e12b.slurm as written (40 minutes)
  go-45   every check passed and the projection is above 36 and at most 43
          minutes: submit both with sbatch --time=00:45:00

Usage: python qual_check.py <e12-q run dir>
"""
import io
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import score_e12 as S  # noqa: E402

QUAL_CONDITIONS = 24
PLAN_QUAL_CONDITIONS_S = 522.0    # section 8: the qualification job's 24 conditions
PLAN_MAIN_CONDITIONS_S = 2002.0   # section 8: a scored job's 86 conditions
PLAN_OVERHEAD_S = 90.0            # environment selection, node facts, srun start-up
PLAN_CPU_QUANT_BLOCK_S = 64.0     # section 8: the ResNet-50 cpu-quant block run
GO_40_MAX_MIN = 36.0
GO_45_MAX_MIN = 43.0


def timeline(qual_dir):
    """{condition: seconds} and the span from the first start to the last end."""
    starts, secs, first, last = {}, {}, None, None
    path = os.path.join(qual_dir, "provenance", "timeline.txt")
    if not os.path.exists(path):
        return secs, None
    for line in io.open(path, encoding="utf-8"):
        parts = line.split()
        if len(parts) < 3:
            continue
        t = time.mktime(time.strptime(parts[0], "%Y-%m-%dT%H:%M:%SZ"))
        first = t if first is None else min(first, t)
        last = t if last is None else max(last, t)
        if parts[1] == "start":
            starts[parts[2]] = t
        elif parts[1] == "done" and parts[2] in starts:
            secs[parts[2]] = t - starts[parts[2]]
    return secs, (last - first) if first is not None else None


def return_codes(qual_dir):
    path = os.path.join(qual_dir, "provenance", "conditions.txt")
    if not os.path.exists(path):
        return {}
    out = {}
    for line in io.open(path, encoding="utf-8"):
        if " rc=" in line:
            name, rc = line.strip().rsplit(" rc=", 1)
            out[name] = int(rc)
    return out


def default_parse(path):
    """Number of int8-hook buckets the frozen replay finds in a trace."""
    return len(S.REF.hook_buckets(S.REF.load(path)))


def default_rev_parse(path):
    """Number of 16-bit-hook buckets the reverse substitution finds in a trace."""
    return len(S.REV.fp16_buckets(S.REF.load(path)))


def check(qual_dir, parse=default_parse, rev_parse=default_rev_parse):
    problems = []
    rcs = return_codes(qual_dir)
    if len(rcs) != QUAL_CONDITIONS:
        problems.append("expected %d conditions, found %d" % (QUAL_CONDITIONS, len(rcs)))
    problems += ["%s returned %d" % (c, rc) for c, rc in sorted(rcs.items()) if rc != 0]
    err = S.fidelity_errors(qual_dir)
    for model in S.MODELS:
        for tag in S.TAGS:
            if (model, tag) not in err:
                problems.append("no complete fidelity record for %s %s" % (model, tag))
        for tag, fn, who in (("ddp-int8-hook-bf16", parse, "the frozen replay"),
                             ("ddp-fp16-hook-bf16", rev_parse, "the reverse substitution")):
            paths = S.trace_paths(qual_dir, model, tag)
            if len(paths) != 2:
                problems.append("%s %s: %d trace files, expected 2" % (model, tag, len(paths)))
            for p in paths:
                try:
                    if fn(p) < 1:
                        problems.append("%s finds no hook bucket in %s" % (who, os.path.basename(p)))
                except Exception as exc:  # noqa: BLE001
                    problems.append("%s cannot parse %s: %r" % (who, os.path.basename(p), exc))
    for model, tag in (("resnet50", "cpu-quant"), ("transformer", "ddp-overlap-bf16")):
        if S.block_pass(qual_dir, S.rid(model, tag, "block", 1), tag) is None:
            problems.append("block run %s %s is not valid" % (model, tag))
    secs, span = timeline(qual_dir)
    cpu_block = secs.get("block-resnet50-cpu-quant-p1")
    projection = None
    if span is None or cpu_block is None:
        problems.append("timeline incomplete: no projection")
    else:
        scale = max(span / PLAN_QUAL_CONDITIONS_S, cpu_block / PLAN_CPU_QUANT_BLOCK_S)
        projection = (PLAN_OVERHEAD_S + PLAN_MAIN_CONDITIONS_S * scale) / 60.0
    if problems or projection is None or projection > GO_45_MAX_MIN:
        decision = "stop"
    elif projection > GO_40_MAX_MIN:
        decision = "go-45"
    else:
        decision = "go-40"
    return {"decision": decision, "projection_min": projection, "problems": problems,
            "span_s": span, "cpu_quant_block_s": cpu_block, "fidelity": {"%s/%s" % k: v for k, v in err.items()}}


if __name__ == "__main__":
    out = check(sys.argv[1])
    print("decision: %s" % out["decision"])
    print("projected scored job: %s minutes" % (None if out["projection_min"] is None else round(out["projection_min"], 1)))
    for p in out["problems"]:
        print("problem: " + p)
