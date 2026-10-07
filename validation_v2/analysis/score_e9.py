# validation_v2/analysis/score_e9.py
# -*- coding: utf-8 -*-
"""E9 scorer (diagnostic/E9_PROTOCOL.md). Frozen before submission.

usage: python score_e9.py <run_dir> [<run_dir> ...] --json OUT.json
"""
import glob
import io
import json
import os
import re
import sys

import rev4_records as R

BASELINES = ("native", "fp16-native", "ddp-overlap")
DDP = ("ddp-overlap", "ddp-fp16-hook", "ddp-int8-hook")
BLOCK = re.compile(r"^redE9-ws(\d+)-(transformer|resnet50)-(.+)-block-p(\d)$")
PHASE = re.compile(r"^redE9-ws(\d+)-(transformer|resnet50)-(.+)-phase$")
FID = re.compile(r"^redE9-ws(\d+)-(transformer|resnet50)-(.+)-fid$")
EQUIV = re.compile(r"^redE9-ws(\d+)-(transformer|resnet50)-(.+)-equiv$")
MARGIN = 0.05


def block_table(runs):
    acc = {}
    for run_id, run in runs.items():
        m = BLOCK.match(run_id)
        if m and R.valid(run["rows"]):
            acc.setdefault((m.group(2), m.group(3)), []).append(R.block_step_ms(run["rows"]))
    return {key: {"T_ms": sum(v) / len(v), "passes": len(v)} for key, v in acc.items()}


def ratios(table):
    out = {}
    for (model, mode), cell in table.items():
        for base in BASELINES:
            ref = table.get((model, base))
            if ref and mode != base:
                out[(model, mode, base)] = cell["T_ms"] / ref["T_ms"]
    return out


def call(values):
    if len(values) < 2:
        return "single allocation"
    if all(v <= 1 - MARGIN for v in values):
        return "faster"
    if all(v >= 1 + MARGIN for v in values):
        return "slower"
    if all(1 - MARGIN < v < 1 + MARGIN for v in values):
        return "within 5%"
    return "allocation-dependent"


def fidelity(runs):
    out = {}
    for run_id, run in runs.items():
        m = FID.match(run_id)
        side = run["sidecar"].get(0)
        if not m or not side or not side["fidelity"]:
            continue
        f = side["fidelity"]
        out[(m.group(2), m.group(3))] = {
            "relative_l2_error": sum(x["relative_l2_error"] for x in f) / len(f),
            "cosine": sum(x["cosine"] for x in f) / len(f),
            "levels_per_side": side["levels_per_side"], "iterations": len(f)}
    return out


def exchange(runs):
    out = {}
    for run_id, run in runs.items():
        m = PHASE.match(run_id)
        if m and R.valid(run["rows"]):
            out[(m.group(2), m.group(3))] = R.exchange_ms(run["rows"])
    return out


def equivalence(runs):
    out = {}
    for run_id, run in runs.items():
        m = EQUIV.match(run_id)
        if m and run["rows"]:
            out[(m.group(2), m.group(3))] = all(r["equivalence_ok"] == "True" for r in run["rows"])
    return out


def probe_checks(probes, p):
    out = {}
    ring, default = probes.get("ring"), probes.get("default")
    if ring:
        b_ag, _ = R.slope(ring["allgather"]["sizes_bytes"], ring["allgather"]["times_s"])
        b_ar, _ = R.slope(ring["allreduce"]["sizes_bytes"], ring["allreduce"]["times_s"])
        ratio = b_ag / b_ar
        out["P1"] = {"slope_ratio": ratio, "model": p / 2.0,
                     "holds": (p / 2.0) / 1.5 <= ratio <= (p / 2.0) * 1.5}
        if default:
            b_default, _ = R.slope(default["allreduce"]["sizes_bytes"], default["allreduce"]["times_s"])
            out["P2"] = {"default_over_ring": b_default / b_ar, "holds": b_default <= 1.05 * b_ar}
    if default:
        errors = {}
        for name in ("allreduce", "allgather"):
            sizes, times = default[name]["sizes_bytes"], default[name]["times_s"]
            b, a = R.slope(sizes, times)
            for s, t in zip(sizes, times):
                if s > (64 << 20):
                    errors["%s %d MiB" % (name, s >> 20)] = abs(a + b * s - t) / t
        out["P3"] = {"errors": errors, "holds": bool(errors) and all(e <= 0.15 for e in errors.values())}
    return out


def ddp_buckets_match(run_dir):
    """Whether the bucket structure in the provenance is identical for the
    three DDP modes: True, False, or None when it is not recorded."""
    seen = {}
    for path in glob.glob(os.path.join(run_dir, "results", "*block-p1__*rank0000.provenance.json")):
        name = os.path.basename(path)
        mode = next((m for m in DDP if "-%s-block" % m in name), None)
        if mode is None:
            continue
        record = json.load(io.open(path, encoding="utf-8"))
        ddp = record.get("ddp")
        buckets = ddp.get("logging_data_buckets") if isinstance(ddp, dict) else None
        if isinstance(buckets, dict):
            # the hook's own name differs by design; compare the bucket fields only
            sizes = {k: v for k, v in buckets.items() if "bucket" in k}
            model = "resnet50" if "-resnet50-" in name else "transformer"
            seen.setdefault(model, {})[mode] = json.dumps(sizes, sort_keys=True)
    if not seen or any(len(v) < 3 for v in seen.values()):
        return None
    return all(len(set(v.values())) == 1 for v in seen.values())


def allocation(run_dir):
    runs = R.read_dir(run_dir)
    worlds = {int(BLOCK.match(k).group(1)) for k in runs if BLOCK.match(k)}
    world = worlds.pop() if len(worlds) == 1 else None
    probes = {}
    for path in glob.glob(os.path.join(run_dir, "probes", "*.json")):
        probes[os.path.splitext(os.path.basename(path))[0]] = json.load(io.open(path, encoding="utf-8"))
    table = block_table(runs)
    gdr = {k: v for k, v in probes.items() if not k.startswith("hs-")}
    host = {k[3:]: v for k, v in probes.items() if k.startswith("hs-")}
    return {"run_dir": os.path.basename(os.path.normpath(run_dir)), "world": world,
            "table": {"%s|%s" % k: v for k, v in table.items()},
            "ratios": {"%s|%s|%s" % k: v for k, v in ratios(table).items()},
            "fidelity": {"%s|%s" % k: v for k, v in fidelity(runs).items()},
            "exchange_ms": {"%s|%s" % k: v for k, v in exchange(runs).items()},
            "equivalence": {"%s|%s" % k: v for k, v in equivalence(runs).items()},
            "ddp_buckets_match": ddp_buckets_match(run_dir),
            "dtype_check": (probes.get("dtype") or {}).get("dtype_check"),
            "probe_checks": {"gdr": probe_checks(gdr, world) if world else {},
                             "host_staged": probe_checks(host, world) if world else {}}}


def score(run_dirs):
    allocations = [allocation(d) for d in run_dirs]
    summary = {}
    for world in sorted({a["world"] for a in allocations if a["world"]}):
        group = [a for a in allocations if a["world"] == world]
        keys = sorted({k for a in group for k in a["ratios"]})
        summary[str(world)] = {k: {"values": [a["ratios"][k] for a in group if k in a["ratios"]],
                                   "call": call([a["ratios"][k] for a in group if k in a["ratios"]])}
                               for k in keys}
    return {"allocations": allocations, "summary": summary}


def main(argv):
    dirs = [a for a in argv if not a.startswith("--")]
    out = argv[argv.index("--json") + 1] if "--json" in argv else None
    dirs = [d for d in dirs if d != out]
    result = score(dirs)
    if out:
        json.dump(result, io.open(out, "w", encoding="utf-8"), indent=1, sort_keys=True)
    for world, table in result["summary"].items():
        print("== %s ranks" % world)
        for key, cell in table.items():
            print("%-48s %s  %s" % (key, " ".join("%.3f" % v for v in cell["values"]), cell["call"]))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
