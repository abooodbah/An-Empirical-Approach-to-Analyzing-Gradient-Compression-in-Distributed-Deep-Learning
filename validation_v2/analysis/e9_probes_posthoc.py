# -*- coding: utf-8 -*-
"""E9, the probes read a second time. POST HOC (2026-10-08): written after
the E9 records were read. Nothing here changes a frozen file or a number the
frozen scorer wrote.

Two readings the frozen scorer (score_e9.py) does not make.

1. The forced-tree probe. Every E9 job ran three probes: the default
   algorithm, NCCL_ALGO=Ring, and NCCL_ALGO=Tree for all-reduce only. The
   frozen scorer fits the first two and does not read the third.

2. The fit range. The protocol's slope is the least-squares line through the
   points from 1 MiB to 64 MiB, and the probe grid has three points there:
   1, 16 and 64 MiB. At 24 ranks over GPUDirect the 1 MiB all-reduce lies
   above the line of the larger payloads. Under the forced ring it took
   longer than the 16 MiB all-reduce; that point lowers the fitted ring
   slope, which is the denominator of P1 and P2. P3 is scored on the default
   probe, whose own 1 MiB point is also high (as long as its 16 MiB point),
   so the forced ring's point does not enter P3. This script repeats every
   ratio and every prediction error with the line through 16 MiB and 64 MiB
   (rev4_records.slope, the same rule on a shorter range), and with two
   other ranges that leave the 1 MiB point out (16 to 256 MiB, 64 to
   256 MiB), so that a reader can see how much the second reading depends on
   the range chosen. It prints the measured points.

Under the protocol's own range this script repeats the frozen scorer's
numbers (tests/test_e9_probes_posthoc.py compares them). A probe file that
cannot be read is named in the output and skipped.

usage: python e9_probes_posthoc.py <run_dir> [<run_dir> ...] --json OUT.json
"""
import io
import json
import os
import sys

import rev4_records as R

MIB = 1 << 20
# name, lowest and highest payload of the line. The first is the protocol's.
RANGES = (("rule_1_to_64_mib", 1 * MIB, 64 * MIB), ("from_16_to_64_mib", 16 * MIB, 64 * MIB),
          ("from_16_to_256_mib", 16 * MIB, 256 * MIB), ("from_64_to_256_mib", 64 * MIB, 256 * MIB))


def read(run_dir, name, unreadable):
    path = os.path.join(run_dir, "probes", name + ".json")
    if not os.path.exists(path):
        return None
    try:
        return json.load(io.open(path, encoding="utf-8"))
    except ValueError:
        unreadable.append(name + ".json")
        return None


def series(probe, collective):
    if not probe or collective not in probe or not probe[collective].get("sizes_bytes"):
        return None
    return probe[collective]["sizes_bytes"], probe[collective]["times_s"]


def line(probe, collective, low, high):
    """(seconds per byte, intercept) of the least-squares line through the
    points from low to high, by the protocol's own rule (rev4_records.slope)."""
    s = series(probe, collective)
    if s is None:
        return None
    try:
        return R.slope(s[0], s[1], low=low, high=high)
    except ValueError:
        return None


def points_ms(probe, collective):
    s = series(probe, collective)
    if s is None:
        return None
    return {"%d" % (size >> 20): 1e3 * t for size, t in zip(*s) if size >= MIB}


def prediction_errors(default, low, high):
    """P3's rule: the line of the default probe against its measured points
    above the line's last point, as a fraction of the measured time."""
    errors = {}
    for name in ("allreduce", "allgather"):
        fit, s = line(default, name, low, high), series(default, name)
        if fit is None:
            continue
        b, a = fit
        for size, t in zip(*s):
            if size > high:
                errors["%s %d MiB" % (name, size >> 20)] = abs(a + b * size - t) / t
    return errors


def reading(default, ring, tree, low, high):
    out = {}
    ar, ag = line(ring, "allreduce", low, high), line(ring, "allgather", low, high)
    de, tr = line(default, "allreduce", low, high), line(tree, "allreduce", low, high)
    if ar:
        out["ring_allreduce_ns_per_byte"] = 1e9 * ar[0]
        if ag:
            out["allgather_ns_per_byte"] = 1e9 * ag[0]
            out["gather_over_ring"] = ag[0] / ar[0]
        if de:
            out["default_over_ring"] = de[0] / ar[0]
        if tr:
            out["tree_allreduce_ns_per_byte"] = 1e9 * tr[0]
            out["tree_over_ring"] = tr[0] / ar[0]
    if ag and tr:
        out["gather_over_tree"] = ag[0] / tr[0]
    errors = prediction_errors(default, low, high)
    if errors:
        out["prediction_errors"] = errors
    return out


def transport(run_dir, prefix, unreadable):
    default, ring, tree = (read(run_dir, prefix + name, unreadable) for name in ("default", "ring", "tree"))
    if not (default or ring or tree):
        return None
    out = {name: reading(default, ring, tree, low, high) for name, low, high in RANGES}
    points = {"default_allreduce": points_ms(default, "allreduce"), "ring_allreduce": points_ms(ring, "allreduce"),
              "tree_allreduce": points_ms(tree, "allreduce"), "ring_allgather": points_ms(ring, "allgather")}
    out["points_ms"] = {k: v for k, v in points.items() if v}
    ring_ms, tree_ms = points["ring_allreduce"] or {}, points["tree_allreduce"] or {}
    if "1" in ring_ms and "16" in ring_ms:
        out["ring_allreduce_1_mib_over_16_mib"] = ring_ms["1"] / ring_ms["16"]
    if "1024" in ring_ms and "1024" in tree_ms:
        out["tree_over_ring_time_at_1024_mib"] = tree_ms["1024"] / ring_ms["1024"]
    return out


def allocation(run_dir):
    unreadable = []
    any_probe = read(run_dir, "ring", unreadable) or read(run_dir, "default", unreadable) or {}
    out = {"run_dir": os.path.basename(os.path.normpath(run_dir)), "world": any_probe.get("world_size")}
    for key, prefix in (("gdr", ""), ("host_staged", "hs-")):
        t = transport(run_dir, prefix, unreadable)
        if t:
            out[key] = t
    if unreadable:
        out["unreadable"] = sorted(set(unreadable))
    return out


def main(argv):
    out = argv[argv.index("--json") + 1] if "--json" in argv else None
    dirs = [a for a in argv if not a.startswith("--") and a != out]
    result = {"label": "POST HOC: the E9 probes read after the records. rule_1_to_64_mib repeats the frozen "
                       "scorer's range; from_16_to_64_mib, from_16_to_256_mib and from_64_to_256_mib leave out "
                       "the 1 MiB point; the tree probe is not part of the frozen scorer",
              "allocations": [allocation(d) for d in dirs]}
    if out:
        json.dump(result, io.open(out, "w", encoding="utf-8", newline="\n"), indent=1, sort_keys=True)
    for a in result["allocations"]:
        for key in ("gdr", "host_staged"):
            if key not in a:
                continue
            rule, late = a[key]["rule_1_to_64_mib"], a[key]["from_16_to_64_mib"]
            nan = float("nan")
            print("%-22s %2s ranks %-11s gather/ring %5.2f by the rule, %5.2f from 16 MiB; default/ring %.2f, %.2f; "
                  "tree/ring %5.2f, %5.2f; ring 1 MiB / 16 MiB time %.2f; tree/ring time at 1 GiB %.2f"
                  % (a["run_dir"], a["world"], key, rule.get("gather_over_ring", nan), late.get("gather_over_ring", nan),
                     rule.get("default_over_ring", nan), late.get("default_over_ring", nan),
                     rule.get("tree_over_ring", nan), late.get("tree_over_ring", nan),
                     a[key].get("ring_allreduce_1_mib_over_16_mib", nan),
                     a[key].get("tree_over_ring_time_at_1024_mib", nan)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
