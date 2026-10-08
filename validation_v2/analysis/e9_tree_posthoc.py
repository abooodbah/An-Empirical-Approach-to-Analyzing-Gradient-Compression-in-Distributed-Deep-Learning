# -*- coding: utf-8 -*-
"""E9, the forced-tree probe. POST HOC (2026-10-08): written after the E9
records were read.

Every E9 job ran three probes: the default algorithm, NCCL_ALGO=Ring, and
NCCL_ALGO=Tree for all-reduce only. The frozen scorer (score_e9.py) fits the
default and ring probes and does not read the tree probe. This script reads
it, with the slope rule the protocol fixed for the other two (least-squares
line through the points from 1 MiB to 64 MiB, rev4_records.slope), and
sets the tree all-reduce slope against the ring all-reduce slope and the
all-gather slope of the same allocation. Nothing here changes a frozen file
or a number the frozen scorer wrote. A probe file that cannot be read is
named in the output and skipped.

usage: python e9_tree_posthoc.py <run_dir> [<run_dir> ...] --json OUT.json
"""
import io
import json
import os
import sys

import rev4_records as R


def read(run_dir, name, unreadable):
    path = os.path.join(run_dir, "probes", name + ".json")
    if not os.path.exists(path):
        return None
    try:
        return json.load(io.open(path, encoding="utf-8"))
    except ValueError:
        unreadable.append(name + ".json")
        return None


def slope(probe, collective):
    if not probe or collective not in probe or not probe[collective].get("sizes_bytes"):
        return None
    b, _ = R.slope(probe[collective]["sizes_bytes"], probe[collective]["times_s"])
    return b


def transport(run_dir, prefix, unreadable):
    ring, tree = read(run_dir, prefix + "ring", unreadable), read(run_dir, prefix + "tree", unreadable)
    b_tree, b_ring, b_gather = slope(tree, "allreduce"), slope(ring, "allreduce"), slope(ring, "allgather")
    if b_tree is None:
        return None
    out = {"tree_allreduce_ns_per_byte": 1e9 * b_tree}
    if b_ring:
        out["ring_allreduce_ns_per_byte"] = 1e9 * b_ring
        out["tree_over_ring"] = b_tree / b_ring
    if b_gather:
        out["allgather_ns_per_byte"] = 1e9 * b_gather
        out["gather_over_tree"] = b_gather / b_tree
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
    result = {"label": "POST HOC: the tree probe, read after the E9 records; not part of the frozen scorer",
              "allocations": [allocation(d) for d in dirs]}
    if out:
        json.dump(result, io.open(out, "w", encoding="utf-8"), indent=1, sort_keys=True)
    for a in result["allocations"]:
        for key in ("gdr", "host_staged"):
            if key in a:
                print("%-22s %2s ranks %-11s tree %.3f ns/B, %.1f times the ring all-reduce, gather %.2f times the tree"
                      % (a["run_dir"], a["world"], key, a[key]["tree_allreduce_ns_per_byte"],
                         a[key].get("tree_over_ring", float("nan")), a[key].get("gather_over_tree", float("nan"))))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
