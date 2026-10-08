# -*- coding: utf-8 -*-
"""E9: how far the two passes of a condition are apart. POST HOC
(2026-10-08): written after the E9 records were read.

The outcome of a condition is the mean of its block-timed step over two
passes, the second run in reverse order (protocol section 2), and the calls
compare such means at 5%. The frozen scorer keeps the mean and the number of
passes. This script keeps each pass, so that a reader can set the 5% of the
calls against the distance between two passes of the same condition in the
same job. It uses the frozen scorer's own run filter and step
(score_e9.BLOCK, rev4_records.valid and block_step_ms) and changes nothing
the scorer wrote.

  spread of a condition = |pass 1 - pass 2| / mean of the two

usage: python e9_pass_spread_posthoc.py <run_dir> [<run_dir> ...] --json OUT.json
"""
import io
import json
import os
import statistics
import sys

import rev4_records as R
import score_e9 as E


def spread(first, second):
    return abs(first - second) / ((first + second) / 2.0)


def conditions(passes):
    """passes: {"model|mode": {"p1": ms, "p2": ms}} -> the same with a spread
    where both passes left a valid record."""
    out = {}
    for key, p in passes.items():
        entry = {"passes_ms": dict(p)}
        if len(p) == 2:
            first, second = (p[k] for k in sorted(p))
            entry["spread"] = spread(first, second)
        out[key] = entry
    return out


def allocation(run_dir):
    passes, world = {}, None
    for run_id, run in R.read_dir(run_dir).items():
        m = E.BLOCK.match(run_id)
        if m and R.valid(run["rows"]):
            world = int(m.group(1))
            passes.setdefault("%s|%s" % (m.group(2), m.group(3)), {})["p" + m.group(4)] = R.block_step_ms(run["rows"])
    return {"run_dir": os.path.basename(os.path.normpath(run_dir)), "world": world, "conditions": conditions(passes)}


def summary(allocations):
    out = {}
    for world in sorted({a["world"] for a in allocations if a.get("world")}):
        found = [(c["spread"], a["run_dir"], key, c["passes_ms"])
                 for a in allocations if a.get("world") == world
                 for key, c in sorted(a["conditions"].items()) if "spread" in c]
        if not found:
            continue
        top = max(found, key=lambda f: f[0])
        out[str(world)] = {
            "two_pass_conditions": len(found),
            "above_5_percent": sum(1 for f in found if f[0] > 0.05),
            "median": statistics.median(f[0] for f in found),
            "largest": {"spread": top[0], "run_dir": top[1], "condition": top[2], "passes_ms": top[3]},
        }
    return out


def main(argv):
    out = argv[argv.index("--json") + 1] if "--json" in argv else None
    dirs = [a for a in argv if not a.startswith("--") and a != out]
    allocations = [allocation(d) for d in dirs]
    result = {"label": "POST HOC: the two passes of each E9 condition, read after the records; "
                       "spread = |pass 1 - pass 2| / mean of the two",
              "allocations": allocations, "summary": summary(allocations)}
    if out:
        json.dump(result, io.open(out, "w", encoding="utf-8", newline="\n"), indent=1, sort_keys=True)
    for world, s in sorted(result["summary"].items(), key=lambda kv: int(kv[0])):
        print("%s ranks: %d two-pass conditions, %d above 5%%, median %.1f%%, largest %.1f%% (%s in %s)"
              % (world, s["two_pass_conditions"], s["above_5_percent"], 100 * s["median"],
                 100 * s["largest"]["spread"], s["largest"]["condition"], s["largest"]["run_dir"]))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
