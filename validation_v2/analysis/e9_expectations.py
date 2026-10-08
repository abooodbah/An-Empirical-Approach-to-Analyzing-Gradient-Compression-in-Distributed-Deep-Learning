# -*- coding: utf-8 -*-
"""E9: the expectations of protocol section 5 and the probe checks of
section 6, set against the scored result.

diagnostic/E9_PROTOCOL.md states them before the jobs ran and does not use
them in scoring. This script reads E9_RESULT.json, the frozen scorer's
output, and says of each whether it held and in which cell it did not. The
last section of the Markdown file repeats the probe ratios from
E9_PROBES_POSTHOC.json, a reading made after the records were read, and is
labeled so. Nothing here changes a frozen file or a scored number.

usage: python e9_expectations.py        (reads and writes validation_v2/results/rev4)
"""
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(os.path.dirname(HERE), "results", "rev4")
MODELS = ("transformer", "resnet50")
CHECKS = ("P1", "P2", "P3")
STATEMENT = {
    "X1": "gpu-quant-sr-int8 is within 10% of gpu-quant-reducible in step time.",
    "X2": "gpu-quant-sr-fp16 is not faster than fp16-native.",
    "X3": "ddp-int8-hook is slower than ddp-overlap at both scales, by more at 24 ranks.",
    "X4": "ddp-fp16-hook is within 5% of ddp-overlap or faster.",
    "X5": "The relative error of gpu-quant-sr-fp16 is below that of gpu-quant-sr-int8.",
    "P1": "With all-reduce forced to Ring, slope(all-gather) / slope(all-reduce) is within a factor 1.5 of p/2.",
    "P2": "The default all-reduce slope is not more than 5% above the Ring slope.",
    "P3": "The 1 MiB to 64 MiB line predicts the 256 MiB and 1 GiB times within 15%.",
}


def live(result):
    return [a for a in result["allocations"] if a.get("world")]


def worlds(result):
    return sorted({a["world"] for a in live(result)})


def cells(result):
    return [(w, m) for w in worlds(result) for m in MODELS]


def summary_cells(result, mode, base):
    out = {}
    for world, model in cells(result):
        cell = result["summary"].get(str(world), {}).get("%s|%s|%s" % (model, mode, base))
        if cell:
            out["%d|%s" % (world, model)] = cell
    return out


def outcomes(result):
    out = {}

    values = {}
    for a in live(result):
        for model in MODELS:
            new, old = (a["table"].get("%s|%s" % (model, m)) for m in ("gpu-quant-sr-int8", "gpu-quant-reducible"))
            if new and old:
                values.setdefault("%d|%s" % (a["world"], model), []).append(new["T_ms"] / old["T_ms"])
    failed = [k for k, v in values.items() if any(abs(r - 1.0) > 0.10 for r in v)]
    out["X1"] = {"statement": STATEMENT["X1"], "values": values, "failed": failed, "held": bool(values) and not failed}

    found = summary_cells(result, "gpu-quant-sr-fp16", "fp16-native")
    failed = [k for k, c in found.items() if c["call"] == "faster"]
    out["X2"] = {"statement": STATEMENT["X2"], "calls": {k: c["call"] for k, c in found.items()},
                 "values": {k: c["values"] for k, c in found.items()}, "failed": failed,
                 "held": bool(found) and not failed}

    found = summary_cells(result, "ddp-int8-hook", "ddp-overlap")
    failed = [k for k, c in found.items() if c["call"] != "slower"]
    scales = worlds(result)
    for model in MODELS:
        low, high = (found.get("%d|%s" % (w, model)) for w in (scales[0], scales[-1]))
        if len(scales) < 2 or not low or not high or min(high["values"]) <= max(low["values"]):
            failed.append("%s: not larger at %d ranks" % (model, scales[-1]))
    out["X3"] = {"statement": STATEMENT["X3"], "calls": {k: c["call"] for k, c in found.items()},
                 "values": {k: c["values"] for k, c in found.items()}, "failed": failed,
                 "held": bool(found) and not failed}

    found = summary_cells(result, "ddp-fp16-hook", "ddp-overlap")
    failed = [k for k, c in found.items() if c["call"] not in ("faster", "within 5%")]
    out["X4"] = {"statement": STATEMENT["X4"], "calls": {k: c["call"] for k, c in found.items()},
                 "values": {k: c["values"] for k, c in found.items()}, "failed": failed,
                 "held": bool(found) and not failed}

    values = {}
    for a in live(result):
        for model in MODELS:
            wide, narrow = (a.get("fidelity", {}).get("%s|%s" % (model, m))
                            for m in ("gpu-quant-sr-fp16", "gpu-quant-sr-int8"))
            if wide and narrow:
                values.setdefault("%d|%s" % (a["world"], model),
                                  [wide["relative_l2_error"], narrow["relative_l2_error"]])
    failed = [k for k, (wide, narrow) in values.items() if not wide < narrow]
    out["X5"] = {"statement": STATEMENT["X5"], "values": values, "failed": failed, "held": bool(values) and not failed}

    for check in CHECKS:
        entry = {"statement": STATEMENT[check], "values": {}}
        for transport in ("gdr", "host_staged"):
            holds, numbers = {}, {}
            for a in live(result):
                c = a["probe_checks"].get(transport, {}).get(check)
                if not c:
                    continue
                holds.setdefault(str(a["world"]), []).append(bool(c["holds"]))
                number = (c["slope_ratio"] if check == "P1" else c["default_over_ring"] if check == "P2"
                          else max(c["errors"].values()) if c["errors"] else None)
                numbers.setdefault(str(a["world"]), []).append(number)
            entry[transport], entry["values"][transport] = holds, numbers
        out[check] = entry
    return out


# ------------------------------------------------------------------ Markdown
def pair(values, fmt="%.2f"):
    return ", ".join(fmt % v for v in values)


def words(key):
    """A cell name in words: the bar of 24|resnet50 would split a table cell."""
    if "|" not in key:
        return key
    world, model = key.split("|")
    return "%s at %s ranks" % (model, world)


def by_cell(values, fmt="%.2f", calls=None):
    parts = []
    for key in sorted(values, key=lambda k: (int(k.split("|")[0]), MODELS.index(k.split("|")[1]))):
        world, model = key.split("|")
        call = " (%s)" % calls[key] if calls else ""
        parts.append("%s ranks %s %s%s" % (world, model, pair(values[key], fmt), call))
    return "; ".join(parts)


def probe_cell(entry, check, transport, world):
    holds = entry[transport].get(world)
    if not holds:
        return "not run"
    numbers = entry["values"][transport][world]
    shown = pair([100 * n for n in numbers], "%.1f%%") if check == "P3" else pair(numbers)
    return "%s: %s" % ("held" if all(holds) else "not held", shown)


def second_reading(posthoc, transport, world, value):
    cells_ = [value(a[transport]) for a in posthoc["allocations"]
              if a.get("world") == world and transport in a]
    return "; ".join(cells_) if cells_ else "not run"


def render(out, posthoc):
    scales = sorted({w for check in CHECKS for w in out[check]["gdr"]}, key=int)
    lines = [
        "# E9: the stated expectations and the probe checks, against the scored result",
        "",
        "Generated by `validation_v2/analysis/e9_expectations.py` from `E9_RESULT.json`, the frozen scorer's",
        "output. `diagnostic/E9_PROTOCOL.md` states the expectations (section 5) and the probe checks",
        "(section 6) before the jobs ran and does not use them in scoring. Where two values are given they",
        "are the two allocations of that scale.",
        "",
        "## Expectations",
        "",
        "| | Statement | Outcome | Values |",
        "|---|---|---|---|",
    ]
    notes = {
        "X1": "step of gpu-quant-sr-int8 over step of gpu-quant-reducible: ",
        "X2": "step of gpu-quant-sr-fp16 over step of fp16-native: ",
        "X3": "step of ddp-int8-hook over step of ddp-overlap: ",
        "X4": "step of ddp-fp16-hook over step of ddp-overlap: ",
        "X5": "relative L2 error of gpu-quant-sr-fp16, then of gpu-quant-sr-int8: ",
    }
    for name in ("X1", "X2", "X3", "X4", "X5"):
        e = out[name]
        outcome = "held" if e["held"] else "not held (%s)" % "; ".join(words(k) for k in e["failed"])
        fmt = "%.3f" if name == "X5" else "%.2f"
        lines.append("| %s | %s | %s | %s%s |" % (name, e["statement"], outcome, notes[name],
                                                  by_cell(e["values"], fmt, e.get("calls"))))
    lines += [
        "",
        "## Probe checks by the protocol's rule (the line through 1 MiB to 64 MiB)",
        "",
        "| | Statement | " + " | ".join("GPUDirect, %s ranks" % w for w in scales) + " | "
        + " | ".join("Host-staged, %s ranks" % w for w in scales) + " |",
        "|---|---|" + "---|" * (2 * len(scales)),
    ]
    for check in CHECKS:
        e = out[check]
        lines.append("| %s | %s | " % (check, e["statement"])
                     + " | ".join(probe_cell(e, check, t, w) for t in ("gdr", "host_staged") for w in scales) + " |")
    lines += ["", "P1 gives the slope ratio (the model is p/2), P2 the default slope over the Ring slope, P3 the",
              "largest prediction error of each allocation."]
    if not all(h for t in ("gdr", "host_staged") for holds in out["P1"][t].values() for h in holds):
        lines += ["P1 does not hold in every allocation. By section 6 the paper says that the traffic model does",
                  "not explain the slope ratio on this system."]
    lines += [
        "",
        "## A second reading of the probes (POST HOC)",
        "",
        "Made after the records were read, by `validation_v2/analysis/e9_probes_posthoc.py` into",
        "`E9_PROBES_POSTHOC.json`. The protocol's line passes through three points of the probe grid: 1, 16",
        "and 64 MiB. The rows below use the line through 16 MiB and 64 MiB, give the measured 1 MiB and",
        "16 MiB times of the forced-ring all-reduce, and add the forced-tree probe, which the frozen scorer",
        "does not read. They replace no number above.",
        "",
        "| | " + " | ".join("GPUDirect, %s ranks" % w for w in scales) + " | "
        + " | ".join("Host-staged, %s ranks" % w for w in scales) + " |",
        "|---|" + "---|" * (2 * len(scales)),
    ]
    late = "from_16_to_64_mib"
    rows = (
        ("All-gather slope over Ring all-reduce slope, from 16 MiB",
         lambda g: "%.2f" % g[late]["gather_over_ring"] if "gather_over_ring" in g[late] else "not run"),
        ("Default all-reduce slope over Ring slope, from 16 MiB",
         lambda g: "%.2f" % g[late]["default_over_ring"] if "default_over_ring" in g[late] else "not run"),
        ("Largest prediction error above 64 MiB, from 16 MiB",
         lambda g: "%.1f%%" % (100 * max(g[late]["prediction_errors"].values()))
         if g[late].get("prediction_errors") else "not run"),
        ("Forced-ring all-reduce, 1 MiB then 16 MiB, ms",
         lambda g: "%.2f then %.2f" % (g["points_ms"]["ring_allreduce"]["1"], g["points_ms"]["ring_allreduce"]["16"])
         if "ring_allreduce" in g["points_ms"] else "not run"),
        ("Forced-tree all-reduce time over forced-ring time at 1 GiB",
         lambda g: "%.2f" % g["tree_over_ring_time_at_1024_mib"]
         if "tree_over_ring_time_at_1024_mib" in g else "not run"),
    )
    for title, value in rows:
        lines.append("| %s | " % title
                     + " | ".join(second_reading(posthoc, t, int(w), value)
                                  for t in ("gdr", "host_staged") for w in scales) + " |")
    return "\n".join(lines) + "\n"


def main(argv):
    result = json.load(io.open(os.path.join(RESULTS, "E9_RESULT.json"), encoding="utf-8"))
    posthoc = json.load(io.open(os.path.join(RESULTS, "E9_PROBES_POSTHOC.json"), encoding="utf-8"))
    out = outcomes(result)
    json.dump(out, io.open(os.path.join(RESULTS, "E9_EXPECTATIONS.json"), "w", encoding="utf-8", newline="\n"),
              indent=1, sort_keys=True)
    text = render(out, posthoc)
    io.open(os.path.join(RESULTS, "E9_EXPECTATIONS.md"), "w", encoding="utf-8", newline="\n").write(text)
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
