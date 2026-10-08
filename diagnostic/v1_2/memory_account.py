# diagnostic/v1_2/memory_account.py
# -*- coding: utf-8 -*-
"""Peak device memory of every 1B-parameter condition on record, beside the
arithmetic the paper uses for the whole-gradient gather. RETROSPECTIVE
(2026-10-07; arithmetic corrected 2026-10-08): it reads peak_memory_mb from
the archived per-rank records. Everything in MEMORY_ACCOUNT.md is written by
this script.

usage: python memory_account.py [folder holding the extracted run directories]
       (default: $REV4_MEM_ROOT, then C:/r5w/mem)"""
import csv
import glob
import io
import os
import sys

GRADIENT_INT8_GB = 1.046030336      # one int8 copy of 1,046,030,336 elements

HOOK_FAILURES = [
    "- Job 3517833, the int8 hook at 16 ranks: the run ended at the final hash on one rank with "
    "\"RuntimeError: Native API failed. Native API returns: -5 (PI_ERROR_OUT_OF_RESOURCES) -5 "
    "(PI_ERROR_OUT_OF_RESOURCES)\" (`raw/control/redctl-V-reruns-3517832-3517833.tgz`, "
    "`redctl-V-2n-3517833/logs/llama-ddp-int8-hook.14.err`).",
    "- Job 3517753, the int8 hook at 32 ranks: ten rank logs end with GPU page faults on write, for "
    "example \"Segmentation fault from GPU at 0xff00000f5a016000, ctx_id: 5 (CCS) type: 0 (NotPresent), "
    "level: 1 (PDE), access: 1 (Write), banned: 1, aborting.\" "
    "(`raw/control/redctl-V-v3-3517750-51-52-53.tgz`, `redctl-V-4n-3517753/logs/llama-ddp-int8-hook.*.err`).",
]

UNITS = (
    "`peak_memory_mb` is bytes divided by 10^6 (the harness's `peak_memory_mb()`), and the table above "
    "divides it by 1000. Section VIII of the 7 October manuscript printed 42.0 GB for the eight-rank "
    "whole-gradient gather and 32.8 GB for the model's footprint. Both came from "
    "`diagnostic/peak_mem_1b.py`, which divides the same records by 1024. In the unit of this table they "
    "are 43.0 GB and 33.5 GB. That manuscript's arithmetic also counted p int8 copies, which gives 41.9 GB "
    "at 8 ranks against 43.0 GB measured, 50.3 GB at 16, and 67.0 GB at 32. The measured peak is the "
    "footprint plus p + 1 copies (the p received and the rank's own encoded one), and the lines under the "
    "table use that count. No archived record holds the device's memory; 64 GB is the vendor's figure "
    "for one tile, and its unit is not recorded here.")


def peaks(run_dir):
    out = {}
    for path in glob.glob(os.path.join(run_dir, "results", "*.csv")):
        with io.open(path, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                value = r.get("peak_memory_mb")
                if value in (None, "", "None"):
                    continue
                key = (r["model"], r["comm_mode"], int(r["world_size"]), r.get("timing_mode") or "phase")
                out[key] = max(out.get(key, 0.0), float(value))
    return out


def gather_peak_gb(footprint_gb, p, copy_gb):
    """Peak of a whole-gradient int8 gather: the model's footprint, the p
    received copies of the gradient, and the rank's own encoded copy."""
    return footprint_gb + (p + 1) * copy_gb


def runs_under(root):
    return [(name, os.path.join(root, name)) for name in sorted(os.listdir(root))
            if os.path.isdir(os.path.join(root, name, "results"))]


def render(named_run_dirs):
    lines = ["# Peak device memory of the 1B conditions on record (RETROSPECTIVE, 2026-10-07)", "",
             "Largest `peak_memory_mb` over ranks and iterations, in GB (1 GB = 1000 MB).", "",
             "| run | mode | ranks | timing | peak GB | above the dense row of the same run |",
             "|---|---|---|---|---|---|"]
    measured = {}       # the Intel 8-rank phase-timed pair the arithmetic is checked against
    for name, run_dir in named_run_dirs:
        table = {k: v for k, v in peaks(run_dir).items() if k[0] == "llama1b"}
        dense = {}
        for (model, mode, ranks, timing), value in table.items():
            if mode in ("native", "ddp-overlap"):
                dense[(ranks, timing)] = value
        for (model, mode, ranks, timing), value in sorted(table.items()):
            base = dense.get((ranks, timing))
            extra = "" if base is None or mode in ("native", "ddp-overlap") else "%.1f" % ((value - base) / 1000.0)
            lines.append("| %s | %s | %d | %s | %.1f | %s |" % (name, mode, ranks, timing, value / 1000.0, extra))
        if ("llama1b", "gpu-quant", 8, "phase") in table and ("llama1b", "native", 8, "phase") in table:
            measured = {"run": name, "footprint": table[("llama1b", "native", 8, "phase")] / 1000.0,
                        "gather": table[("llama1b", "gpu-quant", 8, "phase")] / 1000.0}
    lines += ["", "Whole-gradient int8 gather: the model's footprint, plus p received copies of the gradient",
              "and the rank's own encoded copy (%.3f GB each)." % GRADIENT_INT8_GB]
    if measured:
        f = measured["footprint"]
        lines += ["Footprint %.1f GB (dense exchange, 8 ranks, run %s)." % (f, measured["run"]),
                  "8 ranks: %.1f GB by this arithmetic, %.1f GB measured."
                  % (gather_peak_gb(f, 8, GRADIENT_INT8_GB), measured["gather"]),
                  "16 ranks: %.1f GB. 32 ranks: %.1f GB."
                  % (gather_peak_gb(f, 16, GRADIENT_INT8_GB), gather_peak_gb(f, 32, GRADIENT_INT8_GB))]
    lines += ["", "## The two hook failures on record", ""] + HOOK_FAILURES
    lines += ["", "## Units", "", UNITS, ""]
    return "\n".join(lines)


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("REV4_MEM_ROOT", "C:/r5w/mem")
    text = render(runs_under(root))
    here = os.path.dirname(os.path.abspath(__file__))
    io.open(os.path.join(here, "MEMORY_ACCOUNT.md"), "w", encoding="utf-8", newline="\n").write(text)
    print(text)
