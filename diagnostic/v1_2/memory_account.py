# diagnostic/v1_2/memory_account.py
# -*- coding: utf-8 -*-
"""Peak device memory of every 1B-parameter condition on record, beside the
arithmetic the paper uses for the whole-gradient gather. RETROSPECTIVE
(2026-10-07): it reads peak_memory_mb from the archived per-rank records."""
import csv
import glob
import io
import os
import sys

GRADIENT_INT8_GB = 1.046030336      # one int8 copy of 1,046,030,336 elements


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
    """Footprint plus p received int8 copies of the gradient."""
    return footprint_gb + p * copy_gb


def render(named_run_dirs):
    lines = ["# Peak device memory of the 1B conditions on record (RETROSPECTIVE, 2026-10-07)", "",
             "Largest `peak_memory_mb` over ranks and iterations, in GB (1 GB = 1000 MB).", "",
             "| run | mode | ranks | timing | peak GB | above the dense row of the same run |",
             "|---|---|---|---|---|---|"]
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
    lines += ["", "Whole-gradient gather, Section VIII arithmetic (32.8 GB footprint, 1.05 GB per copy):",
              "16 ranks %.1f GB, 32 ranks %.1f GB, on a 64 GB device."
              % (gather_peak_gb(32.8, 16, 1.05), gather_peak_gb(32.8, 32, 1.05)), ""]
    return "\n".join(lines)


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else "C:/r5w/mem"
    runs = [(name, os.path.join(root, name)) for name in sorted(os.listdir(root))
            if os.path.isdir(os.path.join(root, name, "results"))]
    text = render(runs)
    here = os.path.dirname(os.path.abspath(__file__))
    io.open(os.path.join(here, "MEMORY_ACCOUNT.md"), "w", encoding="utf-8", newline="\n").write(text)
    print(text)
