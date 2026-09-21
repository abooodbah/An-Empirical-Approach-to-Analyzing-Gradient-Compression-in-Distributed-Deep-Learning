# -*- coding: utf-8 -*-
"""Compare identical wall-mode conditions between the J2 and J5 allocations
(features.py keys collapse them, so re-read per run directory here)."""
import csv
import glob
import os
import statistics as st

base = (r"C:\Users\Abdulfatah\AppData\Local\Temp\claude"
        r"\C--Users-Abdulfatah--codex-skills-multi-agent-coordinator-v2"
        r"\90dfc924-e7a7-4c99-bd46-b77f6884c30d\scratchpad\r4")


def op(p):
    return "\\\\?\\" + os.path.abspath(p)


def steps(job):
    out = {}
    for f in glob.glob(os.path.join(base, "r4-*-%s" % job, "results", "*rank0000.csv")):
        rows = list(csv.DictReader(open(op(f), newline="")))
        if rows and rows[0].get("timing_mode") == "wall" and rows[0].get("profile") == "paper":
            cond = os.path.basename(f).split("__")[1].replace(".ws12.tau1.bs32.wall", "")
            out[cond] = st.mean(float(r["observed_step_s"]) for r in rows) * 1e3
    return out


a, b = steps("3459786"), steps("3460182")
print("%-34s %8s %8s  ratio" % ("condition (wall, rank0 mean, ms)", "J2", "J5"))
for k in sorted(set(a) & set(b)):
    print("%-34s %8.1f %8.1f  %5.2f" % (k, a[k], b[k], b[k] / a[k]))
