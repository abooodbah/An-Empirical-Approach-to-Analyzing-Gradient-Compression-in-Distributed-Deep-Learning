# provenance/make_rev4_su.py
# -*- coding: utf-8 -*-
"""Writes provenance/rev4_su.tsv from the sacct dumps of the Revision 4
jobs. A job that ran is charged at least fifteen minutes."""
import io
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCES = (("C:/r5w/rev4/sacct_s3.txt", 3.0), ("C:/r5w/rev4/sacct_ls6.txt", 3.0))
HEAD = "job_id\tname\tnodes\telapsed_s\tpartition\tstate\tstart\tend\trate_su_per_node_hour\tsu"

rows, total = [], 0.0
for path, rate in SOURCES:
    if not os.path.exists(path):
        continue
    for line in io.open(path, encoding="utf-8").read().split("\n"):
        p = line.strip().split("|")
        if len(p) < 8 or not p[0].isdigit():
            continue
        elapsed = int(p[3] or 0)
        su = 0.0 if elapsed <= 0 else int(p[2]) * rate * max(elapsed, 900) / 3600.0
        total += su
        rows.append("\t".join(p[:8] + ["%.1f" % rate, "%.2f" % su]))
io.open(os.path.join(HERE, "rev4_su.tsv"), "w", encoding="utf-8", newline="\n").write(
    "\n".join([HEAD] + rows + ["total\t\t\t\t\t\t\t\t\t%.2f" % total]) + "\n")
print("total %.2f SU over %d jobs" % (total, len(rows)))
