# provenance/make_rev4_su.py
# -*- coding: utf-8 -*-
"""Writes provenance/rev4_su.tsv from the accounting records of the
Revision 4 jobs, which are kept beside it as rev4_sacct_<system>.txt:

    TZ=UTC sacct -n -X -P -o JobID,JobName,NNodes,ElapsedRaw,Partition,State,Start,End -j <ids>

All times are UTC, as everywhere in PROVENANCE.md. SU are computed from the
elapsed time at 3 SU per node-hour, and a job that ran is charged at least
fifteen minutes; the systems' own accounting is the record of charge."""
import io
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCES = (("rev4_sacct_stampede3.txt", 3.0), ("rev4_sacct_lonestar6.txt", 3.0))
HEAD = "job_id\tname\tnodes\telapsed_s\tpartition\tstate\tstart_utc\tend_utc\trate_su_per_node_hour\tsu"


def render():
    rows, total = [], 0.0
    for name, rate in SOURCES:
        path = os.path.join(HERE, name)
        if not os.path.exists(path):
            continue
        for line in io.open(path, encoding="utf-8").read().split("\n"):
            p = line.strip().split("|")
            if len(p) < 8 or not p[0].isdigit():
                continue
            elapsed = int(p[3] or 0)
            su = 0.0 if elapsed <= 0 else int(p[2]) * rate * max(elapsed, 900) / 3600.0
            total += round(su, 3)
            rows.append("\t".join(p[:8] + ["%.1f" % rate, "%.3f" % su]))
    return "\n".join([HEAD] + rows + ["total\t\t\t\t\t\t\t\t\t%.3f" % total]) + "\n"


if __name__ == "__main__":
    text = render()
    io.open(os.path.join(HERE, "rev4_su.tsv"), "w", encoding="utf-8", newline="\n").write(text)
    print(text)
