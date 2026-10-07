# harness/launch/su_ledger.py
# -*- coding: utf-8 -*-
"""Keeps this plan's charges on one system under a cap. Committed SU are
the charges of finished jobs plus the time-limit bounds of jobs that have
not finished. A job that ran is charged at least fifteen minutes.

usage: su_ledger.py check --ledger rev4_jobs.tsv --sacct sacct.txt --system stampede3
                          --rate 3 --cap 20 --nodes N --limit HH:MM:SS
sacct.txt: lines "JobID|NNodes|ElapsedRaw|State" from sacct -n -X -P.
"""
import argparse
import csv
import io
import sys

MINIMUM_S = 900
FINISHED = ("COMPLETED", "FAILED", "TIMEOUT", "CANCELLED", "NODE_FAIL", "OUT_OF_MEMORY", "PREEMPTED")


def seconds(limit):
    parts = [int(x) for x in limit.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    return parts[0] * 3600 + parts[1] * 60 + parts[2]


def bound(nodes, limit_s, rate):
    return nodes * rate * max(limit_s, MINIMUM_S) / 3600.0


def charge(nodes, elapsed_s, rate):
    if elapsed_s <= 0:
        return 0.0
    return nodes * rate * max(elapsed_s, MINIMUM_S) / 3600.0


def committed(jobs, rate):
    total = 0.0
    for job in jobs:
        if job["state"].split()[0] in FINISHED:
            total += charge(job["nodes"], job["elapsed_s"], rate)
        else:
            total += bound(job["nodes"], job["limit_s"], rate)
    return total


def may_submit(jobs, nodes, limit_s, rate, cap):
    return committed(jobs, rate) + bound(nodes, limit_s, rate) <= cap + 1e-9


def load(ledger_path, sacct_path, system):
    states = {}
    for line in io.open(sacct_path, encoding="utf-8"):
        parts = line.strip().split("|")
        if len(parts) >= 4 and parts[0].isdigit():
            states[parts[0]] = (int(parts[2] or 0), parts[3])
    jobs = []
    for row in csv.DictReader(io.open(ledger_path, encoding="utf-8"), delimiter="\t"):
        if row["system"] != system:
            continue
        elapsed, state = states.get(row["job_id"], (0, "PENDING"))
        jobs.append({"job_id": row["job_id"], "nodes": int(row["nodes"]),
                     "limit_s": seconds(row["time_limit"]), "elapsed_s": elapsed, "state": state})
    return jobs


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("check",))
    for name in ("--ledger", "--sacct", "--system", "--limit"):
        parser.add_argument(name, required=True)
    for name in ("--rate", "--cap"):
        parser.add_argument(name, type=float, required=True)
    parser.add_argument("--nodes", type=int, required=True)
    a = parser.parse_args(argv)
    jobs = load(a.ledger, a.sacct, a.system)
    now = committed(jobs, a.rate)
    new = bound(a.nodes, seconds(a.limit), a.rate)
    ok = may_submit(jobs, a.nodes, seconds(a.limit), a.rate, a.cap)
    print("%s: committed %.2f SU, this job up to %.2f SU, cap %.2f SU: %s"
          % (a.system, now, new, a.cap, "ALLOWED" if ok else "REFUSED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
