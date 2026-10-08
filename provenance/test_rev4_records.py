# provenance/test_rev4_records.py
# -*- coding: utf-8 -*-
"""The records of Revision 4 agree with each other: the frozen files of E9
and E10 still hash to what was published before the jobs, the SU record is
what its script writes from the archived accounting, every time in it is
UTC, and every job started after it was submitted and after the push that
published its protocol."""
import csv
import hashlib
import io
import json
import os

import make_rev4_su as S

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROV = os.path.join(ROOT, "provenance")


def read_tsv(name):
    with io.open(os.path.join(PROV, name), encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def pushes():
    return json.load(io.open(os.path.join(PROV, "github_activity_rev4.json"), encoding="utf-8"))


def test_e9_and_e10_frozen_files_still_hash_to_the_published_lists():
    changed = []
    for name in ("E9_FREEZE.sha256", "E10_FREEZE.sha256"):
        lines = io.open(os.path.join(ROOT, "diagnostic", name), encoding="utf-8").read().split("\n")
        entries = [l.split(None, 1) for l in lines if l.strip()]
        assert len(entries) == 7, name
        for digest, rel in entries:
            with open(os.path.join(ROOT, rel.strip().lstrip("*")), "rb") as fh:
                if hashlib.sha256(fh.read()).hexdigest() != digest:
                    changed.append(rel)
    assert changed == []


def test_su_record_is_what_the_script_writes_from_the_archived_accounting():
    committed = io.open(os.path.join(PROV, "rev4_su.tsv"), encoding="utf-8").read()
    assert S.render() == committed


def test_su_record_times_are_utc_and_follow_submission():
    jobs = {r["job_id"]: r for r in read_tsv("rev4_jobs.tsv")}
    rows = [r for r in read_tsv("rev4_su.tsv") if r["job_id"].isdigit()]
    assert rows and "start_utc" in rows[0] and "end_utc" in rows[0]
    for r in rows:
        # ISO strings of one zone compare as times; a local-time start would
        # fall five hours before its own submission
        assert r["start_utc"] >= jobs[r["job_id"]]["submitted_utc"], r["job_id"]
        assert r["end_utc"] > r["start_utc"]


def test_every_job_was_submitted_after_the_push_that_published_its_protocol():
    first = min(p["timestamp"] for p in pushes())
    assert first == "2026-10-08T00:14:19Z"
    for r in read_tsv("rev4_jobs.tsv"):
        assert r["submitted_utc"] + "Z" > first, r["job_id"]


def test_the_resubmission_followed_the_push_that_recorded_its_approval():
    by_after = {p["after"][:7]: p["timestamp"] for p in pushes()}
    resubmitted = [r for r in read_tsv("rev4_jobs.tsv") if r["set"] == "s2-resubmitted"]
    assert len(resubmitted) == 1
    assert resubmitted[0]["submitted_utc"] + "Z" > by_after["516973c"]


def test_su_total_is_the_sum_of_the_rows():
    rows = read_tsv("rev4_su.tsv")
    jobs = [float(r["su"]) for r in rows if r["job_id"].isdigit()]
    total = [float(r["su"]) for r in rows if r["job_id"] == "total"]
    assert len(total) == 1 and abs(sum(jobs) - total[0]) < 0.0015
