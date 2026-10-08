# -*- coding: utf-8 -*-
"""The committed E10 result is what the frozen scorer gives on the three
committed archives. This also exercises score_e10.allocation and first_error
on real harness v5.2 output, including a job that completed no timed run."""
import glob
import io
import json
import os
import shutil
import tarfile
import tempfile

import score_e10 as E

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def extract(archive, target):
    with tarfile.open(archive) as tf:
        try:
            tf.extractall(target, filter="data")
        except TypeError:           # Python before 3.12 has no filter argument
            tf.extractall(target)


def test_the_committed_result_is_the_frozen_scorer_on_the_committed_archives():
    archives = sorted(glob.glob(os.path.join(ROOT, "raw", "rev4", "redctl-E10-*.tgz")))
    assert len(archives) == 3
    # The record file names are about 140 characters long. pytest's tmp_path
    # would push them past the 260-character limit of Windows paths, so the
    # archives go into a short temporary folder (REV4_SHORT_TMP to choose it).
    work = tempfile.mkdtemp(prefix="e10", dir=os.environ.get("REV4_SHORT_TMP"))
    try:
        for archive in archives:
            extract(archive, work)
        dirs = sorted(d.replace("\\", "/") + "/" for d in glob.glob(os.path.join(work, "redctl-E10-*")))
        got = json.loads(json.dumps(E.score(dirs)))
    finally:
        shutil.rmtree(work, ignore_errors=True)
    want = json.load(io.open(os.path.join(ROOT, "validation_v2", "results", "rev4", "E10_RESULT.json"),
                             encoding="utf-8"))
    assert got == want
    states = {a["run_dir"]: {k: c["state"] for k, c in a["conditions"].items()} for a in got["allocations"]}
    assert states["redctl-E10-4n-3579138"] == {"32|gpu-quant-bucketed": "missing"}
    assert set(states["redctl-E10-4n-3579496"].values()) == {"complete"}
