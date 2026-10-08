# -*- coding: utf-8 -*-
"""The frozen E9 scorer on real output: the smallest committed archive scores
to the entry the committed result holds for it. This exercises
score_e9.allocation, block_table, ratios, equivalence, ddp_buckets_match and
the probe checks on harness v5.2 records from Lonestar6, which the unit
tests reach only through fixtures. reproduce_diagnostic.sh re-scores all
four archives and compares the whole file."""
import glob
import io
import json
import os
import shutil
import tarfile
import tempfile

import score_e9 as E

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RESULT = os.path.join(ROOT, "validation_v2", "results", "rev4", "E9_RESULT.json")


def test_the_committed_entry_of_q12b_is_the_frozen_scorer_on_its_archive():
    archive = glob.glob(os.path.join(ROOT, "raw", "rev4", "e9-q12b-*.tgz"))
    assert len(archive) == 1
    # short folder: the record names are long (see test_e10_committed_result.py)
    work = tempfile.mkdtemp(prefix="e9", dir=os.environ.get("REV4_SHORT_TMP"))
    try:
        with tarfile.open(archive[0]) as tf:
            try:
                tf.extractall(work, filter="data")
            except TypeError:
                tf.extractall(work)
        run = glob.glob(os.path.join(work, "e9-q12b-*"))[0].replace("\\", "/") + "/"
        got = json.loads(json.dumps(E.allocation(run)))
    finally:
        shutil.rmtree(work, ignore_errors=True)
    committed = json.load(io.open(RESULT, encoding="utf-8"))
    want = [a for a in committed["allocations"] if a["run_dir"] == got["run_dir"]]
    assert len(want) == 1 and got == want[0]
    assert got["world"] == 12 and got["ddp_buckets_match"] is True
    assert got["equivalence"] == {"transformer|gpu-quant-sr-fp16": True, "transformer|gpu-quant-sr-int8": True}
    # the one condition of this allocation that lost a pass to a start-up failure
    assert {k: v["passes"] for k, v in got["table"].items() if v["passes"] != 2} == {"resnet50|fp16-native": 1}


def test_the_summary_prints_both_allocations_of_each_scale():
    committed = json.load(io.open(RESULT, encoding="utf-8"))
    assert sorted(committed["summary"]) == ["12", "24"]
    for world, table in committed["summary"].items():
        assert all(len(cell["values"]) == 2 for cell in table.values()), world
