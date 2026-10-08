# -*- coding: utf-8 -*-
import io
import json
import os

import pytest

import e9_tree_posthoc as T

MIB = 1 << 20


def probe(tmp_path, run, name, allreduce=None, allgather=None):
    d = tmp_path / run / "probes"
    d.mkdir(parents=True, exist_ok=True)
    sizes = [4, 1024, MIB, 16 * MIB, 64 * MIB, 256 * MIB]
    body = {"world_size": 12}
    for coll, slope in (("allreduce", allreduce), ("allgather", allgather)):
        if slope is not None:
            body[coll] = {"sizes_bytes": sizes, "times_s": [0.001 + slope * s for s in sizes]}
    io.open(str(d / (name + ".json")), "w", encoding="utf-8").write(json.dumps(body))
    return str(tmp_path / run)


def test_tree_slope_is_set_against_the_ring_and_the_gather_of_the_same_allocation(tmp_path):
    probe(tmp_path, "e9-q12a-4n-1", "ring", allreduce=2e-10, allgather=12e-10)
    run = probe(tmp_path, "e9-q12a-4n-1", "tree", allreduce=20e-10)
    out = T.allocation(run)
    assert out["run_dir"] == "e9-q12a-4n-1" and out["world"] == 12
    assert out["gdr"]["tree_over_ring"] == pytest.approx(10.0)
    assert out["gdr"]["gather_over_tree"] == pytest.approx(0.6)
    assert "host_staged" not in out


def test_an_allocation_without_a_tree_probe_reports_nothing_for_it(tmp_path):
    run = probe(tmp_path, "e9-q12b-4n-2", "ring", allreduce=2e-10, allgather=12e-10)
    assert T.allocation(run) == {"run_dir": "e9-q12b-4n-2", "world": 12}


def test_an_unreadable_probe_file_is_named_and_does_not_stop_the_others(tmp_path):
    probe(tmp_path, "e9-q24a-8n-3", "ring", allreduce=2e-10, allgather=12e-10)
    run = probe(tmp_path, "e9-q24a-8n-3", "tree", allreduce=20e-10)
    io.open(os.path.join(run, "probes", "hs-tree.json"), "w", encoding="utf-8").write("")
    out = T.allocation(run)
    assert out["gdr"]["tree_over_ring"] == pytest.approx(10.0)
    assert out["unreadable"] == ["hs-tree.json"]
