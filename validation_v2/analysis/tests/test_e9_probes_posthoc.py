# -*- coding: utf-8 -*-
import io
import json
import os

import pytest

import e9_probes_posthoc as P

MIB = 1 << 20
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RESULTS = os.path.join(ROOT, "validation_v2", "results", "rev4")
SIZES = [4, 1024, MIB, 16 * MIB, 64 * MIB, 256 * MIB, 1024 * MIB]


def probe(tmp_path, run, name, allreduce=None, allgather=None, slow=None, world=12):
    """A probe file on the real grid. slow = {(collective, MiB): seconds}
    replaces single points."""
    d = tmp_path / run / "probes"
    d.mkdir(parents=True, exist_ok=True)
    body = {"world_size": world}
    for coll, slope in (("allreduce", allreduce), ("allgather", allgather)):
        if slope is not None:
            sizes = [s for s in SIZES if coll == "allreduce" or s <= 256 * MIB]
            times = [0.001 + slope * s for s in sizes]
            for (c, mib), seconds in (slow or {}).items():
                if c == coll:
                    times[sizes.index(mib * MIB)] = seconds
            body[coll] = {"sizes_bytes": sizes, "times_s": times}
    io.open(str(d / (name + ".json")), "w", encoding="utf-8").write(json.dumps(body))
    return str(tmp_path / run)


def test_on_points_that_lie_on_a_line_both_ranges_give_the_same_ratios(tmp_path):
    probe(tmp_path, "e9-q12a-4n-1", "default", allreduce=2e-10, allgather=12e-10)
    probe(tmp_path, "e9-q12a-4n-1", "ring", allreduce=2e-10, allgather=12e-10)
    run = probe(tmp_path, "e9-q12a-4n-1", "tree", allreduce=20e-10)
    out = P.allocation(run)
    assert out["run_dir"] == "e9-q12a-4n-1" and out["world"] == 12
    for name in ("rule_1_to_64_mib", "from_16_to_64_mib"):
        r = out["gdr"][name]
        assert r["gather_over_ring"] == pytest.approx(6.0)
        assert r["default_over_ring"] == pytest.approx(1.0)
        assert r["tree_over_ring"] == pytest.approx(10.0)
        assert r["gather_over_tree"] == pytest.approx(0.6)
        assert max(r["prediction_errors"].values()) < 1e-9
    assert out["gdr"]["tree_over_ring_time_at_1024_mib"] == pytest.approx((0.001 + 20e-10 * 1024 * MIB)
                                                                           / (0.001 + 2e-10 * 1024 * MIB))
    assert "host_staged" not in out


def test_a_slow_1_mib_point_moves_the_rule_and_not_the_fit_from_16_mib(tmp_path):
    # the 24-rank situation: under the forced ring the 1 MiB all-reduce takes
    # longer than the 16 MiB one
    probe(tmp_path, "e9-q24a-8n-3", "default", allreduce=2e-10, allgather=24e-10, world=24)
    run = probe(tmp_path, "e9-q24a-8n-3", "ring", allreduce=2e-10, allgather=24e-10, world=24,
                slow={("allreduce", 1): 0.010})
    gdr = P.allocation(run)["gdr"]
    assert gdr["from_16_to_64_mib"]["gather_over_ring"] == pytest.approx(12.0)
    assert gdr["rule_1_to_64_mib"]["gather_over_ring"] > 18.0
    assert gdr["rule_1_to_64_mib"]["default_over_ring"] > 1.5
    assert gdr["from_16_to_64_mib"]["default_over_ring"] == pytest.approx(1.0)
    assert gdr["ring_allreduce_1_mib_over_16_mib"] > 1.0
    assert gdr["points_ms"]["ring_allreduce"]["1"] == pytest.approx(10.0)
    assert gdr["points_ms"]["ring_allreduce"]["16"] == pytest.approx(1e3 * (0.001 + 2e-10 * 16 * MIB))
    assert sorted(gdr["points_ms"]["ring_allgather"], key=int) == ["1", "16", "64", "256"]
    assert "tree_over_ring" not in gdr["rule_1_to_64_mib"]          # no tree probe in this allocation


def test_the_host_staged_probes_are_read_under_their_own_key(tmp_path):
    probe(tmp_path, "e9-q12a-4n-1", "ring", allreduce=2e-10, allgather=12e-10)
    run = probe(tmp_path, "e9-q12a-4n-1", "hs-ring", allreduce=4e-10, allgather=28e-10)
    out = P.allocation(run)
    assert out["host_staged"]["rule_1_to_64_mib"]["gather_over_ring"] == pytest.approx(7.0)
    assert "default_over_ring" not in out["host_staged"]["rule_1_to_64_mib"]


def test_an_unreadable_probe_file_is_named_and_does_not_stop_the_others(tmp_path):
    probe(tmp_path, "e9-q24a-8n-3", "ring", allreduce=2e-10, allgather=12e-10)
    run = probe(tmp_path, "e9-q24a-8n-3", "tree", allreduce=20e-10)
    io.open(os.path.join(run, "probes", "hs-tree.json"), "w", encoding="utf-8").write("")
    out = P.allocation(run)
    assert out["gdr"]["rule_1_to_64_mib"]["tree_over_ring"] == pytest.approx(10.0)
    assert out["unreadable"] == ["hs-tree.json"]


def test_an_allocation_without_probes_reports_only_its_name(tmp_path):
    (tmp_path / "e9-q12b-4n-2" / "probes").mkdir(parents=True)
    assert P.allocation(str(tmp_path / "e9-q12b-4n-2")) == {"run_dir": "e9-q12b-4n-2", "world": None}


# ---- the committed result
def committed():
    return json.load(io.open(os.path.join(RESULTS, "E9_PROBES_POSTHOC.json"), encoding="utf-8"))


def test_under_the_protocols_range_the_committed_file_repeats_the_frozen_scorer():
    # the post hoc script and the frozen scorer agree where they overlap, so
    # the second reading differs from the first in the range and in nothing else
    frozen = {a["run_dir"]: a for a in json.load(io.open(os.path.join(RESULTS, "E9_RESULT.json"),
                                                         encoding="utf-8"))["allocations"]}
    seen = 0
    for a in committed()["allocations"]:
        for key in ("gdr", "host_staged"):
            checks = frozen[a["run_dir"]]["probe_checks"].get(key, {})
            if not checks:
                assert key not in a
                continue
            rule = a[key]["rule_1_to_64_mib"]
            assert rule["gather_over_ring"] == pytest.approx(checks["P1"]["slope_ratio"], rel=1e-12)
            assert rule["default_over_ring"] == pytest.approx(checks["P2"]["default_over_ring"], rel=1e-12)
            assert rule["prediction_errors"] == pytest.approx(checks["P3"]["errors"], rel=1e-12)
            seen += 1
    assert seen == 6                                  # four allocations over GPUDirect, two host-staged


def test_what_the_paper_says_about_the_24_rank_fit_is_in_the_committed_file():
    by_world = {}
    for a in committed()["allocations"]:
        by_world.setdefault(a["world"], []).append(a["gdr"])
    assert sorted(by_world) == [12, 24] and all(len(v) == 2 for v in by_world.values())
    for g in by_world[24]:
        # the forced-ring 1 MiB all-reduce took longer than the 16 MiB one
        assert g["points_ms"]["ring_allreduce"]["1"] > 2.5 * g["points_ms"]["ring_allreduce"]["16"]
        assert g["rule_1_to_64_mib"]["gather_over_ring"] > 30
        assert 11.5 < g["from_16_to_64_mib"]["gather_over_ring"] < 12.5
        assert max(g["from_16_to_64_mib"]["prediction_errors"].values()) <= 0.15
        assert 11.5 < g["tree_over_ring_time_at_1024_mib"] < 12.5
    for g in by_world[12]:
        assert g["points_ms"]["ring_allreduce"]["1"] < g["points_ms"]["ring_allreduce"]["16"]
        assert 5.4 < g["from_16_to_64_mib"]["gather_over_ring"] < 5.8
        assert 5.7 < g["rule_1_to_64_mib"]["gather_over_ring"] < 5.8
        assert 12.0 < g["tree_over_ring_time_at_1024_mib"] < 12.5
