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


# ---- added after the second review (2026-10-08): points that are not on a line
def curved(tmp_path, run="e9-q24a-8n-9"):
    """Probes whose points bend, so that every fit range gives its own ratio
    and a mistake in a range, a divisor or a probe shows."""
    ring_ar = {("allreduce", 1): 0.010, ("allreduce", 16): 0.004, ("allreduce", 64): 0.012,
               ("allreduce", 256): 0.050, ("allreduce", 1024): 0.175}
    ring_ag = {("allgather", 1): 0.003, ("allgather", 16): 0.033, ("allgather", 64): 0.134,
               ("allgather", 256): 0.536}
    default_ar = {("allreduce", 1): 0.003, ("allreduce", 16): 0.0037, ("allreduce", 64): 0.0121,
                  ("allreduce", 256): 0.0525, ("allreduce", 1024): 0.200}
    default_ag = {("allgather", 1): 0.003, ("allgather", 16): 0.040, ("allgather", 64): 0.160,
                  ("allgather", 256): 0.640}
    tree_ar = {("allreduce", 1): 0.003, ("allreduce", 16): 0.011, ("allreduce", 64): 0.124,
               ("allreduce", 256): 0.522, ("allreduce", 1024): 2.100}
    probe(tmp_path, run, "ring", allreduce=1e-10, allgather=1e-10, world=24, slow=dict(ring_ar, **ring_ag))
    probe(tmp_path, run, "default", allreduce=1e-10, allgather=1e-10, world=24, slow=dict(default_ar, **default_ag))
    return probe(tmp_path, run, "tree", allreduce=1e-10, world=24, slow=tree_ar)


def secant(points, a, b):
    return (points[b] - points[a]) / ((b - a) * MIB)


def test_each_fit_range_uses_its_own_points_and_the_ring_probes_own_gather(tmp_path):
    gdr = P.allocation(curved(tmp_path))["gdr"]
    ring_ar = {16: 0.004, 64: 0.012, 256: 0.050}
    ring_ag = {16: 0.033, 64: 0.134, 256: 0.536}
    default_ar = {16: 0.0037, 64: 0.0121, 256: 0.0525}
    # two points: the line is the secant
    assert gdr["from_16_to_64_mib"]["gather_over_ring"] == pytest.approx(
        secant(ring_ag, 16, 64) / secant(ring_ar, 16, 64))                      # 12.625
    assert gdr["from_64_to_256_mib"]["gather_over_ring"] == pytest.approx(
        secant(ring_ag, 64, 256) / secant(ring_ar, 64, 256))                    # 10.579
    assert gdr["from_16_to_64_mib"]["default_over_ring"] == pytest.approx(
        secant(default_ar, 16, 64) / secant(ring_ar, 16, 64))
    # three points: least squares, checked against numpy
    import numpy
    sizes = [16 * MIB, 64 * MIB, 256 * MIB]
    b_ag = numpy.polyfit(sizes, [ring_ag[k] for k in (16, 64, 256)], 1)[0]
    b_ar = numpy.polyfit(sizes, [ring_ar[k] for k in (16, 64, 256)], 1)[0]
    assert gdr["from_16_to_256_mib"]["gather_over_ring"] == pytest.approx(b_ag / b_ar)
    assert not gdr["from_16_to_64_mib"]["gather_over_ring"] == pytest.approx(
        gdr["from_16_to_256_mib"]["gather_over_ring"], rel=0.02)


def test_the_prediction_error_is_relative_to_the_measured_time_above_the_ranges_top(tmp_path):
    gdr = P.allocation(curved(tmp_path))["gdr"]
    default_ar = {16: 0.0037, 64: 0.0121, 256: 0.0525, 1024: 0.200}
    slope = secant(default_ar, 16, 64)
    predicted = {k: default_ar[64] + slope * (k - 64) * MIB for k in (256, 1024)}
    errors = gdr["from_16_to_64_mib"]["prediction_errors"]
    assert sorted(errors) == ["allgather 256 MiB", "allreduce 1024 MiB", "allreduce 256 MiB"]   # 64 MiB is not scored
    assert errors["allreduce 256 MiB"] == pytest.approx(abs(predicted[256] - 0.0525) / 0.0525)
    assert errors["allreduce 1024 MiB"] == pytest.approx(abs(predicted[1024] - 0.200) / 0.200)
    # a range that ends at 256 MiB predicts the 1 GiB point only
    assert sorted(gdr["from_64_to_256_mib"]["prediction_errors"]) == ["allreduce 1024 MiB"]
    assert sorted(gdr["rule_1_to_64_mib"]["prediction_errors"]) == sorted(errors)


def test_the_tree_is_set_against_the_forced_ring_not_the_default_probe(tmp_path):
    gdr = P.allocation(curved(tmp_path))["gdr"]
    assert gdr["tree_over_ring_time_at_1024_mib"] == pytest.approx(2.100 / 0.175)       # the default took 0.200 s
    assert gdr["from_16_to_64_mib"]["tree_over_ring"] == pytest.approx(
        secant({16: 0.011, 64: 0.124}, 16, 64) / secant({16: 0.004, 64: 0.012}, 16, 64))


def test_the_script_on_a_committed_archive_gives_the_committed_entry():
    # ties the script to its committed output; reproduce_diagnostic.sh compares all four
    import glob
    import shutil
    import tarfile
    import tempfile
    archive = glob.glob(os.path.join(ROOT, "raw", "rev4", "e9-q24b-*.tgz"))
    assert len(archive) == 1
    work = tempfile.mkdtemp(prefix="e9p", dir=os.environ.get("REV4_SHORT_TMP"))
    try:
        with tarfile.open(archive[0]) as tf:
            members = [m for m in tf.getmembers() if "/probes/" in m.name]
            try:
                tf.extractall(work, members=members, filter="data")
            except TypeError:
                tf.extractall(work, members=members)
        run = glob.glob(os.path.join(work, "e9-q24b-*"))[0].replace("\\", "/") + "/"
        got = json.loads(json.dumps(P.allocation(run)))
    finally:
        shutil.rmtree(work, ignore_errors=True)
    want = [a for a in committed()["allocations"] if a["run_dir"] == got["run_dir"]]
    assert len(want) == 1 and got == want[0]


def test_the_second_reading_of_all_three_checks_is_in_the_committed_file():
    # what the paper's second paragraph states for 24 ranks over GPUDirect
    late = [a["gdr"] for a in committed()["allocations"] if a["world"] == 24]
    assert [round(g["from_16_to_64_mib"]["default_over_ring"], 3) for g in late] == [0.977, 1.058]
    assert [round(max(g["from_16_to_64_mib"]["prediction_errors"].values()), 3) for g in late] == [0.134, 0.103]
    ratios = [g[name]["gather_over_ring"] for g in late
              for name in ("from_16_to_64_mib", "from_16_to_256_mib", "from_64_to_256_mib")]
    assert round(min(ratios), 1) == 10.5 and round(max(ratios), 1) == 12.1
    assert [round(g["from_16_to_256_mib"]["gather_over_ring"], 2) for g in late] == [10.63, 11.07]
    assert [round(g["from_64_to_256_mib"]["gather_over_ring"], 2) for g in late] == [10.46, 10.91]
    # the host-staged path has no slow point and a different tree ratio
    host = [a["host_staged"] for a in committed()["allocations"] if "host_staged" in a]
    assert all(h["ring_allreduce_1_mib_over_16_mib"] < 1 for h in host) and len(host) == 2
    assert [round(h["tree_over_ring_time_at_1024_mib"], 2) for h in host] == [4.22, 4.05]
