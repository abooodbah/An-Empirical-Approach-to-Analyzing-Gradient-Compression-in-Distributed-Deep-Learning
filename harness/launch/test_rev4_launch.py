# harness/launch/test_rev4_launch.py
# -*- coding: utf-8 -*-
import os
import subprocess
import sys

import pytest

import su_ledger as L

HERE = os.path.dirname(os.path.abspath(__file__))
BASH = os.environ.get("GIT_BASH") or ("C:/PROGRA~1/Git/bin/bash.exe"
                                      if os.path.exists("C:/PROGRA~1/Git/bin/bash.exe") else "bash")


def test_bound_and_charge_use_the_fifteen_minute_minimum():
    assert L.bound(4, 50 * 60, 3) == pytest.approx(10.0)
    assert L.bound(2, 55 * 60, 3) == pytest.approx(5.5)
    assert L.charge(4, 120, 3) == pytest.approx(3.0)          # two minutes still cost fifteen
    assert L.charge(2, 30 * 60, 3) == pytest.approx(3.0)
    assert L.charge(4, 0, 3) == 0.0                           # cancelled while pending


def job(state, nodes, limit_min, elapsed_s=0):
    return {"state": state, "nodes": nodes, "limit_s": limit_min * 60, "elapsed_s": elapsed_s}


def test_committed_counts_bounds_for_unfinished_and_charges_for_finished():
    jobs = [job("PENDING", 4, 50), job("RUNNING", 2, 55, 600), job("COMPLETED", 2, 55, 2940),
            job("FAILED", 4, 50, 200), job("CANCELLED", 4, 50, 0)]
    assert L.committed(jobs, 3) == pytest.approx(10.0 + 5.5 + 4.9 + 3.0 + 0.0)


def test_a_submission_past_the_cap_is_refused():
    both = [job("PENDING", 2, 55), job("PENDING", 4, 50)]
    assert L.committed(both, 3) == pytest.approx(15.5)
    assert L.may_submit(both, 2, 45 * 60, 3, 20.0)            # 15.5 + 4.5
    assert not L.may_submit(both, 2, 55 * 60, 3, 20.0)        # 15.5 + 5.5
    failed = [job("FAILED", 2, 55, 300), job("FAILED", 4, 50, 400)]
    assert L.committed(failed, 3) == pytest.approx(4.5)
    assert L.may_submit(failed, 4, 50 * 60, 3, 20.0)
    assert L.may_submit(failed + [job("PENDING", 4, 50)], 2, 55 * 60, 3, 20.0)
    assert not L.may_submit(failed + [job("PENDING", 4, 50), job("PENDING", 2, 55)], 2, 15 * 60, 3, 20.0)


def test_cli_reads_the_ledger_and_sacct(tmp_path):
    ledger = tmp_path / "jobs.tsv"
    ledger.write_text("submitted_utc\tsystem\tset\tjob_id\tpartition\tnodes\ttime_limit\n"
                      "t\tstampede3\ts2\t111\tpvc\t4\t00:50:00\n"
                      "t\tlonestar6\tq12a\t222\tgpu-a100\t4\t00:50:00\n", encoding="utf-8")
    sacct = tmp_path / "sacct.txt"
    sacct.write_text("111|4|0|PENDING\n", encoding="utf-8")
    base = [sys.executable, os.path.join(HERE, "su_ledger.py"), "check", "--ledger", str(ledger),
            "--sacct", str(sacct), "--system", "stampede3", "--rate", "3", "--cap", "20", "--nodes", "2"]
    assert subprocess.run(base + ["--limit", "00:55:00"]).returncode == 0
    assert subprocess.run(base + ["--limit", "01:45:00"]).returncode == 1


def trace(sets_file, red_set, functions):
    stubs = "; ".join('%s() { echo "%s $*"; }' % (f, f) for f in functions)
    script = 'set -e; RED_SET=%s; %s; source "%s"' % (red_set, stubs, sets_file.replace("\\", "/"))
    done = subprocess.run([BASH, "-c", script], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    return done.stdout.strip().split("\n")


E9_FUNCS = ("node_facts", "preflight", "dtype_check", "probe", "hs_probe", "calib", "equivalence",
            "e9_fidelity", "e9_phase", "e9_block")


def test_e9_b_allocation_runs_probes_then_thirty_six_block_runs_second_pass_reversed():
    lines = trace(os.path.join(HERE, "e9_sets.sh"), "q24b", E9_FUNCS)
    assert lines[:3] == ["node_facts ", "preflight transformer", "dtype_check "]
    assert [l for l in lines if l.startswith("probe")] == ["probe default", "probe ring", "probe tree"]
    assert not any(l.startswith(("hs_probe", "e9_fidelity", "e9_phase")) for l in lines)
    block = [l.split() for l in lines if l.startswith("e9_block")]
    first = [(b[1], b[2]) for b in block if b[3] == "1"]
    second = [(b[1], b[2]) for b in block if b[3] == "2"]
    assert len(first) == 18 and second == first[::-1]
    assert lines.index("calib ") < lines.index("e9_block transformer native 1")


def test_e9_a_allocation_adds_fidelity_phase_and_host_staged_probes_before_the_outcomes():
    lines = trace(os.path.join(HERE, "e9_sets.sh"), "q12a", E9_FUNCS)
    assert sum(l.startswith("hs_probe") for l in lines) == 3
    assert sum(l.startswith("e9_fidelity") for l in lines) == 10
    assert sum(l.startswith("e9_phase") for l in lines) == 12
    last_extra = max(i for i, l in enumerate(lines) if l.startswith(("e9_fidelity", "e9_phase")))
    assert last_extra < min(i for i, l in enumerate(lines) if l.startswith("e9_block"))
    assert not any("ddp" in l for l in lines if l.startswith(("e9_fidelity", "e9_phase")))


def test_e9_rejects_an_unknown_set():
    script = 'RED_SET=q48a; source "%s"' % os.path.join(HERE, "e9_sets.sh").replace("\\", "/")
    assert subprocess.run([BASH, "-c", script], capture_output=True).returncode != 0


E10_FUNCS = ("smoke", "stop_after_smoke", "probe", "e10")


def test_e10_orders_the_gather_first_and_the_flat_dense_run_last():
    s1 = trace(os.path.join(HERE, "e10_sets.sh"), "s1", E10_FUNCS)
    assert s1[0].startswith("smoke") and s1[1] == "probe "
    runs = [l.split()[1:5] for l in s1 if l.startswith("e10 ")]
    assert runs == [["2", "gpu-quant-bucketed", "40", "10"], ["2", "gpu-quant-reducible-bucketed", "40", "10"],
                    ["2", "native-bucketed", "40", "10"], ["1", "gpu-quant", "40", "10"],
                    ["1", "gpu-quant-bucketed", "40", "10"], ["2", "native", "40", "10"]]
    s2 = [l.split()[1:5] for l in trace(os.path.join(HERE, "e10_sets.sh"), "s2", E10_FUNCS) if l.startswith("e10 ")]
    assert s2 == [["4", "gpu-quant-bucketed", "20", "5"], ["4", "gpu-quant-reducible-bucketed", "20", "5"],
                  ["4", "native-bucketed", "20", "5"]]


def test_no_launch_file_cancels_by_user_or_name():
    for name in ("ls6_e9.slurm", "e9_sets.sh", "pvc_e10.slurm", "e10_sets.sh"):
        text = open(os.path.join(HERE, name), encoding="utf-8").read()
        for bad in ("scancel -u", "scancel --user", "scancel -n", "scancel --name", "scontrol hold",
                    "scontrol update"):
            assert bad not in text, (name, bad)
