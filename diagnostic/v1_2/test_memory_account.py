# diagnostic/v1_2/test_memory_account.py
# -*- coding: utf-8 -*-
import io
import os

import pytest

import memory_account as A

HEADER = "model,comm_mode,world_size,timing_mode,rank,peak_memory_mb\n"
HERE = os.path.dirname(os.path.abspath(__file__))
MEM_ROOT = os.environ.get("REV4_MEM_ROOT", "C:/r5w/mem")


def write(tmp_path, name, body):
    d = tmp_path / "run" / "results"
    d.mkdir(parents=True, exist_ok=True)
    io.open(str(d / name), "w", encoding="utf-8", newline="\n").write(HEADER + body)
    return str(tmp_path / "run")


def test_peak_is_the_largest_over_ranks_and_rows(tmp_path):
    write(tmp_path, "a__rank0000.csv", "llama1b,gpu-quant,8,phase,0,41000\nllama1b,gpu-quant,8,phase,0,42000\n")
    run = write(tmp_path, "a__rank0001.csv", "llama1b,gpu-quant,8,phase,1,41500\n")
    assert A.peaks(run) == {("llama1b", "gpu-quant", 8, "phase"): 42000.0}


def test_rows_without_a_peak_are_skipped(tmp_path):
    run = write(tmp_path, "b__rank0000.csv", "llama1b,native,8,phase,0,\nllama1b,native,8,phase,0,None\n")
    assert A.peaks(run) == {}


def test_whole_gradient_gather_holds_the_received_copies_and_the_rank_own_copy():
    # Measured on the Intel system at 8 ranks (jobs 3517751 and 3579139): the
    # dense footprint is 33.541047 GB and the whole-gradient gather peaks at
    # 42.957567 GB. The difference is 9.002 int8 copies of the gradient: the
    # eight received and the rank's own encoded one. An account with p copies
    # gives 41.9 GB and misses the measurement by a whole copy.
    footprint = 33.541047
    assert A.gather_peak_gb(footprint, 8, A.GRADIENT_INT8_GB) == pytest.approx(42.957567, abs=0.01)
    assert A.gather_peak_gb(footprint, 16, A.GRADIENT_INT8_GB) == pytest.approx(51.32, abs=0.01)
    assert A.gather_peak_gb(footprint, 32, A.GRADIENT_INT8_GB) == pytest.approx(68.06, abs=0.01)


def test_the_account_carries_its_notes_and_the_measured_check(tmp_path):
    # Nothing in MEMORY_ACCOUNT.md is written by hand, so a rerun cannot
    # erase the notes; and the arithmetic is printed beside the measurement
    # it has to reproduce.
    write(tmp_path, "a__rank0000.csv", "llama1b,native,8,phase,0,33541.047\n")
    run = write(tmp_path, "b__rank0000.csv", "llama1b,gpu-quant,8,phase,0,42957.567\n")
    text = A.render([("redctl-V-1n-3517751", run)])
    assert "## The two hook failures on record" in text and "## Units" in text
    assert "8 ranks: 43.0 GB by this arithmetic, 43.0 GB measured" in text
    assert "16 ranks: 51.3 GB" in text and "32 ranks: 68.1 GB" in text


def test_the_committed_account_is_what_the_script_writes():
    # Needs the four archives extracted (reproduce_diagnostic.sh does it).
    if not os.path.isdir(os.path.join(MEM_ROOT, "v2-L1-2n-3465529", "results")):
        pytest.skip("extract the archives first (reproduce_diagnostic.sh, or REV4_MEM_ROOT)")
    committed = io.open(os.path.join(HERE, "MEMORY_ACCOUNT.md"), encoding="utf-8").read()
    assert A.render(A.runs_under(MEM_ROOT)) == committed


def test_lonestar6_hook_holds_far_more_than_dense_ddp():
    run = os.path.join(MEM_ROOT, "v2-L1-2n-3465529")
    if not os.path.isdir(run):
        pytest.skip("extract the archives first (reproduce_diagnostic.sh, or REV4_MEM_ROOT)")
    p = A.peaks(run)
    dense = p[("llama1b", "ddp-overlap", 6, "block")]
    hook = p[("llama1b", "ddp-int8-hook", 6, "block")]
    assert dense == pytest.approx(12717, abs=5)
    assert hook == pytest.approx(22144, abs=5)
