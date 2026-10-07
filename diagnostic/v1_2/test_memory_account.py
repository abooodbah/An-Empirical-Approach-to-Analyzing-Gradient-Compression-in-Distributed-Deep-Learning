# diagnostic/v1_2/test_memory_account.py
# -*- coding: utf-8 -*-
import io
import os

import pytest

import memory_account as A

HEADER = "model,comm_mode,world_size,timing_mode,rank,peak_memory_mb\n"


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


def test_whole_gradient_gather_arithmetic_matches_section_viii():
    assert A.gather_peak_gb(32.8, 16, 1.05) == pytest.approx(49.6)
    assert A.gather_peak_gb(32.8, 32, 1.05) == pytest.approx(66.4)


def test_lonestar6_hook_holds_far_more_than_dense_ddp():
    run = "C:/r5w/mem/v2-L1-2n-3465529"
    if not os.path.isdir(run):
        pytest.skip("extract the archives first (Task 6 Step 1)")
    p = A.peaks(run)
    dense = p[("llama1b", "ddp-overlap", 6, "block")]
    hook = p[("llama1b", "ddp-int8-hook", 6, "block")]
    assert dense == pytest.approx(12717, abs=5)
    assert hook == pytest.approx(22144, abs=5)
