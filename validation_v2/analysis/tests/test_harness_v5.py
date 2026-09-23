# -*- coding: utf-8 -*-
"""Contract tests of harness v5 that need no accelerator: CLI validation,
helpers, and result-record validation."""
import copy
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import control_workload_benchmark_v5 as H  # noqa: E402

BASE = ["--campaign-id", "c", "--run-id", "r", "--output-dir", "o", "--model", "transformer",
        "--comm-mode", "gpu-quant-reducible", "--skip-calibration"]
VALID = ["--profile", "validation", "--iters", "100", "--warmup-iters", "20", "--input-pool", "4"]


def parse(extra):
    return H.validate_config(H.build_parser().parse_args(BASE + extra))


def test_validation_profile_accepts_frozen_condition():
    a = parse(VALID + ["--timing-mode", "block"])
    assert a.timing_mode == "block" and a.input_pool == 4


@pytest.mark.parametrize("change", [["--iters", "40"], ["--warmup-iters", "10"], ["--input-pool", "0"],
                                    ["--sync-debug", "warn"], ["--verify-equivalence"]])
def test_validation_profile_rejects_other_settings(change):
    args = VALID + ["--timing-mode", "block"]
    flag = change[0]
    if flag in args:
        i = args.index(flag)
        args = args[:i] + change + args[i + 2:]
    else:
        args = args + change
    with pytest.raises(H.ConfigurationError):
        parse(args)


def test_seq_len_applies_to_llama_only():
    with pytest.raises(H.ConfigurationError):
        parse(["--profile", "qualification", "--seq-len", "512"])
    a = H.validate_config(H.build_parser().parse_args(
        BASE[:6] + ["--model", "llama1b", "--comm-mode", "ddp-overlap", "--skip-calibration",
                    "--profile", "qualification", "--seq-len", "512"]))
    assert a.seq_len == 512


def test_cpu_and_gloo_are_qualification_only():
    with pytest.raises(H.ConfigurationError):
        parse(["--accelerator", "cpu", "--backend", "gloo"])          # paper profile
    with pytest.raises(H.ConfigurationError):
        parse(["--profile", "qualification", "--accelerator", "cuda", "--backend", "gloo"])
    assert parse(["--profile", "qualification", "--accelerator", "cpu", "--backend", "gloo"]).accelerator == "cpu"


def test_calibration_sizes_and_trace_ranks():
    assert H.parse_calibration_sizes("4096,1024,4096") == (1024, 4096)
    assert H.parse_calibration_sizes(None) is None
    with pytest.raises(H.ConfigurationError):
        H.parse_calibration_sizes("2")
    assert H.parse_rank_list("3,0,3") == (0, 3)


def test_block_step_equivalent():
    assert H.block_step_equivalent([1.0, 2.0, 1.5], 100) == (2.0, 0.02)
    for bad in (([], 10), ([1.0], 0), ([float("nan")], 5), ([-1.0], 5)):
        with pytest.raises(ValueError):
            H.block_step_equivalent(*bad)


def _record():
    r = {field: None for field in H.RESULT_FIELDS}
    r.update({
        "schema_version": H.SCHEMA_VERSION, "campaign_id": "c", "track": "control-extension", "run_id": "r",
        "timing_semantics": H.BLOCK_SEMANTICS, "software_versions": "{}", "status": "PASS",
        "provenance_file": "p.json", "model": "transformer", "mode": "gpu-quant-reducible",
        "comm_mode": "gpu-quant-reducible", "timing_mode": "block", "profile": "validation",
        "validated_sample_count": 100, "measured_iterations": 100, "nodes": 4, "ranks_per_node": 3,
        "world_size": 12, "local_batch_size": 32, "global_batch_size": 384, "paper_condition": False,
        "do_communication": True, "accelerator": "cuda", "loss": 0.7, "block_iterations": 100,
        "block_elapsed_s": 2.0, "block_elapsed_max_s": 2.1, "block_step_equiv_s": 0.021,
    })
    return r


def test_block_record_validates():
    H.validate_result_record(_record())


@pytest.mark.parametrize("field,value", [("block_step_equiv_s", None), ("block_iterations", 40),
                                         ("observed_step_s", 0.02), ("validated_sample_count", 99),
                                         ("accelerator", "cpu")])
def test_block_record_rejections(field, value):
    r = copy.deepcopy(_record())
    r[field] = value
    with pytest.raises(ValueError):
        H.validate_result_record(r)
