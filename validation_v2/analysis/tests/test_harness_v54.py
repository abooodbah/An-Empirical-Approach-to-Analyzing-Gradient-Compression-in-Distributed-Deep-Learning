# validation_v2/analysis/tests/test_harness_v54.py
# -*- coding: utf-8 -*-
"""Harness v5.4 (2026-10-14): the int8 hook's asynchronous path keeps its
gathered scales outside HOOK_STATE. In v5.3 it stored them in HOOK_STATE,
which the provenance record copies, and json.dumps failed on the tensors
after every asynchronous run (E11 job 3508877)."""
import hashlib
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
HARNESS = os.path.join(os.path.dirname(os.path.dirname(HERE)), "harness")
sys.path.insert(0, HARNESS)
import control_workload_benchmark_v5_4 as H  # noqa: E402


def source(name):
    return io.open(os.path.join(HARNESS, name), encoding="utf-8").read()


def test_no_tensor_is_stored_in_hook_state():
    src = source("control_workload_benchmark_v5_4.py")
    assert 'HOOK_STATE["s_and_z"]' not in src
    assert 'HOOK_TENSORS["s_and_z"] = gathered_s_and_z' in src
    assert 'gathered_s_and_z = HOOK_TENSORS["s_and_z"]' in src


def test_hook_tensors_live_apart_from_the_recorded_state():
    assert H.HOOK_TENSORS is not H.HOOK_STATE
    assert "HOOK_TENSORS" not in json.dumps(H.HOOK_STATE)


def test_hook_state_with_the_async_fields_serializes():
    state = dict(H.HOOK_STATE, mode="async", backend="nccl", gather_futures="async")
    assert json.loads(json.dumps(state, allow_nan=False))["mode"] == "async"


def test_v53_selector_is_kept():
    assert H.int8_hook_mode("nccl") == "async"
    assert H.int8_hook_mode("ccl") == "sync"


def test_v53_is_untouched():
    raw = open(os.path.join(HARNESS, "control_workload_benchmark_v5_3.py"), "rb").read()
    assert hashlib.sha256(raw).hexdigest().startswith("6f941a652e7da18c")


def test_recorded_hash_matches_the_generated_file():
    raw = open(os.path.join(HARNESS, "control_workload_benchmark_v5_4.py"), "rb").read()
    recorded = io.open(os.path.join(os.path.dirname(HARNESS), "CODE_v54.sha256"), encoding="utf-8").read().split()[0]
    assert hashlib.sha256(raw).hexdigest() == recorded
