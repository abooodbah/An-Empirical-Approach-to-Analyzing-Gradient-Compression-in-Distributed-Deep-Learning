# validation_v2/analysis/tests/test_harness_v53.py
# -*- coding: utf-8 -*-
"""Harness v5.3 (2026-10-13): the int8 hook's synchronous form is selected on
oneCCL only. In v5.2 the test `"ccl" in backend` also matched "nccl", so the
hook ran synchronously on every system."""
import hashlib
import io
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
HARNESS = os.path.join(os.path.dirname(os.path.dirname(HERE)), "harness")
sys.path.insert(0, HARNESS)
import control_workload_benchmark_v5_3 as H  # noqa: E402


def lines(name):
    return io.open(os.path.join(HARNESS, name), encoding="utf-8").read().split("\n")


@pytest.mark.parametrize("backend", ["nccl", "NCCL", "gloo"])
def test_hook_runs_async_off_oneccl(backend, monkeypatch):
    monkeypatch.delenv("RED_HOOK_SYNC", raising=False)
    assert H.int8_hook_mode(backend) == "async"


@pytest.mark.parametrize("backend", ["ccl", "xccl", "CCL"])
def test_hook_runs_sync_on_oneccl(backend, monkeypatch):
    monkeypatch.delenv("RED_HOOK_SYNC", raising=False)
    assert H.int8_hook_mode(backend) == "sync"


def test_environment_still_forces_the_sync_form(monkeypatch):
    monkeypatch.setenv("RED_HOOK_SYNC", "1")
    assert H.int8_hook_mode("nccl") == "sync"


def test_the_hook_uses_the_selector():
    src = "\n".join(lines("control_workload_benchmark_v5_3.py"))
    assert 'HOOK_STATE["mode"] = int8_hook_mode(backend)' in src
    assert '"ccl" in backend' not in src


def test_v52_is_untouched():
    raw = open(os.path.join(HARNESS, "control_workload_benchmark_v5_2.py"), "rb").read()
    assert hashlib.sha256(raw).hexdigest().startswith("6d32e96e210059a9")


def test_recorded_hash_matches_the_generated_file():
    raw = open(os.path.join(HARNESS, "control_workload_benchmark_v5_3.py"), "rb").read()
    recorded = io.open(os.path.join(os.path.dirname(HARNESS), "CODE_v53.sha256"), encoding="utf-8").read().split()[0]
    assert hashlib.sha256(raw).hexdigest() == recorded
