# validation_v2/analysis/tests/test_harness_v52.py
# -*- coding: utf-8 -*-
"""Contract tests of harness v5.2 that need no accelerator."""
import hashlib
import io
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
HARNESS = os.path.join(os.path.dirname(os.path.dirname(HERE)), "harness")
sys.path.insert(0, HARNESS)
import control_workload_benchmark_v5_2 as H  # noqa: E402

BASE = ["--campaign-id", "c", "--run-id", "r", "--output-dir", "o", "--model", "transformer",
        "--skip-calibration", "--profile", "qualification"]


def parse(extra):
    return H.validate_config(H.build_parser().parse_args(BASE + extra))


def lines(name):
    return io.open(os.path.join(HARNESS, name), encoding="utf-8").read().split("\n")


def test_v51_is_untouched_and_is_a_subsequence_of_v52():
    raw = open(os.path.join(HARNESS, "control_workload_benchmark_v5_1.py"), "rb").read()
    assert hashlib.sha256(raw).hexdigest().startswith("b920424cfa25bdd7")
    new = iter(lines("control_workload_benchmark_v5_2.py"))
    missing = [old for old in lines("control_workload_benchmark_v5_1.py") if old not in new]
    assert missing == []


def test_recorded_hash_matches_the_generated_file():
    raw = open(os.path.join(HARNESS, "control_workload_benchmark_v5_2.py"), "rb").read()
    recorded = io.open(os.path.join(os.path.dirname(HARNESS), "CODE_v52.sha256"), encoding="utf-8").read().split()[0]
    assert hashlib.sha256(raw).hexdigest() == recorded


def test_every_v51_mode_is_still_accepted():
    for mode in ("native", "gpu-quant", "gpu-quant-reducible", "gpu-quant-gathersum", "fp16-native",
                 "ddp-overlap", "ddp-int8-hook", "ddp-fp16-hook", "gpu-topk", "powersgd"):
        assert parse(["--comm-mode", mode]).comm_mode == mode


def test_bucketed_modes_need_a_bucket_size_and_others_reject_one():
    for mode in H.BUCKETED_MODES:
        with pytest.raises(H.ConfigurationError):
            parse(["--comm-mode", mode])
        assert parse(["--comm-mode", mode, "--exchange-bucket-elems", "4096"]).exchange_bucket_elems == 4096
    with pytest.raises(H.ConfigurationError):
        parse(["--comm-mode", "gpu-quant", "--exchange-bucket-elems", "4096"])


def test_progress_file_is_refused_inside_a_block_timed_run():
    with pytest.raises(H.ConfigurationError):
        parse(["--comm-mode", "native", "--timing-mode", "block", "--progress-dir", "p"])
    assert parse(["--comm-mode", "native", "--timing-mode", "wall", "--progress-dir", "p"]).progress_dir is not None


def test_fidelity_is_a_qualification_option_and_not_for_ddp_modes():
    with pytest.raises(H.ConfigurationError):
        H.validate_config(H.build_parser().parse_args(
            BASE[:-1] + ["extension", "--comm-mode", "gpu-quant", "--verify-fidelity"]))
    with pytest.raises(H.ConfigurationError):
        parse(["--comm-mode", "ddp-int8-hook", "--verify-fidelity"])
    assert parse(["--comm-mode", "gpu-quant-sr-fp16", "--verify-fidelity"]).verify_fidelity


@pytest.mark.parametrize("n,bucket", [(10, 3), (9, 3), (5, 8), (1, 1), (11430000, 1000003)])
def test_bucket_bounds_cover_the_gradient_once_in_order(n, bucket):
    bounds = H.bucket_bounds(n, bucket)
    assert bounds[0][0] == 0 and bounds[-1][1] == n
    assert all(a[1] == b[0] for a, b in zip(bounds, bounds[1:]))
    assert all(0 < stop - start <= bucket for start, stop in bounds)
    assert len(bounds) == -(-n // bucket)


def test_bucket_bounds_reject_a_nonpositive_size():
    with pytest.raises(H.ConfigurationError):
        H.bucket_bounds(10, 0)


def test_exchange_buffer_accounting():
    assert H.exchange_buffer_bytes("gpu-quant-bucketed", 32, 16777216) == 32 * 16777216 + 4 * 16777216
    assert H.exchange_buffer_bytes("gpu-quant-reducible-bucketed", 32, 16777216) == 0
    assert H.exchange_buffer_bytes("native-bucketed", 32, 16777216) == 0
    with pytest.raises(H.ConfigurationError):
        H.exchange_buffer_bytes("gpu-quant", 8, 1)


@pytest.mark.parametrize("p", range(2, 65))
def test_levels_keep_partial_sums_representable(p):
    assert p * H.levels_int8(p) <= 127 and p * H.levels_fp16(p) <= 2048


def test_stochastic_levels_are_unbiased_bounded_integers():
    torch = H.torch
    generator = torch.Generator().manual_seed(5)
    values = torch.full((400000,), 0.3)
    scale = torch.tensor(1.0)
    q = H.stochastic_levels(values, scale, 3.0, generator)
    assert set(q.unique().tolist()) <= {0.0, 1.0}
    assert abs(float(q.mean()) - 0.3) < 0.005
    edge = H.stochastic_levels(torch.tensor([3.0, -3.0, 2.999, -2.999]), scale, 3.0, generator)
    assert float(edge.max()) <= 3.0 and float(edge.min()) >= -3.0
    assert bool(torch.equal(edge, torch.floor(edge)))


def test_fp16_carries_the_integer_sum_exactly():
    torch = H.torch
    p = 24
    level = H.levels_fp16(p)
    total = torch.zeros(4, dtype=torch.float16)
    for _ in range(p):
        total += torch.tensor([level, -level, level - 1, 1], dtype=torch.float16)
    assert total.to(torch.int32).tolist() == [p * level, -p * level, p * (level - 1), p]
