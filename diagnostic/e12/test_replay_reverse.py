# -*- coding: utf-8 -*-
"""Tests for the E12 reverse substitutions (replay_reverse.py) on synthetic
Kineto-shaped traces whose replay makespans are known by hand. The frozen E8
reference is used, never changed (diagnostic/e8/test_freeze.py)."""
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
E8 = os.path.join(os.path.dirname(HERE), "e8")
for _p in (HERE, E8):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import replay_reference as R  # noqa: E402
import replay_reverse as RV  # noqa: E402
import score_e12 as S  # noqa: E402
from test_replay_reference import COMPUTE, NCCL, TraceBuilder, _wrap  # noqa: E402

SIDE = 89
CAL = {"alpha_allreduce_s": 3e-5, "beta_allreduce_s_per_byte": 1e-10,
       "alpha_allgather_s": 10e-6, "beta_allgather_s_per_byte": 40e-9}


def _collective(b, t, kind, n, dur, world, nccl_name):
    """A c10d collective: syncStreams wait, the NCCL kernel, the end record."""
    _wrap(b, "c10d::%s_" % kind, t, 40.0)
    b.runtime(t + 1, "cudaEventRecord")
    b.runtime(t + 3, "cudaStreamWaitEvent", op="record_param_comms")
    c = b.launch(t + 6, NCCL, dur, name=nccl_name, op="record_param_comms", rt="cudaLaunchKernelExC",
                 args={"In msg nelems": n, "Group size": world})
    b.runtime(t + 30, "cudaEventRecord")
    b.runtime(t + 34, "cudaStreamWaitEvent", op="record_param_comms")
    return c, t + 50


# ---------------------------------------------------------------- G-rev
def _reducible_iteration(b, t0, n=10000, world=4, ar_us=200.0):
    """gpu-quant-reducible: mm, encode (div, round, clamp at 8n bytes, to int8
    at 5n; 1000 bytes per us), the int8 all-reduce, the final cast (100 us),
    mul, optimizer, device sync."""
    b.annotate("DistributedDataParallel.forward", t0, 1.0)
    b.launch(t0, COMPUTE, 100.0, op="aten::mm")
    t = t0 + 10
    for name, dur in (("aten::div", 80.0), ("aten::round", 80.0), ("aten::clamp", 80.0), ("aten::to", 50.0)):
        b.launch(t, COMPUTE, dur, op=name)
        t += 10
    _, t = _collective(b, t, "allreduce", n, ar_us, world, "ncclDevKernel_AllReduce_Sum_u8_RING_LL(x)")
    b.launch(t, COMPUTE, 100.0, op="aten::to")
    b.launch(t + 10, COMPUTE, 10.0, op="aten::mul_")
    b.annotate("Optimizer.step#SGD.step", t + 20, 5.0)
    b.launch(t + 21, COMPUTE, 10.0, op="aten::add_")
    b.runtime(t + 40, "cudaDeviceSynchronize", d=2500.0)


def reducible_trace(iters=4):
    b = TraceBuilder()
    for i in range(iters):
        _reducible_iteration(b, i * 4000.0)
    return b.trace()


def test_allgather_cost_is_the_e12_cost_models_line():
    for world, nbytes in ((6, 11_425_794), (6, 8), (4, 10000)):
        cal = dict(CAL)
        want = S.line(cal, "allgather", nbytes) if world == S.P else None
        got = RV.allgather_s(cal, world, nbytes)
        if want is not None:
            assert got == pytest.approx(want, rel=1e-12)
    assert RV.allgather_s(CAL, 4, 10000) == pytest.approx(3 * 10e-6 + 40e-9 * 0.75 * 10000)


def test_reduce_regions_found():
    regs = RV.reduce_regions(reducible_trace())
    assert [r["n"] for r in regs] == [10000] * 4 and all(r["world"] == 4 for r in regs)
    assert [e["name"] for e in regs[0]["enc"]] == ["aten::div", "aten::round", "aten::clamp", "aten::to"]


def test_reducible_to_gathersum_by_hand():
    tr = reducible_trace()
    base = R.replay(tr).busy_ms
    # mm 100, encode 290, all-reduce 200, final cast 100, mul 10, optimizer 10
    assert base == pytest.approx(0.710, abs=0.003)
    cand = R.replay(RV.substitute_reduce_gather(tr, CAL)).busy_ms
    # mm 100, encode 290, all-gather 30 + 300, 4 copies of 2n at 1000 B/us = 80,
    # (zero-fill 40 runs beside the all-gather), 4 x (cast 50 + add 120) = 680,
    # final cast 100 x 8/5 = 160, mul 10, optimizer 10
    assert cand == pytest.approx(1.660, abs=0.003)


def test_gathersum_decode_waits_for_the_copies():
    sub = RV.substitute_reduce_gather(reducible_trace(), CAL)
    res = R.replay(sub)
    copies = [c for c, g in sub.gpu.items() if g["name"] == "gather copy (substituted)"]
    casts = [c for c, g in sub.gpu.items() if g["name"] == "int8 to int32 (substituted)"]
    assert len(copies) == 16 and len(casts) == 16
    # every iteration's first per-source cast starts after that iteration's last copy
    ends = sorted(res.finish[c] for c in copies)
    starts = sorted(res.start[c] for c in casts)
    for k in range(4):
        assert starts[4 * k] >= ends[4 * k + 3] - 1e-6


def test_g_rev_needs_a_gradient_all_reduce():
    b = TraceBuilder()
    b.launch(0, COMPUTE, 100.0, op="aten::mm")
    with pytest.raises(ValueError):
        RV.substitute_reduce_gather(b.trace(), CAL)


# ---------------------------------------------------------------- H-rev
def _fp16_bucket(b, t, n, world, ar_us=300.0, nccl_name="ncclDevKernel_AllReduce_Sum_f16_RING_LL(x)"):
    """One fp16-hook bucket inside its AccumulateGrad function: cast (6n
    bytes, 600 us at n = 100000), divide (4n, 400 us), the 16-bit all-reduce,
    and the copy back on a side stream."""
    start = t
    b.launch(t, COMPUTE, 6.0 * n / 1000.0, op="aten::to")
    b.launch(t + 10, COMPUTE, 4.0 * n / 1000.0, op="aten::div_")
    _, t = _collective(b, t + 20, "allreduce", n, ar_us, world, nccl_name)
    b.launch(t, SIDE, 6.0 * n / 1000.0, op="aten::copy_")
    t += 10
    _wrap(b, "autograd::engine::evaluate_function: torch::autograd::AccumulateGrad", start - 1, t - start + 1)
    return t


def _fp16_iteration(b, t0, n=100000, world=4, **kw):
    b.annotate("DistributedDataParallel.forward", t0, 1.0)
    b.launch(t0, COMPUTE, 1000.0, op="aten::mm")
    t = _fp16_bucket(b, t0 + 10, n, world, **kw)
    b.launch(t, COMPUTE, 10.0, op="aten::copy_")
    b.annotate("Optimizer.step#SGD.step", t + 10, 5.0)
    b.launch(t + 11, COMPUTE, 10.0, op="aten::add_")
    b.runtime(t + 30, "cudaDeviceSynchronize", d=30000.0)


def fp16_trace(iters=3, **kw):
    b = TraceBuilder()
    for i in range(iters):
        _fp16_iteration(b, i * 40000.0, **kw)
    return b.trace()


def test_int8_hook_op_list_is_the_harness_hook():
    n, p = 1000, 6
    ops = RV.int8_hook_ops(n, p)
    kinds = [k for k, _ in ops]
    assert kinds.count("ag_s") == 1 and kinds.count("ag_d") == 1 and kinds.count("cp") == 2 * p
    assert kinds.count("t") == RV.STATS_OPS - 2 + p and kinds.count("v") == 2 * p
    assert sum(b for k, b in ops if k == "k") == (8 + 37 + p + 4 + 33 * p + 8) * n
    assert [b for k, b in ops if k in ("ag_s", "ag_d")] == [8, n]
    # the scale gather comes before the quantization, the data gather before the dequantization
    assert kinds.index("ag_s") < kinds.index("ag_d")


def test_fp16_buckets_found():
    bk = RV.fp16_buckets(fp16_trace())
    assert [x["n"] for x in bk] == [100000] * 3 and all(x["world"] == 4 for x in bk)
    assert len(bk[0]["removed"]) == 4


def test_bf16_hook_buckets_are_found_too():
    tr = fp16_trace(nccl_name="ncclDevKernel_AllReduce_Sum_bf16_RING_LL(x)")
    assert len(RV.fp16_buckets(tr)) == 3


def test_fp16_to_int8_hook_by_hand():
    tr = fp16_trace()
    n, p, bw = 100000, 4, 1000.0
    cand = R.replay(RV.substitute_fp16_int8(tr, CAL)).busy_ms
    # the synchronous hook serializes on the compute stream: 201 n bytes of
    # elementwise work and copies, 17 scalar ops, 4 copies of the scales at
    # the scalar floor, and the two all-gathers; around it mm 1000, the
    # finalize copy 10 and the optimizer 10
    chain = (201 * n / bw + (RV.STATS_OPS - 2 + p) * RV.TINY_US + p * RV.TINY_US
             + (30 + 0.04 * 0.75 * 8) + (30 + 0.04 * 0.75 * n))
    assert cand == pytest.approx((1000 + chain + 20) / 1e3, abs=0.003)


def test_h_rev_gain_falls_as_the_gather_slows():
    tr = fp16_trace()
    base = R.replay(tr).step_ms
    fast = R.replay(RV.substitute_fp16_int8(tr, CAL)).step_ms
    slow = R.replay(RV.substitute_fp16_int8(tr, dict(CAL, beta_allgather_s_per_byte=80e-9))).step_ms
    assert R.gain(base, slow) < R.gain(base, fast)


def test_h_rev_needs_the_fp16_hook_pattern():
    b = TraceBuilder()
    b.launch(0, COMPUTE, 6.0, op="aten::to")
    _collective(b, 20, "allreduce", 1000, 50.0, 4, "ncclDevKernel_AllReduce_Sum_f16_RING_LL(x)")
    with pytest.raises(ValueError):
        RV.substitute_fp16_int8(b.trace(), CAL)


def test_predict_rejects_a_forward_family():
    with pytest.raises(ValueError):
        RV.predict([], "G", CAL)
    assert RV.FAMILIES == ("G-rev", "H-rev")
