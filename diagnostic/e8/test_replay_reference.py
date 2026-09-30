# -*- coding: utf-8 -*-
"""Tests for the E8 reference (diagnostic/E8_PROTOCOL.md section 7), on
synthetic Kineto-shaped traces whose replay makespans are known by hand."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import replay_reference as R  # noqa: E402

COMPUTE, NCCL = 7, 17


class TraceBuilder:
    """Builds a minimal Kineto trace: CPU runtime calls on one thread, the GPU
    ops they launch, cpu_op wrappers, and iteration annotations."""

    def __init__(self):
        self.ev = []
        self.corr = 0
        self.ext = 0

    def _cpu_op(self, name, t, d, tid=1):
        self.ext += 1
        self.ev.append({"ph": "X", "cat": "cpu_op", "name": name, "ts": t, "dur": d, "tid": tid,
                        "args": {"External id": self.ext}})
        return self.ext

    def launch(self, t, stream, dur, name="kernel_elementwise", start=None, cpu_d=5.0, op="aten::add",
               cat="kernel", args=None, tid=1, rt="cudaLaunchKernel"):
        ext = self._cpu_op(op, t, cpu_d + 2.0, tid)
        self.corr += 1
        self.ev.append({"ph": "X", "cat": "cuda_runtime", "name": rt, "ts": t + 1.0, "dur": cpu_d,
                        "tid": tid, "args": {"correlation": self.corr, "External id": ext}})
        g = {"ph": "X", "cat": cat, "name": name, "ts": start if start is not None else t + 1.0 + 5.0,
             "dur": dur, "tid": stream, "args": {"stream": stream, "correlation": self.corr}}
        if args:
            g["args"].update(args)
        self.ev.append(g)
        return self.corr

    def runtime(self, t, name, d=1.0, tid=1, op=None):
        ext = self._cpu_op(op, t, d + 1.0, tid) if op else 0
        self.corr += 1
        self.ev.append({"ph": "X", "cat": "cuda_runtime", "name": name, "ts": t, "dur": d, "tid": tid,
                        "args": {"correlation": self.corr, "External id": ext}})

    def annotate(self, name, t, d, tid=1):
        self.ev.append({"ph": "X", "cat": "user_annotation", "name": name, "ts": t, "dur": d, "tid": tid,
                        "args": {}})

    def trace(self):
        return R.Trace({"traceEvents": self.ev})


def two_stream_iteration(b, t0, comm_dur, overlap):
    """One iteration: compute A (100us), collective (comm_dur) waiting on A,
    compute B (100us, independent of the collective if overlap, else after a
    wait on it), optimizer (10us) after a wait on the collective, then a
    device sync. GPU times are what an ideal machine would produce."""
    b.annotate("DistributedDataParallel.forward", t0, 1.0)
    a = b.launch(t0 + 2, COMPUTE, 100.0, start=t0 + 10)
    b.runtime(t0 + 10, "cudaEventRecord")
    b.runtime(t0 + 11, "cudaStreamWaitEvent", op="record_param_comms")
    c = b.launch(t0 + 12, NCCL, comm_dur, name="ncclDevKernel_AllGather_RING_LL", start=t0 + 110,
                 op="record_param_comms", rt="cudaLaunchKernelExC")
    b.runtime(t0 + 20, "cudaEventRecord")
    if not overlap:
        b.runtime(t0 + 21, "cudaStreamWaitEvent", op="record_param_comms")
        bc = b.launch(t0 + 22, COMPUTE, 100.0, start=t0 + 110 + comm_dur)
        end_b = t0 + 110 + comm_dur + 100
    else:
        bc = b.launch(t0 + 22, COMPUTE, 100.0, start=t0 + 110)
        end_b = t0 + 210
        b.runtime(t0 + 30, "cudaEventRecord")
        b.runtime(t0 + 31, "cudaStreamWaitEvent", op="record_param_comms")
    opt_start = max(end_b, t0 + 110 + comm_dur)
    b.annotate("Optimizer.step#SGD.step", t0 + 40, 5.0)
    b.launch(t0 + 41, COMPUTE, 10.0, start=opt_start)
    b.runtime(t0 + 50, "cudaDeviceSynchronize", d=max(1.0, opt_start + 10 - (t0 + 50)))
    return c, bc


def build(comm_dur, overlap, iters=4, period=1000.0):
    b = TraceBuilder()
    colls, bs = [], []
    for i in range(iters):
        c, bc = two_stream_iteration(b, i * period, comm_dur, overlap)
        colls.append(c)
        bs.append(bc)
    b.bs = bs
    return b, colls


def test_known_makespan_serial():
    b, _ = build(300.0, overlap=False)
    tr = b.trace()
    it = R.replay(tr)
    # A 100 + collective 300 + B 100 + optimizer 10 = 510us of GPU after the first launch
    assert it.busy_ms == pytest.approx(0.510, abs=0.02)


def test_known_makespan_overlapped():
    b, _ = build(300.0, overlap=True)
    it = R.replay(b.trace())
    # B runs beside the collective: A 100 + max(collective 300, B 100) + optimizer 10
    assert it.busy_ms == pytest.approx(0.410, abs=0.02)


def test_shorter_serial_collective_shortens_by_the_same_amount():
    b, colls = build(300.0, overlap=False)
    tr = b.trace()
    base = R.replay(tr).busy_ms
    faster = R.replay(tr, durations={c: 100.0 for c in colls}).busy_ms
    assert base - faster == pytest.approx(0.200, abs=0.01)


def test_shorter_hidden_collective_changes_nothing():
    b, colls = build(80.0, overlap=True)   # 80us collective fully hidden behind B (100us)
    tr = b.trace()
    base = R.replay(tr).busy_ms
    faster = R.replay(tr, durations={c: 20.0 for c in colls}).busy_ms
    assert faster == pytest.approx(base, abs=0.005)


def test_replay_of_the_original_matches_the_trace():
    b, _ = build(300.0, overlap=False)
    tr = b.trace()
    it = R.replay(tr)
    assert it.intervals_ms and all(v == pytest.approx(1.0, abs=0.01) for v in it.intervals_ms)
    assert it.fidelity == pytest.approx(1.0, abs=0.02)


def test_cross_stream_dependency_inferred():
    b, colls = build(300.0, overlap=False)
    tr = b.trace()
    deps = R.infer_dependencies(tr)
    # the collective waits on compute A; compute B waits on the collective
    assert 1 in deps.get(colls[0], ())            # compute A has correlation 1
    assert colls[0] in deps.get(b.bs[0], ())


def test_prediction_is_one_minus_ratio():
    assert R.gain(10.0, 7.5) == pytest.approx(0.25)
    assert R.recommends(0.05) and not R.recommends(0.0499)
