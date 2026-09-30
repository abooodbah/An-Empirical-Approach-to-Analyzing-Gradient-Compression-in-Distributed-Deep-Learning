# -*- coding: utf-8 -*-
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import trace_overlap as T  # noqa: E402


def test_overlap_on_synthetic_trace(tmp_path):
    ev = [{"ph": "X", "cat": "kernel", "name": "ncclDevKernel_AllReduce_Sum_f16_RING_LL", "ts": 0, "dur": 10},
          {"ph": "X", "cat": "kernel", "name": "volta_sgemm", "ts": 5, "dur": 15},
          {"ph": "X", "cat": "cpu_op", "name": "nccl:all_reduce", "ts": 0, "dur": 30}]
    p = tmp_path / "t.json"
    p.write_text(json.dumps({"traceEvents": ev}))
    o = T.overlap(str(p))
    assert (o["nccl_ms"], o["overlapped_ms"], o["exposed_ms"]) == (0.01, 0.005, 0.005)
    assert o["overlap_frac"] == 0.5
