# -*- coding: utf-8 -*-
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import replicate_j3 as J  # noqa: E402


def r(mode, timing, alloc, step, exch=None, codec=0.0, kind="gather", vol=3.0, tag="H"):
    return {"tag": tag, "model": "resnet50", "mode": mode, "p": 24, "timing": timing, "tau": 1,
            "barrier": "every_iteration", "alloc": alloc, "step_ms": step, "exchange_share": exch,
            "codec_share": codec, "barrier_share": 0.0, "shape_ratio": 20.0, "volume_ratio": vol,
            "kind": kind}


def test_delta_and_outcome_per_allocation():
    rows = [r("gpu-quant-gathersum", "wall", "rep1", 100.0),
            r("gpu-quant-reducible", "wall", "rep1", 45.0, kind="reduce", vol=1.0)]
    line = J.pair_delta(rows, "resnet50", "gpu-quant-gathersum", "gpu-quant-reducible", "rep1")
    assert abs(line - 0.55) < 1e-9


def test_missing_pair_is_none():
    assert J.pair_delta([], "resnet50", "gpu-quant", "gpu-quant-reducible", "rep1") is None
