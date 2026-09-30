# -*- coding: utf-8 -*-
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import score_e7 as E  # noqa: E402


def row(model, mode, timing, alloc, step, exch=None, codec=0.0, shape=None, kind=None, vol=None):
    k = kind or ("gather" if mode in ("gpu-quant", "cpu-quant", "gpu-topk",
                                      "ddp-int8-hook", "gpu-quant-gathersum") else "reduce")
    return {"tag": "H", "model": model, "mode": mode, "p": 6, "timing": timing,
            "tau": 1, "barrier": "every_iteration", "alloc": alloc, "step_ms": step,
            "exchange_share": exch, "codec_share": codec, "barrier_share": 0.0,
            "shape_ratio": shape, "volume_ratio": vol if vol is not None else (0.75 if k == "gather" else 1.0),
            "kind": k}


def by(results, case, rule_set):
    return [r for r in results if r[0] == case and r[2] == rule_set]


def test_ratio_rule_false_positive_is_null():
    rows = [row("transformer", "gpu-quant-gathersum", "phase", "a", 100, exch=0.1, shape=3.0),
            row("transformer", "gpu-quant-gathersum", "wall", "a", 100),
            row("transformer", "gpu-quant-reducible", "wall", "a", 98)]
    r = by(E.score(rows, ["a"]), "E7-T1", "A2")[0]
    assert (r[3], r[6]) == ("shape", "null")


def test_shape_on_dense_reduce_is_inapplicable():
    rows = [row("transformer", "fp16-native", "phase", "a", 100, exch=0.3, shape=3.2)]
    r = by(E.score(rows, ["a"]), "E7-T6", "A2")[0]
    assert r[6] == "inapplicable"


def test_placement_positive_control_correct():
    rows = [row("resnet50", "cpu-quant", "phase", "a", 900, exch=0.05, codec=0.7),
            row("resnet50", "cpu-quant", "wall", "a", 900),
            row("resnet50", "gpu-quant", "wall", "a", 300)]
    r = by(E.score(rows, ["a"]), "E7-R4", "full")[0]
    assert (r[3], r[6]) == ("placement", "correct")


def test_failed_intervention_run_is_missing_not_abstention():
    rows = [row("transformer", "ddp-int8-hook", "phase", "a", 100),
            row("transformer", "ddp-int8-hook", "wall", "a", 100)]
    r = by(E.score(rows, ["a"]), "E7-T5", "full")[0]
    assert r[6] == "missing (run failed)"


def test_rows_from_other_allocations_are_ignored():
    rows = [row("transformer", "gpu-quant-gathersum", "phase", "z", 100, exch=0.1, shape=3.0)]
    r = by(E.score(rows, ["a"]), "E7-T1", "A2")[0]
    assert r[6] == "missing (M1 failed)"


def test_two_allocations_weigh_half_each():
    rows = []
    for al, int_step in (("a", 98), ("b", 60)):
        rows += [row("transformer", "gpu-quant-gathersum", "phase", al, 100, exch=0.1, shape=3.0),
                 row("transformer", "gpu-quant-gathersum", "wall", al, 100),
                 row("transformer", "gpu-quant-reducible", "wall", al, int_step)]
    s = E.summary([r for r in E.score(rows, ["a", "b"]) if r[0] == "E7-T1"])
    assert s["A2"]["cleared"] == 0.5 and s["A2"]["false_positive"] == 0.5
