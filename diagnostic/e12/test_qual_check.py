# -*- coding: utf-8 -*-
"""Tests for the qualification go/no-go rule (E12_PROTOCOL.md section 8)."""
import io
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import qual_check as Q  # noqa: E402
import score_e12 as S  # noqa: E402
import test_score_e12 as T  # noqa: E402

PARSE_TRACES = ("ddp-int8-hook-bf16", "ddp-fp16-hook-bf16")


def ok(_):
    return 16


def qual(tmp_path, span_s=522, cpu_block_s=60, fail=None, drop_fid=None, traces=PARSE_TRACES):
    q = str(tmp_path / "e12-q-2n-5")
    for model in S.MODELS:
        for tag, e in T.ERR.items():
            if (model, tag) != drop_fid:
                T.write_fid(q, model, tag, [e] * S.FIDELITY_ITERS)
        for tag in traces:
            T.write_trace(q, model, tag)
    T.write_block(q, "resnet50", "cpu-quant", 1, 0.38)
    T.write_block(q, "transformer", "ddp-overlap-bf16", 1, 0.014)
    prov = os.path.join(q, "provenance")
    os.makedirs(prov, exist_ok=True)
    names = ["c%02d" % i for i in range(23)] + ["block-resnet50-cpu-quant-p1"]
    with io.open(os.path.join(prov, "conditions.txt"), "w", encoding="utf-8") as fh:
        for n in names:
            fh.write("%s rc=%d\n" % (n, 1 if n == fail else 0))
    with io.open(os.path.join(prov, "timeline.txt"), "w", encoding="utf-8") as fh:
        fh.write("2026-10-20T10:00:00Z start c00\n2026-10-20T10:00:10Z done c00 rc=0\n")
        end = span_s
        start_cpu = end - cpu_block_s
        fh.write("2026-10-20T10:%02d:%02dZ start block-resnet50-cpu-quant-p1\n" % divmod(start_cpu, 60))
        fh.write("2026-10-20T10:%02d:%02dZ done block-resnet50-cpu-quant-p1 rc=0\n" % divmod(end, 60))
    return q


def test_go_40_on_plan(tmp_path):
    out = Q.check(qual(tmp_path), parse=ok, rev_parse=ok)
    assert out["decision"] == "go-40" and out["problems"] == []
    assert out["projection_min"] == pytest.approx((90 + 2002) / 60)


def test_slow_qualification_moves_to_45_then_stops(tmp_path):
    assert Q.check(qual(tmp_path, span_s=560), parse=ok, rev_parse=ok)["decision"] == "go-45"   # 37.3 min
    assert Q.check(qual(tmp_path, cpu_block_s=90), parse=ok, rev_parse=ok)["decision"] == "stop"  # 48.4


def test_boundaries():
    assert Q.GO_40_MAX_MIN == 36.0 and Q.GO_45_MAX_MIN == 43.0
    # the projection at the plan is the plan: 90 s + 2002 s = 34.9 minutes
    assert (Q.PLAN_OVERHEAD_S + Q.PLAN_MAIN_CONDITIONS_S) / 60 == pytest.approx(34.87, abs=0.01)
    assert Q.PLAN_QUAL_CONDITIONS_S == 522.0 and Q.QUAL_CONDITIONS == 24


@pytest.mark.parametrize("kw,needle", [({"fail": "c03"}, "c03 returned 1"),
                                       ({"drop_fid": ("resnet50", "ddp-int8-hook-bf16")}, "no complete fidelity"),
                                       ({"traces": ()}, "0 trace files"),
                                       ({"traces": ("ddp-int8-hook-bf16",)}, "ddp-fp16-hook-bf16: 0 trace files")])
def test_a_failed_check_stops(tmp_path, kw, needle):
    out = Q.check(qual(tmp_path, **kw), parse=ok, rev_parse=ok)
    assert out["decision"] == "stop" and any(needle in p for p in out["problems"])


def test_an_unparsable_trace_stops(tmp_path):
    def bad(_):
        raise ValueError("hook start (aten::min) not found")
    out = Q.check(qual(tmp_path), parse=bad, rev_parse=ok)
    assert out["decision"] == "stop" and any("cannot parse" in p for p in out["problems"])
    assert Q.check(qual(tmp_path), parse=lambda p: 0, rev_parse=ok)["decision"] == "stop"


def test_an_unparsable_mirror_trace_stops(tmp_path):
    def bad(_):
        raise ValueError("fp16 hook pattern (cast, div_, all-reduce) not found")
    out = Q.check(qual(tmp_path), parse=ok, rev_parse=bad)
    assert out["decision"] == "stop" and any("reverse substitution cannot parse" in p for p in out["problems"])
    assert Q.check(qual(tmp_path), parse=ok, rev_parse=lambda p: 0)["decision"] == "stop"
