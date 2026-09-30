# -*- coding: utf-8 -*-
import os
import subprocess

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
BASH = os.environ.get("GIT_BASH", "C:/PROGRA~1/Git/bin/bash.exe")
SETS = os.path.join(HERE, "rev_sets.sh").replace("\\", "/")
STUBS = ("calib(){ echo calib; }\npreflight(){ echo \"preflight $1\"; }\n"
         "paper(){ echo \"paper $1 $2\"; }\nwall(){ echo \"wall $1 $2\"; }\n"
         "ext_phase(){ echo \"ext_phase $1 $2\"; }\next_wall(){ echo \"ext_wall $1 $2\"; }\n"
         "trace_run(){ echo \"trace $1 $2\"; }\n")


def expand(red_set):
    r = subprocess.run([BASH, "-c", STUBS + "RED_SET=%s\nsource \"%s\"\n" % (red_set, SETS)],
                       capture_output=True, text=True)
    return r.returncode, r.stdout.strip().splitlines()


def test_e7a():
    rc, lines = expand("e7a")
    assert rc == 0 and lines[:2] == ["preflight transformer", "calib"] and len(lines) == 36
    assert [l for l in lines if l.startswith("trace")] == [
        "trace %s %s" % (m, mode) for m in ("transformer", "resnet50")
        for mode in ("gpu-quant-gathersum", "gpu-quant-reducible", "ddp-int8-hook", "ddp-fp16-hook")]
    assert lines.index("trace transformer gpu-quant-gathersum") > max(
        i for i, l in enumerate(lines) if l.startswith("ext_"))


def test_e7b_has_no_traces():
    rc, lines = expand("e7b")
    assert rc == 0 and len(lines) == 28 and not any(l.startswith("trace") for l in lines)


def test_every_e7_pair_is_measured():
    lines = set(expand("e7b")[1])
    for model in ("transformer", "resnet50"):
        for orig, intv in (("gpu-quant-gathersum", "gpu-quant-reducible"),
                           ("gpu-quant", "gpu-quant-reducible"),
                           ("gpu-topk", "gpu-quant-reducible"),
                           ("cpu-quant", "gpu-quant"), ("ddp-int8-hook", "ddp-fp16-hook")):
            assert {"ext_phase %s %s" % (model, orig), "ext_wall %s %s" % (model, orig),
                    "ext_wall %s %s" % (model, intv)} <= lines
        assert "ext_phase %s fp16-native" % model in lines


@pytest.mark.parametrize("red_set,count", [("j3r", 20), ("j3rfull", 26)])
def test_j3r_sets(red_set, count):
    rc, lines = expand(red_set)
    assert rc == 0 and len(lines) == count
    needed = {"wall transformer gpu-quant", "wall transformer gpu-quant-reducible",
              "wall transformer ddp-int8-hook", "wall transformer ddp-fp16-hook",
              "wall transformer ddp-overlap", "wall resnet50 gpu-quant",
              "wall resnet50 gpu-quant-reducible", "wall resnet50 gpu-quant-gathersum",
              "wall resnet50 gpu-topk", "wall resnet50 ddp-overlap",
              "paper transformer gpu-quant", "paper transformer ddp-int8-hook",
              "paper transformer fp16-native", "paper resnet50 gpu-quant",
              "paper resnet50 gpu-quant-gathersum", "paper resnet50 gpu-topk"}
    assert needed <= set(lines)


def test_unknown_set_fails():
    rc, _ = expand("bogus")
    assert rc != 0
