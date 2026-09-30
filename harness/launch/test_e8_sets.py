# -*- coding: utf-8 -*-
"""E8 condition sets (diagnostic/E8_PROTOCOL.md section 2): 26 runs per
allocation, the methods' inputs before any outcome, pass 2 reversed."""
import os
import subprocess

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
BASH = os.environ.get("GIT_BASH", "C:/PROGRA~1/Git/bin/bash.exe")
SETS = os.path.join(HERE, "e8_sets.sh").replace("\\", "/")
STUBS = ("calib(){ echo calib; }\npreflight(){ echo \"preflight $1\"; }\n"
         "e8_phase(){ echo \"phase $1 $2 b$E8_BATCH\"; }\n"
         "e8_trace(){ echo \"trace $1 $2 b$E8_BATCH\"; }\n"
         "e8_block(){ echo \"block $1 $2 p$3 b$E8_BATCH\"; }\n")
MODELS = ("transformer", "resnet50")
ORIGINALS = ("gpu-quant-gathersum", "ddp-int8-hook")
MODES = ("gpu-quant-gathersum", "gpu-quant-reducible", "ddp-int8-hook", "ddp-fp16-hook")


def expand(red_set):
    r = subprocess.run([BASH, "-c", STUBS + "RED_SET=%s\nsource \"%s\"\n" % (red_set, SETS)],
                       capture_output=True, text=True)
    return r.returncode, r.stdout.strip().splitlines()


@pytest.mark.parametrize("red_set,batch", [("e8s", 32), ("e8l", 128)])
def test_e8_set(red_set, batch):
    rc, lines = expand(red_set)
    assert rc == 0 and len(lines) == 26
    assert lines[:2] == ["preflight transformer", "calib"]
    b = "b%d" % batch
    assert lines[2:6] == ["phase %s %s %s" % (m, o, b) for m in MODELS for o in ORIGINALS]
    assert lines[6:10] == ["trace %s %s %s" % (m, o, b) for m in MODELS for o in ORIGINALS]
    pass1 = ["block %s %s p1 %s" % (m, mode, b) for m in MODELS for mode in MODES]
    pass2 = ["block %s %s p2 %s" % (m, mode, b) for m in reversed(MODELS) for mode in reversed(MODES)]
    assert lines[10:] == pass1 + pass2


def test_unknown_set_fails():
    rc, _ = expand("e7a")
    assert rc != 0
