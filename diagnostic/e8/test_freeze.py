# -*- coding: utf-8 -*-
"""Every file listed in E8_FREEZE.md still has its frozen hash."""
import hashlib
import io
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.dirname(os.path.dirname(HERE))


def test_frozen_hashes():
    text = io.open(os.path.join(HERE, "E8_FREEZE.md"), encoding="utf-8").read()
    pairs = re.findall(r"sha256 ([0-9a-f]{64}) `([^`]+)`", text)
    assert len(pairs) == 7
    for digest, rel in pairs:
        with open(os.path.join(ART, rel), "rb") as fh:
            assert hashlib.sha256(fh.read()).hexdigest() == digest, rel
