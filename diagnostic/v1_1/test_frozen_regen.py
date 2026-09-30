# -*- coding: utf-8 -*-
"""The frozen E6 pipeline must regenerate its committed outputs exactly."""
import hashlib
import io
import json
import os
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
DIAG = os.path.dirname(HERE)
WORK = os.environ.get("R4W", "C:/r4w")
RULES_SHA = "71329c336cc3e9646dbb6e6dd4ce20e12305d666c663f2a3e4b35195b535a822"
E6_EXTRACTOR_PREFIX = "95e240d86e79301e"


def sha(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def test_rules_unchanged():
    assert sha(os.path.join(DIAG, "rules.py")) == RULES_SHA


def test_extractor_is_the_version_that_scored_e6():
    assert sha(os.path.join(DIAG, "features.py")).startswith(E6_EXTRACTOR_PREFIX)


def key(r):
    return (r["model"], r["mode"], r["p"], r["timing"], r["tau"], r["barrier"], r["alloc"])


@pytest.mark.skipif(not os.path.isdir(WORK), reason="round-4 records not extracted")
def test_regeneration_is_exact(tmp_path):
    out = str(tmp_path / "features_regen.json")
    subprocess.run([sys.executable, os.path.join(DIAG, "features.py"), WORK, "", out],
                   check=True, capture_output=True)
    regen = {key(r): r for r in json.load(io.open(out, encoding="utf-8")) if r["tag"] == "R"}
    committed = [r for r in json.load(io.open(os.path.join(DIAG, "features_r4.json"),
                                              encoding="utf-8")) if r["tag"] == "R"]
    assert len(committed) == 92 and len(regen) == 92
    for c in committed:
        r = regen[key(c)]
        for k, v in c.items():
            if isinstance(v, float):
                assert r[k] == pytest.approx(v, rel=1e-12, abs=1e-15), (key(c), k)
            else:
                assert r[k] == v, (key(c), k)
    scored = subprocess.run([sys.executable, os.path.join(DIAG, "score_e6.py"), out],
                            check=True, capture_output=True, text=True).stdout
    final = io.open(os.path.join(DIAG, "E6_FINAL.txt"), encoding="utf-8").read()
    norm = lambda s: "\n".join(line.rstrip() for line in s.replace("\r\n", "\n").strip().splitlines())
    assert norm(scored) == norm(final)
