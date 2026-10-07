# provenance/test_rev4_frozen.py
# -*- coding: utf-8 -*-
"""Every file listed in rev4_baseline.sha256 still has the hash it had when
Revision 4 began (2026-10-07)."""
import hashlib
import io
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_frozen_files_unchanged():
    lines = io.open(os.path.join(ROOT, "provenance", "rev4_baseline.sha256"),
                    encoding="utf-8").read().split("\n")
    entries = [l.split("  ", 1) for l in lines if l.strip()]
    assert len(entries) >= 40
    changed = []
    for digest, rel in entries:
        with open(os.path.join(ROOT, rel), "rb") as fh:
            if hashlib.sha256(fh.read()).hexdigest() != digest:
                changed.append(rel)
    assert changed == []
