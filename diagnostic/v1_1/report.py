# -*- coding: utf-8 -*-
"""Writes E6_CORRECTED.txt: every rule set under every reading, counts first,
then the per-case lines. RETROSPECTIVE (2026-09-29). Usage: python report.py"""
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rescore as R  # noqa: E402

CONFIGS = [("frozen", False, False), ("described", True, True),
           ("nominal_frozen_hooks", False, False), ("nominal", True, True),
           ("as_frozen_extractor", False, False)]


def build():
    rows = R.load_rows()
    out = ["E6 held-out evaluation, corrected readings (RETROSPECTIVE, 2026-09-29).",
           "Frozen rules and extractor unchanged; see CORRECTIONS.md and EXTRACTOR_FREEZE.md.",
           ""]
    for reading, e_same, c5 in CONFIGS:
        out.append("== reading %s (e_same=%s, c5=%s) ==" % (reading, e_same, c5))
        for name in ("full", "A1", "A2", "A3"):
            counts, lines = R.rescore(rows, R.METHODS[name], reading=reading,
                                      e_same=e_same, c5=c5)
            out.append("%-4s cleared 5%%: %g | %s" % (name, R.cleared(counts), R.fmt(counts)))
        out.append("")
    out.append("== per-case lines, reading described (e_same, c5) ==")
    for name in ("full", "A1", "A2", "A3"):
        for l in R.rescore(rows, R.METHODS[name], reading="described", e_same=True, c5=True)[1]:
            d = "   -  " if l["delta"] is None else "%+6.2f" % l["delta"]
            out.append("%-4s case %2d alloc=%-8s %-13s e=%5.3f delta=%s %s" % (
                name, l["case"], l["alloc"], l["verdict"], l["e"], d, l["outcome"]))
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    io.open(os.path.join(HERE, "E6_CORRECTED.txt"), "w", encoding="utf-8",
            newline="\n").write(build())
    print("wrote E6_CORRECTED.txt")
