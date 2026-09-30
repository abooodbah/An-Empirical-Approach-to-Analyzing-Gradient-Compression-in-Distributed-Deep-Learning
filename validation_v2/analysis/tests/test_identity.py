# -*- coding: utf-8 -*-
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import identity as I  # noqa: E402

CSV = os.path.join(os.path.dirname(os.path.dirname(HERE)), "results",
                   "analysis_2026-09-29", "v2_expA.csv")


def test_identity_reproduces_every_cell():
    rows = I.table(CSV)
    assert rows and max(abs(r["g_pred"] - r["g_obs"]) for r in rows) < 1e-5


def test_directions_match_the_audit():
    d = {(r["job"], r["model"], r["gain"]): r["direction"]
         for r in I.table(CSV) if r["convention"] == "E"}
    assert d[("V1", "transformer", "h_topk")] == "overstates"
    assert d[("V2", "transformer", "h_topk")] == "overstates"
    assert d[("V3", "transformer", "h_topk")] == "understates"
    assert all(d[(j, "transformer", "g_matched")] == "understates" for j in ("V1", "V2", "V3"))
    assert d[("R1", "resnet50", "g_matched")] == "understates"
    assert d[("R1", "resnet50", "h_topk")] == "understates"
