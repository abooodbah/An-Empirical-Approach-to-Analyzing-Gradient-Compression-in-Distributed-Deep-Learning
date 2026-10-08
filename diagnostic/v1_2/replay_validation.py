# diagnostic/v1_2/replay_validation.py
# -*- coding: utf-8 -*-
"""The overlap-aware replay's predictions beside the measurements, case by
case, from the frozen E8 result. RETROSPECTIVE (2026-10-07): a table of what
E8_RESULT.json already holds."""
import csv
import io
import json
import os
import statistics as st
import sys

FIELDS = ("case", "alloc", "model", "family", "regime", "gain_predicted", "gain_measured",
          "error_points", "orig_predicted_ms", "orig_measured_ms", "cand_predicted_ms",
          "cand_measured_ms")


def rows(path):
    out = []
    for r in json.load(io.open(path, encoding="utf-8"))["rows"]:
        if r["method"] != "reference" or r["label"].startswith("missing"):
            continue
        detail = r["detail"]
        out.append({"case": r["case"], "alloc": r["alloc"], "model": r["model"],
                    "family": r["family"], "regime": r["regime"],
                    "gain_predicted": r["e"], "gain_measured": r["g"],
                    "error_points": 100.0 * abs(r["e"] - r["g"]),
                    "orig_predicted_ms": detail["orig_ms"], "orig_measured_ms": r["T_orig"],
                    "cand_predicted_ms": detail["cand_ms"], "cand_measured_ms": r["T_cand"]})
    return out


def summary(table):
    errors = [r["error_points"] for r in table]
    signed = [100.0 * (r["gain_predicted"] - r["gain_measured"]) for r in table]
    return {"n": len(table), "median_error_points": st.median(errors),
            "max_error_points": max(errors),
            "within_5_points": sum(1 for e in errors if e <= 5.0),
            "mean_signed_points": st.mean(signed)}


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    table = rows(os.path.join(os.path.dirname(here), "E8_RESULT.json"))
    with io.open(os.path.join(here, "REPLAY_VALIDATION.csv"), "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, FIELDS)
        writer.writeheader()
        writer.writerows(table)
    print(json.dumps(summary(table), indent=1))
    sys.exit(0)
