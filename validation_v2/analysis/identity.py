# -*- coding: utf-8 -*-
"""g_E = 1 - (rho_1 / rho_0) (1 - g_B): phase- or wall-timed gain from the
block-timed gain and the perturbation ratios (rho = T_conv / T_block) of the
original (0) and the replacement (1). RETROSPECTIVE (2026-09-29).
Usage: python identity.py <v2_expA.csv> <out.csv>"""
import csv
import sys

MARGIN = 0.05
PAIRS = {"g_matched": "rho_gpu-quant-gathersum", "h_topk": "rho_gpu-topk"}
REPLACEMENT = "rho_gpu-quant-reducible"


def table(path):
    rows = list(csv.DictReader(open(path, newline="")))
    block = {(r["job"], r["model"]): r for r in rows if r["convention"] == "B"}
    out = []
    for r in rows:
        if r["convention"] not in ("E", "S"):
            continue
        b = block[(r["job"], r["model"])]
        for gain, rho0_key in PAIRS.items():
            ratio = float(r[REPLACEMENT]) / float(r[rho0_key])
            g_b, g_obs = float(b[gain]), float(r[gain])
            direction = "equal" if abs(g_obs - g_b) < 1e-9 else \
                ("overstates" if g_obs > g_b else "understates")
            out.append({"job": r["job"], "model": r["model"], "gain": gain,
                        "convention": r["convention"], "g_B": g_b, "g_obs": g_obs,
                        "g_pred": 1.0 - ratio * (1.0 - g_b), "ratio": ratio,
                        "direction": direction,
                        "crosses_margin": (g_obs >= MARGIN) != (g_b >= MARGIN)})
    return out


if __name__ == "__main__":
    rows = table(sys.argv[1])
    with open(sys.argv[2], "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("wrote %d rows" % len(rows))
