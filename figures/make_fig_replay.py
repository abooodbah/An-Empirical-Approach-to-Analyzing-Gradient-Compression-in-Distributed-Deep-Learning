# -*- coding: utf-8 -*-
"""Predicted against measured speedup of the overlap-aware replay (v12,
2026-10-19). One point per case and allocation: the third held-out set
(E8, 15 predictions; one trace failed) and the fourth (E12, 24
predictions, including the six reverse change types on both workloads).

Speedup = T(original) / T(change) = 1 / (1 - gain). The predicted speedup
uses the replay's predicted gain e; the measured one the block-timed gain g.
Data come from the frozen scorers' result files in the artifact repo.

usage: python make_fig_replay.py <artifact repo> <output folder>
"""
import json
import os
import statistics as st
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
plt.rcParams["mathtext.fontset"] = "stix"

FS = 10.0                       # the figure prints 1:1 at column width
BLUE, ORANGE, GRAY = "#0072B2", "#E69F00", "#555555"
STYLE = {"G": (BLUE, "o"), "H": (ORANGE, "X")}   # as Figs. 3 to 5


def points(repo):
    d8 = json.load(open(os.path.join(repo, "diagnostic", "E8_RESULT.json")))
    p8 = [(x["family"], x["e"], x["g"]) for x in d8["rows"]
          if x["method"] == "reference" and x["verdict"] != "missing"]
    d12 = json.load(open(os.path.join(repo, "diagnostic", "E12_RESULT.json")))
    p12 = [(x["family"][0], x["e"], x["g"]) for x in d12["rows"]
           if x["method"] == "replay" and x["e"] is not None]
    return p8, p12


def speed(gain):
    return 1.0 / (1.0 - gain)


def main(repo, out):
    p8, p12 = points(repo)
    assert len(p8) == 15 and len(p12) == 24, (len(p8), len(p12))
    for name, p in (("third", p8), ("fourth", p12)):
        err = [abs(e - g) for _, e, g in p]
        print("%s set: %d predictions, median |e-g| %.3f" % (name, len(p), st.median(err)))

    fig, ax = plt.subplots(figsize=(3.5, 3.45))
    lo, hi = 0.6, 2.5
    ax.plot([lo, hi], [lo, hi], color=GRAY, lw=0.8, ls="--", zorder=1)
    thr = speed(0.05)
    ax.axvline(thr, color="#bbbbbb", lw=0.6, zorder=0)
    ax.axhline(thr, color="#bbbbbb", lw=0.6, zorder=0)
    for p, filled in ((p8, False), (p12, True)):
        for fam, e, g in p:
            col, mk = STYLE[fam]
            ax.plot(speed(g), speed(e), marker=mk, ms=5.5, ls="none", color=col,
                    mfc=col if filled else "white", mew=1.0, zorder=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ticks = [0.6, 0.8, 1.0, 1.25, 1.6, 2.0, 2.5]
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(FixedLocator(ticks))
        axis.set_major_formatter(FuncFormatter(lambda v, _: "%g" % v))
        axis.set_minor_formatter(NullFormatter())
        axis.set_minor_locator(FixedLocator([]))
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("measured speedup", fontsize=FS)
    ax.set_ylabel("speedup predicted by the replay", fontsize=FS)
    ax.tick_params(labelsize=FS)
    # quadrant labels sit where the quadrant has no points
    ax.text(0.62, 1.02, "slowed and\ndeclined", fontsize=FS, color=GRAY, ha="left", va="top")
    ax.text(2.45, 1.08, "paid and\nrecommended", fontsize=FS, color=GRAY, ha="right", va="bottom")
    ax.text(2.45, 0.62, "dashed: predicted\n= measured", fontsize=FS, color=GRAY, ha="right",
            va="bottom")
    handles = [
        Line2D([], [], marker="o", ls="none", color=BLUE, mfc=BLUE, ms=5.5, label="gather and reduce arms"),
        Line2D([], [], marker="X", ls="none", color=ORANGE, mfc=ORANGE, ms=5.5, label="int8 and 16-bit hooks"),
        Line2D([], [], marker="o", ls="none", color=GRAY, mfc="white", ms=5.5, label="third set"),
        Line2D([], [], marker="o", ls="none", color=GRAY, mfc=GRAY, ms=5.5, label="fourth set"),
    ]
    fig.legend(handles=handles, fontsize=FS, frameon=False, loc="lower center", ncol=2,
               columnspacing=0.8, handletextpad=0.3, borderaxespad=0.1)
    fig.subplots_adjust(left=0.17, right=0.97, top=0.98, bottom=0.30)
    os.makedirs(out, exist_ok=True)
    fig.savefig(os.path.join(out, "fig_replay.pdf"))
    print("wrote", os.path.join(out, "fig_replay.pdf"))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
