# -*- coding: utf-8 -*-
"""Build the four TPDS-paper figures from the July campaign data.

Inputs: the aggregated figure CSVs in processed/ and the raw per-rank
CSVs of the validated ndrhost 2x4 seeds (extracted under RAW_SEEDS).
Outputs: vector PDFs under figures/figures/ plus printed numbers used
in the manuscript.

Usage: TPDS_RAW_SEEDS=<folder> python figures/make_figures.py"""
import csv
import glob
import os
import statistics as st
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
# TPDS_FIG_DATA: folder of the aggregated CSVs (default: processed/).
# TPDS_RAW_SEEDS: folder holding only the three extracted H100 seed archives
# raw/july/h100-3328658, -3328659 and -3328660.tar.gz (default: work/july).
DATA = os.environ.get("TPDS_FIG_DATA") or os.path.join(ROOT, "processed")
RAW_SEEDS = os.environ.get("TPDS_RAW_SEEDS") or os.path.join(ROOT, "work", "july")
if os.name == "nt" and not RAW_SEEDS.startswith("\\\\?\\"):
    RAW_SEEDS = "\\\\?\\" + os.path.abspath(RAW_SEEDS)
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]


def rows(name):
    with open(os.path.join(DATA, name), newline="") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------- figure 1
def _sketch_collectives(ax):
    """Top panel: combining during the exchange (all-reduce) against
    collecting for local reconstruction (all-gather), four ranks each."""
    import numpy as np
    from matplotlib.patches import FancyArrowPatch, Circle
    ax.axis("off")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    # data units are anisotropic (panel is ~7.1 x 1.8 in); scale node
    # radii and the ring so circles render round
    aspect = (7.1 / 1.8) / (100 / 100)  # x-units per y-unit on screen

    def ranks(cx, cy, r):
        ang = np.deg2rad([90, 0, -90, 180])
        return [(cx + (r / aspect) * np.cos(a), cy + r * np.sin(a)) for a in ang]

    def node(x, y, label, fc):
        from matplotlib.patches import Ellipse
        ax.add_patch(Ellipse((x, y), 2 * 6.0 / aspect, 2 * 6.0, facecolor=fc,
                             edgecolor="#444444", linewidth=0.7, zorder=3))
        ax.text(x, y, label, ha="center", va="center", fontsize=6.4, zorder=4)

    def arrow(p, q, color, lw=0.9, rad=0.0, ls="-"):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=7,
                                     linewidth=lw, color=color, linestyle=ls,
                                     connectionstyle=f"arc3,rad={rad}",
                                     shrinkA=6, shrinkB=6, zorder=2))

    # left: ring all-reduce, each hop carries a partial sum
    ax.text(25, 96, "combine during the exchange (all-reduce)",
            ha="center", va="top", fontsize=7.2, fontweight="bold")
    L = ranks(25, 55, 30)
    for i, (x, y) in enumerate(L):
        node(x, y, f"$g_{i}$", "#e8eef5")
    for i in range(4):
        arrow(L[i], L[(i + 1) % 4], "#2d5f8b", rad=-0.3)
    # one label per hop, placed on the bowed arc's outer side
    for i in range(4):
        (x1, y1), (x2, y2) = L[i], L[(i + 1) % 4]
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ox, oy = mx - 25, my - 55
        nx, ny = ox / (abs(ox) + 1e-9) * 4.0 if abs(ox) > 1 else 0, oy / (abs(oy) + 1e-9) * 9 if abs(oy) > 1 else 0
        ax.text(mx + nx, my + ny, "partial sum", ha="center", va="center", fontsize=5.4,
                color="#2d5f8b",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"))
    ax.text(25, 14, r"each rank receives $\approx 2\frac{p-1}{p}G$",
            ha="center", va="center", fontsize=6.6)
    ax.text(25, 5, "result: the sum, already reduced", ha="center",
            va="center", fontsize=6.2, color="#333333")

    # right: all-gather, every payload to every rank, then a local sum
    ax.text(75, 96, "collect, then reconstruct locally (all-gather)",
            ha="center", va="top", fontsize=7.2, fontweight="bold")
    R = ranks(75, 55, 30)
    for i, (x, y) in enumerate(R):
        node(x, y, f"$q_{i}$", "#f6d7cd")
    for i in range(4):
        for j in range(4):
            if i != j:
                arrow(R[i], R[j], "#b5573f", lw=0.6, rad=0.12 if (j - i) % 4 in (1, 3) else 0.0)
    ax.text(75, 14, r"each rank receives $(p-1)\,\frac{G}{c}$",
            ha="center", va="center", fontsize=6.6)
    ax.text(75, 5, "result: $p$ payloads, summed after arrival",
            ha="center", va="center", fontsize=6.2, color="#333333")


def fig_paths():
    fig = plt.figure(figsize=(7.1, 4.3))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.35], hspace=0.08)
    top = fig.add_subplot(gs[0])
    _sketch_collectives(top)
    ax = fig.add_subplot(gs[1])
    ax.axis("off")
    ax.set_xlim(0, 100)
    ax.set_ylim(-4, 100)

    def box(x, y, w, h, title, lines, fc, fs=7.0):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                     boxstyle="round,pad=0.4,rounding_size=1.4",
                     linewidth=0.7, edgecolor="#444444", facecolor=fc))
        ax.text(x + w / 2, y + h - 8, title, ha="center", va="center",
                fontsize=fs + 0.6, fontweight="bold")
        for i, t in enumerate(lines):
            ax.text(x + w / 2, y + h - 20 - 11.5 * i, t, ha="center",
                    va="center", fontsize=fs)

    box(0.5, 46, 23.5, 50, "dense all-reduce",
        ["FP32 gradient", "ring reduce in transit",
         r"$V_{\mathrm{AR}}\approx 2\frac{p-1}{p}G$ / rank"], "#e8eef5")
    box(25.5, 46, 24.5, 50, "rank-local int8",
        ["per-rank scale + encode", "all-gather, local decode",
         r"$V_{\mathrm{AG}}\approx \frac{p-1}{c}G$ / rank"], "#f6d7cd")
    box(51.5, 46, 23.5, 50, "Top-k sparse",
        ["select k, pack indices", "gather sparse support",
         "reconstruct + residual"], "#fdf3d4")
    box(76.5, 46, 23, 50, "PowerSGD",
        ["rank-r factors", "factor all-reduce", "(reducible in transit)"],
        "#dcead9")
    box(0.5, 0, 48.5, 38, "GPU-resident path",
        ["gradient $\\rightarrow$ encode $\\rightarrow$ collective "
         "$\\rightarrow$ decode", "no host copies on the timed path"],
        "#f2f2f2", 7.2)
    box(51.5, 0, 48, 38, "CPU-staged path",
        ["gradient $\\rightarrow$ D2H copy $\\rightarrow$ host encode "
         "$\\rightarrow$ network", "$\\rightarrow$ host decode "
         "$\\rightarrow$ H2D copy"], "#f2f2f2", 7.2)
    fig.subplots_adjust(left=0.005, right=0.995, top=0.99, bottom=0.01)
    fig.savefig(os.path.join(FIG, "fig_mechanisms.pdf"))
    print("fig_mechanisms.pdf")


# ---------------------------------------------------------------- figure 2
def fig_calibration():
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.1, 2.7))

    ic = rows("intel_collective_scaling.csv")
    ranks = [int(r["accelerators"]) for r in ic]
    ar = [float(r["allreduce_time_ms"]) for r in ic]
    ag = [float(r["allgather_time_ms"]) for r in ic]
    a.plot(ranks, ar, marker="o", ms=4, lw=1.2, color="#0072B2",
           label="all-reduce")
    a.plot(ranks, ag, marker="s", ms=4, lw=1.2, color="#D55E00",
           label="all-gather")
    for x, y1, y2 in zip(ranks, ar, ag):
        a.annotate(f"{y2/y1:.1f}x", (x, y2), textcoords="offset points",
                   xytext=(0, 5), ha="center", fontsize=6.6,
                   color="#D55E00")
    a.set_yscale("log")
    a.set_xticks(ranks)
    a.set_xlabel("ranks", fontsize=8)
    a.set_ylabel("16 MiB collective time (ms)", fontsize=8)
    a.set_title("(a) Intel, 16 MiB payload vs. ranks", fontsize=8.5)
    a.legend(fontsize=6.8, frameon=False, loc="upper left")

    hc = rows("h100_fabric_collectives.csv")
    sizes = [r["payload_label"] for r in hc]
    har = [float(r["allreduce_time_ms_mean"]) for r in hc]
    hag = [float(r["allgather_time_ms_mean"]) for r in hc]
    x = range(len(sizes))
    b.plot(x, har, marker="o", ms=4, lw=1.2, color="#0072B2",
           label="all-reduce")
    b.plot(x, hag, marker="s", ms=4, lw=1.2, color="#D55E00",
           label="all-gather")
    for xi, y1, y2 in zip(x, har, hag):
        if y2 / y1 > 1.5:
            b.annotate(f"{y2/y1:.1f}x", (xi, y2),
                       textcoords="offset points", xytext=(0, 5),
                       ha="center", fontsize=6.6, color="#D55E00")
    b.set_yscale("log")
    b.set_xticks(list(x))
    b.set_xticklabels(sizes, fontsize=7.5)
    b.set_xlabel("payload per rank", fontsize=8)
    b.set_ylabel("collective time (ms)", fontsize=8)
    b.set_title("(b) H100 2x4 (8 ranks), payload sweep", fontsize=8.5)
    b.legend(fontsize=6.8, frameon=False, loc="upper left")
    for ax in (a, b):
        ax.tick_params(labelsize=7.5)
        ax.set_axisbelow(True)
        ax.grid(axis="y", linewidth=0.35, color="#dddddd")
    fig.subplots_adjust(left=0.08, right=0.99, top=0.92, bottom=0.17,
                        wspace=0.3)
    fig.savefig(os.path.join(FIG, "fig_calibration.pdf"))
    print("fig_calibration.pdf")


# ---------------------------------------------------------------- figure 3
def fig_scaling():
    sc = rows("intel_step_scaling.csv")
    ranks = [int(r["accelerators"]) for r in sc]
    methods = [("native", "native", "#0072B2", "o", "-"),
               ("gpu_quant", "GPU int8 (all-gather)", "#D55E00", "s", "-"),
               ("topk", "Top-k 1%", "#009E73", "^", "-"),
               ("powersgd", "PowerSGD r1", "#CC79A7", "D", "-"),
               ("ddp_overlap", "DDP overlap", "#555555", "v", "--")]
    fig, axs = plt.subplots(1, 2, figsize=(7.1, 2.7))
    for ax, model, title in ((axs[0], "resnet", "(a) ResNet-50"),
                             (axs[1], "transformer", "(b) Transformer")):
        for key, lab, col, mk, ls in methods:
            ts = [float(r[f"{model}_{key}"]) for r in sc]
            eff = [ts[0] / t for t in ts]
            ax.plot(ranks, eff, marker=mk, ms=4, lw=1.2, color=col, ls=ls,
                    mfc="white", label=lab)
            if model == "transformer":
                ax.annotate(f"{eff[-1]:.2f}", (ranks[-1], eff[-1]),
                            textcoords="offset points", xytext=(6, -2),
                            fontsize=6.6, color=col)
        ax.axhline(1.0, color="#999999", lw=0.7, ls=":")
        ax.set_xticks(ranks)
        ax.set_xlabel("ranks (8 per node)", fontsize=8)
        ax.set_title(title, fontsize=8.5)
        ax.set_ylim(0, 1.35)
        ax.tick_params(labelsize=7.5)
        ax.set_axisbelow(True)
        ax.grid(axis="y", linewidth=0.35, color="#dddddd")
    axs[0].set_ylabel("weak-scaling efficiency", fontsize=8)
    axs[1].legend(fontsize=6.2, frameon=False, loc="lower left",
                  labelspacing=0.3)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.92, bottom=0.17,
                        wspace=0.22)
    fig.savefig(os.path.join(FIG, "fig_scaling.pdf"))
    print("fig_scaling.pdf")


# ---------------------------------------------------------------- figure 4
PHASES = [("forward_s", "#5b8db8", "forward"),
          ("backward_compute_s", "#7fcdbb", "backward"),
          ("serialize_s", "#c85c5c", "encode"),
          ("network_s", "#f5d76e", "network"),
          ("deserialize_s", "#e0a838", "decode"),
          ("barrier_wait_s", "#f2b27e", "sync wait"),
          ("optimizer_step_s", "#b6a6ca", "optimizer")]


def seed_condition_means():
    """mean seconds per phase and observed step, per (model, mode, tau),
    averaged rank-iterations within a job then across the three jobs."""
    acc = defaultdict(lambda: defaultdict(list))
    for job in sorted(os.listdir(RAW_SEEDS)):
        per = defaultdict(lambda: defaultdict(list))
        for f in glob.glob(os.path.join(RAW_SEEDS, job, "results", "*.csv")):
            with open(f, newline="") as fh:
                for r in csv.DictReader(fh):
                    key = (r["model"], r["comm_mode"], int(r["sync_every"]))
                    for ph, _, _ in PHASES:
                        v = r.get(ph, "")
                        per[key][ph].append(float(v) if v else 0.0)
                    per[key]["observed_step_s"].append(
                        float(r["observed_step_s"]))
        for key, d in per.items():
            for ph, vals in d.items():
                acc[key][ph].append(st.mean(vals))
    return {k: {ph: st.mean(v) for ph, v in d.items()} for k, d in acc.items()}


def fig_phases(cm):
    conds = [("native", 1, "native"), ("gpu-quant", 1, "GPU int8"),
             ("gpu-topk", 1, "Top-k"), ("cpu-quant", 1, "CPU int8"),
             ("ddp-overlap", 1, "DDP overlap")]
    fig, axs = plt.subplots(1, 2, figsize=(7.1, 2.9))
    for ax, model, title in ((axs[0], "resnet50", "(a) ResNet-50"),
                             (axs[1], "transformer", "(b) Transformer")):
        for i, (mode, tau, lab) in enumerate(conds):
            key = (model, mode, tau)
            if key not in cm:
                alt = [k for k in cm if k[0] == model and k[1] == mode]
                key = alt[0] if alt else None
            d = cm[key]
            step = d["observed_step_s"] * 1e3
            if mode == "ddp-overlap":
                ax.bar(i, step, 0.6, color="#dddddd", edgecolor="#333333",
                       linewidth=0.5, hatch="//",
                       label="observed step (non-additive)"
                       if (i == len(conds) - 1 and model == "resnet50")
                       else None)
            else:
                bot = 0.0
                for ph, col, plab in PHASES:
                    v = d[ph] * 1e3
                    ax.bar(i, v, 0.6, bottom=bot, color=col,
                           edgecolor="#333333", linewidth=0.4,
                           label=plab if (i == 0 and model == "resnet50")
                           else None)
                    bot += v
                ax.plot([i - 0.36, i + 0.36], [step, step], color="#222222",
                        lw=1.0)
        ax.set_xticks(range(len(conds)))
        ax.set_xticklabels([c[2] for c in conds], fontsize=7,
                           rotation=12)
        ax.set_ylabel("ms per iteration", fontsize=8)
        ax.set_title(title, fontsize=8.5)
        ax.tick_params(labelsize=7.5)
        ax.set_axisbelow(True)
        ax.grid(axis="y", linewidth=0.35, color="#dddddd")
    fig.legend(fontsize=6.2, frameon=False, loc="lower center", ncol=8,
               columnspacing=0.9, handlelength=1.2,
               bbox_to_anchor=(0.5, -0.015))
    fig.subplots_adjust(left=0.075, right=0.99, top=0.92, bottom=0.22,
                        wspace=0.25)
    fig.savefig(os.path.join(FIG, "fig_phases.pdf"))
    print("fig_phases.pdf")


def print_numbers(cm):
    print("\n=== manuscript numbers (TACC 2x4, mean of three jobs)")
    for model in ("resnet50", "transformer"):
        for mode, tau in (("native", 1), ("native", 2), ("native", 4),
                          ("gpu-quant", 1), ("cpu-quant", 1),
                          ("gpu-topk", 1), ("gpu-topk", 4),
                          ("powersgd", 1), ("ddp-overlap", 1)):
            key = (model, mode, tau)
            if key not in cm:
                continue
            d = cm[key]
            step = d["observed_step_s"]
            ser = d["serialize_s"] + d["deserialize_s"]
            print(f"{model:11s} {mode:11s} tau{tau} "
                  f"step={step*1e3:8.2f}ms net={d['network_s']/step*100:5.1f}% "
                  f"ser={ser/step*100:5.1f}% sync={d['barrier_wait_s']/step*100:4.1f}%")


fig_paths()
fig_calibration()
fig_scaling()
cm = seed_condition_means()
fig_phases(cm)
print_numbers(cm)
