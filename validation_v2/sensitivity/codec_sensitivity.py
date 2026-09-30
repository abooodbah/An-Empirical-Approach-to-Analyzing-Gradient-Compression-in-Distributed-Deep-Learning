# -*- coding: utf-8 -*-
"""Input-distribution sensitivity of the compression codecs on one device.

Times the harness's own encode and reconstruct functions (imported from
harness v5.1; the shared-scale encoder is a verbatim copy of its inline
closure, checked by the tests) on four distributions at the gradient
sizes of the two workloads. Inputs and state are prepared before timing;
cells run in a seeded random order; each cell is timed in BLOCKS blocks
of a fixed number of calls between device synchronizations.

Pre-registered margin: a codec is insensitive to the input distribution
at a workload size when, for every non-Gaussian distribution, the 95%
bootstrap interval of its median per-call time over the Gaussian median
lies inside [0.95, 1.05]. The collective, the scale all-reduce, and the
Top-k residual (which needs the global reconstruction) are not timed here.

Usage: python codec_sensitivity.py run <harness.py> <out.csv> <cuda|cpu>
       python codec_sensitivity.py analyze <out.csv>
"""
import csv
import importlib.util
import random
import statistics as st
import sys
import time
from collections import defaultdict

import torch

SIZES = {"transformer": 11425794, "resnet50": 25557032}
DISTS = ("gaussian", "heavy", "zero_tied", "outlier")
SEEDS = (1, 2, 3, 4, 5)
BLOCKS = 5
MARGIN = 0.05
P_LEVELS = 12
TOPK_FRAC = 0.01
CODECS = ("rank_local_encode", "rank_local_decode", "shared_encode", "shared_decode",
          "topk_select", "topk_reconstruct", "cpu_encode", "cpu_decode")
VERBATIM_LINES = (
    "quantized = torch.clamp(",
    "torch.round(flat / scale_tensor), min=-levels, max=levels",
    "return quantized.to(torch.int8)",
    "reconstructed = encoded.to(torch.float32)",
    "reconstructed.mul_(scale_tensor / runtime.world_size)",
)


def load_harness(path):
    spec = importlib.util.spec_from_file_location("harness_v51", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod      # the harness's dataclasses resolve their module here
    spec.loader.exec_module(mod)
    return mod


def make_input(dist, n, seed, device):
    g = torch.Generator().manual_seed(seed)
    if dist == "gaussian":
        x = torch.randn(n, generator=g) * 1e-3
    elif dist == "heavy":
        z = torch.randn(n, generator=g)
        chi2 = torch.randn(n, generator=g) ** 2 + torch.randn(n, generator=g) ** 2
        x = z / torch.sqrt(chi2 / 2.0) * 1e-3            # Student t, 2 degrees of freedom
    elif dist == "zero_tied":
        x = torch.round(torch.randn(n, generator=g) * 10.0) * 1e-4
        x[torch.rand(n, generator=g) < 0.9] = 0.0
    elif dist == "outlier":
        x = torch.randn(n, generator=g) * 1e-3
        idx = torch.randperm(n, generator=g)[: max(1, n // 10000)]
        x[idx] *= 1000.0
    else:
        raise ValueError(dist)
    return x.to(device)


def prepare(h, codec, x, device):
    n = x.numel()
    levels = float(max(1, 127 // P_LEVELS))

    class _RT:
        world_size = P_LEVELS

    runtime = _RT()
    scale_tensor = (x.detach().abs().max().to(torch.float32) / levels).clamp_min(1e-30)
    flat = x

    def shared_encode():
        quantized = torch.clamp(
            torch.round(flat / scale_tensor), min=-levels, max=levels
        )
        return quantized.to(torch.int8)

    if codec == "rank_local_encode":
        return lambda: h.gpu_quantize(x)
    if codec == "rank_local_decode":
        scale, enc = h.gpu_quantize(x)
        return lambda: h.gpu_dequantize(scale, enc, n)
    if codec == "shared_encode":
        return shared_encode
    if codec == "shared_decode":
        encoded = shared_encode()

        def shared_decode():
            reconstructed = encoded.to(torch.float32)
            reconstructed.mul_(scale_tensor / runtime.world_size)
            return reconstructed
        return shared_decode
    k = max(1, int(TOPK_FRAC * n))
    if codec == "topk_select":
        return lambda: h.gpu_topk(x, k)
    if codec == "topk_reconstruct":
        idx, vals = h.gpu_topk(x, k)
        return lambda: h._reconstruct_topk([idx], [vals], [idx.numel()], x, 1)
    if codec == "cpu_encode":
        return lambda: h.cpu_quantize(x)
    if codec == "cpu_decode":
        payload = h.cpu_quantize(x)
        return lambda: h.cpu_dequantize(payload, n, torch.device(device))
    raise ValueError(codec)


def run(harness, out, device, sizes=None, seeds=SEEDS, blocks=BLOCKS):
    h = load_harness(harness)
    sizes = sizes or SIZES
    sync = torch.cuda.synchronize if device == "cuda" else (lambda: None)
    cells = [(c, w, d, s) for c in CODECS for w in sizes for d in DISTS for s in seeds]
    random.Random(20261001).shuffle(cells)
    with open(out, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["codec", "workload", "n", "dist", "seed", "block", "calls",
                     "per_call_s", "device", "threads"])
        for codec, wl, dist, seed in cells:
            x = make_input(dist, sizes[wl], seed, device)
            call = prepare(h, codec, x, device)
            for _ in range(3):
                call()
            sync()
            t0 = time.perf_counter(); call(); sync()
            calls = max(3, min(200, int(0.05 / max(time.perf_counter() - t0, 1e-6))))
            for b in range(blocks):
                sync(); t0 = time.perf_counter()
                for _ in range(calls):
                    call()
                sync()
                wr.writerow([codec, wl, sizes[wl], dist, seed, b, calls,
                             (time.perf_counter() - t0) / calls, device, torch.get_num_threads()])


def verdicts(path):
    by = defaultdict(list)
    for r in csv.DictReader(open(path, newline="")):
        by[(r["codec"], r["workload"], r["dist"])].append(float(r["per_call_s"]))
    rng = random.Random(7)
    ok = {}
    for (codec, wl, dist), vals in sorted(by.items()):
        if dist == "gaussian":
            continue
        base = by[(codec, wl, "gaussian")]
        boots = sorted(st.median([rng.choice(vals) for _ in vals]) /
                       st.median([rng.choice(base) for _ in base]) for _ in range(2000))
        inside = (1 - MARGIN) <= boots[49] and boots[1949] <= (1 + MARGIN)
        ok[(codec, wl)] = ok.get((codec, wl), True) and inside
        print("%-18s %-11s %-9s ratio %.3f  95%% [%.3f, %.3f] %s" % (
            codec, wl, dist, st.median(vals) / st.median(base), boots[49], boots[1949],
            "inside" if inside else "OUTSIDE"))
    return ok


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        for key, good in verdicts(sys.argv[2]).items():
            print("%s %s: %s" % (key[0], key[1], "insensitive" if good else "SENSITIVE"))
