# -*- coding: utf-8 -*-
"""E12 mirror cases (diagnostic/E12_PROTOCOL.md section 5): the reverse
substitutions of the frozen E8 replay's two families, so that the replay can
also be scored on changes expected to hurt.

The frozen reference (diagnostic/e8/replay_reference.py) is imported and not
changed: its trace model, replay and forward substitutions are used as they
are. This module only adds two rewrites of an original's trace.

- G-rev, gpu-quant-reducible to gpu-quant-gathersum: the gradient all-reduce
  of the int8 bytes becomes an all-gather of the same bytes per rank on the
  same stream, followed by p device-to-device copies of n bytes (the copies
  c10d makes out of the gathered buffer, as in the traced gather-sum). The
  decode gains an int32 zero-fill (4n bytes) and, per source, a cast from
  int8 to int32 (5n) and an add (12n), all on the compute stream and after
  the copies; the final cast reads int32 (8n bytes) instead of int8 (5n).
- H-rev, ddp-fp16-hook to ddp-int8-hook (synchronous form, as E12 runs it):
  each bucket's fp16 hook (cast, divide, all-reduce, copy back) becomes the
  int8 hook on the compute stream: 15 statistics ops (two reductions of 4n
  bytes, the rest scalar), p zero-fills of the scale buffers, an all-gather
  of 8 bytes per rank and its p copies, the quantization (8n, 8n, 8n, 8n and
  5n bytes), p zero-fills of n bytes, the all-gather of n bytes per rank and
  its p copies (2n each), a zero-fill of 4n, per source a cast (5n), a
  subtract (8n), a multiply (8n) and an add (12n), and a final divide (8n).
  The compute stream waits for each all-gather's copies before the next op,
  as the synchronous hook does.

Costs: an all-gather of b bytes per rank takes the allocation's fitted
all-gather line, (p - 1) alpha_AG + beta_AG (p - 1) / p b, with the harness's
alpha and beta as recorded (the same line the E12 cost model uses).
Elementwise kernels take bytes / the elementwise bandwidth measured in the
same trace (as the frozen reference does); scalar kernels take TINY_US.
Every new op costs OP_CPU on the CPU, the median CPU time of the original's
own hook or codec ops; views cost VIEW_CPU_US.

Developed only on the e7a traces (DEV_REPORT_REVERSE.md); no E12 record was
read.
"""
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
_E8 = os.path.join(os.path.dirname(HERE), "e8")
if _E8 not in sys.path:
    sys.path.insert(0, _E8)
import replay_reference as R  # noqa: E402  frozen E8 reference

FAMILIES = ("G-rev", "H-rev")
TINY_US = 1.5        # scalar kernel on an A100 (the int8 hook's statistics ops in the e7a traces: 1.2 to 2.1 us)
VIEW_CPU_US = 3.0    # a view op with no kernel (aten::select, aten::empty_like)
STATS_OPS = 15       # zeros, min, minimum, max, maximum, sub, div, gt, full_like, where, neg, div, round, clamp, stack


def allgather_s(cal, world, nbytes):
    """Fitted all-gather time (s) of nbytes per rank."""
    return cal["alpha_allgather_s"] * (world - 1) + cal["beta_allgather_s_per_byte"] * (world - 1.0) / world * nbytes


def _launch(t, d, corr):
    return R._new_launch(t, d, corr)


def _nccl_launch(tr, idx):
    return R._nccl_of(tr, idx)


def _op_cpu(ops, default=15.0):
    durs = [float(e.get("dur", 0.0)) for e in ops if float(e.get("dur", 0.0)) > 0]
    return st.median(durs) if durs else default


# ---------------------------------------------------------------- G-rev
def reduce_regions(tr):
    """Every gradient all-reduce (the largest element count; one per
    iteration) with its items, the encode ops before it (bandwidth) and the
    final cast after it."""
    found = []
    for g in (e for e in tr.cpu_ops if e.get("name") == "c10d::allreduce_"):
        idx = R._items_in(tr, g.get("tid"), *R._span(g))
        nc = _nccl_launch(tr, idx)
        if nc is not None:
            found.append((float(g["ts"]), g, idx, int(tr.gpu[nc]["args"].get("In msg nelems", 0)), nc))
    if not found:
        raise ValueError("no all-reduce with a NCCL kernel")
    nmax = max(f[3] for f in found)
    if nmax <= 1:
        raise ValueError("no gradient all-reduce")
    regions = []
    for _, g, idx, n, nc in sorted((f for f in found if f[3] == nmax), key=lambda f: f[0]):
        world = int(tr.gpu[nc]["args"].get("Group size", 0)) or None
        sib = R._siblings(tr, g)
        before = [e for e in sib if float(e["ts"]) < float(g["ts"])]
        after = [e for e in sib if float(e["ts"]) > float(g["ts"])]
        enc = [e for e in before if e.get("name") in ("aten::div", "aten::round", "aten::clamp", "aten::to")
               and R._launches(tr, R._items_in(tr, g.get("tid"), *R._span(e)))][-4:]
        cast = next((e for e in after if e.get("name") in ("aten::to", "aten::_to_copy")), None)
        if world is None or cast is None or len(enc) < 4:
            raise ValueError("reducible pattern not found (world %s, cast %s, encode ops %d)"
                             % (world, cast is not None, len(enc)))
        cast_idx = R._items_in(tr, g.get("tid"), *R._span(cast))
        regions.append({"op": g, "idx": idx, "n": n, "nccl": nc, "world": world, "enc": enc,
                        "cast_idx": cast_idx, "cast": R._launches(tr, cast_idx)})
    return regions


def substitute_reduce_gather(tr, cal):
    """G-rev: the reducible all-reduce becomes the gather-sum (module doc)."""
    regions = reduce_regions(tr)
    big = regions[0]
    tid = big["op"].get("tid")
    kb = []
    for e, mult in zip(big["enc"], (8, 8, 8, 5)):
        cs = R._launches(tr, R._items_in(tr, tid, *R._span(e)))
        kb.append((cs[-1], mult * big["n"]))
    bw = R._elementwise_bandwidth(tr, kb)
    oc = _op_cpu(big["enc"])
    compute = tr.gpu[kb[0][0]]["stream"]
    deps = R.infer_dependencies(tr)
    rw = R._Rewrite(tr)
    casts = []
    for r in regions:
        n, p = r["n"], r["world"]
        nccl = tr.gpu[r["nccl"]]["stream"]
        ag = rw.op(nccl, 1e6 * allgather_s(cal, p, n), "ncclAllGather int8 (substituted)")
        rw.new.forced_deps[ag] = set(deps.get(r["nccl"], ()))  # the all-reduce's waits
        copies = [rw.op(nccl, 2.0 * n / bw, "gather copy (substituted)") for _ in range(p)]
        new = []
        t_last = None
        for k in r["idx"]:
            it = tr.items[k]
            if it["corr"] == r["nccl"]:
                new.append(_launch(it["t"], it["d"], ag))
                t_last = it["t"] + it["d"]
            else:
                new.append(it)
        new += [_launch(t_last + VIEW_CPU_US * p + i * oc, oc, c) for i, c in enumerate(copies)]
        rw.remove.update(r["idx"])
        rw.insert[min(r["idx"])] = new + [R._shift_item(tr.items[min(r["idx"])]["t"], p * (oc + VIEW_CPU_US))]
        rw.redirect[r["nccl"]] = copies[-1]
        zero = rw.op(compute, 4.0 * n / bw, "int32 zero-fill (substituted)")
        dec = [zero]
        for _ in range(p):
            dec.append(rw.op(compute, 5.0 * n / bw, "int8 to int32 (substituted)"))
            dec.append(rw.op(compute, 12.0 * n / bw, "int32 add (substituted)"))
        rw.new.forced_deps[dec[1]] = {copies[-1]}
        t0 = tr.items[min(r["cast_idx"])]["t"]
        rw.insert[min(r["cast_idx"])] = [_launch(t0 + i * oc, oc, c) for i, c in enumerate(dec)] + [
            R._shift_item(t0, len(dec) * oc)]
        for c in r["cast"][:1]:
            rw.new.forced_deps.setdefault(c, set()).add(dec[-1])
        casts.extend(r["cast"])
    out = rw.build()
    for c in casts:
        g = dict(out.gpu[c])
        g["dur"] = g["dur"] * 8.0 / 5.0
        out.gpu[c] = g
    return out


# ---------------------------------------------------------------- H-rev
def _is_half_allreduce(name):
    low = name.lower()
    return "allreduce" in low and ("f16" in low or "bf16" in low)


def fp16_buckets(tr):
    """Each bucket's fp16-hook region: the cast and divide before the
    16-bit all-reduce, the all-reduce, and the copy back after it."""
    ars = []
    for e in tr.cpu_ops:
        if e.get("name") != "c10d::allreduce_":
            continue
        idx = R._items_in(tr, e.get("tid"), *R._span(e))
        nc = _nccl_launch(tr, idx)
        if nc is not None and _is_half_allreduce(tr.gpu[nc]["name"]):
            ars.append((float(e["ts"]), e, idx, nc))
    buckets = []
    for _, e, idx, nc in sorted(ars, key=lambda x: x[0]):
        tid = e.get("tid")
        sib = R._siblings(tr, e)
        k = next(i for i, x in enumerate(sib) if x is e)
        if k < 2 or sib[k - 1].get("name") != "aten::div_" or sib[k - 2].get("name") not in ("aten::to", "aten::_to_copy"):
            raise ValueError("fp16 hook pattern (cast, div_, all-reduce) not found")
        cast, div = sib[k - 2], sib[k - 1]
        c_cast = R._launches(tr, R._items_in(tr, tid, *R._span(cast)))
        c_div = R._launches(tr, R._items_in(tr, tid, *R._span(div)))
        if not c_cast or not c_div:
            raise ValueError("fp16 hook cast or divide has no kernel")
        compute = tr.gpu[c_cast[-1]]["stream"]
        copy = next((x for x in sib[k + 1:] if x.get("name") == "aten::copy_"), None)
        if copy is None:
            raise ValueError("fp16 hook copy back not found")
        c_copy = R._launches(tr, R._items_in(tr, tid, *R._span(copy)))
        n = int(tr.gpu[nc]["args"].get("In msg nelems", 0))
        world = int(tr.gpu[nc]["args"].get("Group size", 0)) or None
        if world is None or n <= 0:
            raise ValueError("bucket size or world size not recorded")
        t0, t1 = float(cast["ts"]), R._span(copy)[1]
        buckets.append({"idx": R._items_in(tr, tid, t0, t1), "n": n, "world": world, "nccl": nc,
                        "compute": compute, "kernels": [(c_cast[-1], 6 * n), (c_div[-1], 4 * n)],
                        "removed": c_cast + c_div + c_copy + [nc], "t_start": t0, "t_end": t1,
                        "coll_cpu": max(sum(tr.items[j]["d"] for j in idx), 1.0),
                        "op_cpu": _op_cpu([cast, div, copy])})
    return buckets


def int8_hook_ops(n, p):
    """The synchronous int8 hook as (kind, bytes) in launch order; kind is
    'k' (elementwise, bytes), 't' (scalar), 'v' (view, CPU only),
    'ag_s' / 'ag_d' (the two all-gathers) and 'cp' (a copy of bytes)."""
    ops = [("t", 0), ("k", 4 * n)] + [("t", 0)] + [("k", 4 * n)] + [("t", 0)] * (STATS_OPS - 4)
    ops += [("t", 0)] * p + [("ag_s", 8)] + [("cp", 8)] * p
    ops += [("k", 8 * n)] * 4 + [("k", 5 * n)]
    ops += [("k", n)] * p + [("ag_d", n)] + [("cp", 2 * n)] * p
    ops += [("k", 4 * n)]
    for _ in range(p):
        ops += [("k", 5 * n), ("v", 0), ("k", 8 * n), ("v", 0), ("k", 8 * n), ("k", 12 * n)]
    ops += [("k", 8 * n)]
    return ops


def substitute_fp16_int8(tr, cal):
    """H-rev: each fp16-hook bucket becomes the synchronous int8 hook (module doc)."""
    buckets = fp16_buckets(tr)
    if not buckets:
        raise ValueError("no fp16-hook bucket found")
    big = max(buckets, key=lambda b: b["n"])
    bw = R._elementwise_bandwidth(tr, big["kernels"])
    rw = R._Rewrite(tr)
    for b in buckets:
        n, p, compute = b["n"], b["world"], b["compute"]
        nccl = tr.gpu[b["nccl"]]["stream"]
        oc = b["op_cpu"]
        t = b["t_start"]
        new_items = []
        last_compute, wait_on = None, None
        for kind, nbytes in int8_hook_ops(n, p):
            if kind == "v":
                t += VIEW_CPU_US
                continue
            if kind in ("ag_s", "ag_d"):
                c = rw.op(nccl, 1e6 * allgather_s(cal, p, nbytes), "ncclAllGather (substituted)")
                if last_compute is not None:
                    rw.new.forced_deps[c] = {last_compute}
                d = b["coll_cpu"]
            elif kind == "cp":
                c = rw.op(nccl, max(nbytes / bw, TINY_US), "gather copy (substituted)")
                d = oc
                wait_on = c
            else:
                dur = TINY_US if kind == "t" else max(nbytes / bw, TINY_US)
                c = rw.op(compute, dur, "int8 hook op (substituted)")
                if wait_on is not None:
                    rw.new.forced_deps[c] = {wait_on}
                    wait_on = None
                last_compute = c
                d = oc
            new_items.append(_launch(t, d, c))
            t += d
        old_cpu = b["t_end"] - b["t_start"]
        redirect = {c: last_compute for c in b["removed"]}
        rw.replace(b["idx"], new_items, (t - b["t_start"]) - old_cpu, redirect)
    return rw.build()


SUBSTITUTE = {"G-rev": substitute_reduce_gather, "H-rev": substitute_fp16_int8}


def predict(paths, family, cal):
    """g_pred for one mirror case from the original's traces (ranks 0 and 3)
    and the allocation's calibration (both collectives)."""
    if family not in SUBSTITUTE:
        raise ValueError("unknown reverse family %r" % (family,))
    per_rank = []
    for p in paths:
        tr = R.load(p)
        base = R.replay(tr)
        lat = R._traced_dependencies(tr, tr.deps)
        cand = R.replay(SUBSTITUTE[family](tr, cal), latencies=lat)
        per_rank.append({"path": os.path.basename(p), "orig_ms": base.step_ms, "cand_ms": cand.step_ms,
                         "fidelity": base.fidelity, "busy_orig_ms": base.busy_ms, "busy_cand_ms": cand.busy_ms})
    orig = max(r["orig_ms"] for r in per_rank)
    cand = max(r["cand_ms"] for r in per_rank)
    return R.gain(orig, cand), {"orig_ms": orig, "cand_ms": cand, "ranks": per_rank}
