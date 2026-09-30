# -*- coding: utf-8 -*-
"""E8 reference (diagnostic/E8_PROTOCOL.md section 7): an overlap-aware replay
of the original's Kineto trace with the candidate substituted.

The replay adapts the dependency-graph approach of dPRO (Hu et al., MLSys
2022) to the harness's Kineto traces. It is not dPRO, which reads only its
own byteprofile/BytePS traces.

Model
- CPU: the CUDA runtime calls of all threads, merged in time order, keep
  their durations and the gaps between them. A blocking call (device or
  stream synchronize, a copy to pageable host memory) instead ends when the
  replayed GPU work it waits on ends, plus the trace's post-sync latency;
  every later call moves by the same amount.
- GPU: an op starts no earlier than its launch plus the trace's launch
  latency, after the previous op on its stream, and after the ops it waits
  on. Waits come from cudaEventRecord / cudaStreamWaitEvent pairs: the first
  op launched after a wait waits on the last op launched, before the paired
  record, on the recording stream (the stream of the last launch before the
  record). A c10d wait whose record shares the waiting stream waits on the
  last collective-stream op instead.
- Iterations end at the first blocking synchronize after each
  Optimizer.step annotation. The step is the median interval between
  consecutive ends; a prediction uses the larger of the traced ranks.

Substitutions (costs from the allocation's all-reduce calibration, alpha +
beta x bytes, and from the elementwise bandwidth measured in the same trace)
- H, ddp-int8-hook to ddp-fp16-hook: each bucket's hook (encode, the scale
  and data all-gathers with their copies, and the per-source dequantization)
  becomes a cast to fp16 and a divide on the compute stream, an all-reduce of
  2n bytes after them, and a copy back on a side stream after the
  all-reduce, where n is the bucket's element count. As in the traced fp16
  hook, the compute stream waits for the copies once, at the first compute
  op after the iteration's last hook (DDP's end of backward).
- G, gpu-quant-gathersum to gpu-quant-reducible: the gradient all-gather and
  its copies become an all-reduce of the same int8 bytes; the int32 zero-fill
  and the per-source casts and adds are removed; the final cast reads int8
  (5n bytes) instead of int32 (8n bytes).
"""
import bisect
import copy
import io
import json
import os
import statistics as st
from collections import defaultdict

LAUNCH_CALLS = {"cudaLaunchKernel", "cudaLaunchKernelExC", "cudaMemcpyAsync", "cudaMemsetAsync",
                "cudaMemcpy", "cudaMemset", "cuLaunchKernel", "cuLaunchKernelEx"}
SYNC_CALLS = {"cudaDeviceSynchronize", "cudaStreamSynchronize", "cudaEventSynchronize"}
GPU_CATS = ("kernel", "gpu_memcpy", "gpu_memset")
C10D_WAIT_OPS = {"record_param_comms"}
THRESHOLD = 0.05
DEFAULT_LAUNCH_LATENCY_US = 5.0
DEFAULT_POST_SYNC_US = 10.0
MAX_LAUNCH_LATENCY_US = 100.0


def openable(path):
    if os.name == "nt":
        path = os.path.abspath(path).replace("/", "\\")
        if not path.startswith("\\\\?\\"):
            path = "\\\\?\\" + path
    return path


def gain(original, candidate):
    """Predicted gain of the candidate: 1 - candidate / original."""
    return 1.0 - candidate / original


def recommends(g_pred):
    return g_pred >= THRESHOLD


class Trace:
    """CPU runtime calls (items), GPU ops keyed by correlation id, cpu_op
    events, and user annotations of one Kineto trace."""

    def __init__(self, data):
        ev = data["traceEvents"]
        self.gpu = {}
        for e in ev:
            a = e.get("args") or {}
            if e.get("cat") in GPU_CATS and "correlation" in a:
                self.gpu[a["correlation"]] = {"stream": a.get("stream"), "ts": float(e["ts"]),
                                              "dur": float(e.get("dur", 0.0)), "name": e.get("name", ""),
                                              "args": a, "cat": e["cat"]}
        calls = sorted((e for e in ev if e.get("cat") in ("cuda_runtime", "cuda_driver")),
                       key=lambda e: float(e["ts"]))
        self.items = []
        for e in calls:
            a = e.get("args") or {}
            name, corr = e.get("name", ""), a.get("correlation")
            if name in LAUNCH_CALLS and corr in self.gpu:
                kind = "launch"
            elif name in SYNC_CALLS:
                kind = "sync"
            elif name == "cudaEventRecord":
                kind = "record"
            elif name == "cudaStreamWaitEvent":
                kind = "wait"
            else:
                kind = "other"
            blocking = kind == "sync" or (kind == "launch" and (
                name == "cudaMemcpy" or "Pageable" in self.gpu[corr]["name"]))
            self.items.append({"t": float(e["ts"]), "d": float(e.get("dur", 0.0)), "name": name,
                               "kind": kind, "corr": corr, "ext": a.get("External id"),
                               "tid": e.get("tid"), "blocking": blocking})
        self.cpu_ops = [e for e in ev if e.get("cat") == "cpu_op"]
        self.ann = sorted((e for e in ev if e.get("cat") == "user_annotation"), key=lambda e: float(e["ts"]))
        self.ext_name = {}
        for e in self.cpu_ops:
            self.ext_name.setdefault((e.get("args") or {}).get("External id"), e.get("name"))
        self.nccl_streams = {g["stream"] for g in self.gpu.values() if "nccl" in g["name"].lower()}
        self.deps = None
        self.forced_deps = {}

    def clone(self):
        c = copy.copy(self)
        c.gpu = dict(self.gpu)
        c.items = list(self.items)
        c.deps = None if self.deps is None else {k: set(v) for k, v in self.deps.items()}
        c.forced_deps = {k: set(v) for k, v in self.forced_deps.items()}
        return c


def load(path):
    with io.open(openable(path), encoding="utf-8") as fh:
        return Trace(json.load(fh))


def infer_dependencies(tr):
    """Cross-stream waits from event record / stream wait pairs (see module doc)."""
    deps = defaultdict(set)
    last_on = {}
    last_stream = None
    rec_stream, rec_src = None, None
    last_coll = None
    pending = []
    for it in tr.items:
        kind = it["kind"]
        if kind == "launch":
            c = it["corr"]
            s = tr.gpu[c]["stream"]
            for src_stream, src, fallback in pending:
                if src is not None and src_stream is not None and src_stream != s:
                    deps[c].add(src)
                elif fallback is not None and tr.gpu[fallback]["stream"] != s:
                    deps[c].add(fallback)
            pending = []
            last_on[s] = c
            last_stream = s
            if s in tr.nccl_streams:
                last_coll = c
        elif kind == "record":
            rec_stream = last_stream
            rec_src = last_on.get(last_stream) if last_stream is not None else None
        elif kind == "wait":
            fallback = last_coll if tr.ext_name.get(it["ext"]) in C10D_WAIT_OPS else None
            pending.append((rec_stream, rec_src, fallback))
    for c, ps in tr.forced_deps.items():
        deps[c] |= ps
    return deps


def _traced_dependencies(tr, deps):
    """Latencies measured in the original: launch-to-start of ops that were
    not held by a predecessor, and sync return after the awaited GPU work."""
    prev_on = {}
    lat = []
    for it in tr.items:
        if it["kind"] != "launch":
            continue
        c = it["corr"]
        g = tr.gpu[c]
        s = g["stream"]
        if g["ts"] is None:  # a substituted op has no traced time
            continue
        held = max([prev_on.get(s, -1e30)] + [tr.gpu[p]["ts"] + tr.gpu[p]["dur"] for p in deps.get(c, ())
                                              if p in tr.gpu and tr.gpu[p]["ts"] is not None])
        # launch-limited ops only: started soon after their launch and not
        # right behind a predecessor (queued kernels leave gaps of a few us)
        if 0.0 <= g["ts"] - it["t"] < MAX_LAUNCH_LATENCY_US and g["ts"] - held > 1.0:
            lat.append(g["ts"] - it["t"])
        prev_on[s] = g["ts"] + g["dur"]
    post = []
    fin = -1e30
    for it in tr.items:
        if it["kind"] == "launch":
            g = tr.gpu[it["corr"]]
            if g["ts"] is not None:
                fin = max(fin, g["ts"] + g["dur"])
        elif it["kind"] == "sync" and it["d"] > 50.0:
            end = it["t"] + it["d"]
            if end >= fin:
                post.append(end - fin)
    launch_latency = st.median(lat) if lat else DEFAULT_LAUNCH_LATENCY_US
    post_sync = min(st.median(post), 50.0) if post else DEFAULT_POST_SYNC_US
    return launch_latency, post_sync


def iteration_ends(tr):
    """Indices of the items that end iterations: the first blocking
    synchronize after each Optimizer.step annotation."""
    opt_ends = [float(a["ts"]) + float(a.get("dur", 0.0)) for a in tr.ann
                if str(a.get("name", "")).startswith("Optimizer.step")]
    ends = []
    j = 0
    for k, it in enumerate(tr.items):
        if j < len(opt_ends) and it["kind"] == "sync" and it["name"] == "cudaDeviceSynchronize" and it["t"] >= opt_ends[j]:
            ends.append(k)
            j += 1
            while j < len(opt_ends) and opt_ends[j] <= it["t"]:
                j += 1
    return ends


class Result:
    def __init__(self):
        self.intervals_ms = []
        self.traced_intervals_ms = []
        self.busy_ms = None
        self.step_ms = None
        self.fidelity = None
        self.start = {}
        self.finish = {}


def replay(tr, durations=None, latencies=None):
    """Replay the trace (optionally with GPU durations overridden, in us)."""
    if tr.deps is None:
        tr.deps = infer_dependencies(tr)
    deps = tr.deps
    launch_latency, post_sync = latencies or _traced_dependencies(tr, deps)
    durations = durations or {}
    res = Result()
    shift = 0.0
    stream_last = {}
    item_end = []
    for it in tr.items:
        c0 = it["t"] + shift
        kind = it["kind"]
        if kind == "shift":
            shift += it["delta"]
        elif kind == "launch":
            c = it["corr"]
            g = tr.gpu[c]
            ready = max(c0 + launch_latency, stream_last.get(g["stream"], -1e30))
            for p in deps.get(c, ()):
                if p in res.finish:
                    ready = max(ready, res.finish[p])
            res.start[c] = ready
            res.finish[c] = ready + durations.get(c, g["dur"])
            stream_last[g["stream"]] = res.finish[c]
            if it["blocking"]:
                new_end = max(c0 + min(it["d"], 20.0), res.finish[c] + post_sync)
                shift = new_end - (it["t"] + it["d"])
        elif kind == "sync":
            awaited = max(stream_last.values()) if stream_last else c0
            new_end = max(c0 + min(it["d"], 10.0), awaited + post_sync)
            shift = new_end - (it["t"] + it["d"])
        item_end.append(it["t"] + it["d"] + shift)
    ends = iteration_ends(tr)
    res.intervals_ms = [(item_end[b] - item_end[a]) / 1e3 for a, b in zip(ends, ends[1:])]
    res.traced_intervals_ms = [(tr.items[b]["t"] + tr.items[b]["d"] - tr.items[a]["t"] - tr.items[a]["d"]) / 1e3
                               for a, b in zip(ends, ends[1:])]
    busy = []
    for a, b in zip(ends, ends[1:]):
        cs = [it["corr"] for it in tr.items[a + 1:b] if it["kind"] == "launch"]
        if cs:
            busy.append((max(res.finish[c] for c in cs) - min(res.start[c] for c in cs)) / 1e3)
    res.busy_ms = st.median(busy) if busy else None
    if res.intervals_ms:
        res.step_ms = st.median(res.intervals_ms)
        traced = [v for v in res.traced_intervals_ms if v > 0]
        res.fidelity = res.step_ms / st.median(traced) if traced else None
    return res


# ---------------------------------------------------------------- substitution
def _siblings(tr, op):
    """The direct children of op's innermost enclosing cpu_op on its thread
    (the thread's top-level ops if none), in time order. A DDP comm hook
    runs inside the AccumulateGrad function of its bucket's last gradient."""
    tid = op.get("tid")
    cache = tr.__dict__.setdefault("_trees", {})
    if tid not in cache:
        ops = sorted((e for e in tr.cpu_ops if e.get("tid") == tid),
                     key=lambda e: (float(e["ts"]), -float(e.get("dur", 0.0))))
        parent, stack = {}, []
        for e in ops:
            while stack and float(e["ts"]) >= float(stack[-1]["ts"]) + float(stack[-1].get("dur", 0.0)):
                stack.pop()
            parent[id(e)] = id(stack[-1]) if stack else None
            stack.append(e)
        cache[tid] = (ops, parent)
    ops, parent = cache[tid]
    p = parent[id(op)]
    return [e for e in ops if parent[id(e)] == p]


def _span(e):
    return float(e["ts"]), float(e["ts"]) + float(e.get("dur", 0.0))


def _items_in(tr, tid, t0, t1):
    return [k for k, it in enumerate(tr.items) if it["tid"] == tid and t0 <= it["t"] < t1]


def _nccl_of(tr, idx):
    for k in idx:
        it = tr.items[k]
        if it["kind"] == "launch" and tr.gpu[it["corr"]]["stream"] in tr.nccl_streams \
                and "nccl" in tr.gpu[it["corr"]]["name"].lower():
            return it["corr"]
    return None


def _elementwise_bandwidth(tr, kernels_bytes):
    """Bytes per microsecond of elementwise kernels with known traffic."""
    rates = [b / tr.gpu[c]["dur"] for c, b in kernels_bytes if tr.gpu[c]["dur"] > 0]
    if not rates:
        raise ValueError("no elementwise kernel to measure bandwidth")
    return st.median(rates)


def _launches(tr, idx):
    return [tr.items[k]["corr"] for k in idx if tr.items[k]["kind"] == "launch"]


class _Rewrite:
    """Accumulates region replacements, then builds the substituted trace."""

    def __init__(self, tr):
        self.tr = tr
        self.new = tr.clone()
        self.remove = set()
        self.insert = {}          # item index -> list of new items placed before it
        self.redirect = {}        # removed corr -> replacement corr
        self.next_corr = -1

    def op(self, stream, dur_us, name):
        c = self.next_corr
        self.next_corr -= 1
        self.new.gpu[c] = {"stream": stream, "ts": None, "dur": float(dur_us), "name": name, "args": {}, "cat": "kernel"}
        return c

    def replace(self, idx, new_items, delta_us, redirect):
        self.remove.update(idx)
        first = min(idx)
        self.insert[first] = new_items + [{"t": self.tr.items[first]["t"], "d": 0.0, "name": "shift", "kind": "shift",
                                           "delta": delta_us, "corr": None, "ext": None, "tid": None, "blocking": False}]
        self.redirect.update(redirect)

    def build(self):
        tr = self.tr
        items = []
        for k, it in enumerate(tr.items):
            if k in self.insert:
                items.extend(self.insert[k])
            if k not in self.remove:
                items.append(it)
        self.new.items = items
        deps = infer_dependencies(tr)
        new_deps = defaultdict(set)
        removed = {tr.items[k]["corr"] for k in self.remove if tr.items[k]["kind"] == "launch"}
        for c, ps in deps.items():
            if c in removed:
                continue
            for p in ps:
                q = self.redirect.get(p, None if p in removed else p)
                if q is not None:
                    new_deps[c].add(q)
        for c, ps in self.new.forced_deps.items():
            new_deps[c] |= ps
        self.new.deps = new_deps
        return self.new


def _new_launch(t, d, corr):
    return {"t": t, "d": d, "name": "cudaLaunchKernel", "kind": "launch", "corr": corr, "ext": None,
            "tid": None, "blocking": False}


def hook_buckets(tr):
    """Each bucket's int8-hook region on the thread that runs the hook:
    (first item index, last item index, n, scale all-gather, data all-gather,
    quantize kernels, dequantize end time)."""
    gathers = [e for e in tr.cpu_ops if e.get("name") == "c10d::allgather_"]
    if not gathers:
        return []
    tid = gathers[0].get("tid")
    tg = sorted((e for e in gathers if e.get("tid") == tid), key=lambda e: float(e["ts"]))
    buckets = []
    prev_end = -1e30
    i = 0
    while i + 1 < len(tg):
        s_op, d_op = tg[i], tg[i + 1]
        top = _siblings(tr, s_op)
        if not any(e is d_op for e in top):
            i += 1
            continue
        s_idx = _items_in(tr, tid, *_span(s_op))
        d_idx = _items_in(tr, tid, *_span(d_op))
        s_n, d_n = _nccl_of(tr, s_idx), _nccl_of(tr, d_idx)
        if s_n is None or d_n is None or int(tr.gpu[s_n]["args"].get("In msg nelems", 0)) != 2:
            i += 1
            continue
        n = int(tr.gpu[d_n]["args"].get("In msg nelems", 0))
        world = int(tr.gpu[d_n]["args"].get("Group size", 0)) or None
        starts = [e for e in top if e.get("name") == "aten::min" and prev_end <= float(e["ts"]) < float(s_op["ts"])]
        if not starts:
            raise ValueError("hook start (aten::min) not found before a scale all-gather")
        t_start = float(starts[-1]["ts"])
        between = [e for e in top if float(s_op["ts"]) < float(e["ts"]) < float(d_op["ts"])]
        quant = [e for e in between if e.get("name") in ("aten::div", "aten::round", "aten::add", "aten::clamp", "aten::to")]
        after = [e for e in top if float(e["ts"]) > float(d_op["ts"])]
        adds, t_end = 0, None
        for e in after:
            if e.get("name") in ("aten::add_", "aten::add"):
                adds += 1
            elif e.get("name") == "aten::div" and world and adds >= world:
                t_end = _span(e)[1]
                break
        if t_end is None:
            raise ValueError("hook end (final aten::div after %s adds) not found" % world)
        idx = _items_in(tr, tid, t_start, t_end)
        qk = []
        for e, mult in zip(quant, (8, 8, 8, 8, 5)):
            cs = _launches(tr, _items_in(tr, tid, *_span(e)))
            if cs:
                qk.append((cs[-1], mult * n))
        coll_cpu = sum(tr.items[k]["d"] for k in d_idx) - sum(
            tr.items[k]["d"] for k in d_idx if tr.items[k]["kind"] == "launch" and tr.gpu[tr.items[k]["corr"]]["cat"] == "gpu_memcpy")
        op_cpu = st.median([float(e.get("dur", 0.0)) for e in quant]) if quant else 15.0
        buckets.append({"idx": idx, "n": n, "world": world, "scale": s_n, "data": d_n, "quant": qk,
                        "t_start": t_start, "t_end": t_end, "coll_cpu": max(coll_cpu, 1.0), "op_cpu": op_cpu})
        prev_end = t_end
        i += 2
    return buckets


def substitute_hook_fp16(tr, alpha_s, beta_s_per_byte):
    """H: each int8-hook bucket becomes the fp16 hook (see module doc)."""
    buckets = hook_buckets(tr)
    if not buckets:
        raise ValueError("no int8-hook bucket found")
    big = max(buckets, key=lambda b: b["n"])
    bw = _elementwise_bandwidth(tr, big["quant"])
    compute = tr.gpu[big["quant"][0][0]]["stream"]
    nccl = tr.gpu[big["data"]]["stream"]
    side = max(g["stream"] for g in tr.gpu.values() if isinstance(g["stream"], int)) + 1000
    end_times = [tr.items[k]["t"] for k in iteration_ends(tr)]
    backs, last_end = defaultdict(list), {}
    removed_idx = set()
    rw = _Rewrite(tr)
    for b in buckets:
        n = b["n"]
        cast = rw.op(compute, 6.0 * n / bw, "fp16 cast (substituted)")
        div = rw.op(compute, 4.0 * n / bw, "fp16 divide (substituted)")
        ar = rw.op(nccl, 1e6 * (alpha_s + beta_s_per_byte * 2.0 * n), "ncclAllReduce fp16 (substituted)")
        back = rw.op(side, 6.0 * n / bw, "fp16 copy back (substituted)")
        rw.new.forced_deps[ar] = {div}
        rw.new.forced_deps[back] = {ar}
        it = bisect.bisect_left(end_times, b["t_end"])
        backs[it].append(back)
        last_end[it] = max(last_end.get(it, -1e30), b["t_end"])
        removed_idx.update(b["idx"])
        t = b["t_start"]
        oc = b["op_cpu"]
        new_items = [_new_launch(t, oc, cast), _new_launch(t + oc, oc, div),
                     _new_launch(t + 2 * oc, b["coll_cpu"], ar), _new_launch(t + 2 * oc + b["coll_cpu"], oc, back)]
        old_cpu = b["t_end"] - b["t_start"]
        new_cpu = 3 * oc + b["coll_cpu"]
        removed = _launches(tr, b["idx"])
        redirect = {c: (ar if tr.gpu[c]["stream"] == nccl else back) for c in removed}
        rw.replace(b["idx"], new_items, new_cpu - old_cpu, redirect)
    for it, t_last in last_end.items():
        k = next((k for k, x in enumerate(tr.items) if x["t"] > t_last and x["kind"] == "launch"
                  and k not in removed_idx and tr.gpu[x["corr"]]["stream"] == compute), None)
        if k is not None:
            rw.new.forced_deps.setdefault(tr.items[k]["corr"], set()).update(backs[it])
    return rw.build()


def gather_regions(tr):
    """G: every gradient all-gather (int8, the largest element count; one per
    iteration) with its item indices, the decode items to remove (int32
    zero-fill, per-source casts and adds), and the final int32-to-float cast."""
    found = []
    for g in (e for e in tr.cpu_ops if e.get("name") == "c10d::allgather_"):
        idx = _items_in(tr, g.get("tid"), *_span(g))
        nc = _nccl_of(tr, idx)
        if nc is not None:
            found.append((float(g["ts"]), g, idx, int(tr.gpu[nc]["args"].get("In msg nelems", 0)), nc))
    if not found:
        raise ValueError("no all-gather with a NCCL kernel")
    nmax = max(f[3] for f in found)
    regions = []
    for _, g, idx, n, nc in sorted((f for f in found if f[3] == nmax), key=lambda f: f[0]):
        world = int(tr.gpu[nc]["args"].get("Group size", 0)) or None
        after = [e for e in _siblings(tr, g) if float(e["ts"]) > float(g["ts"])]
        zi = next((i for i, e in enumerate(after) if e.get("name") == "aten::zeros"), None)
        if zi is None or world is None:
            raise ValueError("decode start (aten::zeros) or world size not found")
        remove_ops, adds, final_cast = [after[zi]], 0, None
        for e in after[zi + 1:]:
            name = e.get("name")
            if adds < world and name in ("aten::to", "aten::_to_copy", "aten::add_", "aten::add"):
                remove_ops.append(e)
                adds += name in ("aten::add_", "aten::add")
            elif adds >= world and name in ("aten::to", "aten::_to_copy"):
                final_cast = e
                break
        if adds < world or final_cast is None:
            raise ValueError("decode pattern not found (adds %d of %s)" % (adds, world))
        tid = g.get("tid")
        dec_idx = []
        for e in remove_ops:
            dec_idx.extend(_items_in(tr, tid, *_span(e)))
        regions.append({"gather": g, "gather_idx": idx, "n": n, "nccl": nc, "world": world, "decode_idx": dec_idx,
                        "decode_cpu": max(_span(e)[1] for e in remove_ops) - min(_span(e)[0] for e in remove_ops),
                        "final_cast": _launches(tr, _items_in(tr, tid, *_span(final_cast)))})
    return regions


def _shift_item(t, delta):
    return {"t": t, "d": 0.0, "name": "shift", "kind": "shift", "delta": delta, "corr": None, "ext": None,
            "tid": None, "blocking": False}


def substitute_gather_reduce(tr, alpha_s, beta_s_per_byte):
    """G: gather-sum becomes the reducible all-reduce (see module doc)."""
    regions = gather_regions(tr)
    deps = infer_dependencies(tr)
    rw = _Rewrite(tr)
    casts = []
    for r in regions:
        nccl = tr.gpu[r["nccl"]]["stream"]
        ar = rw.op(nccl, 1e6 * (alpha_s + beta_s_per_byte * r["n"]), "ncclAllReduce int8 (substituted)")
        rw.new.forced_deps[ar] = set(deps.get(r["nccl"], ()))  # the gather's waits
        gidx = r["gather_idx"]
        memcpy_cpu = sum(tr.items[k]["d"] for k in gidx if tr.items[k]["kind"] == "launch"
                         and tr.gpu[tr.items[k]["corr"]]["cat"] == "gpu_memcpy")
        new_gather = []
        for k in gidx:
            it = tr.items[k]
            if it["kind"] == "launch" and tr.gpu[it["corr"]]["cat"] == "gpu_memcpy":
                continue
            new_gather.append(_new_launch(it["t"], it["d"], ar) if it["corr"] == r["nccl"] else it)
        rw.remove.update(gidx)
        rw.insert[min(gidx)] = new_gather + [_shift_item(tr.items[min(gidx)]["t"], -memcpy_cpu)]
        rw.redirect.update({c: ar for c in _launches(tr, gidx)})
        dec = r["decode_idx"]
        rw.remove.update(dec)
        rw.insert[min(dec)] = [_shift_item(tr.items[min(dec)]["t"], -r["decode_cpu"])]
        rw.redirect.update({c: ar for c in _launches(tr, dec)})
        for c in r["final_cast"][:1]:
            rw.new.forced_deps.setdefault(c, set()).add(ar)
        casts.extend(r["final_cast"])
    out = rw.build()
    for c in casts:
        g = dict(out.gpu[c])
        g["dur"] = g["dur"] * 5.0 / 8.0
        out.gpu[c] = g
    return out


def predict(paths, family, alpha_s, beta_s_per_byte):
    """g_pred for one case from the original's traces (ranks 0 and 3)."""
    per_rank = []
    for p in paths:
        tr = load(p)
        base = replay(tr)
        lat = _traced_dependencies(tr, tr.deps)
        sub = substitute_hook_fp16 if family == "H" else substitute_gather_reduce
        cand = replay(sub(tr, alpha_s, beta_s_per_byte), latencies=lat)
        per_rank.append({"path": os.path.basename(p), "orig_ms": base.step_ms, "cand_ms": cand.step_ms,
                         "fidelity": base.fidelity, "busy_orig_ms": base.busy_ms, "busy_cand_ms": cand.busy_ms})
    orig = max(r["orig_ms"] for r in per_rank)
    cand = max(r["cand_ms"] for r in per_rank)
    return gain(orig, cand), {"orig_ms": orig, "cand_ms": cand, "ranks": per_rank}
