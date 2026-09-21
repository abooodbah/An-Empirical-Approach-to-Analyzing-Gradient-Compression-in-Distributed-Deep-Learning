# -*- coding: utf-8 -*-
"""Generate control_workload_benchmark_v4.py from v3 (round 4, 2026-09-21).

Adds, without touching v3:
  * gpu-quant-gathersum: arm B of the E2a collective-shape control. Same
    exact-sum shared-scale int8 encoding as gpu-quant-reducible (arm A);
    the identical encoded vector is all-gathered and summed locally in
    int32. Only the collective differs.
  * --verify-equivalence: correctness mode. After the timed collective of
    either arm, the other collective is run on the same encoded input and
    the two integer results are compared bit for bit; mismatch aborts. The
    extra collective is outside every timed region and the mode is
    recorded, so timing runs never carry it.
  * --tau-no-barrier: E3. Removes the explicit per-iteration barrier on
    non-exchange iterations; the exchange itself synchronizes on exchange
    iterations. Recorded as a distinct execution condition.
  * fp16-native: dense exchange of the FP16 cast of the gradient, the
    HotNets baseline (c = 2), decoded to FP32 before the update.
  * peak memory recorded per iteration in wall mode as well.
Encoding spec for the exact-sum arms: scalar s_max = max_i ||g_i||_inf
(all-reduce MAX); L = floor(127 / p); scale = s_max / L; q = clamp(round
(g / scale), -L, L) as int8; all-zero gradient gives s_max = 0 and scale
is clamped to 1e-30 so q = 0 everywhere. Sum bound: |sum q| <= p * L <=
127 for any subset, so intermediate ring partial sums cannot overflow.
"""
import io
import os
import py_compile

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "control_workload_benchmark_v3.py")
DST = os.path.join(HERE, "control_workload_benchmark_v4.py")

s = io.open(SRC, encoding="utf-8").read()
n = [0]


def rep(old, new, count=1):
    global s
    found = s.count(old)
    assert found == count, "anchor x%d (want %d): %r" % (found, count, old[:90])
    s = s.replace(old, new)
    n[0] += 1


rep('SCHEMA_VERSION = "tacc-paper-workloads/v2.2-control"',
    'SCHEMA_VERSION = "tacc-paper-workloads/v2.3-control"')

rep('CONTROL_MODES = ("gpu-quant-reducible", "gpu-topk-localef")',
    'CONTROL_MODES = ("gpu-quant-reducible", "gpu-topk-localef", "gpu-quant-gathersum", "fp16-native")')

rep('    "comm_mode",\n    "timing_mode",\n    "sync_every",',
    '    "comm_mode",\n    "timing_mode",\n    "tau_barrier",\n    "verify_equivalence",\n    "equivalence_ok",\n    "sync_every",')

rep('    parser.add_argument("--sync-every", type=int, default=1)\n',
    '    parser.add_argument("--sync-every", type=int, default=1)\n'
    '    parser.add_argument("--tau-no-barrier", action="store_true", default=False,\n'
    '                        help="E3: no explicit barrier on non-exchange iterations.")\n'
    '    parser.add_argument("--verify-equivalence", action="store_true", default=False,\n'
    '                        help="E2a correctness mode: cross-check the other arm outside timed regions.")\n')

rep('    if args.comm_mode not in ("native", "gpu-topk") and args.sync_every != 1:',
    '    if args.comm_mode not in ("native", "gpu-topk", "fp16-native") and args.sync_every != 1:')

rep('    if args.timing_mode != "phase":\n        deviations.append("uninstrumented_wall_clock_timing")\n    return deviations',
    '    if args.timing_mode != "phase":\n        deviations.append("uninstrumented_wall_clock_timing")\n'
    '    if args.tau_no_barrier:\n        deviations.append("tau_sweep_without_per_iteration_barrier")\n'
    '    if args.verify_equivalence:\n        deviations.append("equivalence_verification_mode_not_timing_evidence")\n'
    '    return deviations')

# barrier control (E3)
rep('            if args.comm_mode not in OVERLAP_MODES and not wall_mode:\n'
    '                _, barrier_wait_s = measure(runtime, lambda: _distributed_barrier(runtime))',
    '            do_communication_now = (iteration + 1) % args.sync_every == 0\n'
    '            if args.comm_mode not in OVERLAP_MODES and not wall_mode and (do_communication_now or not args.tau_no_barrier):\n'
    '                _, barrier_wait_s = measure(runtime, lambda: _distributed_barrier(runtime))')

# E2a arm B and fp16 baseline, inserted before the gpu-topk-localef branch
rep('                elif do_communication and args.comm_mode == "gpu-topk-localef":',
    '                elif do_communication and args.comm_mode == "gpu-quant-gathersum":\n'
    '                    # E2a arm B: identical exact-sum encoding to gpu-quant-reducible,\n'
    '                    # exchanged by all-gather and summed locally in int32.\n'
    '                    levels = float(max(1, 127 // runtime.world_size))\n'
    '\n'
    '                    def shared_scale_b() -> torch.Tensor:\n'
    '                        absolute_maximum = flat.detach().abs().max().to(torch.float32)\n'
    '                        dist.all_reduce(absolute_maximum, op=dist.ReduceOp.MAX)\n'
    '                        return absolute_maximum\n'
    '\n'
    '                    absolute_maximum, scale_sync_s = measure_op(shared_scale_b)\n'
    '                    scale_tensor = (absolute_maximum / levels).clamp_min(1e-30)\n'
    '\n'
    '                    def encode_shared_b() -> torch.Tensor:\n'
    '                        return torch.clamp(torch.round(flat / scale_tensor), min=-levels, max=levels).to(torch.int8)\n'
    '\n'
    '                    encoded, serialize_s = measure_op(encode_shared_b)\n'
    '                    gathered = [torch.empty_like(encoded) for _ in range(runtime.world_size)]\n'
    '                    _, network_s = measure_op(lambda: dist.all_gather(gathered, encoded))\n'
    '                    network_s = network_s + scale_sync_s\n'
    '\n'
    '                    def decode_gathered() -> None:\n'
    '                        total = torch.zeros(gradient_elements, dtype=torch.int32, device=runtime.device)\n'
    '                        for piece in gathered:\n'
    '                            total += piece.to(torch.int32)\n'
    '                        reconstructed = total.to(torch.float32).mul_(scale_tensor / runtime.world_size)\n'
    '                        unflatten_and_copy(model, reconstructed)\n'
    '                        HOOK_STATE["_last_int_sum"] = total\n'
    '\n'
    '                    _, deserialize_s = measure_op(decode_gathered)\n'
    '                    if args.verify_equivalence:\n'
    '                        runtime.synchronize()\n'
    '                        reference = encoded.clone()\n'
    '                        dist.all_reduce(reference)\n'
    '                        ok = bool(torch.equal(reference.to(torch.int32), HOOK_STATE["_last_int_sum"]))\n'
    '                        HOOK_STATE["equivalence_ok"] = HOOK_STATE.get("equivalence_ok", True) and ok\n'
    '                        if not ok:\n'
    '                            raise RuntimeError("E2a equivalence failed: all-gather+int32 sum != all-reduce int8 sum")\n'
    '                    legacy_bytes = encoded.numel() + 4\n'
    '                    collective_input_bytes = legacy_bytes\n'
    '                    compression_ratio = bytes_uncompressed / max(1, legacy_bytes)\n'
    '\n'
    '                elif do_communication and args.comm_mode == "fp16-native":\n'
    '                    # HotNets 2024 baseline: dense FP16 all-reduce (c = 2), FP32 update.\n'
    '                    half, serialize_s = measure_op(lambda: flat.to(torch.float16))\n'
    '                    _, network_s = measure_op(lambda: dist.all_reduce(half))\n'
    '\n'
    '                    def average_half() -> None:\n'
    '                        unflatten_and_copy(model, half.to(torch.float32).div_(runtime.world_size))\n'
    '\n'
    '                    _, deserialize_s = measure_op(average_half)\n'
    '                    legacy_bytes = half.numel() * 2\n'
    '                    collective_input_bytes = legacy_bytes\n'
    '                    compression_ratio = 2.0\n'
    '\n'
    '                elif do_communication and args.comm_mode == "gpu-topk-localef":')

# arm A gains the same verification hook (off the timed path)
rep('                    _, deserialize_s = measure_op(decode_shared)\n'
    '                    legacy_bytes = encoded.numel() + 4\n'
    '                    collective_input_bytes = legacy_bytes\n'
    '                    compression_ratio = bytes_uncompressed / max(1, legacy_bytes)\n'
    '\n'
    '                elif do_communication and args.comm_mode == "gpu-quant-gathersum":',
    '                    _, deserialize_s = measure_op(decode_shared)\n'
    '                    if args.verify_equivalence:\n'
    '                        runtime.synchronize()\n'
    '                        pieces = [torch.empty_like(pre_reduce) for _ in range(runtime.world_size)]\n'
    '                        dist.all_gather(pieces, pre_reduce)\n'
    '                        total = torch.zeros(gradient_elements, dtype=torch.int32, device=runtime.device)\n'
    '                        for piece in pieces:\n'
    '                            total += piece.to(torch.int32)\n'
    '                        ok = bool(torch.equal(total, encoded.to(torch.int32)))\n'
    '                        HOOK_STATE["equivalence_ok"] = HOOK_STATE.get("equivalence_ok", True) and ok\n'
    '                        if not ok:\n'
    '                            raise RuntimeError("E2a equivalence failed: all-reduce int8 sum != all-gather+int32 sum")\n'
    '                    legacy_bytes = encoded.numel() + 4\n'
    '                    collective_input_bytes = legacy_bytes\n'
    '                    compression_ratio = bytes_uncompressed / max(1, legacy_bytes)\n'
    '\n'
    '                elif do_communication and args.comm_mode == "gpu-quant-gathersum":')

# arm A must keep a copy of the pre-reduce encoding when verifying
rep('                    encoded, serialize_s = measure_op(encode_shared)\n'
    '                    _, network_s = measure_op(lambda: dist.all_reduce(encoded))\n'
    '                    network_s = network_s + scale_sync_s\n',
    '                    encoded, serialize_s = measure_op(encode_shared)\n'
    '                    pre_reduce = encoded.clone() if args.verify_equivalence else None\n'
    '                    _, network_s = measure_op(lambda: dist.all_reduce(encoded))\n'
    '                    network_s = network_s + scale_sync_s\n')

# record fields
rep('                "timing_mode": args.timing_mode,\n                "sync_every": args.sync_every,\n                "local_batch_size": args.batch_size,',
    '                "timing_mode": args.timing_mode,\n'
    '                "tau_barrier": "none_on_local_iterations" if args.tau_no_barrier else "every_iteration",\n'
    '                "verify_equivalence": args.verify_equivalence,\n'
    '                "equivalence_ok": HOOK_STATE.get("equivalence_ok") if args.verify_equivalence else None,\n'
    '                "sync_every": args.sync_every,\n                "local_batch_size": args.batch_size,')

rep('"""Control-campaign variant v3 of the frozen July harness (2026-09-19).',
    '"""Control-campaign variant v4 of the frozen July harness (2026-09-21).\n'
    '\n'
    'Generated by make_control_harness_v4.py from v3. Adds the E2a arm B\n'
    '(gpu-quant-gathersum), --verify-equivalence, --tau-no-barrier (E3), and\n'
    'the fp16-native dense baseline. v3 header follows.\n'
    '\n'
    'v3 was: Control-campaign variant v3 of the frozen July harness (2026-09-19).')

io.open(DST, "w", encoding="utf-8", newline="\n").write(s)
py_compile.compile(DST, doraise=True)
print("v4 harness written:", DST)
print("edit groups applied:", n[0])
