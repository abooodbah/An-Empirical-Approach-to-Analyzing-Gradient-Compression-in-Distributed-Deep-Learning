# validation_v2/harness/build_v52.py
# -*- coding: utf-8 -*-
"""Build harness v5.2 (2026-10-07) from v5.1 by insertion only: three
bucketed exchange modes, two stochastic shared-scale quantizers, a progress
file, a fidelity check, and a sidecar record. No v5.1 line is changed or
removed; test_harness_v52.py checks that."""
import hashlib
import io
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "control_workload_benchmark_v5_1.py")
DST = os.path.join(HERE, "control_workload_benchmark_v5_2.py")

MODES_OLD = '''COMM_MODES = LEGACY_MODES + ("ddp-overlap",) + CONTROL_MODES + HOOK_MODES
'''
MODES_NEW = MODES_OLD + '''# v5.2 (2026-10-07): bucketed exchanges and stochastic shared-scale quantizers.
BUCKETED_MODES = ("native-bucketed", "gpu-quant-bucketed", "gpu-quant-reducible-bucketed")
STOCHASTIC_MODES = ("gpu-quant-sr-int8", "gpu-quant-sr-fp16")
CONTROL_MODES = CONTROL_MODES + BUCKETED_MODES + STOCHASTIC_MODES
COMM_MODES = LEGACY_MODES + ("ddp-overlap",) + CONTROL_MODES + HOOK_MODES
'''

HELPERS_OLD = '''    signed = encoded.to(torch.int16) - 128
    return signed.to(torch.float32) * scale.reshape(1)[0]
'''
HELPERS_NEW = HELPERS_OLD + '''

def bucket_bounds(n: int, bucket_elems: int) -> List[Tuple[int, int]]:
    """Half-open slices that cover n elements in order; the last may be short."""
    if bucket_elems < 1:
        raise ConfigurationError("--exchange-bucket-elems must be positive for a bucketed mode")
    return [(start, min(n, start + bucket_elems)) for start in range(0, n, bucket_elems)]


def exchange_buffer_bytes(mode: str, world_size: int, bucket_elems: int) -> int:
    """Bytes of exchange buffers this harness holds at once in a bucketed
    mode, beyond the flat gradient, its encoded copy, and the FP32 result,
    which the whole-gradient modes hold too. The backend may hold more
    inside a collective; the measured peak is recorded beside this."""
    if mode == "gpu-quant-bucketed":
        return world_size * bucket_elems + 4 * bucket_elems
    if mode in ("gpu-quant-reducible-bucketed", "native-bucketed"):
        return 0
    raise ConfigurationError(f"{mode} is not a bucketed mode")


def bucket_buffers(runtime: Any, bucket_elems: int, n: int, received_count: int) -> Dict[str, Any]:
    """Buffers of the bucketed gather, allocated once per run and reused for
    every bucket and iteration."""
    state = HOOK_STATE.get("_bucket_buffers")
    width = min(bucket_elems, n)
    if state is None or state["result"].numel() != n or state["scratch"].numel() != width:
        state = {
            "received": [torch.empty(width, dtype=torch.uint8, device=runtime.device)
                         for _ in range(received_count)],
            "scratch": torch.empty(width, dtype=torch.float32, device=runtime.device),
            "result": torch.empty(n, dtype=torch.float32, device=runtime.device),
        }
        HOOK_STATE["_bucket_buffers"] = state
    return state


def levels_int8(world_size: int) -> int:
    """Levels per side whose exact sum over the ranks fits a signed byte."""
    return max(1, 127 // world_size)


def levels_fp16(world_size: int) -> int:
    """Levels per side whose exact sum fits float16, exact to 2048."""
    return 2048 // world_size


def stochastic_generator(runtime: Any, rank_seed: int) -> Optional[Any]:
    """One generator per run and rank, seeded apart from the model and data
    streams. A backend without device generators uses the global stream,
    which seed_everything already seeds per rank."""
    if "_stochastic_generator" not in HOOK_STATE:
        try:
            generator = torch.Generator(device=runtime.device)
            generator.manual_seed(rank_seed + 7919)
        except (RuntimeError, TypeError):
            generator = None
        HOOK_STATE["_stochastic_generator"] = generator
    return HOOK_STATE["_stochastic_generator"]


def stochastic_levels(values: Any, scale: Any, levels: float, generator: Optional[Any]) -> Any:
    """floor(values / scale + u), u uniform on [0, 1), clamped to the level
    range: an unbiased shared-scale quantizer."""
    noise = torch.rand(values.shape, generator=generator, device=values.device, dtype=torch.float32)
    return torch.clamp(torch.floor(values / scale + noise), min=-levels, max=levels)


def write_v52_sidecar(args: argparse.Namespace, runtime: Any, prefix: str, elements: int,
                      peak_memory_mb: Optional[float]) -> None:
    bucketed = args.comm_mode in BUCKETED_MODES
    if args.comm_mode == "gpu-quant-sr-fp16":
        levels: Optional[int] = levels_fp16(runtime.world_size)
    elif args.comm_mode in ("gpu-quant-sr-int8", "gpu-quant-reducible", "gpu-quant-reducible-bucketed",
                            "gpu-quant-gathersum"):
        levels = levels_int8(runtime.world_size)
    else:
        levels = None
    payload = {
        "harness": "v5.2",
        "run_id": args.run_id,
        "comm_mode": args.comm_mode,
        "world_size": runtime.world_size,
        "rank": runtime.rank,
        "gradient_elements": elements,
        "exchange_bucket_elems": args.exchange_bucket_elems if bucketed else None,
        "bucket_count": len(bucket_bounds(elements, args.exchange_bucket_elems)) if bucketed else None,
        "exchange_buffer_bytes": (exchange_buffer_bytes(args.comm_mode, runtime.world_size,
                                                        min(args.exchange_bucket_elems, elements))
                                  if bucketed else None),
        "levels_per_side": levels,
        "peak_memory_mb": peak_memory_mb,
        "fidelity": HOOK_STATE.get("fidelity", []),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(str(args.output_dir / f"{prefix}.v52.json"), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=1, sort_keys=True)
'''

ARGS_OLD = '''    parser.add_argument("--powersgd-rank", type=int, default=1)
'''
ARGS_NEW = ARGS_OLD + '''    parser.add_argument("--exchange-bucket-elems", type=int, default=0,
                        help="Elements per bucket for the bucketed modes (0: not bucketed).")
    parser.add_argument("--progress-dir", type=Path, default=None,
                        help="Write one JSON line per measured iteration per rank (phase and wall timing).")
    parser.add_argument("--verify-fidelity", action="store_true", default=False,
                        help="Compare the exchanged gradient with the dense average (qualification only).")
'''

VALIDATE_DEF_OLD = '''def validate_config(args: argparse.Namespace) -> argparse.Namespace:
'''
VALIDATE_DEF_NEW = '''def validate_v52(args: argparse.Namespace) -> None:
    bucketed = args.comm_mode in BUCKETED_MODES
    if bucketed and args.exchange_bucket_elems < 1:
        raise ConfigurationError(f"{args.comm_mode} requires --exchange-bucket-elems")
    if not bucketed and args.exchange_bucket_elems:
        raise ConfigurationError("--exchange-bucket-elems applies to the bucketed modes only")
    if args.progress_dir is not None and args.timing_mode == "block":
        raise ConfigurationError("--progress-dir would write inside a block-timed block")
    if args.verify_fidelity and args.profile != "qualification":
        raise ConfigurationError("--verify-fidelity is for qualification runs only")
    if args.verify_fidelity and args.comm_mode in DDP_MODES:
        raise ConfigurationError("--verify-fidelity needs the gradient before the exchange; DDP modes hide it")
    if args.comm_mode in STOCHASTIC_MODES and args.profile == "paper":
        raise ConfigurationError("the stochastic quantizers are not paper-profile conditions")


''' + VALIDATE_DEF_OLD

VALIDATE_CALL_OLD = '''                                     + "; ".join(mismatches))
    return args
'''
VALIDATE_CALL_NEW = '''                                     + "; ".join(mismatches))
    validate_v52(args)
    return args
'''

SETUP_OLD = '''        # Outer block timer, identical in every timing mode: align the ranks,
        # finish pending device work, time the block, synchronize at its end.
'''
SETUP_NEW = '''        # v5.2: allocate the bucketed gather's buffers and open the progress
        # file before the measured block.
        if args.comm_mode == "gpu-quant-bucketed":
            bucket_buffers(runtime, args.exchange_bucket_elems, parameter_count, runtime.world_size)
        progress_handle = None
        if args.progress_dir is not None:
            args.progress_dir.mkdir(parents=True, exist_ok=True)
            progress_handle = open(str(args.progress_dir / f"{prefix}.progress.jsonl"), "w", encoding="utf-8")
        flat_reference: Optional[torch.Tensor] = None
''' + SETUP_OLD

REFERENCE_OLD = '''                gradient_elements = int(flat.numel())
                bytes_uncompressed = gradient_elements * 4
'''
REFERENCE_NEW = REFERENCE_OLD + '''                flat_reference = flat.detach().clone() if args.verify_fidelity else None
'''

BRANCH_OLD = '''                    compression_ratio = bytes_uncompressed / max(1, legacy_bytes)

            _, optimizer_step_s = measure_op(optimizer.step)
'''
BRANCH_NEW = '''                    compression_ratio = bytes_uncompressed / max(1, legacy_bytes)

                elif do_communication and args.comm_mode == "native-bucketed":
                    # Dense FP32 reduced bucket by bucket in place: the dense
                    # arm with the bucketed arms' collective boundaries.
                    network_s = 0.0
                    for start, stop in bucket_bounds(gradient_elements, args.exchange_bucket_elems):
                        piece = flat[start:stop]
                        _, elapsed = measure_op(lambda: dist.all_reduce(piece))
                        network_s += elapsed

                    def average_and_copy_bucketed() -> None:
                        flat.div_(runtime.world_size)
                        unflatten_and_copy(model, flat)

                    _, deserialize_s = measure_op(average_and_copy_bucketed)
                    legacy_bytes = bytes_uncompressed
                    collective_input_bytes = bytes_uncompressed
                    compression_ratio = 1.0

                elif do_communication and args.comm_mode == "gpu-quant-bucketed":
                    # The gpu-quant encoding (one rank-local scale over the
                    # whole gradient) exchanged bucket by bucket. At most
                    # world_size received buckets are alive at once, in
                    # buffers allocated once and reused.
                    (scale, encoded), serialize_s = measure_op(lambda: gpu_quantize(flat))
                    buffers = bucket_buffers(runtime, args.exchange_bucket_elems, gradient_elements,
                                             runtime.world_size)
                    scales = [torch.empty_like(scale) for _ in range(runtime.world_size)]
                    _, network_s = measure_op(lambda: dist.all_gather(scales, scale))
                    result = buffers["result"]
                    _, deserialize_s = measure_op(result.zero_)
                    for start, stop in bucket_bounds(gradient_elements, args.exchange_bucket_elems):
                        width = stop - start
                        received = [buffer[:width] for buffer in buffers["received"]]
                        piece = encoded[start:stop]
                        _, elapsed = measure_op(lambda: dist.all_gather(received, piece))
                        network_s += elapsed

                        def decode_bucket() -> None:
                            scratch = buffers["scratch"][:width]
                            target = result[start:stop]
                            for remote_scale, payload in zip(scales, received):
                                scratch.copy_(payload)
                                scratch.sub_(128.0)
                                scratch.mul_(remote_scale.reshape(1)[0])
                                target.add_(scratch)

                        _, elapsed = measure_op(decode_bucket)
                        deserialize_s += elapsed

                    def finish_bucketed() -> None:
                        result.div_(runtime.world_size)
                        unflatten_and_copy(model, result)

                    _, elapsed = measure_op(finish_bucketed)
                    deserialize_s += elapsed
                    if args.verify_equivalence:
                        runtime.synchronize()
                        whole_scales, whole_payloads = _collect_gpu_quant(scale, encoded, runtime)
                        reference = torch.zeros_like(flat)
                        for remote_scale, remote_payload in zip(whole_scales, whole_payloads):
                            reference += gpu_dequantize(remote_scale, remote_payload, gradient_elements)
                        reference /= runtime.world_size
                        ok = bool(torch.equal(reference, result))
                        HOOK_STATE["equivalence_ok"] = HOOK_STATE.get("equivalence_ok", True) and ok
                        if not ok:
                            raise RuntimeError("bucketed gather differs from the whole-gradient gather")
                    legacy_bytes = encoded.numel() + 4
                    collective_input_bytes = legacy_bytes
                    compression_ratio = bytes_uncompressed / max(1, legacy_bytes)

                elif do_communication and args.comm_mode == "gpu-quant-reducible-bucketed":
                    # The gpu-quant-reducible encoding (shared scale, exact
                    # int8 sum) reduced bucket by bucket.
                    levels = float(levels_int8(runtime.world_size))

                    def shared_scale_c() -> torch.Tensor:
                        absolute_maximum = flat.detach().abs().max().to(torch.float32)
                        dist.all_reduce(absolute_maximum, op=dist.ReduceOp.MAX)
                        return absolute_maximum

                    absolute_maximum, scale_sync_s = measure_op(shared_scale_c)
                    scale_tensor = (absolute_maximum / levels).clamp_min(1e-30)

                    def encode_shared_c() -> torch.Tensor:
                        return torch.clamp(torch.round(flat / scale_tensor), min=-levels, max=levels).to(torch.int8)

                    encoded, serialize_s = measure_op(encode_shared_c)
                    pre_reduce = encoded.clone() if args.verify_equivalence else None
                    network_s = scale_sync_s
                    for start, stop in bucket_bounds(gradient_elements, args.exchange_bucket_elems):
                        piece = encoded[start:stop]
                        _, elapsed = measure_op(lambda: dist.all_reduce(piece))
                        network_s += elapsed

                    def decode_shared_c() -> None:
                        reconstructed = encoded.to(torch.float32)
                        reconstructed.mul_(scale_tensor / runtime.world_size)
                        unflatten_and_copy(model, reconstructed)

                    _, deserialize_s = measure_op(decode_shared_c)
                    if args.verify_equivalence:
                        runtime.synchronize()
                        dist.all_reduce(pre_reduce)
                        ok = bool(torch.equal(pre_reduce, encoded))
                        HOOK_STATE["equivalence_ok"] = HOOK_STATE.get("equivalence_ok", True) and ok
                        if not ok:
                            raise RuntimeError("bucketed int8 all-reduce differs from the whole all-reduce")
                    legacy_bytes = encoded.numel() + 4
                    collective_input_bytes = legacy_bytes
                    compression_ratio = bytes_uncompressed / max(1, legacy_bytes)

                elif do_communication and args.comm_mode in STOCHASTIC_MODES:
                    # Shared-scale stochastic quantizer, all-reduce compatible.
                    # sr-int8: a signed byte on the wire. sr-fp16: integers
                    # carried in float16, exact to 2048.
                    fp16_wire = args.comm_mode == "gpu-quant-sr-fp16"
                    levels = float(levels_fp16(runtime.world_size) if fp16_wire
                                   else levels_int8(runtime.world_size))
                    wire_dtype = torch.float16 if fp16_wire else torch.int8

                    def shared_scale_s() -> torch.Tensor:
                        absolute_maximum = flat.detach().abs().max().to(torch.float32)
                        dist.all_reduce(absolute_maximum, op=dist.ReduceOp.MAX)
                        return absolute_maximum

                    absolute_maximum, scale_sync_s = measure_op(shared_scale_s)
                    scale_tensor = (absolute_maximum / levels).clamp_min(1e-30)
                    generator = stochastic_generator(runtime, rank_seed)

                    def encode_stochastic() -> torch.Tensor:
                        return stochastic_levels(flat, scale_tensor, levels, generator).to(wire_dtype)

                    encoded, serialize_s = measure_op(encode_stochastic)
                    pre_reduce = encoded.clone() if args.verify_equivalence else None
                    _, network_s = measure_op(lambda: dist.all_reduce(encoded))
                    network_s = network_s + scale_sync_s

                    def decode_stochastic() -> None:
                        reconstructed = encoded.to(torch.float32)
                        reconstructed.mul_(scale_tensor / runtime.world_size)
                        unflatten_and_copy(model, reconstructed)

                    _, deserialize_s = measure_op(decode_stochastic)
                    if args.verify_equivalence:
                        runtime.synchronize()
                        pieces = [torch.empty_like(pre_reduce) for _ in range(runtime.world_size)]
                        dist.all_gather(pieces, pre_reduce)
                        total = torch.zeros(gradient_elements, dtype=torch.int32, device=runtime.device)
                        for piece in pieces:
                            total += piece.to(torch.int32)
                        ok = bool(torch.equal(total, encoded.to(torch.int32)))
                        HOOK_STATE["equivalence_ok"] = HOOK_STATE.get("equivalence_ok", True) and ok
                        if not ok:
                            raise RuntimeError("stochastic quantizer: the wire sum differs from the int32 sum")
                    legacy_bytes = encoded.numel() * (2 if fp16_wire else 1) + 4
                    collective_input_bytes = legacy_bytes
                    compression_ratio = bytes_uncompressed / max(1, legacy_bytes)

            if args.verify_fidelity and do_communication and flat_reference is not None:
                # Outside every timed phase: the exchanged gradient against
                # the dense average of the same local gradients.
                runtime.synchronize()
                dist.all_reduce(flat_reference)
                flat_reference.div_(runtime.world_size)
                exchanged = flatten_grads(model, runtime.device)
                dense_norm = float(flat_reference.norm().item())
                exchanged_norm = float(exchanged.norm().item())
                HOOK_STATE.setdefault("fidelity", []).append({
                    "iteration": iteration,
                    "relative_l2_error": float((exchanged - flat_reference).norm().item()) / max(1e-30, dense_norm),
                    "cosine": float(torch.dot(exchanged, flat_reference).item())
                    / max(1e-30, dense_norm * exchanged_norm),
                })
                runtime.synchronize()

            _, optimizer_step_s = measure_op(optimizer.step)
'''

PROGRESS_OLD = '''                "compression_ratio": compression_ratio,
            })

        in_block_syncs = SYNC_COUNTER[0] - syncs_before_block
'''
PROGRESS_NEW = '''                "compression_ratio": compression_ratio,
            })
            if progress_handle is not None:
                progress_handle.write(json.dumps({
                    "iteration": iteration, "observed_step_s": observed_step_s,
                    "serialize_s": serialize_s, "network_s": network_s, "deserialize_s": deserialize_s,
                    "peak_memory_mb": runtime.peak_memory_mb()}) + "\\n")
                progress_handle.flush()

        in_block_syncs = SYNC_COUNTER[0] - syncs_before_block
'''

FINAL_OLD = '''            records.append(record)

        final_hash = model_sha256(model)
'''
FINAL_NEW = '''            records.append(record)

        if progress_handle is not None:
            progress_handle.close()
        if args.comm_mode in BUCKETED_MODES + STOCHASTIC_MODES or args.verify_fidelity:
            write_v52_sidecar(args, runtime, prefix, parameter_count, peak_memory_block_mb)
            HOOK_STATE.pop("_bucket_buffers", None)
            release = getattr(torch, runtime.kind, None) if runtime.kind in ("cuda", "xpu") else None
            if release is not None and hasattr(release, "empty_cache"):
                release.empty_cache()
        final_hash = model_sha256(model)
'''

SUBS = [(MODES_OLD, MODES_NEW), (HELPERS_OLD, HELPERS_NEW), (ARGS_OLD, ARGS_NEW),
        (VALIDATE_DEF_OLD, VALIDATE_DEF_NEW), (VALIDATE_CALL_OLD, VALIDATE_CALL_NEW),
        (SETUP_OLD, SETUP_NEW), (REFERENCE_OLD, REFERENCE_NEW), (BRANCH_OLD, BRANCH_NEW),
        (PROGRESS_OLD, PROGRESS_NEW), (FINAL_OLD, FINAL_NEW)]


def keeps_every_line(old, new):
    remaining = iter(new.split("\n"))
    return all(line in remaining for line in old.split("\n"))


def build():
    text = io.open(SRC, encoding="utf-8").read()
    for index, (old, new) in enumerate(SUBS):
        if text.count(old) != 1:
            raise SystemExit("anchor %d occurs %d times in v5.1" % (index, text.count(old)))
        if not keeps_every_line(old, new):
            raise SystemExit("replacement %d removes a v5.1 line" % index)
        text = text.replace(old, new)
    io.open(DST, "w", encoding="utf-8", newline="\n").write(text)
    digest = hashlib.sha256(open(DST, "rb").read()).hexdigest()
    io.open(os.path.join(os.path.dirname(HERE), "CODE_v52.sha256"), "w", encoding="utf-8", newline="\n").write(
        "%s  harness/control_workload_benchmark_v5_2.py\n" % digest)
    return digest


if __name__ == "__main__":
    print(build())
