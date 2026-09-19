# -*- coding: utf-8 -*-
"""Generate control_workload_benchmark_v3.py from the v2 control harness.

Adds, without touching v2 (whose hash the September records cite):
  * llama1b workload: LLaMA-style decoder (18 layers, d 2048, 16 heads,
    SwiGLU 5504, vocab 32000, seq 2048; about 1.05e9 parameters);
  * ddp-fp16-hook: PyTorch's official fp16_compress_hook (reducible, c=2)
    under bucketed DDP overlap;
  * ddp-int8-hook: PyTorch's quantization_pertensor_hook algorithm ported
    with device-agnostic placement (rank-local int8 over all-gather) under
    bucketed DDP overlap;
  * profile "extension" for non-legacy conditions.
"""
import io
import os
import py_compile

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "control_workload_benchmark.py")
DST = os.path.join(HERE, "control_workload_benchmark_v3.py")

s = io.open(SRC, encoding="utf-8").read()
n = [0]


def rep(old, new, count=1):
    global s
    found = s.count(old)
    assert found == count, "anchor x%d (want %d): %r" % (found, count, old[:90])
    s = s.replace(old, new)
    n[0] += 1


rep('SCHEMA_VERSION = "tacc-paper-workloads/v2.1-control"',
    'SCHEMA_VERSION = "tacc-paper-workloads/v2.2-control"')

rep('MODELS = ("resnet50", "transformer")',
    'MODELS = ("resnet50", "transformer", "llama1b")')

rep('CONTROL_MODES = ("gpu-quant-reducible", "gpu-topk-localef")\n'
    'COMM_MODES = LEGACY_MODES + ("ddp-overlap",) + CONTROL_MODES',
    'CONTROL_MODES = ("gpu-quant-reducible", "gpu-topk-localef")\n'
    'HOOK_MODES = ("ddp-int8-hook", "ddp-fp16-hook")\n'
    'OVERLAP_MODES = ("ddp-overlap",) + HOOK_MODES\n'
    'DDP_MODES = ("powersgd",) + OVERLAP_MODES\n'
    'COMM_MODES = LEGACY_MODES + ("ddp-overlap",) + CONTROL_MODES + HOOK_MODES\n'
    'HOOK_STATE: Dict[str, Any] = {}')

rep('    parser.add_argument("--profile", choices=("paper", "qualification"), default="paper")',
    '    parser.add_argument("--profile", choices=("paper", "qualification", "extension"), default="paper")')

rep('    if args.profile != "paper":\n'
    '        deviations.append("qualification_profile_not_paper_evidence")',
    '    if args.profile != "paper":\n'
    '        deviations.append(f"{args.profile}_profile_not_paper_evidence")')

rep('    if args.comm_mode == "ddp-overlap":\n'
    '        deviations.append("supplemental_overlap_baseline_not_in_legacy_matrix")',
    '    if args.comm_mode in OVERLAP_MODES:\n'
    '        deviations.append("supplemental_overlap_baseline_not_in_legacy_matrix")\n'
    '    if args.comm_mode in HOOK_MODES:\n'
    '        deviations.append(f"production_comm_hook_{args.comm_mode}")')

rep('    if comm_mode == "ddp-overlap":\n'
    '        return (\n'
    '            "supplemental_overlap_baseline",',
    '    if comm_mode in OVERLAP_MODES:\n'
    '        return (\n'
    '            "supplemental_overlap_baseline",')

# --- LLaMA-style workload ------------------------------------------------------
rep('def build_workload(\n'
    '    model_name: str,',
    'class RMSNorm(nn.Module):\n'
    '    def __init__(self, dim: int, eps: float = 1e-5):\n'
    '        super().__init__()\n'
    '        self.weight = nn.Parameter(torch.ones(dim))\n'
    '        self.eps = eps\n'
    '\n'
    '    def forward(self, x: torch.Tensor) -> torch.Tensor:\n'
    '        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight\n'
    '\n'
    '\n'
    'class LlamaBlock(nn.Module):\n'
    '    def __init__(self, d_model: int, n_heads: int, ffn: int):\n'
    '        super().__init__()\n'
    '        self.norm1 = RMSNorm(d_model)\n'
    '        self.norm2 = RMSNorm(d_model)\n'
    '        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)\n'
    '        self.out = nn.Linear(d_model, d_model, bias=False)\n'
    '        self.w1 = nn.Linear(d_model, ffn, bias=False)\n'
    '        self.w3 = nn.Linear(d_model, ffn, bias=False)\n'
    '        self.w2 = nn.Linear(ffn, d_model, bias=False)\n'
    '        self.n_heads = n_heads\n'
    '\n'
    '    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:\n'
    '        batch, length, width = x.shape\n'
    '        head_dim = width // self.n_heads\n'
    '        hidden = self.norm1(x)\n'
    '        q, k, v = self.qkv(hidden).view(batch, length, 3, self.n_heads, head_dim).unbind(2)\n'
    '        q = q.transpose(1, 2)\n'
    '        k = k.transpose(1, 2)\n'
    '        v = v.transpose(1, 2)\n'
    '        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)\n'
    '        scores = scores.masked_fill(mask, float("-inf"))\n'
    '        attention = torch.softmax(scores, dim=-1)\n'
    '        mixed = torch.matmul(attention, v).transpose(1, 2).reshape(batch, length, width)\n'
    '        x = x + self.out(mixed)\n'
    '        hidden = self.norm2(x)\n'
    '        return x + self.w2(torch.nn.functional.silu(self.w1(hidden)) * self.w3(hidden))\n'
    '\n'
    '\n'
    'class LlamaStyleLM(nn.Module):\n'
    '    """Decoder-only Transformer in the LLaMA family: RMSNorm, SwiGLU, causal\n'
    '    attention, learned positions. Attention is written out explicitly so the\n'
    '    same code runs on the torch 1.13 XPU stack and the torch 2.4 CUDA stack."""\n'
    '\n'
    '    def __init__(self, d_model: int = 2048, n_layers: int = 18, n_heads: int = 16,\n'
    '                 ffn: int = 5504, vocab: int = 32000, seq: int = 2048):\n'
    '        super().__init__()\n'
    '        self.tok = nn.Embedding(vocab, d_model)\n'
    '        self.pos = nn.Embedding(seq, d_model)\n'
    '        self.blocks = nn.ModuleList([LlamaBlock(d_model, n_heads, ffn) for _ in range(n_layers)])\n'
    '        self.norm = RMSNorm(d_model)\n'
    '        self.head = nn.Linear(d_model, vocab, bias=False)\n'
    '\n'
    '    def forward(self, tokens: torch.Tensor) -> torch.Tensor:\n'
    '        batch, length = tokens.shape\n'
    '        positions = torch.arange(length, device=tokens.device)\n'
    '        x = self.tok(tokens) + self.pos(positions)\n'
    '        mask = torch.triu(torch.ones(length, length, dtype=torch.bool, device=tokens.device), diagonal=1)\n'
    '        for block in self.blocks:\n'
    '            x = block(x, mask)\n'
    '        logits = self.head(self.norm(x))\n'
    '        return logits.transpose(1, 2)\n'
    '\n'
    '\n'
    'def build_workload(\n'
    '    model_name: str,')

rep('            "data": "fresh synthetic random tensors per rank and iteration",\n'
    '        }\n'
    '    else:  # pragma: no cover - argparse constrains this',
    '            "data": "fresh synthetic random tensors per rank and iteration",\n'
    '        }\n'
    '    elif model_name == "llama1b":\n'
    '        model = LlamaStyleLM()\n'
    '\n'
    '        def get_data() -> Tuple[torch.Tensor, torch.Tensor]:\n'
    '            return (\n'
    '                torch.randint(0, 32000, (batch_size, 2048), dtype=torch.int64, device=device),\n'
    '                torch.randint(0, 32000, (batch_size, 2048), dtype=torch.int64, device=device),\n'
    '            )\n'
    '\n'
    '        spec = {\n'
    '            "architecture": "LLaMA-style decoder: 18 layers, d_model 2048, 16 heads, "\n'
    '                            "SwiGLU ffn 5504, vocab 32000, seq 2048, learned positions, "\n'
    '                            "explicit causal attention, next-token cross-entropy",\n'
    '            "input": [batch_size, 2048],\n'
    '            "vocabulary": 32000,\n'
    '            "parameters_approx": 1.046e9,\n'
    '            "data": "fresh synthetic random tokens per rank and iteration",\n'
    '        }\n'
    '    else:  # pragma: no cover - argparse constrains this')

# --- production hooks ---------------------------------------------------------
rep('def configure_model_for_mode(',
    'def _gather_future(outputs: List[torch.Tensor], value: torch.Tensor, group: Any) -> Any:\n'
    '    """all_gather as a future; falls back to a completed future if the backend\n'
    '    does not expose futures for all_gather (recorded in HOOK_STATE)."""\n'
    '    if HOOK_STATE.get("gather_futures") != "unsupported":\n'
    '        try:\n'
    '            fut = dist.all_gather(outputs, value, group=group, async_op=True).get_future()\n'
    '            HOOK_STATE["gather_futures"] = "async"\n'
    '            return fut\n'
    '        except Exception as exc:  # pragma: no cover - backend dependent\n'
    '            HOOK_STATE["gather_futures"] = "unsupported"\n'
    '            HOOK_STATE["gather_futures_error"] = repr(exc)[:200]\n'
    '    dist.all_gather(outputs, value, group=group)\n'
    '    done = torch.futures.Future()\n'
    '    done.set_result([outputs])\n'
    '    return done\n'
    '\n'
    '\n'
    'def _half_compress_hook(process_group, bucket):  # unannotated: DDP checks hook annotations\n'
    '    """Port of torch.distributed.algorithms.ddp_comm_hooks.default_hooks.\n'
    '    fp16_compress_hook with a selectable 16-bit dtype (HOOK_STATE["half_dtype"]).\n'
    '    Algorithm unchanged: cast the bucket to 16 bits, divide by world size,\n'
    '    all-reduce, copy back into the fp32 bucket."""\n'
    '    group_to_use = process_group if process_group is not None else dist.group.WORLD\n'
    '    world_size = group_to_use.size()\n'
    '    dtype = getattr(torch, HOOK_STATE.get("half_dtype", "float16"))\n'
    '    buffer = bucket.buffer()\n'
    '    compressed = buffer.to(dtype).div_(world_size)\n'
    '    fut = dist.all_reduce(compressed, group=group_to_use, async_op=True).get_future()\n'
    '\n'
    '    def decompress(fut):\n'
    '        value = fut.value()\n'
    '        reduced = value[0] if isinstance(value, list) else value\n'
    '        buffer.copy_(reduced)\n'
    '        return buffer\n'
    '\n'
    '    return fut.then(decompress)\n'
    '\n'
    '\n'
    'def _pertensor_int8_hook(process_group, bucket):  # unannotated: DDP checks hook annotations\n'
    '    """Port of torch.distributed.algorithms.ddp_comm_hooks.quantization_hooks.\n'
    '    quantization_pertensor_hook with device-agnostic tensor placement (the\n'
    '    upstream hook hardcodes .cuda). Algorithm unchanged: per-bucket min/max\n'
    '    uint8 parameters, all-gather of the parameters, all-gather of the\n'
    '    quantized buckets, per-source dequantization, mean over ranks."""\n'
    '    group_to_use = process_group if process_group is not None else dist.group.WORLD\n'
    '    world_size = group_to_use.size()\n'
    '    tensor = bucket.buffer()\n'
    '    device = tensor.device\n'
    '    if "mode" not in HOOK_STATE:\n'
    '        # oneCCL does not run nested future callbacks reliably; on that\n'
    '        # backend the hook performs its collectives synchronously and the\n'
    '        # provenance records it.\n'
    '        backend = str(dist.get_backend(group_to_use)).lower()\n'
    '        HOOK_STATE["mode"] = "sync" if "ccl" in backend or os.environ.get("RED_HOOK_SYNC") == "1" else "async"\n'
    '        HOOK_STATE["backend"] = backend\n'
    '    if HOOK_STATE["mode"] == "sync":\n'
    '        zero = torch.zeros((), device=device, dtype=tensor.dtype)\n'
    '        min_val = torch.minimum(tensor.min(), zero)\n'
    '        max_val = torch.maximum(tensor.max(), zero)\n'
    '        scale = (max_val - min_val) / 255.0\n'
    '        scale = torch.where(scale > 0, scale, torch.full_like(scale, 1e-8))\n'
    '        zero_point = torch.clamp(torch.round(-min_val / scale), 0.0, 255.0)\n'
    '        s_and_z = torch.stack([scale, zero_point]).to(torch.float32)\n'
    '        gathered_s_and_z = [torch.zeros(2, dtype=torch.float32, device=device) for _ in range(world_size)]\n'
    '        dist.all_gather(gathered_s_and_z, s_and_z, group=group_to_use)\n'
    '        quantized = torch.clamp(torch.round(tensor / scale) + zero_point, 0.0, 255.0).to(torch.uint8)\n'
    '        all_ranks_quantized = [torch.zeros_like(quantized) for _ in range(world_size)]\n'
    '        dist.all_gather(all_ranks_quantized, quantized, group=group_to_use)\n'
    '        aggregated = torch.zeros_like(tensor, dtype=torch.float32)\n'
    '        for source, source_quantized in enumerate(all_ranks_quantized):\n'
    '            aggregated += (source_quantized.to(torch.float32) - gathered_s_and_z[source][1]) * gathered_s_and_z[source][0]\n'
    '        done = torch.futures.Future()\n'
    '        done.set_result(aggregated / world_size)\n'
    '        return done\n'
    '    zero = torch.zeros((), device=device, dtype=tensor.dtype)\n'
    '    min_val = torch.minimum(tensor.min(), zero)\n'
    '    max_val = torch.maximum(tensor.max(), zero)\n'
    '    scale = (max_val - min_val) / 255.0\n'
    '    scale = torch.where(scale > 0, scale, torch.full_like(scale, 1e-8))\n'
    '    zero_point = torch.clamp(torch.round(-min_val / scale), 0.0, 255.0)\n'
    '    s_and_z = torch.stack([scale, zero_point]).to(torch.float32)\n'
    '    all_ranks_s_and_z = [torch.zeros(2, dtype=torch.float32, device=device) for _ in range(world_size)]\n'
    '    fut = _gather_future(all_ranks_s_and_z, s_and_z, group_to_use)\n'
    '\n'
    '    def quantize_and_allgather(fut: Any) -> Any:\n'
    '        gathered_s_and_z = fut.wait()[0]\n'
    '        HOOK_STATE["s_and_z"] = gathered_s_and_z\n'
    '        quantized = torch.clamp(torch.round(tensor / scale) + zero_point, 0.0, 255.0).to(torch.uint8)\n'
    '        all_ranks_quantized = [torch.zeros_like(quantized) for _ in range(world_size)]\n'
    '        return _gather_future(all_ranks_quantized, quantized, group_to_use).wait()[0]\n'
    '\n'
    '    def dequantize_and_aggregate(fut: Any) -> torch.Tensor:\n'
    '        all_ranks_quantized = fut.wait()\n'
    '        if isinstance(all_ranks_quantized, list) and len(all_ranks_quantized) == 1 \\\n'
    '                and isinstance(all_ranks_quantized[0], list):\n'
    '            all_ranks_quantized = all_ranks_quantized[0]\n'
    '        gathered_s_and_z = HOOK_STATE["s_and_z"]\n'
    '        aggregated = torch.zeros_like(tensor, dtype=torch.float32)\n'
    '        for source, quantized in enumerate(all_ranks_quantized):\n'
    '            source_scale = gathered_s_and_z[source][0]\n'
    '            source_zero = gathered_s_and_z[source][1]\n'
    '            aggregated += (quantized.to(torch.float32) - source_zero) * source_scale\n'
    '        return aggregated / world_size\n'
    '\n'
    '    return fut.then(quantize_and_allgather).then(dequantize_and_aggregate)\n'
    '\n'
    '\n'
    'def configure_model_for_mode(')

rep('    if args.comm_mode not in ("powersgd", "ddp-overlap"):\n'
    '        return model, None, {"ddp": False}',
    '    if args.comm_mode not in DDP_MODES:\n'
    '        return model, None, {"ddp": False}')

rep('        details.update({"hook": "official_pytorch_powersgd", **state_kwargs})\n'
    '        details["process_group"] = "WORLD"\n'
    '    return ddp_model, ddp_model, details',
    '        details.update({"hook": "official_pytorch_powersgd", **state_kwargs})\n'
    '        details["process_group"] = "WORLD"\n'
    '    elif args.comm_mode == "ddp-fp16-hook":\n'
    '        # 16-bit reducible hook: the official fp16/bf16 hooks are name-gated to\n'
    '        # NCCL by torch 1.13 and oneCCL exposes no float16 ring all-reduce, so\n'
    '        # this is the official five-line implementation with a selectable dtype:\n'
    '        # bfloat16 on the ccl backend, float16 on CUDA. Same two-byte wire width.\n'
    '        wanted = os.environ.get("RED_HOOK16_DTYPE") or ("bfloat16" if runtime.backend == "ccl" else "float16")\n'
    '        HOOK_STATE["half_dtype"] = wanted\n'
    '        ddp_model.register_comm_hook(dist.group.WORLD, _half_compress_hook)\n'
    '        details.update({\n'
    '            "hook": "pytorch_fp16_compress_hook_port_selectable_dtype",\n'
    '            "dtype": wanted,\n'
    '            "process_group": "WORLD",\n'
    '            "hook_state": HOOK_STATE,\n'
    '        })\n'
    '    elif args.comm_mode == "ddp-int8-hook":\n'
    '        ddp_model.register_comm_hook(dist.group.WORLD, _pertensor_int8_hook)\n'
    '        details.update({\n'
    '            "hook": "pytorch_quantization_pertensor_hook_device_agnostic_port",\n'
    '            "process_group": "WORLD",\n'
    '            "hook_state": HOOK_STATE,\n'
    '        })\n'
    '    return ddp_model, ddp_model, details')

# --- loop and validators: hook modes join the overlap family --------------------
rep('            if args.comm_mode != "ddp-overlap" and not wall_mode:',
    '            if args.comm_mode not in OVERLAP_MODES and not wall_mode:')
rep('            if args.comm_mode in ("powersgd", "ddp-overlap"):\n'
    '                if not do_communication:',
    '            if args.comm_mode in DDP_MODES:\n'
    '                if not do_communication:')
rep('    if record["comm_mode"] == "ddp-overlap" and record["network_s"] is not None:',
    '    if record["comm_mode"] in OVERLAP_MODES and record["network_s"] is not None:')
rep('        if args.comm_mode == "ddp-overlap":\n'
    '            required_finite.append("backward_plus_comm_s")',
    '        if args.comm_mode in OVERLAP_MODES:\n'
    '            required_finite.append("backward_plus_comm_s")')
rep('        elif args.comm_mode == "ddp-overlap":\n'
    '            track = "overlap-baseline"',
    '        elif args.comm_mode in OVERLAP_MODES:\n'
    '            track = "overlap-baseline"')

rep('"""Control-campaign variant of the frozen July harness (2026-09-18).',
    '"""Control-campaign variant v3 of the frozen July harness (2026-09-19).\n'
    '\n'
    'Generated by make_control_harness_v3.py from the v2 control harness (which\n'
    'the September 18 records cite by hash). Adds the llama1b workload, the\n'
    'ddp-fp16-hook and ddp-int8-hook production overlap modes, and the\n'
    '"extension" profile.\n'
    '\n'
    'v2 header follows.')

io.open(DST, "w", encoding="utf-8", newline="\n").write(s)
py_compile.compile(DST, doraise=True)
print("v3 harness written:", DST)
print("edit groups applied:", n[0])
