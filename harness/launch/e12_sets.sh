# e12_sets.sh - condition order for the E12 (A-plus) jobs, sourced by
# ls6_e12q.slurm, ls6_e12a.slurm and ls6_e12b.slurm. diagnostic/E12_PROTOCOL.md
# section 3 lists the same order. The caller defines node_facts, preflight,
# calib, e12_equiv, e12_phase, e12_trace, e12_block and e12_fid.
#
# A tag is a harness mode, with "-bf16" when the run uses --dtype bf16. The
# int8 hook always runs its synchronous form (RED_HOOK_SYNC=1, set by the
# caller's tag_args): the form of E7 and E8, the one the frozen replay parses,
# and on NCCL at 6 ranks the faster of the two (E11).
#
# Sets: q (qualification), e12a and e12b (the two scored allocations; the same
# list). In e12a and e12b every input of every method (M1 phase runs, traces)
# is measured before any outcome (block runs); pass 2 is pass 1 reversed.
E12_BATCH=64
export E12_BATCH
unset RED_HOOK16_DTYPE RED_HOOK_SYNC
E12_MODELS=(transformer resnet50)
E12_TAGS=(ddp-overlap ddp-fp16-hook ddp-int8-hook gpu-quant cpu-quant gpu-quant-gathersum gpu-quant-reducible
          ddp-overlap-bf16 ddp-fp16-hook-bf16 ddp-int8-hook-bf16)
# M1 (phase-timed) runs: the seven originals of section 2, plus the reducible
# arm of the matched control, whose phase-timed step gives rho.
E12_PHASE=(gpu-quant-gathersum gpu-quant cpu-quant ddp-int8-hook ddp-overlap ddp-overlap-bf16 ddp-int8-hook-bf16
           gpu-quant-reducible)
# Per-exchange error (qualification job only). Dense DDP sends the dense
# average itself; its error is zero by definition and it is not run.
E12_FIDELITY=(gpu-quant cpu-quant gpu-quant-gathersum gpu-quant-reducible ddp-fp16-hook ddp-int8-hook
              ddp-fp16-hook-bf16 ddp-int8-hook-bf16)

case "$RED_SET" in
  q)
    node_facts
    preflight transformer
    calib
    for model in "${E12_MODELS[@]}"; do
      for tag in "${E12_FIDELITY[@]}"; do e12_fid "$model" "$tag"; done
    done
    # parse check of the frozen replay on a BF16 int8-hook trace, and the
    # longest outcome condition, timed once (neither is scored)
    for model in "${E12_MODELS[@]}"; do e12_trace "$model" ddp-int8-hook-bf16; done
    e12_block resnet50 cpu-quant 1
    e12_block transformer ddp-overlap-bf16 1
    ;;
  e12a|e12b)
    node_facts
    preflight transformer
    calib
    for model in "${E12_MODELS[@]}"; do
      e12_equiv "$model" gpu-quant-gathersum
      e12_equiv "$model" gpu-quant-reducible
    done
    for model in "${E12_MODELS[@]}"; do
      for tag in "${E12_PHASE[@]}"; do e12_phase "$model" "$tag"; done
    done
    for model in "${E12_MODELS[@]}"; do
      for tag in "${E12_TAGS[@]}"; do e12_trace "$model" "$tag"; done
    done
    for model in "${E12_MODELS[@]}"; do
      for tag in "${E12_TAGS[@]}"; do e12_block "$model" "$tag" 1; done
    done
    for ((i = ${#E12_MODELS[@]} - 1; i >= 0; i--)); do
      for ((j = ${#E12_TAGS[@]} - 1; j >= 0; j--)); do e12_block "${E12_MODELS[i]}" "${E12_TAGS[j]}" 2; done
    done
    ;;
  *) echo "FATAL: unknown RED_SET=$RED_SET (q, e12a or e12b)" >&2; exit 1 ;;
esac
