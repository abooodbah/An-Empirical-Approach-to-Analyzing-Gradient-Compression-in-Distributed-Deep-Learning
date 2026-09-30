# rev_sets.sh - condition sets for the revision jobs (sourced by ls6_rev.slurm).
# The caller defines: calib preflight paper wall ext_phase ext_wall trace_run.
case "$RED_SET" in
  e7a|e7b)
    preflight transformer
    calib
    for model in transformer resnet50; do
      for mode in gpu-quant-gathersum gpu-quant gpu-topk cpu-quant ddp-int8-hook fp16-native; do
        ext_phase "$model" "$mode"
      done
      for mode in gpu-quant-gathersum gpu-quant-reducible gpu-quant gpu-topk cpu-quant ddp-int8-hook ddp-fp16-hook; do
        ext_wall "$model" "$mode"
      done
    done
    if [[ "$RED_SET" == e7a ]]; then
      for model in transformer resnet50; do
        for mode in gpu-quant-gathersum gpu-quant-reducible ddp-int8-hook ddp-fp16-hook; do
          trace_run "$model" "$mode"
        done
      done
    fi
    ;;
  j3r|j3rfull)
    preflight transformer
    calib
    for mode in gpu-quant ddp-int8-hook fp16-native; do paper transformer "$mode"; done
    for mode in gpu-quant gpu-quant-gathersum gpu-topk; do paper resnet50 "$mode"; done
    for mode in gpu-quant gpu-quant-reducible ddp-int8-hook ddp-fp16-hook ddp-overlap; do
      wall transformer "$mode"
    done
    for mode in gpu-quant gpu-quant-reducible gpu-quant-gathersum gpu-topk ddp-int8-hook ddp-fp16-hook ddp-overlap; do
      wall resnet50 "$mode"
    done
    if [[ "$RED_SET" == j3rfull ]]; then
      for mode in native fp16-native gpu-quant-gathersum gpu-topk; do wall transformer "$mode"; done
      for mode in native fp16-native; do wall resnet50 "$mode"; done
    fi
    ;;
  *) echo "FATAL: unknown RED_SET=$RED_SET" >&2; exit 1 ;;
esac
