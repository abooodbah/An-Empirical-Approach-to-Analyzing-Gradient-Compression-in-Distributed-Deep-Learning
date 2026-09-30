# e8_sets.sh - condition sets for the E8 jobs (sourced by ls6_e8.slurm).
# The caller defines: preflight calib e8_phase e8_trace e8_block, which read
# E8_BATCH. Order: the methods' inputs (M1 phase runs and traces of the
# originals) come before every outcome (block runs), and pass 2 is pass 1
# reversed. See diagnostic/E8_PROTOCOL.md section 2.
case "$RED_SET" in
  e8s) E8_BATCH=32 ;;
  e8l) E8_BATCH=128 ;;
  *) echo "FATAL: unknown RED_SET=$RED_SET" >&2; exit 1 ;;
esac
export E8_BATCH
E8_MODELS=(transformer resnet50)
E8_ORIGINALS=(gpu-quant-gathersum ddp-int8-hook)
E8_MODES=(gpu-quant-gathersum gpu-quant-reducible ddp-int8-hook ddp-fp16-hook)
preflight transformer
calib
for model in "${E8_MODELS[@]}"; do
  for mode in "${E8_ORIGINALS[@]}"; do e8_phase "$model" "$mode"; done
done
for model in "${E8_MODELS[@]}"; do
  for mode in "${E8_ORIGINALS[@]}"; do e8_trace "$model" "$mode"; done
done
for model in "${E8_MODELS[@]}"; do
  for mode in "${E8_MODES[@]}"; do e8_block "$model" "$mode" 1; done
done
for ((i = ${#E8_MODELS[@]} - 1; i >= 0; i--)); do
  for ((j = ${#E8_MODES[@]} - 1; j >= 0; j--)); do e8_block "${E8_MODELS[i]}" "${E8_MODES[j]}" 2; done
done
