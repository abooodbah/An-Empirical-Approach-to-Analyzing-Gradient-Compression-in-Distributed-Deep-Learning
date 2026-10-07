# e9_sets.sh - condition order for the E9 jobs (sourced by ls6_e9.slurm).
# RED_SET=q12a|q12b|q24a|q24b. The caller defines node_facts, preflight,
# dtype_check, probe, hs_probe, calib, equivalence, e9_fidelity, e9_phase,
# e9_block. The "a" allocation of each scale also runs the fidelity runs,
# the phase-timed runs, and the host-staged probes. Pass 2 is pass 1
# reversed. See diagnostic/E9_PROTOCOL.md section 2.
case "$RED_SET" in q12a|q12b|q24a|q24b) ;; *) echo "FATAL: unknown RED_SET=$RED_SET" >&2; exit 1 ;; esac
E9_MODELS=(transformer resnet50)
E9_MODES=(native fp16-native ddp-overlap ddp-fp16-hook ddp-int8-hook gpu-quant gpu-quant-reducible gpu-quant-sr-int8 gpu-quant-sr-fp16)
E9_MANUAL=(native fp16-native gpu-quant gpu-quant-reducible gpu-quant-sr-int8 gpu-quant-sr-fp16)
E9_FIDELITY=(fp16-native gpu-quant gpu-quant-reducible gpu-quant-sr-int8 gpu-quant-sr-fp16)
node_facts
preflight transformer
dtype_check
probe default
probe ring
probe tree
if [[ "$RED_SET" == *a ]]; then hs_probe default; hs_probe ring; hs_probe tree; fi
calib
equivalence transformer gpu-quant-sr-int8
equivalence transformer gpu-quant-sr-fp16
if [[ "$RED_SET" == *a ]]; then
  for model in "${E9_MODELS[@]}"; do
    for mode in "${E9_FIDELITY[@]}"; do e9_fidelity "$model" "$mode"; done
  done
  for model in "${E9_MODELS[@]}"; do
    for mode in "${E9_MANUAL[@]}"; do e9_phase "$model" "$mode"; done
  done
fi
for model in "${E9_MODELS[@]}"; do
  for mode in "${E9_MODES[@]}"; do e9_block "$model" "$mode" 1; done
done
for ((i = ${#E9_MODELS[@]} - 1; i >= 0; i--)); do
  for ((j = ${#E9_MODES[@]} - 1; j >= 0; j--)); do e9_block "${E9_MODELS[i]}" "${E9_MODES[j]}" 2; done
done
