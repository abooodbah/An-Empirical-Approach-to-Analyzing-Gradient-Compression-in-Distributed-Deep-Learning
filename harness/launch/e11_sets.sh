# e11_sets.sh - condition order for the E11 development job (sourced by
# ls6_e11.slurm). E11 asks one question: does PyTorch's int8 hook, run in its
# asynchronous form on NCCL, still lose to the 16-bit hook and to dense DDP?
# Each workload runs dense DDP, the 16-bit hook, the int8 hook asynchronous
# (harness v5.3) and the int8 hook synchronous (RED_HOOK_SYNC=1, the form of
# every earlier run), block-timed, in two passes; pass 2 is pass 1 reversed.
# The caller defines node_facts, preflight, calib and run_script.
E11_MODELS=(transformer resnet50)
E11_VARIANTS=("ddp-overlap:" "ddp-fp16-hook:" "ddp-int8-hook:async" "ddp-int8-hook:sync")

e11_block() {  # e11_block <model> <mode> <variant: empty, async or sync> <pass>
  local model="$1" mode="$2" variant="$3" pass="$4" tag="$2"
  [[ -n "$variant" ]] && tag="$mode-$variant"
  if [[ "$variant" == sync ]]; then export RED_HOOK_SYNC=1; else unset RED_HOOK_SYNC; fi
  run_script "block-$model-$tag-p$pass" "$BENCH" --profile validation --model "$model" --comm-mode "$mode" \
    --timing-mode block --sync-every 1 --batch-size 32 --iters 100 --warmup-iters 20 --input-pool 4 \
    "${reuse[@]}" --run-id "${rid}-$model-$tag-block-p$pass" "${common[@]}" || true
  unset RED_HOOK_SYNC
}

node_facts
preflight transformer
calib
for model in "${E11_MODELS[@]}"; do
  for item in "${E11_VARIANTS[@]}"; do e11_block "$model" "${item%%:*}" "${item#*:}" 1; done
done
for ((i = ${#E11_MODELS[@]} - 1; i >= 0; i--)); do
  for ((j = ${#E11_VARIANTS[@]} - 1; j >= 0; j--)); do
    item="${E11_VARIANTS[j]}"; e11_block "${E11_MODELS[i]}" "${item%%:*}" "${item#*:}" 2
  done
done
