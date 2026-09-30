# -*- coding: utf-8 -*-
"""Generate ls6_rev.slurm from ls6_r4.slurm (left untouched) with asserted edits."""
import io
import os

HERE = os.path.dirname(os.path.abspath(__file__))
src = io.open(os.path.join(HERE, "ls6_r4.slurm"), encoding="utf-8").read()


def rep(old, new):
    global src
    assert src.count(old) == 1, old[:60]
    src = src.replace(old, new)


rep("#SBATCH --job-name=redctl-ls6-gdr", "#SBATCH --job-name=redctl-rev")
rep('CAMPAIGN_ID="${CAMPAIGN_ID:-redctl-r4-2026-09-21}"',
    'CAMPAIGN_ID="${CAMPAIGN_ID:-redctl-rev-2026-10}"')
rep('BENCHMARK="$RED_ROOT/control_workload_benchmark_v4.py"',
    'BENCHMARK="$RED_ROOT/control_workload_benchmark_v4.py"\n'
    'BENCH_V51="$RED_ROOT/control_workload_benchmark_v5_1.py"')
rep('RUN_DIR="$RED_ROOT/runs/r4-${RED_SET}-${NODES}n-${SLURM_JOB_ID}"',
    'RUN_DIR="$RED_ROOT/runs/rev-${RED_SET}-${NODES}n-${SLURM_JOB_ID}"')
rep('mkdir -p "$RUN_DIR/results" "$RUN_DIR/logs" "$RUN_DIR/provenance"',
    'mkdir -p "$RUN_DIR/results" "$RUN_DIR/logs" "$RUN_DIR/provenance" "$RUN_DIR/traces"')
rep('sha256sum "$BENCHMARK" >"$RUN_DIR/provenance/benchmark.sha256"',
    'sha256sum "$BENCHMARK" "$BENCH_V51" "$RED_ROOT/rev_sets.sh" "$RED_ROOT"/prereg/* '
    '>"$RUN_DIR/provenance/benchmark.sha256"')
rep("""      --max_restarts=0 "$BENCHMARK" "$@"' _ "$@"
}""",
    """      --max_restarts=0 "$BENCHMARK" "$@"' _ "$@" || rc=$?
  echo "${condition} rc=${rc}" >>"$RUN_DIR/provenance/conditions.txt"
  return "$rc"
}""")
rep('  export CONDITION_ID="$condition" MASTER_PORT="$port" HEAD_NODE RUN_DIR BENCHMARK\n',
    '  export CONDITION_ID="$condition" MASTER_PORT="$port" HEAD_NODE RUN_DIR BENCHMARK\n'
    '  local rc=0\n')
rep("""    --run-id "${rid}-${model}-${mode}" "${common[@]}"
}""", """    --run-id "${rid}-${model}-${mode}" "${common[@]}" || true
}""")
rep("""    --run-id "${rid}-${model}-${mode}-wall" "${common[@]}"
}""", """    --run-id "${rid}-${model}-${mode}-wall" "${common[@]}" || true
}""")
start = src.index('case "$RED_SET" in')
end = src.index("esac", start) + len("esac")
src = src[:start] + r'''calib() {
  run_condition "paper-resnet50-native" --profile paper --model resnet50 --comm-mode native --sync-every 1 \
    --batch-size 32 --iters 40 --warmup-iters 10 --run-id "$cal_ref" "${common[@]}" \
    || { echo "FATAL: calibration failed; no scored run was attempted" >&2; exit 1; }
}
preflight() {
  run_condition "preflight-$1" --profile qualification --model "$1" --comm-mode native --sync-every 1 \
    --batch-size 4 --iters 3 --warmup-iters 1 --skip-calibration --distributed-timeout-s 120 \
    --run-id "${rid}-preflight-$1" "${common[@]}" \
    || { echo "FATAL: preflight failed; no scored run was attempted" >&2; exit 1; }
}
ext_phase() {
  local model="$1" mode="$2"
  run_condition "ext-${model}-${mode}" --profile extension --model "$model" --comm-mode "$mode" \
    --sync-every 1 --batch-size 128 --iters 40 --warmup-iters 10 "${reuse[@]}" \
    --run-id "${rid}-${model}-${mode}-b128" "${common[@]}" || true
}
ext_wall() {
  local model="$1" mode="$2"
  run_condition "extwall-${model}-${mode}" --profile extension --model "$model" --comm-mode "$mode" \
    --timing-mode wall --sync-every 1 --batch-size 128 --iters 40 --warmup-iters 10 "${reuse[@]}" \
    --run-id "${rid}-${model}-${mode}-b128-wall" "${common[@]}" || true
}
trace_run() {
  local model="$1" mode="$2" saved="$BENCHMARK"
  BENCHMARK="$BENCH_V51"
  run_condition "trace-${model}-${mode}" --profile qualification --model "$model" --comm-mode "$mode" \
    --timing-mode wall --batch-size 128 --iters 8 --warmup-iters 20 --input-pool 4 \
    --trace-dir "$RUN_DIR/traces" --trace-ranks 0,3 --skip-calibration \
    --run-id "${rid}-trace-${model}-${mode}" "${common[@]}" || true
  BENCHMARK="$saved"
}
# shellcheck source=/dev/null
source "$RED_ROOT/rev_sets.sh"''' + src[end:]
assert "r4-${RED_SET}" not in src
io.open(os.path.join(HERE, "ls6_rev.slurm"), "w", encoding="utf-8", newline="\n").write(src)
print("ls6_rev.slurm written")
