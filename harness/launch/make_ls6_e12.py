# -*- coding: utf-8 -*-
"""Write the three E12 (A-plus) job scripts from ls6_e11b.slurm:
ls6_e12q.slurm (qualification, 15 minutes), ls6_e12a.slurm and
ls6_e12b.slurm (the two scored allocations, 40 minutes each). All run on
the Lonestar6 development queue on 2 nodes (6 ranks) with GPUDirect
(NCCL_NET_GDR_LEVEL=SYS). Changes from E11b: harness v5.5 and its hash
file, a check of the staged files against diagnostic/E12_FREEZE.sha256, the
E12 campaign id, run directory, run-id prefix, condition functions and
condition file (e12_sets.sh). Environment selection, transport settings and
run_script are E11b's. See diagnostic/E12_PROTOCOL.md."""
import io
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "ls6_e11b.slurm")

FREEZE_CHECK = r'''want="$(cut -d' ' -f1 "$RED_ROOT/CODE_v55.sha256")"
[[ "$(sha256sum "$BENCH" | cut -d' ' -f1)" == "$want" ]] || { echo "FATAL: harness v5.5 differs from CODE_v55.sha256" >&2; exit 1; }
# E12 freeze (diagnostic/E12_PROTOCOL.md section 9): every frozen file is
# staged in $RED_ROOT or $RED_ROOT/prereg with its frozen hash, and this
# script is one of the frozen job scripts.
FREEZE="$RED_ROOT/prereg/E12_FREEZE.sha256"
[[ -f "$FREEZE" ]] || { echo "FATAL: prereg/E12_FREEZE.sha256 is not staged" >&2; exit 1; }
slurm_digests=" "
while read -r digest path; do
  [[ -n "$digest" ]] || continue
  path="${path%$'\r'}"
  base="${path##*/}"
  if [[ "$base" == *.slurm ]]; then slurm_digests+="$digest "; continue; fi
  staged="$RED_ROOT/$base"; [[ -f "$staged" ]] || staged="$RED_ROOT/prereg/$base"
  [[ -f "$staged" ]] || { echo "FATAL: frozen file $base is not staged" >&2; exit 1; }
  [[ "$(sha256sum "$staged" | cut -d' ' -f1)" == "$digest" ]] || { echo "FATAL: staged $base differs from E12_FREEZE.sha256" >&2; exit 1; }
done <"$FREEZE"
[[ "$slurm_digests" == *" $(sha256sum "$0" | cut -d' ' -f1) "* ]] || { echo "FATAL: this job script is not a frozen E12 script" >&2; exit 1; }
'''

FUNCTIONS = r'''# E12 conditions (diagnostic/E12_PROTOCOL.md section 3). A tag is a harness
# mode, with "-bf16" for --dtype bf16; the int8 hook always runs synchronously.
tag_args() {  # tag_args <tag>: sets MODE and DTYPE_ARGS, and the int8 hook's form
  local tag="$1"
  MODE="${tag%-bf16}"
  DTYPE_ARGS=()
  if [[ "$tag" == *-bf16 ]]; then DTYPE_ARGS=(--dtype bf16); fi
  if [[ "$MODE" == ddp-int8-hook ]]; then export RED_HOOK_SYNC=1; else unset RED_HOOK_SYNC; fi
}
e12_equiv() {  # e12_equiv <model> <tag>: the matched control's arms cross-check each other
  tag_args "$2"
  run_script "equiv-$1-$2" "$BENCH" --profile qualification --model "$1" --comm-mode "$MODE" \
    ${DTYPE_ARGS[@]+"${DTYPE_ARGS[@]}"} --batch-size "$E12_BATCH" --iters 3 --warmup-iters 1 --skip-calibration \
    --verify-equivalence --run-id "${rid}-$1-$2-b${E12_BATCH}-equiv" "${common[@]}" || true
  unset RED_HOOK_SYNC
}
e12_phase() {  # e12_phase <model> <tag>: M1, the rules' input (extension profile, phase timing)
  tag_args "$2"
  run_script "phase-$1-$2" "$BENCH" --profile extension --model "$1" --comm-mode "$MODE" \
    ${DTYPE_ARGS[@]+"${DTYPE_ARGS[@]}"} --sync-every 1 --batch-size "$E12_BATCH" --iters 40 --warmup-iters 10 \
    "${reuse[@]}" --run-id "${rid}-$1-$2-b${E12_BATCH}-phase" "${common[@]}" || true
  unset RED_HOOK_SYNC
}
e12_trace() {  # e12_trace <model> <tag>: the replay's input, as E8's trace runs (ranks 0 and 3)
  tag_args "$2"
  run_script "trace-$1-$2" "$BENCH" --profile qualification --model "$1" --comm-mode "$MODE" \
    ${DTYPE_ARGS[@]+"${DTYPE_ARGS[@]}"} --timing-mode wall --batch-size "$E12_BATCH" --iters 8 --warmup-iters 20 \
    --input-pool 4 --trace-dir "$RUN_DIR/traces" --trace-ranks 0,3 --skip-calibration \
    --run-id "${rid}-trace-$1-$2-b${E12_BATCH}" "${common[@]}" || true
  unset RED_HOOK_SYNC
}
e12_block() {  # e12_block <model> <tag> <pass>: the outcome (validation profile, block timing)
  tag_args "$2"
  run_script "block-$1-$2-p$3" "$BENCH" --profile validation --model "$1" --comm-mode "$MODE" \
    ${DTYPE_ARGS[@]+"${DTYPE_ARGS[@]}"} --timing-mode block --sync-every 1 --batch-size "$E12_BATCH" --iters 100 \
    --warmup-iters 20 --input-pool 4 "${reuse[@]}" --run-id "${rid}-$1-$2-b${E12_BATCH}-block-p$3" "${common[@]}" || true
  unset RED_HOOK_SYNC
}
e12_fid() {  # e12_fid <model> <tag>: per-exchange error against the dense average (qualification job)
  tag_args "$2"
  run_script "fid-$1-$2" "$BENCH" --profile qualification --model "$1" --comm-mode "$MODE" \
    ${DTYPE_ARGS[@]+"${DTYPE_ARGS[@]}"} --batch-size "$E12_BATCH" --iters 8 --warmup-iters 2 --skip-calibration \
    --verify-fidelity --run-id "${rid}-$1-$2-b${E12_BATCH}-fid" "${common[@]}" || true
  unset RED_HOOK_SYNC
}
'''

JOBS = {
    "q": ("ls6_e12q.slurm", "00:15:00", "qualification: per-exchange error of the eight compressed tags, a\n"
          "# parse check of the frozen replay on BF16 int8-hook traces, and the longest\n"
          "# outcome condition timed once. Nothing here is a scored outcome."),
    "e12a": ("ls6_e12a.slurm", "00:40:00", "scored allocation a: 24 held-out cases on two workloads, all traced,\n"
             "# with the BF16 production baseline and the matched-encoding control."),
    "e12b": ("ls6_e12b.slurm", "00:40:00", "scored allocation b: the same list as e12a in a second allocation."),
}


def make(red_set):
    name, limit, what = JOBS[red_set]
    src = io.open(SRC, encoding="utf-8").read()
    subs = [
        ("#SBATCH --job-name=redctl-e11b", "#SBATCH --job-name=redctl-e12%s" % red_set.replace("e12", "")),
        ("#SBATCH --time=00:45:00", "#SBATCH --time=%s" % limit),
        ("# E11 (diagnostic/E11_NOTE.md): the int8 hook asynchronous and synchronous\n"
         "# beside the 16-bit hook and dense DDP, development queue, 6 ranks. Written\n"
         "# by make_ls6_e11.py from ls6_e9.slurm.",
         "# E12 (diagnostic/E12_PROTOCOL.md), set %s, %s\n"
         "# Development queue, 2 nodes, 6 ranks. Written by make_ls6_e12.py from\n"
         "# ls6_e11b.slurm; do not edit by hand." % (red_set, what)),
        ('RED_SET="${RED_SET:-dev6b}"', 'RED_SET="%s"' % red_set),
        ('BENCH="$RED_ROOT/control_workload_benchmark_v5_4.py"', 'BENCH="$RED_ROOT/control_workload_benchmark_v5_5.py"'),
        ('CAMPAIGN_ID="${CAMPAIGN_ID:-redctl-e11-2026-10}"', 'CAMPAIGN_ID="${CAMPAIGN_ID:-redctl-e12-2026-10}"'),
        ('want="$(cut -d\' \' -f1 "$RED_ROOT/CODE_v54.sha256")"\n'
         '[[ "$(sha256sum "$BENCH" | cut -d\' \' -f1)" == "$want" ]] || { echo "FATAL: harness v5.4 differs from CODE_v54.sha256" >&2; exit 1; }\n',
         FREEZE_CHECK),
        ('RUN_DIR="$RED_ROOT/runs/e11-${RED_SET}-${NODES}n-${SLURM_JOB_ID}"',
         'RUN_DIR="$RED_ROOT/runs/e12-${RED_SET}-${NODES}n-${SLURM_JOB_ID}"'),
        ('"$RED_ROOT/e11_sets.sh" "$0"', '"$RED_ROOT/e12_sets.sh" "$0"'),
        ('rid="redE11-ws${WORLD}"', 'rid="redE12-ws${WORLD}"'),
        ('# shellcheck source=/dev/null\nsource "$RED_ROOT/e11_sets.sh"',
         FUNCTIONS + '# shellcheck source=/dev/null\nsource "$RED_ROOT/e12_sets.sh"'),
    ]
    for old, new in subs:
        assert src.count(old) == 1, old
        src = src.replace(old, new)
    io.open(os.path.join(HERE, name), "w", encoding="utf-8", newline="\n").write(src)
    return name


if __name__ == "__main__":
    for key in ("q", "e12a", "e12b"):
        print("wrote", make(key))
