# -*- coding: utf-8 -*-
"""Write harness/launch/ls6_e11.slurm from ls6_e9.slurm. Changes: the
development queue (2 nodes, 45 minutes), harness v5.3 and its hash file,
the E11 campaign id, run directory and condition file. Everything else,
environment selection and transport settings included, is E9's."""
import io
import os

HERE = os.path.dirname(os.path.abspath(__file__))
src = io.open(os.path.join(HERE, "ls6_e9.slurm"), encoding="utf-8").read()
subs = [
    ("#SBATCH --job-name=redctl-e9", "#SBATCH --job-name=redctl-e11"),
    ("#SBATCH --partition=gpu-a100", "#SBATCH --partition=gpu-a100-dev"),
    ("#SBATCH --nodes=4", "#SBATCH --nodes=2"),
    ("#SBATCH --time=00:50:00", "#SBATCH --time=00:45:00"),
    ('RED_SET="${RED_SET:?set RED_SET to q12a, q12b, q24a, or q24b}"', 'RED_SET="${RED_SET:-dev6}"'),
    ('BENCH="$RED_ROOT/control_workload_benchmark_v5_2.py"', 'BENCH="$RED_ROOT/control_workload_benchmark_v5_3.py"'),
    ('CAMPAIGN_ID="${CAMPAIGN_ID:-redctl-e9-2026-10}"', 'CAMPAIGN_ID="${CAMPAIGN_ID:-redctl-e11-2026-10}"'),
    ('want="$(cut -d\' \' -f1 "$RED_ROOT/CODE_v52.sha256")"', 'want="$(cut -d\' \' -f1 "$RED_ROOT/CODE_v53.sha256")"'),
    ('echo "FATAL: harness v5.2 differs from CODE_v52.sha256"', 'echo "FATAL: harness v5.3 differs from CODE_v53.sha256"'),
    ('RUN_DIR="$RED_ROOT/runs/e9-${RED_SET}-${NODES}n-${SLURM_JOB_ID}"', 'RUN_DIR="$RED_ROOT/runs/e11-${RED_SET}-${NODES}n-${SLURM_JOB_ID}"'),
    ('"$RED_ROOT/e9_sets.sh" "$0"', '"$RED_ROOT/e11_sets.sh" "$0"'),
    ('rid="redE9-ws${WORLD}"', 'rid="redE11-ws${WORLD}"'),
    ('source "$RED_ROOT/e9_sets.sh"', 'source "$RED_ROOT/e11_sets.sh"'),
    ("# E9 (diagnostic/E9_PROTOCOL.md): reducible quantizers and a matched DDP\n"
     "# baseline, with the collective probes. RED_SET=q12a|q12b|q24a|q24b; the node\n"
     "# count comes from sbatch -N.",
     "# E11 (diagnostic/E11_NOTE.md): the int8 hook asynchronous and synchronous\n"
     "# beside the 16-bit hook and dense DDP, development queue, 6 ranks. Written\n"
     "# by make_ls6_e11.py from ls6_e9.slurm."),
]
for old, new in subs:
    assert src.count(old) == 1, old
    src = src.replace(old, new)
io.open(os.path.join(HERE, "ls6_e11.slurm"), "w", encoding="utf-8", newline="\n").write(src)
print("wrote ls6_e11.slurm")
