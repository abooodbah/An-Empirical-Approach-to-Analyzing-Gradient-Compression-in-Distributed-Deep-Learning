#!/bin/bash
# Regenerates every diagnostic table from the frozen files and the archived
# records and compares each with the committed copy. Needs no cluster.
# Usage: bash reproduce_diagnostic.sh        (work folder: $REPRO_WORK or /c/reprow)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
W="${REPRO_WORK:-/c/reprow}"
PYW="$(cygpath -m "$W" 2>/dev/null || echo "$W")"
cd "$ROOT"
mkdir -p "$W/r4w" "$W/revw"
echo "== frozen files"; python -m pytest provenance/test_rev4_frozen.py -q -p no:cacheprovider | tail -1
for a in raw/round4/*; do case "$a" in *.tgz|*.tar.gz) tar xzf "$a" -C "$W/r4w";; esac; done
for a in raw/rev/*.tgz; do tar xzf "$a" -C "$W/revw"; done
id() { awk -F'\t' -v s="$1" '$2 == s {print $3}' "$ROOT/provenance/rev_jobs.tsv" | sed -n "${2:-1}p"; }
echo "== first set: the frozen scorer output and the corrected readings regenerate"
R4W="$PYW/r4w" python -m pytest diagnostic/v1_1 -q -p no:cacheprovider | tail -1
echo "== second set"
python diagnostic/features.py "$PYW/revw" "" "$PYW/revw/features_rev.json" > /dev/null
(cd diagnostic && python score_e7.py "$PYW/revw/features_rev.json" "$(id e7a),$(id e7b)" frozen > "$W/E7.txt" \
  && cmp "$W/E7.txt" E7_RESULT.txt && echo "E7_RESULT.txt identical")
(cd diagnostic && python score_e7.py "$PYW/revw/features_rev.json" "$(id e7a),$(id e7b)" described > "$W/E7d.txt" \
  && cmp "$W/E7d.txt" E7_RESULT_described.txt && echo "E7_RESULT_described.txt identical")
echo "== third set"
# the committed result lists the allocations in job-id order: the two small ones, then the two large
python diagnostic/e8/score_e8.py "$PYW"/revw/e8-e8s-*/ "$PYW"/revw/e8-e8l-*/ --json "$PYW/E8.json" > /dev/null
python - "$PYW/E8.json" diagnostic/E8_RESULT.json <<'PY'
import json, sys
def strip(x):
    if isinstance(x, dict):
        return {k: strip(v) for k, v in x.items() if k not in ("seconds", "median_measurement_s")}
    if isinstance(x, list):
        return [strip(v) for v in x]
    return x
a, b = (strip(json.load(open(p, encoding="utf-8"))) for p in sys.argv[1:3])
assert a == b, "E8 result differs from the committed file"
print("E8_RESULT.json identical apart from measurement seconds")
PY

# ---- Revision 4: every generated file is rebuilt and compared with the
# committed copy. keep saves the committed copy; same compares the rebuilt
# file with it and puts the committed copy back if they differ.
mkdir -p "$W/keep" "$W/mem" "$W/rev4" "$W/july"
keep() { cp "$1" "$W/keep/$(basename "$1")"; }
same() {
  if cmp -s "$1" "$W/keep/$(basename "$1")"; then echo "$(basename "$1") identical"
  else cp "$W/keep/$(basename "$1")" "$1"; echo "DIFFERS from the committed copy: $1 (committed copy restored)"; exit 1; fi
}
echo "== frozen files of E9 and E10, and the records of Revision 4"
python -m pytest provenance/test_rev4_records.py -q -p no:cacheprovider | tail -1
echo "== selection metrics and replay validation (about half a minute)"
for f in diagnostic/v1_2/METRICS.json diagnostic/v1_2/METRICS.txt diagnostic/v1_2/REPLAY_VALIDATION.csv; do keep "$f"; done
python diagnostic/v1_2/metrics.py > /dev/null
python diagnostic/v1_2/replay_validation.py > /dev/null
for f in diagnostic/v1_2/METRICS.json diagnostic/v1_2/METRICS.txt diagnostic/v1_2/REPLAY_VALIDATION.csv; do same "$f"; done
echo "== allocation-level intervals"
for f in validation_v2/results/rev4/INTERVALS.json validation_v2/results/rev4/INTERVALS.md; do keep "$f"; done
python validation_v2/analysis/intervals.py > /dev/null
for f in validation_v2/results/rev4/INTERVALS.json validation_v2/results/rev4/INTERVALS.md; do same "$f"; done
echo "== memory account of the 1B conditions"
for a in raw/control/redctl-V-v3-3517750-51-52-53.tgz raw/control/redctl-V-reruns-3517832-3517833.tgz \
         raw/control/redctl-V-r16b-3517873.tgz raw/v2/v2-L1-2n-3465529.tgz; do tar xzf "$a" -C "$W/mem"; done
keep diagnostic/v1_2/MEMORY_ACCOUNT.md
python diagnostic/v1_2/memory_account.py "$PYW/mem" > /dev/null
same diagnostic/v1_2/MEMORY_ACCOUNT.md
echo "== E10: the frozen scorer on the three committed archives"
for a in raw/rev4/redctl-E10-*.tgz; do tar xzf "$a" -C "$W/rev4"; done
python validation_v2/analysis/score_e10.py "$PYW"/rev4/redctl-E10-*/ --json "$PYW/E10.json" > /dev/null
cmp "$W/E10.json" validation_v2/results/rev4/E10_RESULT.json && echo "E10_RESULT.json identical"
echo "== E9: the frozen scorer on the four committed archives, and the post hoc tree probe"
mkdir -p "$W/e9"
for a in raw/rev4/e9-*.tgz; do tar xzf "$a" -C "$W/e9"; done
python validation_v2/analysis/score_e9.py "$PYW"/e9/e9-*/ --json "$PYW/E9.json" > /dev/null
cmp "$W/E9.json" validation_v2/results/rev4/E9_RESULT.json && echo "E9_RESULT.json identical"
python validation_v2/analysis/e9_tree_posthoc.py "$PYW"/e9/e9-*/ --json "$PYW/E9_TREE.json" > /dev/null
cmp "$W/E9_TREE.json" validation_v2/results/rev4/E9_TREE_POSTHOC.json && echo "E9_TREE_POSTHOC.json identical"
echo "== job-to-job spread behind the H100 phase figure"
for j in 3328658 3328659 3328660; do tar xzf "raw/july/h100-$j.tar.gz" -C "$W/july"; done
keep validation_v2/results/rev4/H100_STEP_SPREAD.json
python validation_v2/analysis/step_spread.py "$PYW/july" > /dev/null
same validation_v2/results/rev4/H100_STEP_SPREAD.json
echo "== SU record of Revision 4"
keep provenance/rev4_su.tsv
python provenance/make_rev4_su.py > /dev/null
same provenance/rev4_su.tsv
echo "== tests of the retrospective modules"
(cd diagnostic/v1_2 && REV4_MEM_ROOT="$PYW/mem" python -m pytest -q -p no:cacheprovider | tail -1)
(cd validation_v2/analysis && REV4_JULY_ROOT="$PYW/july" python -m pytest tests/test_step_spread.py tests/test_intervals.py tests/test_traffic_models.py -q -p no:cacheprovider | tail -1)
echo "ALL REGENERATED"
