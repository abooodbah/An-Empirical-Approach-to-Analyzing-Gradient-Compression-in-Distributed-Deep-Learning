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
echo "== metrics"
(cd diagnostic/v1_2 && python -m pytest -q -p no:cacheprovider | tail -1)
echo "ALL REGENERATED"
