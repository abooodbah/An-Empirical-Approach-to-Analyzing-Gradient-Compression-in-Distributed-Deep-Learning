# e10_sets.sh - condition order for the E10 jobs (sourced by pvc_e10.slurm).
# The caller defines smoke, stop_after_smoke, probe, e10. Arguments of e10:
# nodes, mode, measured iterations, warm-up iterations, step time limit.
# The most valuable run comes first, so a job cut off by its time limit
# loses the cheapest baseline. See diagnostic/E10_PROTOCOL.md section 2.
case "$RED_SET" in
  s1)
    smoke 00:08:00 || stop_after_smoke "$?"
    probe
    e10 2 gpu-quant-bucketed 40 10 00:22:00
    e10 2 gpu-quant-reducible-bucketed 40 10 00:08:00
    e10 2 native-bucketed 40 10 00:12:00
    e10 1 gpu-quant 40 10 00:10:00
    e10 1 gpu-quant-bucketed 40 10 00:10:00
    e10 2 native 40 10 00:12:00
    ;;
  s2)
    smoke 00:10:00 || stop_after_smoke "$?"
    probe
    e10 4 gpu-quant-bucketed 20 5 00:30:00
    e10 4 gpu-quant-reducible-bucketed 20 5 00:06:00
    e10 4 native-bucketed 20 5 00:08:00
    ;;
  *) echo "FATAL: unknown RED_SET=$RED_SET" >&2; exit 1 ;;
esac
