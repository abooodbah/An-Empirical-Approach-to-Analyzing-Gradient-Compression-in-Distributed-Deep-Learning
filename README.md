# An Empirical Approach to Analyzing Gradient Compression in Distributed Deep Learning

## What the paper shows

Compression shortens training only if its saving survives the collective, the encoding and the schedule. The paper measures all three on NVIDIA and Intel systems with up to 32 workers.

- **The collective sets the cost.** With identical encoded values, replacing all-gather with all-reduce cut the Transformer's step by 51% and 79% in two allocations.
- **CPU-side encoding dominates.** Host copies, conversion and reconstruction take 86.2% of the CPU-staged ResNet-50 step, and the network only 5.1%.
- **Phase timing hides overlap.** Its synchronization made the Transformer's Top-k step up to 1.59 times as long as one timer around the whole run.
- **The replay finds the beneficial changes.** It replays a recorded step with its overlap intact, within a median 2.3% of the measured step. On held-out configurations it recommended 7.5 of 8 beneficial changes (each allocation counts half), and every recommendation shortened the step. Its median gain error was 5.2 percentage points.
- **The replay rejects the harmful changes.** On a fourth held-out set it rejected all six changes it models that slowed the step.

## What is here

| Folder | Contents |
|---|---|
| `harness/` | Benchmark programs and launch scripts |
| `raw/` | Per-rank records, one archive per job |
| `processed/` | Tables behind the figures |
| `diagnostic/` | Rules, replay, and held-out protocols, scorers and results |
| `validation_v2/` | Validation campaign and results |
| `figures/` | Figure scripts |
| `provenance/` | Job lists and protocol push times |
| `contract/` | Experiment contract |
| `environment/` | Software and node facts |

## Reproduce

```
bash reproduce_diagnostic.sh
```

Needs no cluster: `bash`, `tar`, `sha256sum`, and Python 3 with `numpy`, `matplotlib` and `pytest`. It regenerates the diagnostic tables of all four held-out sets and checks them against the committed results. `REPRO_WORK` sets the work folder.

`RECORDS.md` lists where each held-out set, job and result lives.

## Citation

Under review at IEEE Transactions on Parallel and Distributed Systems.

```bibtex
@unpublished{bahbouh2026gradient,
  author = {Bahbouh, Abdulfatah and Ahmad, Ishfaq},
  title  = {An Empirical Approach to Analyzing Gradient Compression
            in Distributed Deep Learning},
  note   = {Under review at IEEE Transactions on Parallel and
            Distributed Systems},
  year   = {2026}
}
```

## License

MIT, copyright 2026 Abdulfatah Bahbouh.
