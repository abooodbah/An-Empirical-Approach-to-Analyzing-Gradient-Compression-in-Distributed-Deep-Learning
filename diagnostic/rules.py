# -*- coding: utf-8 -*-
"""FROZEN diagnostic rules, version 1 (2026-09-21).

Developed from the Stampede3 September records only (tags A, B, C, V).
Lonestar6 records (tag L) were excluded from development and are the
first held-out set. This file is versioned in the artifact repository;
any change after E6 begins is a new version and E6 must be rerun.

Inputs per configuration (all from Tier 1 instrumented records; none
from intervention outcomes):
  exchange_share   network_s / step
  codec_share      (serialize_s + deserialize_s) / step
  barrier_share    barrier_wait_s / step
  kind             gather | reduce (from the method's collective)
  volume_ratio     Eq. 1 received-traffic ratio at this p (gather modes)
  trend            exchange_share at this p minus exchange_share at the
                   smallest p measured for the same method, when available
  shape_ratio      exchange / calibrated reduce time for the same payload,
                   when a calibration exists on the same run (secondary)

Output: one of
  shape          the collective's shape limits the step; predicted
                 intervention: change the collective (gather -> reduce-
                 shaped exchange of the same encoding); expected effect:
                 step falls by roughly exchange_share * (1 - 1/volume_ratio)
  placement      encode/reconstruct work limits the step; predicted
                 intervention: move or replace the encoder (GPU kernel,
                 cheaper selection); expected effect: up to codec_share
  coordination   synchronization limits the step; predicted intervention:
                 reduce exchange frequency or enable overlap; expected
                 effect: up to barrier_share, or the hidden fraction
  none           the exchange is not the limiting phase; no compression-
                 side intervention is predicted to pay
  mixed          two mechanisms each exceed their threshold
  insufficient   features missing or below all thresholds

Thresholds (fixed here from the development table; see FEATURE NOTES):
  T_EXCH = 0.40   exchange share above which the exchange can be limiting
  T_CODEC = 0.20  codec share above which placement can be limiting
  T_BAR = 0.10    barrier share above which coordination can be limiting
  T_VOL = 1.5     volume ratio above which a gather-shaped exchange is
                  predicted to be amplification-limited rather than
                  bandwidth-limited
  T_TREND = 0.10  exchange-share growth with p that marks shape (as
                  opposed to a large but flat share, which marks a
                  bandwidth-bound reduce exchange: none)

FEATURE NOTES (development table, Stampede3):
  gather int8 Transformer: exch 0.63/0.75/0.87 at p=8/16/32, vol 1/2/4,
    codec 0.16/0.05/0.03 -> shape at 16 and 32; at 8 (vol 1.0, codec
    0.16) the rules must not say shape: expected 'none' or 'insufficient'
  gather int8 ResNet-50: exch 0.11/0.31/0.57, vol 1/2/4 -> shape at 32
    (exch > 0.40 and vol > 1.5); at 16 exch 0.31 < T_EXCH -> none
  Top-k both workloads: codec 0.25-0.88, exch <= 0.27 -> placement
  reducible int8: exch 0.05-0.58, flat with p (trend <= 0.16), vol 1.0
    -> none (a bandwidth-bound reduce exchange is not a shape problem)
  native Transformer: exch 0.78-0.84, flat, vol 1.0 -> none
  barrier shares never exceed 0.06 in the development set; T_BAR = 0.10
    is set above every observed value so 'coordination' is never
    emitted for the development set. The tau sweep (E3) is the only
    held-out family that can exercise it.

SCORING PROTOCOL FOR E6 (fixed before any held-out run):
  A configuration is scored only if it has both an uninstrumented
  baseline T_orig^U and an uninstrumented intervention T_int^U from the
  same allocation, paired. delta = (T_orig^U - T_int^U) / T_orig^U.
  The rule's prediction is (mechanism, intervention, expected_effect e).
  Correct: the intervention the rule named was applied and
     delta >= max(0.05, 0.5 * e)      (direction and at least half the
                                        predicted magnitude, above noise)
  Harmful: delta <= -0.05 for a named intervention
  Null: |delta| < 0.05 for a named intervention (counts as incorrect)
  Abstained: rule said none/mixed/insufficient; scored separately as
     coverage, and checked against whether any tried intervention gave
     delta >= 0.05 (an abstention where an intervention would have paid
     is a miss; an abstention where none paid is correct)
  Report: per-configuration table (features, verdict, intervention,
     T_orig^U, T_int^U, delta, outcome), then correct / null / harmful /
     abstain-correct / abstain-miss counts, and coverage. No single
     headline percentage without the table.
  Ablations scored identically on the same configurations:
     A1 payload-only: predict shape iff volume_ratio > T_VOL (Eq. 1 alone)
     A2 calibration-only: predict shape iff shape_ratio > 2.0, no
        attribution features
     A3 timeline-only: attribution shares with T_* thresholds but no
        volume_ratio and no trend (cannot separate gather amplification
        from a large flat reduce share)
"""

T_EXCH, T_CODEC, T_BAR, T_VOL, T_TREND = 0.40, 0.20, 0.10, 1.5, 0.10
RULES_VERSION = "v1-2026-09-21"


def diagnose(f, trend=None):
    """f: dict with exchange_share, codec_share, barrier_share, kind,
    volume_ratio (any may be None). trend: exchange_share growth from the
    smallest measured p for this method, or None."""
    ex, co, ba = f.get("exchange_share"), f.get("codec_share"), f.get("barrier_share")
    kind, vol = f.get("kind"), f.get("volume_ratio")
    if ex is None and co is None:
        return "insufficient", None, 0.0
    hits = []
    if co is not None and co > T_CODEC:
        hits.append(("placement", "move or replace the encoder", co))
    if ba is not None and ba > T_BAR:
        hits.append(("coordination", "reduce exchange frequency or enable overlap", ba))
    if ex is not None and ex > T_EXCH and kind == "gather" and vol is not None and vol > T_VOL \
            and (trend is None or trend > T_TREND):
        hits.append(("shape", "reduce-shaped exchange of the same encoding", ex * (1.0 - 1.0 / vol)))
    if len(hits) == 0:
        return "none", None, 0.0
    if len(hits) == 1:
        return hits[0]
    hits.sort(key=lambda h: -h[2])
    if hits[0][2] >= 2.0 * hits[1][2]:
        return hits[0]
    return "mixed", None, 0.0


def ablation_payload_only(f):
    vol = f.get("volume_ratio")
    return ("shape", "reduce-shaped exchange", 0.0) if (f.get("kind") == "gather" and vol and vol > T_VOL) else ("none", None, 0.0)


def ablation_calibration_only(f):
    sr = f.get("shape_ratio")
    if sr is None:
        return "insufficient", None, 0.0
    return ("shape", "reduce-shaped exchange", 0.0) if sr > 2.0 else ("none", None, 0.0)


def ablation_timeline_only(f):
    ex, co, ba = f.get("exchange_share"), f.get("codec_share"), f.get("barrier_share")
    if ex is None and co is None:
        return "insufficient", None, 0.0
    hits = []
    if co is not None and co > T_CODEC:
        hits.append(("placement", "move or replace the encoder", co))
    if ba is not None and ba > T_BAR:
        hits.append(("coordination", "reduce exchange frequency or enable overlap", ba))
    if ex is not None and ex > T_EXCH:
        hits.append(("shape", "reduce-shaped exchange", ex))
    if not hits:
        return "none", None, 0.0
    hits.sort(key=lambda h: -h[2])
    return hits[0] if len(hits) == 1 or hits[0][2] >= 2.0 * hits[1][2] else ("mixed", None, 0.0)


if __name__ == "__main__":
    import io, json, os, sys
    from collections import defaultdict
    rows = json.load(io.open(sys.argv[1], encoding="utf-8"))
    base = {}
    for r in rows:
        k = (r["tag"], r["model"], r["mode"])
        if k not in base or r["p"] < base[k][0]:
            base[k] = (r["p"], r["exchange_share"])
    print(f"rules {RULES_VERSION}")
    print("tag model       mode                  p   verdict       payload  calib   timeline")
    for r in rows:
        p0, e0 = base[(r["tag"], r["model"], r["mode"])]
        trend = (r["exchange_share"] - e0) if (r["exchange_share"] is not None and e0 is not None and r["p"] != p0) else None
        v = diagnose(r, trend)[0]
        a1, a2, a3 = ablation_payload_only(r)[0], ablation_calibration_only(r)[0], ablation_timeline_only(r)[0]
        print(f"{r['tag']} {r['model']:11s} {r['mode']:20s} {r['p']:2d}   {v:13s} {a1:8s} {a2:7s} {a3}")
