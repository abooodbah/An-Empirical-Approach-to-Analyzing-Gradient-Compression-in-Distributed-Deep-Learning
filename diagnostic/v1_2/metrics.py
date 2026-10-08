# diagnostic/v1_2/metrics.py
# -*- coding: utf-8 -*-
"""Selection quality of every rule set on the three held-out sets, with
cluster-bootstrap intervals over cases. RETROSPECTIVE (2026-10-07): devised
after every outcome was known. It re-reads the labels the frozen scorers
wrote and changes none of them. Categories follow score_e7.summary and
score_e8.summary, so the totals equal the published tables."""
import io
import json
import os
import random
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DIAG = os.path.dirname(HERE)
for _p in (DIAG, os.path.join(DIAG, "v1_1")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

ABSTAIN = ("none", "mixed", "insufficient")
NO_OUTCOME = ("abstain (no intervention run)", "unscored (no paired wall run)")
PAY = 0.05
SEED = 20261007
DRAWS = 10000
E7_LINE = re.compile(r"^(E7-\S+)\s+(\d+)\s+(\S+)\s+(\S+)\s+e=[0-9.]+\s+delta=\s*(\S+)\s+(.+?)\s+\[.*\]$")
E6_READINGS = {"as_scored": ("frozen", False, False),
               "extractor_as_frozen": ("as_frozen_extractor", False, False),
               "corrected": ("described", True, True)}
RULE_SETS = ("full", "A1", "A2", "A3")


def _delta(token):
    return None if token == "-" else float(token)


def e6_rows(reading):
    import rescore as R
    name, e_same, c5 = E6_READINGS[reading]
    rows = R.load_rows()
    out = []
    for method in RULE_SETS:
        # rescore returns one record per case and allocation, with the unrounded delta
        for line in R.rescore(rows, R.METHODS[method], reading=name, e_same=e_same, c5=c5)[1]:
            out.append({"case": "E6-%s" % line["case"], "alloc": str(line["alloc"]), "method": method,
                        "verdict": line["verdict"], "delta": line["delta"], "label": line["outcome"]})
    return out


def e7_rows(path):
    out = []
    for line in io.open(path, encoding="utf-8"):
        m = E7_LINE.match(line.strip())
        if m:
            out.append({"case": m.group(1), "alloc": m.group(2), "method": m.group(3),
                        "verdict": m.group(4), "delta": _delta(m.group(5)),
                        "label": m.group(6).strip()})
    return out


def e8_rows(path):
    data = json.load(io.open(path, encoding="utf-8"))
    return [{"case": r["case"], "alloc": r["alloc"],
             "method": "replay" if r["method"] == "reference" else r["method"],
             "verdict": r["verdict"], "delta": r["g"], "label": r["label"]}
            for r in data["rows"]]


def category(row):
    label = row["label"]
    if label.startswith("missing"):
        return "missing"
    if label in ("correct", "short"):
        return "cleared"
    if label in ("null", "harmful", "inapplicable"):
        return "false_positive"
    if label == "not tested":
        return "not_tested"
    if label == "abstain-correct":
        return "abstain_correct"
    if label == "abstain-miss":
        return "abstain_miss"
    if label in NO_OUTCOME:
        return "no_outcome"
    raise ValueError("unknown label %r" % label)


def paid(row):
    """Whether a tested change reduced the step by at least 5% on this row."""
    cat = category(row)
    if cat in ("cleared", "abstain_miss"):
        return True
    if cat == "not_tested":
        return row["delta"] is not None and row["delta"] >= PAY
    return False


def tally(rows):
    """Weighted counts for one rule set on one set. An allocation of a case
    with k allocations counts 1/k."""
    k = {}
    for r in rows:
        k[r["case"]] = k.get(r["case"], 0) + 1
    keys = ("cases", "recommended", "cleared", "false_positive", "harmful", "not_tested",
            "abstain_correct", "abstain_miss", "no_outcome", "missing", "paying", "not_paying")
    c = dict.fromkeys(keys, 0.0)
    for r in rows:
        w = 1.0 / k[r["case"]]
        cat = category(r)
        c[cat] += w
        if cat == "missing":
            continue
        c["cases"] += w
        if r["verdict"] not in ABSTAIN:
            c["recommended"] += w
        if r["label"] == "harmful":
            c["harmful"] += w
        c["paying" if paid(r) else "not_paying"] += w
    return c


def rates(c):
    def div(a, b):
        return None if b <= 0 else a / b
    return {"coverage": div(c["recommended"], c["cases"]),
            "precision": div(c["cleared"], c["cleared"] + c["false_positive"]),
            "recall": div(c["cleared"], c["paying"]),
            "false_positive_rate": div(c["false_positive"], c["not_paying"]),
            "harmful_rate": div(c["harmful"], c["recommended"])}


def support(c):
    """Numerator and denominator of every rate, in the weighted counts."""
    return {"coverage": (c["recommended"], c["cases"]),
            "precision": (c["cleared"], c["cleared"] + c["false_positive"]),
            "recall": (c["cleared"], c["paying"]),
            "false_positive_rate": (c["false_positive"], c["not_paying"]),
            "harmful_rate": (c["harmful"], c["recommended"])}


def intervals(estimates, resampled):
    """The resampling interval of each rate, or None for a rate of exactly 0
    or 1. A percentile bootstrap cannot return a value it has not drawn, so
    at the boundary it gives a zero-width interval that reads as certainty.
    Such a rate is reported with its count (see support) and no interval."""
    return {name: (None if estimates.get(name) in (0.0, 1.0) else ci)
            for name, ci in resampled.items()}


def bootstrap(rows, draws=DRAWS, seed=SEED):
    """Percentile intervals from resampling cases with replacement. A drawn
    case keeps all its allocations. A draw whose denominator is zero is
    dropped for that rate and counted."""
    cases = sorted({r["case"] for r in rows})
    by_case = {c: [r for r in rows if r["case"] == c] for c in cases}
    rng = random.Random(seed)
    samples, dropped = {}, {}
    for _ in range(draws):
        picked = []
        for i in range(len(cases)):
            for r in by_case[rng.choice(cases)]:
                picked.append(dict(r, case="%s#%d" % (r["case"], i)))
        for name, value in rates(tally(picked)).items():
            if value is None:
                dropped[name] = dropped.get(name, 0) + 1
            else:
                samples.setdefault(name, []).append(value)
    out = {}
    for name, values in samples.items():
        values.sort()
        lo = values[int(0.025 * (len(values) - 1))]
        hi = values[int(round(0.975 * (len(values) - 1)))]
        out[name] = (lo, hi, dropped.get(name, 0))
    return out


def aggregate(rows, rule):
    """Whole-case counts under another way of combining a case's
    allocations: 'all' needs every allocation in the category, 'any' one."""
    test = all if rule == "all" else any
    by_case = {}
    for r in rows:
        by_case.setdefault(r["case"], []).append(r)
    out = dict.fromkeys(("recommended", "cleared", "false_positive"), 0)
    for rs in by_case.values():
        live = [r for r in rs if category(r) != "missing"]
        if not live:
            continue
        out["recommended"] += int(test(r["verdict"] not in ABSTAIN for r in live))
        out["cleared"] += int(test(category(r) == "cleared" for r in live))
        out["false_positive"] += int(test(category(r) == "false_positive" for r in live))
    return out


def build(draws=DRAWS):
    e7 = e7_rows(os.path.join(DIAG, "E7_RESULT.txt"))
    e8 = e8_rows(os.path.join(DIAG, "E8_RESULT.json"))
    sets = {"E6 as scored (post hoc)": e6_rows("as_scored"),
            "E6 extractor as frozen": e6_rows("extractor_as_frozen"),
            "E6 corrected (post hoc)": e6_rows("corrected"),
            "E7": e7, "E8": e8, "E7 and E8": e7 + e8}
    result = {}
    for name, rows in sets.items():
        result[name] = {}
        for method in sorted({r["method"] for r in rows}):
            mine = [r for r in rows if r["method"] == method]
            counts = tally(mine)
            estimates = rates(counts)
            result[name][method] = {
                "counts": counts, "rates": estimates, "support": support(counts),
                "intervals": intervals(estimates, bootstrap(mine, draws=draws)),
                "all_allocations": aggregate(mine, "all"),
                "any_allocation": aggregate(mine, "any")}
    return result


def render(result):
    lines = ["Selection metrics (RETROSPECTIVE, 2026-10-07). Half-counts: an allocation",
             "of a case with k allocations counts 1/k. Intervals: 95%% percentile, %d" % DRAWS,
             "resamples of cases, seed %d. A dash is an undefined rate. A rate of 0 or 1" % SEED,
             "has no resampling interval; its count is printed in parentheses.", ""]
    for name, by_method in result.items():
        lines.append("== %s ==" % name)
        for method, block in by_method.items():
            c, r, ci, n = block["counts"], block["rates"], block["intervals"], block["support"]
            lines.append("%-6s cases %g | recommended %g, cleared %g, did not pay %g, not tested %g | "
                         "abstained right %g, missed %g"
                         % (method, c["cases"], c["recommended"], c["cleared"], c["false_positive"],
                            c["not_tested"], c["abstain_correct"], c["abstain_miss"]))
            for rate in ("coverage", "precision", "recall", "false_positive_rate"):
                if r[rate] is None:
                    lines.append("         %-20s -" % rate)
                elif ci[rate] is None:
                    lines.append("         %-20s %.2f (%g of %g)" % (rate, r[rate], n[rate][0], n[rate][1]))
                else:
                    lo, hi, dropped = ci[rate]
                    lines.append("         %-20s %.2f [%.2f, %.2f]%s"
                                 % (rate, r[rate], lo, hi,
                                    "" if not dropped else " (%d draws undefined)" % dropped))
            lines.append("         all allocations agree: %s; any allocation: %s"
                         % (block["all_allocations"], block["any_allocation"]))
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    out = build()
    json.dump(out, io.open(os.path.join(HERE, "METRICS.json"), "w", encoding="utf-8"),
              indent=1, sort_keys=True)
    io.open(os.path.join(HERE, "METRICS.txt"), "w", encoding="utf-8", newline="\n").write(render(out))
    print(render(out))
