# validation_v2/analysis/tests/test_rev4_scorers.py
# -*- coding: utf-8 -*-
import io
import json

import pytest

import rev4_records as R
import score_e9 as E9
import score_e10 as E10


def rows(run_id, n_iter=2, ranks=2, **over):
    base = {"run_id": run_id, "status": "PASS", "final_param_hashes_equal": "True",
            "block_step_equiv_s": "", "observed_step_s": "", "network_s": "", "peak_memory_mb": ""}
    base.update(over)
    return [dict(base, iteration=str(i), rank=str(r)) for i in range(n_iter) for r in range(ranks)]


def test_block_value_is_one_number_in_milliseconds():
    assert R.block_step_ms(rows("x", block_step_equiv_s="0.0125")) == pytest.approx(12.5)
    mixed = rows("x", block_step_equiv_s="0.0125") + rows("x", block_step_equiv_s="0.02")
    with pytest.raises(ValueError):
        R.block_step_ms(mixed)


def test_phase_step_takes_the_slowest_rank_per_iteration():
    data = [{"iteration": "0", "rank": "0", "observed_step_s": "1.0", "network_s": "0.2"},
            {"iteration": "0", "rank": "1", "observed_step_s": "3.0", "network_s": "0.4"},
            {"iteration": "1", "rank": "0", "observed_step_s": "2.0", "network_s": "0.2"},
            {"iteration": "1", "rank": "1", "observed_step_s": "1.0", "network_s": "0.4"}]
    assert R.phase_step_ms(data) == pytest.approx(2500.0)
    assert R.exchange_ms(data) == pytest.approx(300.0)


def test_validity_needs_equal_parameters_and_an_allowed_status():
    assert R.valid(rows("x"))
    assert not R.valid(rows("x", final_param_hashes_equal="False"))
    assert not R.valid(rows("x", status="QUALIFICATION_ONLY"))
    assert R.valid(rows("x", status="QUALIFICATION_ONLY"), statuses=("PASS", "QUALIFICATION_ONLY"))
    assert not R.valid([])


def test_read_dir_groups_records_sidecars_and_progress(tmp_path):
    (tmp_path / "results").mkdir()
    (tmp_path / "progress").mkdir()
    io.open(str(tmp_path / "results" / "runA__c__rank0000.csv"), "w", encoding="utf-8", newline="\n").write(
        "run_id,iteration,rank\nrunA,0,0\n")
    io.open(str(tmp_path / "results" / "runA__c__rank0000.v52.json"), "w", encoding="utf-8").write(
        json.dumps({"run_id": "runA", "rank": 0, "fidelity": []}))
    io.open(str(tmp_path / "progress" / "runB__c__rank0001.progress.jsonl"), "w", encoding="utf-8").write(
        '{"iteration": 0, "observed_step_s": 1.0, "network_s": 0.5, "peak_memory_mb": 10.0}\n')
    runs = R.read_dir(str(tmp_path))
    assert len(runs["runA"]["rows"]) == 1 and 0 in runs["runA"]["sidecar"]
    assert runs["runB"]["progress"][1][0]["network_s"] == 0.5


def e9_runs(scale, times):
    out = {}
    for (model, mode), (p1, p2) in times.items():
        for n, t in ((1, p1), (2, p2)):
            rid = "redE9-ws%d-%s-%s-block-p%d" % (scale, model, mode, n)
            out[rid] = {"rows": rows(rid, block_step_equiv_s=str(t / 1000.0)), "sidecar": {}, "progress": {}}
    return out


def test_e9_table_averages_the_two_passes_and_drops_invalid_runs():
    runs = e9_runs(12, {("transformer", "native"): (16.0, 18.0), ("transformer", "gpu-quant-sr-fp16"): (14.0, 14.0)})
    bad = "redE9-ws12-transformer-gpu-quant-block-p1"
    runs[bad] = {"rows": rows(bad, block_step_equiv_s="0.03", final_param_hashes_equal="False"),
                 "sidecar": {}, "progress": {}}
    table = E9.block_table(runs)
    assert table[("transformer", "native")] == {"T_ms": pytest.approx(17.0), "passes": 2}
    assert ("transformer", "gpu-quant") not in table
    assert E9.ratios(table)[("transformer", "gpu-quant-sr-fp16", "native")] == pytest.approx(14.0 / 17.0)


@pytest.mark.parametrize("values,word", [([0.80, 0.94], "faster"), ([1.06, 1.30], "slower"),
                                         ([0.97, 1.04], "within 5%"), ([0.90, 1.02], "allocation-dependent"),
                                         ([0.95, 0.95], "faster"), ([1.05, 1.05], "slower"),
                                         ([0.80], "single allocation")])
def test_e9_wording_follows_the_five_percent_rule(values, word):
    assert E9.call(values) == word


def test_e9_fidelity_reads_rank_zero_sidecars():
    rid = "redE9-ws12-resnet50-gpu-quant-sr-int8-fid"
    runs = {rid: {"rows": [], "progress": {}, "sidecar": {0: {"levels_per_side": 10, "fidelity": [
        {"relative_l2_error": 0.2, "cosine": 0.98}, {"relative_l2_error": 0.4, "cosine": 0.96}]}}}}
    f = E9.fidelity(runs)[("resnet50", "gpu-quant-sr-int8")]
    assert f["relative_l2_error"] == pytest.approx(0.3) and f["cosine"] == pytest.approx(0.97)
    assert f["levels_per_side"] == 10 and f["iterations"] == 2


def probe(slope_ar, slope_ag, intercept=0.001, bend=1.0):
    sizes = [4, 1024, 1 << 20, 16 << 20, 64 << 20, 256 << 20, 1 << 30]
    line = lambda b: [intercept + b * s * (bend if s > (64 << 20) else 1.0) for s in sizes]
    return {"allreduce": {"sizes_bytes": sizes, "times_s": line(slope_ar)},
            "allgather": {"sizes_bytes": sizes, "times_s": line(slope_ag)}}


def test_slope_recovers_a_line():
    p = probe(2e-11, 1.2e-10)
    b, a = R.slope(p["allreduce"]["sizes_bytes"], p["allreduce"]["times_s"])
    assert b == pytest.approx(2e-11) and a == pytest.approx(0.001)


def test_probe_checks_hold_for_a_ring_that_follows_the_traffic_model():
    checks = E9.probe_checks({"ring": probe(2e-11, 1.2e-10), "default": probe(1e-11, 1.2e-10)}, 12)
    assert checks["P1"]["slope_ratio"] == pytest.approx(6.0) and checks["P1"]["holds"]
    assert checks["P2"]["holds"] and checks["P3"]["holds"]


def test_probe_checks_fail_when_the_data_say_so():
    checks = E9.probe_checks({"ring": probe(2e-11, 4e-10), "default": probe(3e-11, 4e-10, bend=1.5)}, 12)
    assert not checks["P1"]["holds"] and not checks["P2"]["holds"] and not checks["P3"]["holds"]


def e10_run(rid, step, net, peak, n_iter=20, ranks=2, **over):
    return {"rows": rows(rid, n_iter=n_iter, ranks=ranks, observed_step_s=str(step), network_s=str(net),
                         peak_memory_mb=str(peak), status="QUALIFICATION_ONLY", **over),
            "sidecar": {}, "progress": {}}


def test_e10_complete_run():
    c = E10.condition(e10_run("redE10-ws16-llama1b-gpu-quant-bucketed", 20.0, 15.0, 36000.0), 2)
    assert c["state"] == "complete" and c["iterations"] == 20
    assert c["step_ms"] == pytest.approx(20000.0) and c["exchange_ms"] == pytest.approx(15000.0)
    assert c["peak_gb"] == pytest.approx(36.0)


def test_e10_cut_off_run_is_reported_from_its_progress_file():
    line = {"observed_step_s": 50.0, "network_s": 45.0, "peak_memory_mb": 35000.0}
    progress = {0: [dict(line, iteration=i) for i in range(7)], 1: [dict(line, iteration=i) for i in range(6)]}
    c = E10.condition({"rows": [], "sidecar": {}, "progress": progress}, 2)
    assert c["state"] == "incomplete" and c["iterations"] == 6 and c["step_ms"] == pytest.approx(50000.0)
    short = {0: progress[0][:4], 1: progress[1][:4]}
    assert E10.condition({"rows": [], "sidecar": {}, "progress": short}, 2)["state"] == "missing"
    assert E10.condition({"rows": [], "sidecar": {}, "progress": {0: progress[0]}}, 2)["state"] == "missing"


def test_e10_unequal_parameters_make_a_run_invalid():
    run = e10_run("redE10-ws16-llama1b-native-bucketed", 7.0, 5.0, 33000.0, final_param_hashes_equal="False")
    assert E10.condition(run, 2)["state"] == "invalid"


def test_e10_prediction_sums_the_buckets_including_the_short_last_one():
    elements = 2 * 16777216 + 100
    got = E10.predicted_exchange_ms(0.001, 1e-9, elements, 1)
    want = 1000.0 * (3 * 0.001 + 1e-9 * elements)
    assert got == pytest.approx(want)
