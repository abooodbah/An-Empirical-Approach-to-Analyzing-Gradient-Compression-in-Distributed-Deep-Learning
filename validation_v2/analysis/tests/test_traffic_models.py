# validation_v2/analysis/tests/test_traffic_models.py
# -*- coding: utf-8 -*-
import random

import pytest

import traffic_models as T


@pytest.mark.parametrize("p", [2, 3, 4, 8, 12, 24, 32])
def test_ring_link_loads_match_the_closed_forms(p):
    M = 1000.0
    assert T.ring_loads(p, M, "allgather") == pytest.approx([(p - 1) * M] * p)
    assert T.ring_loads(p, M, "allreduce") == pytest.approx([2 * (p - 1) * M / p] * p)
    assert T.ring_loads(p, M, "reduce_scatter") == pytest.approx([(p - 1) * M / p] * p)


def test_two_level_ring_crossings_keep_the_volume_ratio():
    gather = T.inter_node_loads(8, 4, 1.0, "allgather")
    reduce = T.inter_node_loads(8, 4, 1.0, "allreduce")
    assert len(gather) == 2 and gather == pytest.approx([7.0, 7.0])
    assert reduce == pytest.approx([1.75, 1.75])
    assert gather[0] / reduce[0] == pytest.approx(8 / 2)


def test_tree_reduce_between_two_nodes_moves_one_payload_per_node():
    loads = T.tree_nic_loads([[(0, 1)], [(1, 0)]], 1.0)
    assert loads == pytest.approx({0: 1.0, 1: 1.0})
    assert T.inter_node_loads(8, 4, 1.0, "allgather")[0] / loads[0] == pytest.approx(7.0)


def test_tree_reduce_over_four_nodes_stays_bounded():
    first = [(0, 1), (0, 2), (1, 3)]
    mirror = [(3, 2), (3, 1), (2, 0)]
    loads = T.tree_nic_loads([first, mirror], 1.0)
    assert loads == pytest.approx({0: 1.5, 1: 1.5, 2: 1.5, 3: 1.5})
    assert max(loads.values()) <= 2.0


@pytest.mark.parametrize("p,c", [(8, 4), (16, 4), (32, 4), (12, 2)])
def test_received_byte_ratios(p, c):
    G = 4.2e9
    dense_ar = T.received("allreduce", p, G)
    dense_rs = T.received("reduce_scatter", p, G)
    assert T.received("allgather", p, G, c) / dense_ar == pytest.approx(p / (2.0 * c))
    assert T.received("allgather", p, G, c) / dense_rs == pytest.approx(p / float(c))
    assert T.received("shard_alltoall", p, G, c) / dense_rs == pytest.approx(1.0 / c)


def test_sharded_saving_bound():
    assert T.sharded_saving_bound(4, 2) == pytest.approx(0.25)
    assert T.sharded_saving_bound(4, 1) == pytest.approx(0.375)
    assert T.sharded_saving_bound(2, 2) == pytest.approx(1.0 / 6.0)


@pytest.mark.parametrize("p", range(2, 65))
def test_level_counts_keep_every_partial_sum_representable(p):
    assert 1 <= T.levels_int8(p) and p * T.levels_int8(p) <= 127
    assert 1 <= T.levels_fp16(p) and p * T.levels_fp16(p) <= 2048


def test_level_counts_at_the_measured_rank_counts():
    assert [T.levels_int8(p) for p in (8, 12, 24, 32)] == [15, 10, 5, 3]
    assert [T.levels_fp16(p) for p in (12, 24)] == [170, 85]


def test_stochastic_rounding_is_unbiased_with_bounded_variance():
    rng = random.Random(11)
    for value in (-2.75, -0.5, 0.0, 0.2, 3.999):
        draws = [T.stochastic_round(value, rng) for _ in range(200000)]
        mean = sum(draws) / len(draws)
        var = sum((d - mean) ** 2 for d in draws) / len(draws)
        assert mean == pytest.approx(value, abs=0.01)
        assert var <= 0.25 + 0.005
        assert set(draws) <= {int(value // 1), int(value // 1) + 1}
