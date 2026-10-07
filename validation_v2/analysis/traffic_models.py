# validation_v2/analysis/traffic_models.py
# -*- coding: utf-8 -*-
"""Byte counts of the collectives the paper compares: closed forms, and a
link-load counter the tests use to check them. ANALYSIS (2026-10-07). These
are traffic models. They predict bytes, not time, and nothing here is a
measurement."""
import math
from collections import defaultdict


def ring_loads(p, payload, kind):
    """Bytes on each directed ring link i -> i+1 for one collective of
    `payload` bytes per rank. An all-reduce splits the payload into p chunks;
    chunk k is reduced along p-1 links ending at rank k and then broadcast
    along p-1 links starting at rank k. A reduce-scatter is the reduce half.
    In an all-gather each rank's block travels p-1 links."""
    loads = [0.0] * p
    if kind == "allgather":
        for source in range(p):
            for hop in range(p - 1):
                loads[(source + hop) % p] += payload
        return loads
    if kind not in ("allreduce", "reduce_scatter"):
        raise ValueError(kind)
    chunk = payload / p
    for k in range(p):
        for hop in range(p - 1):
            loads[(k + 1 + hop) % p] += chunk
        if kind == "allreduce":
            for hop in range(p - 1):
                loads[(k + hop) % p] += chunk
    return loads


def inter_node_loads(p, per_node, payload, kind):
    """The ring links that cross a node boundary when ranks are laid out
    node by node, per_node ranks to a node."""
    loads = ring_loads(p, payload, kind)
    return [loads[i] for i in range(p) if i // per_node != ((i + 1) % p) // per_node]


def tree_nic_loads(trees, payload):
    """Bytes each node sends (it receives the same) when `payload` is
    reduced and then broadcast over spanning trees of the nodes, each tree
    carrying an equal share. On every tree edge the child sends its share up
    once and the parent sends it down once."""
    share = payload / len(trees)
    loads = defaultdict(float)
    for edges in trees:
        for a, b in edges:
            loads[a] += share
            loads[b] += share
    return dict(loads)


def received(kind, p, G, c=1.0):
    """Bytes one rank receives per exchange of a gradient of G dense bytes
    compressed by c."""
    if kind == "allreduce":
        return 2.0 * (p - 1) / p * G / c
    if kind == "reduce_scatter":
        return (p - 1) / p * G / c
    if kind == "allgather":
        return (p - 1) * G / c
    if kind == "shard_alltoall":
        return (p - 1) * G / (p * c)
    raise ValueError(kind)


def sharded_saving_bound(c, param_gathers):
    """Largest fraction of a sharded step's collective bytes that compressing
    the gradient exchange by c can remove, when each iteration also gathers
    the parameters `param_gathers` times at the gradient's dense size."""
    return (1.0 - 1.0 / c) / (param_gathers + 1.0)


def levels_int8(p):
    """Levels per side of a shared-scale quantizer whose exact sum over p
    ranks must fit a signed byte."""
    return max(1, 127 // p)


def levels_fp16(p):
    """Levels per side when the integers travel in float16, which holds
    every integer up to 2048 exactly."""
    return 2048 // p


def stochastic_round(value, rng):
    """floor(value + u) with u uniform on [0, 1): unbiased, variance at most
    one quarter of a level squared."""
    return math.floor(value + rng.random())
