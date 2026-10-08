# environment/test_platform_facts.py
# -*- coding: utf-8 -*-
import io
import json

import platform_facts as P


def test_distinct_values_are_collected_by_dotted_key(tmp_path):
    a = {"libraries": {"torch": "2.4.1+cu121"}, "allowlisted_environment": {"NCCL_NET_GDR_LEVEL": "SYS", "HOME": "/x"},
         "host_layout": {"ranks_per_node": 3}}
    b = {"libraries": {"torch": "2.4.1+cu121"}, "allowlisted_environment": {"NCCL_NET_GDR_LEVEL": "LOC"},
         "host_layout": {"ranks_per_node": 3}}
    paths = []
    for i, d in enumerate((a, b)):
        p = tmp_path / ("r%d.provenance.json" % i)
        io.open(str(p), "w", encoding="utf-8").write(json.dumps(d))
        paths.append(str(p))
    facts = P.collect(paths)
    assert facts["libraries.torch"] == ["2.4.1+cu121"]
    assert facts["allowlisted_environment.NCCL_NET_GDR_LEVEL"] == ["LOC", "SYS"]
    assert facts["host_layout.ranks_per_node"] == ["3"]
    assert "allowlisted_environment.HOME" not in facts
