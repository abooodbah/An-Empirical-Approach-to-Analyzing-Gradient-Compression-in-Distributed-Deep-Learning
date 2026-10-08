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


def test_render_names_each_source_and_says_which_jobs_record_node_facts(tmp_path):
    # Only the Lonestar6 launch script of Revision 4 writes node_facts; the
    # Stampede3 one does not, so the Intel system's topology stays unrecorded.
    d = tmp_path / "run-123"
    d.mkdir()
    io.open(str(d / "x__rank0000.provenance.json"), "w", encoding="utf-8").write(json.dumps({"libraries": {"torch": "2"}}))
    text = P.render([("System A", str(d))])
    assert "Read from the run directory `run-123`." in text
    assert "The Lonestar6 jobs of Revision 4 (E9) record these in `provenance/node_facts.*.txt`;" in text
    assert "`environment/node_facts.py` counts each value over their node records in" in text
    assert "`environment/E9_NODE_FACTS.json`." in text
    assert "The Stampede3 jobs (E10) do not" in text
