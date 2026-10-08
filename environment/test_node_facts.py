# environment/test_node_facts.py
# -*- coding: utf-8 -*-
import io
import json
import os

import node_facts as N

HERE = os.path.dirname(os.path.abspath(__file__))

# one record of job 3495231 (node index 0), shortened in the numa section
RECORD = (
    "c302-004.ls6.tacc.utexas.edu\n"
    "== gpu\n"
    "index, name, memory.total [MiB], driver_version, vbios_version, pci.bus_id\n"
    "0, NVIDIA A100-PCIE-40GB, 40960 MiB, 570.195.03, 92.00.25.00.08, 00000000:21:00.0\n"
    "1, NVIDIA A100-PCIE-40GB, 40960 MiB, 570.195.03, 92.00.25.00.08, 00000000:81:00.0\n"
    "2, NVIDIA A100-PCIE-40GB, 40960 MiB, 570.195.03, 92.00.25.00.08, 00000000:E2:00.0\n"
    "== topology\n"
    "\t\x1b[4mGPU0\tGPU1\tGPU2\tNIC0\tCPU Affinity\tNUMA Affinity\tGPU NUMA ID\x1b[0m\n"
    "GPU0\t X \tSYS\tSYS\tNODE\t0-63\t0\t\tN/A\n"
    "GPU1\tSYS\t X \tNODE\tSYS\t64-127\t1\t\tN/A\n"
    "GPU2\tSYS\tNODE\t X \tSYS\t64-127\t1\t\tN/A\n"
    "NIC0\tNODE\tSYS\tSYS\t X \t\t\t\t\n"
    "\n"
    "Legend:\n"
    "\n"
    "  X    = Self\n"
    "NIC Legend:\n"
    "\n"
    "  NIC0: mlx5_0\n"
    "\n"
    "== cpu\n"
    "Thread(s) per core:  1\n"
    "Core(s) per socket:  64\n"
    "Socket(s):           2\n"
    "NUMA node(s):        2\n"
    "Model name:          AMD EPYC 7763 64-Core Processor\n"
    "== numa\n"
    "available: 2 nodes (0-1)\n"
    "== adapter\n"
    "hca_id:\tmlx5_0\n"
    "\tfw_ver:\t\t\t\t20.32.2004\n"
    "\tboard_id:\t\t\tDEL0000000010\n"
    "/sys/class/infiniband/mlx5_0 numa_node=0\n"
    "== pci\n"
    "21:00.0 3D controller: NVIDIA Corporation GA100 [A100 PCIe 40GB] (rev a1)\n"
    "41:00.0 Infiniband controller: Mellanox Technologies MT28908 Family [ConnectX-6]\n"
    "81:00.0 3D controller: NVIDIA Corporation GA100 [A100 PCIe 40GB] (rev a1)\n"
    "== kernel\n"
    "4.18.0-553.150.1.el8_10.x86_64\n"
)


def test_one_record_gives_the_gpu_cpu_adapter_and_placement_facts():
    f = N.parse(RECORD)
    assert f["gpu"] == "NVIDIA A100-PCIE-40GB, 40960 MiB"
    assert f["gpu_driver"] == "570.195.03" and f["gpu_vbios"] == "92.00.25.00.08"
    assert f["cpu"] == "2 x AMD EPYC 7763 64-Core Processor"
    assert f["adapter"] == "Mellanox Technologies MT28908 Family [ConnectX-6]"
    assert f["adapter_firmware"] == "20.32.2004" and f["adapter_numa_node"] == "0"
    assert f["gpu_numa_nodes"] == "0, 1, 1"
    assert f["gpu_to_adapter"] == "NODE, SYS, SYS"
    assert f["kernel"] == "4.18.0-553.150.1.el8_10.x86_64"


def test_a_section_that_is_absent_gives_no_key_and_no_error():
    f = N.parse("host\n== gpu\nindex, name, memory.total [MiB], driver_version, vbios_version, pci.bus_id\n"
                "0, NVIDIA A100-PCIE-40GB, 40960 MiB, 570.195.03, 92.00.25.00.08, 00000000:21:00.0\n")
    assert f == {"gpu": "NVIDIA A100-PCIE-40GB, 40960 MiB", "gpu_driver": "570.195.03",
                 "gpu_vbios": "92.00.25.00.08"}


def test_gpus_that_differ_on_one_node_are_all_listed():
    text = RECORD.replace("1, NVIDIA A100-PCIE-40GB, 40960 MiB, 570.195.03,", "1, NVIDIA A100-PCIE-40GB, 40960 MiB, 571.0,")
    assert N.parse(text)["gpu_driver"] == "570.195.03; 571.0"


def test_the_summary_counts_the_node_records_that_carry_each_value(tmp_path):
    for run, node, text in (("e9-q12a-4n-1", 0, RECORD), ("e9-q12a-4n-1", 1, RECORD),
                            ("e9-q24a-8n-3", 0, RECORD.replace("20.32.2004", "24.33.1048"))):
        d = tmp_path / run / "provenance"
        d.mkdir(parents=True, exist_ok=True)
        io.open(str(d / ("node_facts.%d.txt" % node)), "w", encoding="utf-8").write(text)
    out = N.summary([str(tmp_path / "e9-q12a-4n-1"), str(tmp_path / "e9-q24a-8n-3")])
    assert out["records"] == 3
    assert out["runs"] == {"e9-q12a-4n-1": 2, "e9-q24a-8n-3": 1}
    assert out["facts"]["adapter_firmware"] == {"20.32.2004": 2, "24.33.1048": 1}
    assert out["facts"]["gpu_numa_nodes"] == {"0, 1, 1": 3}
    assert out["facts"]["cpu"] == {"2 x AMD EPYC 7763 64-Core Processor": 3}


# ---- the committed record of the four E9 jobs
def test_the_committed_record_of_the_e9_jobs():
    out = json.load(io.open(os.path.join(HERE, "E9_NODE_FACTS.json"), encoding="utf-8"))
    assert out["records"] == 24 and sorted(out["runs"].values()) == [4, 4, 8, 8]
    facts = out["facts"]
    assert facts["gpu"] == {"NVIDIA A100-PCIE-40GB, 40960 MiB": 24}
    assert facts["gpu_driver"] == {"570.195.03": 24} and facts["gpu_vbios"] == {"92.00.25.00.08": 24}
    assert facts["cpu"] == {"2 x AMD EPYC 7763 64-Core Processor": 24}
    assert facts["adapter_firmware"] == {"20.32.2004": 22, "24.33.1048": 2}
    assert facts["adapter_numa_node"] == {"0": 24}
    assert facts["gpu_numa_nodes"] == {"0, 1, 1": 24} and facts["gpu_to_adapter"] == {"NODE, SYS, SYS": 24}
    assert facts["kernel"] == {"4.18.0-553.150.1.el8_10.x86_64": 24}
