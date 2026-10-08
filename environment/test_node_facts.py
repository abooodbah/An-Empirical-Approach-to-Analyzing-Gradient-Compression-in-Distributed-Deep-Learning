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
CX6 = "Mellanox Technologies MT28908 Family [ConnectX-6]"
BF2 = "Mellanox Technologies MT42822 BlueField-2 integrated ConnectX-6 Dx network controller"


def on(host, text=RECORD):
    return text.replace("c302-004", host)


def other_adapter(host):
    """The record of the one node that differs: another adapter and firmware,
    with the revision suffix lspci prints for it."""
    return on(host).replace("20.32.2004", "24.33.1048").replace(CX6, BF2 + " (rev 01)")


def write_runs(tmp_path, layout):
    dirs = []
    for run, records in layout:
        d = tmp_path / run / "provenance"
        d.mkdir(parents=True, exist_ok=True)
        for i, text in enumerate(records):
            io.open(str(d / ("node_facts.%d.txt" % i)), "w", encoding="utf-8").write(text)
        dirs.append(str(tmp_path / run))
    return dirs


def test_one_record_gives_the_gpu_cpu_adapter_and_placement_facts():
    f = N.parse(RECORD)
    assert f["host"] == "c302-004"
    assert f["gpu"] == "NVIDIA A100-PCIE-40GB, 40960 MiB"
    assert f["gpu_driver"] == "570.195.03" and f["gpu_vbios"] == "92.00.25.00.08"
    assert f["cpu"] == "2 x AMD EPYC 7763 64-Core Processor"
    assert f["adapter"] == CX6
    assert f["adapter_firmware"] == "20.32.2004" and f["adapter_numa_node"] == "0"
    assert f["gpu_numa_nodes"] == "0, 1, 1"
    assert f["gpu_to_adapter"] == "NODE, SYS, SYS"
    assert f["kernel"] == "4.18.0-553.150.1.el8_10.x86_64"


def test_a_section_that_is_absent_gives_no_key_and_no_error():
    f = N.parse("c315-001.ls6.tacc.utexas.edu\n== gpu\n"
                "index, name, memory.total [MiB], driver_version, vbios_version, pci.bus_id\n"
                "0, NVIDIA A100-PCIE-40GB, 40960 MiB, 570.195.03, 92.00.25.00.08, 00000000:21:00.0\n")
    assert f == {"host": "c315-001", "gpu": "NVIDIA A100-PCIE-40GB, 40960 MiB", "gpu_driver": "570.195.03",
                 "gpu_vbios": "92.00.25.00.08"}


def test_gpus_that_differ_on_one_node_are_all_listed():
    text = RECORD.replace("1, NVIDIA A100-PCIE-40GB, 40960 MiB, 570.195.03,", "1, NVIDIA A100-PCIE-40GB, 40960 MiB, 571.0,")
    assert N.parse(text)["gpu_driver"] == "570.195.03; 571.0"


def test_the_adapter_name_drops_the_revision_and_an_ethernet_mode_card_is_still_an_adapter():
    assert N.parse(other_adapter("c309-001"))["adapter"] == BF2
    ethernet = RECORD.replace("Infiniband controller: " + CX6, "Ethernet controller: " + CX6 + " (rev 02)")
    assert N.parse(ethernet)["adapter"] == CX6


def test_a_processor_count_that_is_not_recorded_is_left_out_not_guessed():
    f = N.parse(RECORD.replace("Socket(s):           2\n", ""))
    assert f["cpu"] == "AMD EPYC 7763 64-Core Processor"


def test_two_adapters_on_one_node_are_both_reported():
    text = RECORD.replace("/sys/class/infiniband/mlx5_0 numa_node=0\n",
                          "/sys/class/infiniband/mlx5_0 numa_node=0\n"
                          "hca_id:\tmlx5_1\n\tfw_ver:\t\t\t\t24.33.1048\n"
                          "/sys/class/infiniband/mlx5_1 numa_node=1\n")
    f = N.parse(text)
    assert f["adapter_firmware"] == "20.32.2004; 24.33.1048" and f["adapter_numa_node"] == "0; 1"


def test_the_summary_counts_nodes_and_not_the_records_of_jobs_that_shared_them(tmp_path):
    # two jobs on the same two nodes; the second job has a third node with another adapter
    dirs = write_runs(tmp_path, (("e9-q12a-4n-1", [on("c315-001"), on("c315-003")]),
                                 ("e9-q24a-8n-3", [other_adapter("c309-001"), on("c315-001"), on("c315-003")])))
    out = N.summary(dirs)
    assert out["records"] == 5 and out["nodes"] == 3
    assert out["runs"] == {"e9-q12a-4n-1": ["c315-001", "c315-003"],
                           "e9-q24a-8n-3": ["c309-001", "c315-001", "c315-003"]}
    assert out["facts"]["adapter_firmware"] == {"20.32.2004": 2, "24.33.1048": 1}     # records would say 4 and 1
    assert out["facts"]["adapter"] == {CX6: 2, BF2: 1}
    assert out["facts"]["cpu"] == {"2 x AMD EPYC 7763 64-Core Processor": 3}
    assert out["by_node"]["c309-001"]["adapter"] == BF2
    assert out["by_node"]["c315-001"]["adapter_firmware"] == "20.32.2004"
    assert "host" not in out["by_node"]["c315-001"] and "host" not in out["facts"]


def test_a_node_whose_records_disagree_shows_both_values(tmp_path):
    dirs = write_runs(tmp_path, (("e9-q12a-4n-1", [on("c315-001")]),
                                 ("e9-q12b-4n-2", [on("c315-001").replace("20.32.2004", "20.40.1000")])))
    out = N.summary(dirs)
    assert out["nodes"] == 1 and out["records"] == 2
    assert out["by_node"]["c315-001"]["adapter_firmware"] == "20.32.2004 | 20.40.1000"
    assert out["facts"]["adapter_firmware"] == {"20.32.2004 | 20.40.1000": 1}
    assert out["facts"]["gpu_driver"] == {"570.195.03": 1}


def test_a_record_without_a_host_line_is_not_merged_with_another(tmp_path):
    nameless = RECORD[RECORD.index("== gpu"):]
    dirs = write_runs(tmp_path, (("e9-q12a-4n-1", [nameless, nameless]),))
    out = N.summary(dirs)
    assert out["nodes"] == 2
    assert out["runs"] == {"e9-q12a-4n-1": ["e9-q12a-4n-1/node_facts.0.txt", "e9-q12a-4n-1/node_facts.1.txt"]}


# ---- the committed record of the four E9 jobs
def test_the_committed_record_of_the_e9_jobs():
    out = json.load(io.open(os.path.join(HERE, "E9_NODE_FACTS.json"), encoding="utf-8"))
    small = ["c315-001", "c315-003", "c315-010", "c315-011"]
    large = sorted(small + ["c302-004", "c309-001", "c316-012", "c317-014"])
    assert out["records"] == 24 and out["nodes"] == 8
    # the two jobs of a scale ran on the same nodes, and the 24-rank jobs contain the 12-rank nodes
    assert out["runs"] == {"e9-q12a-4n-3495229": small, "e9-q12b-4n-3495230": small,
                           "e9-q24a-8n-3495231": large, "e9-q24b-8n-3495232": large}
    facts = out["facts"]
    assert facts["gpu"] == {"NVIDIA A100-PCIE-40GB, 40960 MiB": 8}
    assert facts["gpu_driver"] == {"570.195.03": 8} and facts["gpu_vbios"] == {"92.00.25.00.08": 8}
    assert facts["cpu"] == {"2 x AMD EPYC 7763 64-Core Processor": 8}
    assert facts["adapter"] == {CX6: 7, BF2: 1}
    assert facts["adapter_firmware"] == {"20.32.2004": 7, "24.33.1048": 1}
    assert [h for h, f in sorted(out["by_node"].items()) if f["adapter"] == BF2] == ["c309-001"]
    assert out["by_node"]["c309-001"]["adapter_firmware"] == "24.33.1048"
    assert facts["adapter_numa_node"] == {"0": 8}
    assert facts["gpu_numa_nodes"] == {"0, 1, 1": 8} and facts["gpu_to_adapter"] == {"NODE, SYS, SYS": 8}
    assert facts["kernel"] == {"4.18.0-553.150.1.el8_10.x86_64": 8}
