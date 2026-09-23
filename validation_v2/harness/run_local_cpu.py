# -*- coding: utf-8 -*-
"""Run the v5 harness on CPU with gloo in N local processes (qualification
only). Exits nonzero if any rank fails.

usage: python run_local_cpu.py <harness.py> <world_size> <port> <harness args...>
"""
import os
import subprocess
import sys

harness, world, port = sys.argv[1], int(sys.argv[2]), sys.argv[3]
args = sys.argv[4:]
procs = []
for rank in range(world):
    env = dict(os.environ, RANK=str(rank), WORLD_SIZE=str(world), LOCAL_RANK=str(rank),
               MASTER_ADDR="127.0.0.1", MASTER_PORT=port, USE_LIBUV="0",
               OMP_NUM_THREADS="2", PYTHONDONTWRITEBYTECODE="1")
    procs.append(subprocess.Popen([sys.executable, harness] + args, env=env,
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True))
bad = 0
for rank, p in enumerate(procs):
    out, _ = p.communicate(timeout=600)
    if p.returncode != 0:
        bad += 1
        print("rank %d exit %d\n%s" % (rank, p.returncode, out[-3000:]))
    elif rank == 0:
        print(out.strip().splitlines()[-1] if out.strip() else "")
sys.exit(1 if bad else 0)
