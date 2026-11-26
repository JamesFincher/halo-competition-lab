#!/usr/bin/env python3
"""Parallel experiment swarm runner — launches all experiments concurrently on 1x H100.

Each experiment runs as its own python process inside the stock halo:hopper container.
GPU utilization/VRAM is logged continuously via nvidia-smi dmon into results/swarm_gpu_log.csv.
Per-experiment stdout+stderr stream into results/swarm_logs/<name>.log.
"""
from __future__ import annotations
import json, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

EXPERIMENTS = [
    ("exp19_crosstalk", "experiments/surgical_crosstalk_audit/scripts/run_crosstalk_audit.py"),
    ("exp20_randomctrl", "experiments/random_node_control_proof/scripts/run_random_control_proof.py"),
    ("exp21_dla_disjrl", "experiments/ig5_dla_closed_loop_rl/scripts/run_dla_disjoint_rl.py"),
    ("exp22_pareto", "experiments/multi_seed_stability_sweep/scripts/run_pareto_sweep.py"),
    ("exp23_pushpull", "experiments/dual_polarity_circuit_surgery/scripts/run_pushpull.py"),
    ("exp24_adaptivek", "experiments/adaptive_k_proportional_edit/scripts/run_adaptive_k.py"),
    ("exp25_evolution", "experiments/gradfree_evolution_subspace/scripts/run_evolution.py"),
    ("exp26_entropy_rl", "experiments/entropy_weighted_fork_rl/scripts/run_entropy_grpo.py"),
    ("exp27_invpatch", "experiments/inverse_causal_patching/scripts/run_inverse_patching.py"),
    ("exp28_synergy", "experiments/crosslayer_head_mlp_synergy/scripts/run_synergy.py"),
    ("exp29_annealed_rl", "experiments/momentum_annealed_rl/scripts/run_annealed_rl.py"),
    ("exp30_steering_rl", "experiments/steering_oracle_guided_rl/scripts/run_steering_rl.py"),
]

def main():
    t0 = time.time()
    logdir = ROOT / "results/swarm_logs"
    logdir.mkdir(parents=True, exist_ok=True)
    gpu_log = ROOT / "results/swarm_gpu_log.csv"
    print(f"[SWARM] Launching {len(EXPERIMENTS)} experiments in parallel at {time.strftime('%H:%M:%S')}", flush=True)

    gpu_mon = subprocess.Popen(
        ["nvidia-smi", "dmon", "-s", "pucm", "-d", "2", "-o", "DT"],
        stdout=open(gpu_log, "w"), stderr=subprocess.DEVNULL,
    )

    procs = {}
    for name, script in EXPERIMENTS:
        lf = open(logdir / f"{name}.log", "w")
        p = subprocess.Popen([sys.executable, str(ROOT / script)], stdout=lf, stderr=subprocess.STDOUT, cwd=str(ROOT))
        procs[name] = (p, lf)
        print(f"[SWARM] launched {name} pid={p.pid}", flush=True)

    status = {}
    while procs:
        for name, (p, lf) in list(procs.items()):
            rc = p.poll()
            if rc is not None:
                lf.close()
                status[name] = {"rc": rc, "wall": round(time.time() - t0, 1)}
                print(f"[SWARM] {name} finished rc={rc} at +{status[name]['wall']}s", flush=True)
                del procs[name]
        time.sleep(3)
    gpu_mon.terminate()

    summary = {"total_wall_sec": round(time.time() - t0, 1), "experiments": status,
               "gpu_log": str(gpu_log), "n_experiments": len(EXPERIMENTS)}
    (ROOT / "results/swarm_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[SWARM DONE] all {len(EXPERIMENTS)} experiments in {summary['total_wall_sec']}s", flush=True)

if __name__ == "__main__":
    main()
