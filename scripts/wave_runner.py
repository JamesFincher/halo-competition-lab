#!/usr/bin/env python3
"""Rolling wave runner — reads experiments from a wave queue JSON, keeps GPU saturated.

Usage: python scripts/wave_runner.py <wave_queue.json>
Queue format: {"experiments": [{"name": "exp31", "script": "experiments/.../run_x.py"}, ...]}
Launches all in parallel, logs per-experiment to results/swarm_logs/<name>.log,
gpu telemetry to results/wave_gpu_log.csv, summary to results/wave_summary.json.
"""
from __future__ import annotations
import json, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

def main():
    queue_path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "scripts/wave2_queue.json"
    queue = json.load(open(queue_path))
    exps = [(e["name"], e["script"]) for e in queue["experiments"]]
    t0 = time.time()
    logdir = ROOT / "results/swarm_logs"
    logdir.mkdir(parents=True, exist_ok=True)
    gpu_log = ROOT / "results/wave_gpu_log.csv"
    print(f"[WAVE] Launching {len(exps)} experiments at {time.strftime('%H:%M:%S')}", flush=True)
    gpu_mon = subprocess.Popen(["nvidia-smi", "dmon", "-s", "pucm", "-d", "2", "-o", "DT"],
                               stdout=open(gpu_log, "w"), stderr=subprocess.DEVNULL)
    procs = {}
    for name, script in exps:
        lf = open(logdir / f"{name}.log", "w")
        lf.write(f"[WAVE LOG] experiment={name} script={script} wave_started_epoch={t0:.3f} log_started_epoch={time.time():.3f}\n")
        lf.flush()
        p = subprocess.Popen([sys.executable, "-u", str(ROOT / script)], stdout=lf, stderr=subprocess.STDOUT, cwd=str(ROOT))
        lf.write(f"[WAVE LOG] pid={p.pid}\n")
        lf.flush()
        procs[name] = (p, lf)
        print(f"[WAVE] launched {name} pid={p.pid}", flush=True)
    status = {}
    status_path = ROOT / "results/wave_status.json"

    def write_status(final=False):
        payload = {
            "wave": str(queue_path),
            "started_at_epoch": t0,
            "elapsed_sec": round(time.time() - t0, 1),
            "running": sorted(procs),
            "finished": status,
            "final": bool(final),
        }
        tmp = status_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
        tmp.replace(status_path)

    write_status()
    while procs:
        for name, (p, lf) in list(procs.items()):
            rc = p.poll()
            if rc is not None:
                lf.close()
                status[name] = {"rc": rc, "wall": round(time.time() - t0, 1)}
                print(f"[WAVE] {name} finished rc={rc} at +{status[name]['wall']}s", flush=True)
                if rc != 0:
                    # Preserve crash evidence without creating a .json file that the
                    # queue scanner would mistake for a completed experiment result.
                    log_path = logdir / f"{name}.log"
                    crash_dir = ROOT / "experiments" / name / "results" / "crashes"
                    try:
                        crash_dir.mkdir(parents=True, exist_ok=True)
                        lines = log_path.read_text(errors="replace").splitlines() if log_path.exists() else []
                        tail = "\n".join(lines[-200:])
                        (crash_dir / "last_crash.log").write_text(
                            f"experiment={name}\nrc={rc}\nwall_sec={status[name]['wall']}\nlog={log_path}\n\n{tail}\n"
                        )
                    except Exception as exc:
                        print(f"[WAVE] crash artifact write failed for {name}: {type(exc).__name__}: {exc}", flush=True)
                del procs[name]
                write_status()
        time.sleep(3)
    gpu_mon.terminate()
    summary = {"wave": str(queue_path), "total_wall_sec": round(time.time() - t0, 1), "experiments": status}
    (ROOT / "results/wave_summary.json").write_text(json.dumps(summary, indent=2))
    write_status(final=True)
    print(f"[WAVE DONE] {len(exps)} experiments in {summary['total_wall_sec']}s", flush=True)

if __name__ == "__main__":
    main()
