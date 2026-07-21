"""Load-test harness — measures throughput vs CPU/GPU footprint.

Run on the RunPod box against the running container to find the smallest
(cores, batch, instances) config that hits target throughput. Those numbers are
then pinned on EC2. Uses the sync /preprocess endpoint so each request's
completion is observable.

Example:
    python -m gateway.bench --host http://localhost:8000 \
        --dir "/workspace/Lafayette Final" --type upper --concurrency 32 --n 500
"""
from __future__ import annotations

import argparse
import asyncio
import glob
import os
import statistics
import threading
import time

import httpx


def _sample(stop: threading.Event, out: dict) -> None:
    """Background sampler: GPU util/mem (pynvml) + CPU% (psutil)."""
    try:
        import psutil
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(int(os.getenv("BENCH_GPU", "0")))
    except Exception as exc:
        out["error"] = f"sampling disabled: {exc}"
        return
    cpu, gpu, mem = [], [], []
    psutil.cpu_percent()  # prime
    while not stop.is_set():
        cpu.append(psutil.cpu_percent(interval=None))
        u = pynvml.nvmlDeviceGetUtilizationRates(h)
        m = pynvml.nvmlDeviceGetMemoryInfo(h)
        gpu.append(u.gpu)
        mem.append(m.used / 2**20)
        time.sleep(0.5)
    cores = psutil.cpu_count()
    out.update(
        cpu_mean_cores=(statistics.mean(cpu) / 100 * cores) if cpu else 0,
        cpu_peak_cores=(max(cpu) / 100 * cores) if cpu else 0,
        gpu_mean=statistics.mean(gpu) if gpu else 0,
        gpu_peak=max(gpu) if gpu else 0,
        vram_peak_mb=max(mem) if mem else 0,
    )


async def run(args) -> None:
    paths = [p for p in glob.glob(f"{args.dir}/**/*.jpg", recursive=True)
             if "_thumb" in p or "_vton" not in p]
    if not paths:
        raise SystemExit(f"no images under {args.dir}")
    blobs = [open(paths[i % len(paths)], "rb").read() for i in range(min(args.n, 200))]

    lat: list[float] = []
    sent = 0
    sem = asyncio.Semaphore(args.concurrency)

    async with httpx.AsyncClient(timeout=120.0) as client:
        async def one(i: int):
            nonlocal sent
            async with sem:
                t0 = time.perf_counter()
                r = await client.post(
                    f"{args.host}/preprocess",
                    files={"image": (f"{i}.jpg", blobs[i % len(blobs)], "image/jpeg")},
                    data={"type": args.type},
                )
                lat.append(time.perf_counter() - t0)
                if r.status_code == 200:
                    sent += 1

        stop = threading.Event()
        res: dict = {}
        sampler = threading.Thread(target=_sample, args=(stop, res), daemon=True)
        sampler.start()

        t0 = time.perf_counter()
        await asyncio.gather(*(one(i) for i in range(args.n)))
        elapsed = time.perf_counter() - t0

        stop.set()
        sampler.join(timeout=2)

    lat_ms = sorted(x * 1000 for x in lat)
    def pct(p): return lat_ms[min(len(lat_ms) - 1, int(len(lat_ms) * p))]

    print(f"\n=== bench: {args.type}  concurrency={args.concurrency}  n={sent}/{args.n} ===")
    print(f"throughput   {sent / elapsed:6.1f} img/s   ({elapsed:.1f}s wall)")
    print(f"latency ms   p50={pct(.5):.0f}  p90={pct(.9):.0f}  p99={pct(.99):.0f}  max={lat_ms[-1]:.0f}")
    if "error" in res:
        print(res["error"])
    else:
        print(f"cpu cores    mean={res['cpu_mean_cores']:.1f}  peak={res['cpu_peak_cores']:.1f}")
        print(f"gpu util %   mean={res['gpu_mean']:.0f}  peak={res['gpu_peak']:.0f}")
        print(f"vram peak    {res['vram_peak_mb']:.0f} MB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="http://localhost:8000")
    ap.add_argument("--dir", required=True)
    ap.add_argument("--type", default="upper", choices=["full", "upper", "lower", "layered"])
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--n", type=int, default=500)
    asyncio.run(run(ap.parse_args()))
