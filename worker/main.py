"""Worker entrypoint — one Redis consumer per process.

The container launches WORKER_PROCESSES copies of this (each with a distinct
WORKER_INDEX → unique consumer name) so CPU-bound preprocessing scales across cores,
all sharing the co-located Triton engine.
"""
from __future__ import annotations

import asyncio
import logging
import signal
import sys
from concurrent.futures import ThreadPoolExecutor

import httpx
import redis.asyncio as aioredis

from gateway.triton_client import TritonParser
from worker import config as C
from worker.clients import Clients
from worker.redis_worker import Worker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("worker.main")


async def _wait_for_triton(triton: TritonParser, tries: int = 60) -> bool:
    for _ in range(tries):
        try:
            if await triton.ready():
                return True
        except Exception:
            pass
        await asyncio.sleep(1.0)
    return False


async def main() -> None:
    consumer = f"{C.CONSUMER_BASE}-{C.WORKER_INDEX}"
    triton = TritonParser()
    if not await _wait_for_triton(triton):
        logger.error("Triton model not ready — exiting so the container restarts.")
        raise SystemExit(1)

    pool = ThreadPoolExecutor(max_workers=C.CPU_THREADS)
    http = httpx.AsyncClient(
        timeout=httpx.Timeout(C.HTTP_TIMEOUT, connect=10.0),
        limits=httpx.Limits(max_connections=40, max_keepalive_connections=20),
        follow_redirects=True,
    )
    redis = aioredis.from_url(C.REDIS_URL, decode_responses=True)
    worker = Worker(redis, Clients(http), triton, pool, consumer)

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, worker.stop)

    try:
        await worker.run()
    finally:
        await http.aclose()
        await triton.close()
        await redis.aclose()
        pool.shutdown(wait=False)
        logger.info("worker %s stopped.", consumer)


if __name__ == "__main__":
    asyncio.run(main())
