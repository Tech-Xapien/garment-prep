"""Async job queue + process-pool inference dispatcher.

Architecture:
  - asyncio.Queue (bounded) provides back-pressure.
  - N async "consumer" tasks drain the queue concurrently.
  - CPU-bound inference is dispatched to a ProcessPoolExecutor
    so the event loop is never blocked.
  - On completion the result is delivered via callback.py.
"""

from __future__ import annotations

import asyncio
import io
import logging
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from typing import Optional

import numpy as np
from PIL import Image, ImageCms, ImageOps

from callback import deliver
from config import INFERENCE_WORKERS, QUEUE_SIZE, QUEUE_WORKERS
from schemas import JobPayload
from pipelines import full, garment
from services.canvas import place_on_canvas

logger = logging.getLogger("preprocessor.worker")

_SRGB_ICC = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()

# ── Module state (initialised via start / stopped via shutdown) ──────
_queue: Optional[asyncio.Queue] = None
_consumers: list[asyncio.Task] = []
_pool: Optional[ProcessPoolExecutor] = None
_shutdown_event: Optional[asyncio.Event] = None


# ── Image helpers (duplicated from main to keep worker self-contained)
def _decode(raw: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(raw))
    img = ImageOps.exif_transpose(img)
    return np.array(img.convert("RGB"))


def _encode(arr: np.ndarray) -> bytes:
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG", icc_profile=_SRGB_ICC)
    return buf.getvalue()


# ── Function executed *inside* a worker process ──────────────────────
def _run_inference(image_bytes: bytes, pipeline_type: str) -> bytes:
    """Decode → pipeline → canvas → encode.  Runs in the process pool."""
    from inference.local import get_backend

    backend = get_backend()
    image_rgb = _decode(image_bytes)

    if pipeline_type == "full":
        result = full.run(image_rgb, backend.face)
    else:
        result = garment.run(
            image_rgb, pipeline_type, backend.face, backend.parser,
        )

    result = place_on_canvas(result)
    return _encode(result)


# ── Queue consumer coroutine ─────────────────────────────────────────
async def _consumer(name: str) -> None:
    """Drain jobs from the queue until shutdown."""
    loop = asyncio.get_running_loop()

    while True:
        try:
            job: JobPayload = await asyncio.wait_for(
                _queue.get(), timeout=1.0,
            )
        except asyncio.TimeoutError:
            if _shutdown_event and _shutdown_event.is_set():
                logger.info("Consumer %s shutting down (queue empty).", name)
                return
            continue

        logger.info("Consumer %s picked job_id=%s", name, job.job_id)

        try:
            png_bytes: bytes = await loop.run_in_executor(
                _pool,
                partial(_run_inference, job.image_bytes, job.pipeline_type),
            )

            ok = await deliver(
                callback_url=job.callback_url,
                user_id=job.user_id,
                job_id=job.job_id,
                provider=job.provider,
                garment_png=png_bytes,
            )

            if ok:
                logger.info("Job %s completed and delivered.", job.job_id)
            else:
                logger.error("Job %s processed but delivery failed.", job.job_id)

        except Exception:
            logger.exception("Job %s failed during processing.", job.job_id)

        finally:
            _queue.task_done()


# ── Lifecycle ────────────────────────────────────────────────────────
def queue_depth() -> int:
    return _queue.qsize() if _queue else 0


async def enqueue(job: JobPayload) -> bool:
    """Put a job on the queue.  Returns False if the queue is full."""
    try:
        _queue.put_nowait(job)
        return True
    except asyncio.QueueFull:
        return False


async def start() -> None:
    """Spin up the process pool and consumer tasks."""
    global _queue, _pool, _shutdown_event  # noqa: PLW0603

    from inference.local import init_worker

    _shutdown_event = asyncio.Event()
    _queue = asyncio.Queue(maxsize=QUEUE_SIZE)
    _pool = ProcessPoolExecutor(
        max_workers=INFERENCE_WORKERS,
        initializer=init_worker,
    )

    for i in range(QUEUE_WORKERS):
        task = asyncio.create_task(_consumer(f"worker-{i}"))
        _consumers.append(task)

    logger.info(
        "Queue started: pool=%d workers, %d consumers, capacity=%d",
        INFERENCE_WORKERS, QUEUE_WORKERS, QUEUE_SIZE,
    )


async def shutdown() -> None:
    """Drain in-flight jobs then tear down pool + consumers."""
    global _pool  # noqa: PLW0603

    if _shutdown_event:
        _shutdown_event.set()

    if _queue and not _queue.empty():
        logger.info("Draining %d remaining jobs …", _queue.qsize())
        try:
            await asyncio.wait_for(_queue.join(), timeout=120.0)
        except asyncio.TimeoutError:
            logger.warning("Drain timed out — some jobs may be lost.")

    for task in _consumers:
        task.cancel()
    if _consumers:
        await asyncio.gather(*_consumers, return_exceptions=True)
    _consumers.clear()

    if _pool:
        _pool.shutdown(wait=False)
        _pool = None

    from callback import close_client
    await close_client()

    logger.info("Queue shutdown complete.")
