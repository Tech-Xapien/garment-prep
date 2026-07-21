"""Garment-Prep gateway — CPU orchestration in front of a co-located Triton engine.

Flow per job:
  download → [thread] decode+preprocess → Triton infer (GPU, batched) →
  [thread] crop+canvas+encode → callback

CPU-heavy stages run in a threadpool; the GPU parse is awaited while other jobs
use the cores. Concurrency = QUEUE_CONSUMERS × uvicorn workers (both env knobs).
"""
from __future__ import annotations

import asyncio
import io
import logging
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Optional

import httpx
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image
from pydantic import BaseModel

from gateway import callback, crop, imaging
from gateway.config import (
    CPU_THREADS, IMAGE_DOWNLOAD_TIMEOUT, IMAGE_URL_BASE, QUEUE_CONSUMERS,
    QUEUE_SIZE, CALLBACK_URL,
)
from gateway.triton_client import TritonParser

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("gateway")

_VALID_TYPES = {"full", "upper", "lower", "layered"}


# ── CPU stage helpers (run in threadpool) ────────────────────────────
def _decode_prep(raw: bytes) -> tuple[np.ndarray, np.ndarray]:
    rgb = imaging.decode(raw)
    return rgb, imaging.preprocess(rgb)


def _postprocess(rgb: np.ndarray, seg: np.ndarray, ptype: str) -> bytes:
    box = crop.compute_crop(seg, rgb.shape[0], rgb.shape[1], ptype)
    cropped = imaging.crop(rgb, box)
    return imaging.encode_png(imaging.place_on_canvas(cropped))


# ── Job model ────────────────────────────────────────────────────────
class _Job(BaseModel):
    garment_id: str | None
    pipeline_type: str
    callback_url: str
    image_bytes: bytes

    class Config:
        arbitrary_types_allowed = True


# ── Lifespan ─────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.pool = ThreadPoolExecutor(max_workers=CPU_THREADS)
    app.state.queue = asyncio.Queue(maxsize=QUEUE_SIZE)
    app.state.triton = TritonParser()
    app.state.download = httpx.AsyncClient(
        timeout=httpx.Timeout(IMAGE_DOWNLOAD_TIMEOUT, connect=10.0),
        limits=httpx.Limits(max_connections=40, max_keepalive_connections=20),
        follow_redirects=True,
    )

    # Wait for the Triton engine to be ready before accepting work.
    for _ in range(60):
        try:
            if await app.state.triton.ready():
                break
        except Exception:
            pass
        await asyncio.sleep(1.0)
    else:
        logger.error("Triton model not ready after 60s — starting anyway.")

    app.state.consumers = [
        asyncio.create_task(_consumer(app, i)) for i in range(QUEUE_CONSUMERS)
    ]
    app.state.ready = True
    logger.info("Gateway ready: %d consumers, %d cpu threads", QUEUE_CONSUMERS, CPU_THREADS)

    yield

    app.state.ready = False
    for t in app.state.consumers:
        t.cancel()
    await asyncio.gather(*app.state.consumers, return_exceptions=True)
    await app.state.triton.close()
    await app.state.download.aclose()
    await callback.close_client()
    app.state.pool.shutdown(wait=False)


app = FastAPI(title="Garment-Prep Gateway", version="3.0.0", lifespan=lifespan)


# ── Core pipeline ────────────────────────────────────────────────────
async def _run(app: FastAPI, raw: bytes, ptype: str) -> bytes:
    loop = asyncio.get_running_loop()
    rgb, x = await loop.run_in_executor(app.state.pool, _decode_prep, raw)
    seg = await app.state.triton.infer(x)
    return await loop.run_in_executor(app.state.pool, _postprocess, rgb, seg, ptype)


async def _consumer(app: FastAPI, idx: int) -> None:
    q: asyncio.Queue = app.state.queue
    while True:
        job: _Job = await q.get()
        try:
            png = await _run(app, job.image_bytes, job.pipeline_type)
            await callback.deliver(
                callback_url=job.callback_url, garment_id=job.garment_id, garment_png=png,
            )
        except Exception:
            logger.exception("Job %s failed", job.garment_id)
        finally:
            q.task_done()


def _normalize_url(url: str) -> str:
    if url.startswith(("http://", "https://")):
        return url
    base = IMAGE_URL_BASE or "https://"
    return base + url.lstrip("/") if base.endswith("://") else base + "/" + url.lstrip("/")


# ── Endpoints ────────────────────────────────────────────────────────
class InferRequest(BaseModel):
    garment_id: str | None = None
    image_url: str
    pipeline_type: str
    callback_url: str | None = None


class InferResponse(BaseModel):
    garment_id: str | None = None
    status: str = "queued"


@app.post("/infer", response_model=InferResponse)
async def infer(body: InferRequest):
    if body.pipeline_type not in _VALID_TYPES:
        raise HTTPException(422, f"pipeline_type must be one of {_VALID_TYPES}")
    url = _normalize_url(body.image_url)
    try:
        resp = await app.state.download.get(url)
        resp.raise_for_status()
        raw = resp.content
    except httpx.HTTPError as exc:
        raise HTTPException(400, f"download failed: {exc}")
    if not raw:
        raise HTTPException(400, "empty image at URL")

    job = _Job(
        garment_id=body.garment_id,
        pipeline_type=body.pipeline_type,
        callback_url=body.callback_url or CALLBACK_URL,
        image_bytes=raw,
    )
    try:
        app.state.queue.put_nowait(job)
    except asyncio.QueueFull:
        raise HTTPException(503, "queue full — retry shortly")
    return InferResponse(garment_id=body.garment_id)


@app.post("/preprocess", response_class=Response,
          responses={200: {"description": "Processed PNG"}})
async def preprocess(
    image: UploadFile = File(...),
    type: str = Form(...),
):
    """Synchronous path — returns the PNG directly. For QA + benchmarking."""
    if type not in _VALID_TYPES:
        raise HTTPException(422, f"type must be one of {_VALID_TYPES}")
    raw = await image.read()
    if not raw:
        raise HTTPException(400, "empty image payload")
    png = await _run(app, raw, type)
    return Response(content=png, media_type="image/png")


@app.get("/health")
async def health():
    return {
        "status": "ok" if getattr(app.state, "ready", False) else "starting",
        "queue_depth": app.state.queue.qsize() if hasattr(app.state, "queue") else 0,
        "consumers": QUEUE_CONSUMERS,
    }
