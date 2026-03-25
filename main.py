"""Garment-Prep Service — FastAPI application.

Endpoints:
  POST /infer       — async job queue with callback delivery
  POST /preprocess  — legacy synchronous request/response
  GET  /health      — liveness + queue depth
"""

import asyncio
import io
import logging
import sys
from contextlib import asynccontextmanager
from functools import partial

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image, ImageCms, ImageOps

import queue_worker
from config import CALLBACK_URL, INFERENCE_WORKERS
from pipelines import full, garment
from schemas import (
    HealthResponse,
    InferResponse,
    JobPayload,
    PipelineType,
)
from services.canvas import place_on_canvas
from services.face import FaceService
from services.parser import ParserService

# ── Logging ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("preprocessor")


# ── Lifespan ─────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Models for the legacy /preprocess endpoint (main process only)
    logger.info("Loading models for /preprocess (main process) …")
    app.state.face = FaceService()
    app.state.parser = ParserService()
    app.state.models_loaded = True
    logger.info("Models loaded.")

    # Start async queue + process pool for /infer
    logger.info("Starting inference queue …")
    await queue_worker.start()
    logger.info("Ready to serve.")

    yield

    # Graceful shutdown
    logger.info("Shutting down queue …")
    await queue_worker.shutdown()
    app.state.face.close()
    app.state.models_loaded = False
    logger.info("Shutdown complete.")


app = FastAPI(
    title="Garment-Prep Service",
    version="2.0.0",
    lifespan=lifespan,
)


# ── Shared helpers ───────────────────────────────────────────────────
_SRGB_ICC = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()


def _decode_image(raw: bytes) -> tuple[np.ndarray, str]:
    img = Image.open(io.BytesIO(raw))
    fmt = img.format or "JPEG"
    img = ImageOps.exif_transpose(img)
    return np.array(img.convert("RGB")), fmt


def _encode_image(arr: np.ndarray) -> bytes:
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG", icc_profile=_SRGB_ICC)
    return buf.getvalue()


def _process(
    image_rgb: np.ndarray,
    pipeline_type: str,
    face: FaceService,
    parser: ParserService,
) -> np.ndarray | None:
    if pipeline_type == "full":
        return full.run(image_rgb, face)
    return garment.run(image_rgb, pipeline_type, face, parser)


# ── POST /infer — async queue + callback ─────────────────────────────
@app.post("/infer", response_model=InferResponse)
async def infer(
    image: UploadFile = File(..., description="Input image (JPEG/PNG/WebP)"),
    user_id: str = Form(..., description="Caller user ID"),
    job_id: str = Form(..., description="Unique job identifier"),
    provider: str = Form(..., description="Provider identifier"),
    pipeline_type: PipelineType = Form(
        ..., description="Pipeline: full | upper | lower",
    ),
    callback_url: str | None = Form(
        None, description="Override callback URL",
    ),
):
    """Accept an image for async preprocessing.

    Returns immediately with job status.  The processed garment image is
    delivered to the callback URL as multipart/form-data upon completion.
    """
    raw = await image.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty image payload.")

    # Validate the image is decodable before queuing
    try:
        img = Image.open(io.BytesIO(raw))
        img.verify()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid or corrupt image.")

    job = JobPayload(
        user_id=user_id,
        job_id=job_id,
        provider=provider,
        pipeline_type=pipeline_type.value,
        callback_url=callback_url or CALLBACK_URL,
        image_bytes=raw,
    )

    ok = await queue_worker.enqueue(job)
    if not ok:
        raise HTTPException(
            status_code=503,
            detail="Queue full — try again shortly.",
        )

    logger.info("Job queued job_id=%s user_id=%s type=%s", job_id, user_id, pipeline_type.value)

    return InferResponse(job_id=job_id)


# ── POST /preprocess — legacy sync endpoint (backward compat) ────────
@app.post(
    "/preprocess",
    response_class=Response,
    responses={200: {"description": "Processed image (PNG, 1024x1024, sRGB)"}},
)
async def preprocess(
    image: UploadFile = File(..., description="Input image (JPEG/PNG/WebP)"),
    type: PipelineType = Form(..., description="Pipeline type: full, upper, lower"),
):
    """Synchronous preprocessing — returns the image directly."""
    raw = await image.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty image payload.")

    image_rgb, fmt = _decode_image(raw)

    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        None,
        partial(
            _process,
            image_rgb,
            type.value,
            app.state.face,
            app.state.parser,
        ),
    )

    result = place_on_canvas(result)
    out_bytes = _encode_image(result)
    return Response(content=out_bytes, media_type="image/png")


# ── GET /health ──────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        models_loaded=getattr(app.state, "models_loaded", False),
        queue_depth=queue_worker.queue_depth(),
        inference_workers=INFERENCE_WORKERS,
    )
