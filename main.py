"""Garment-Prep Service — FastAPI application.

Endpoints:
  POST /infer       — async job queue with callback delivery
  POST /preprocess  — legacy synchronous request/response
  GET  /health      — liveness + queue depth
"""

import asyncio
import io
import logging
import os
import sys
from contextlib import asynccontextmanager
from functools import partial
from typing import Optional

import httpx
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel
from fastapi.responses import RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageCms, ImageOps

import queue_worker
from config import CALLBACK_URL, INFERENCE_WORKERS, IMAGE_DOWNLOAD_TIMEOUT, IMAGE_URL_BASE
from schemas import (
    HealthResponse,
    InferResponse,
    JobPayload,
    PipelineType,
)

TEST_MODE = os.getenv("TEST_MODE", "0") == "1"

if TEST_MODE:
    from pipelines import full, garment
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

# ── Shared httpx client for image downloads ───────────────────────────
_download_client: Optional[httpx.AsyncClient] = None


async def _get_download_client() -> httpx.AsyncClient:
    global _download_client  # noqa: PLW0603
    if _download_client is None or _download_client.is_closed:
        _download_client = httpx.AsyncClient(
            timeout=httpx.Timeout(IMAGE_DOWNLOAD_TIMEOUT, connect=10.0),
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
            follow_redirects=True,
        )
    return _download_client


async def _close_download_client() -> None:
    global _download_client  # noqa: PLW0603
    if _download_client and not _download_client.is_closed:
        await _download_client.aclose()
    _download_client = None


# ── Lifespan ─────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    if TEST_MODE:
        # Models for the legacy /preprocess endpoint (main process only)
        logger.info("TEST_MODE: Loading models for /preprocess (main process) …")
        app.state.face = FaceService()
        app.state.parser = ParserService()
        logger.info("Models loaded in main process.")

    # Start async queue + process pool for /infer
    logger.info("Starting inference queue …")
    await queue_worker.start()
    app.state.models_loaded = True
    logger.info("Ready to serve.")

    yield

    # Graceful shutdown
    logger.info("Shutting down queue …")
    await queue_worker.shutdown()
    await _close_download_client()
    if TEST_MODE:
        app.state.face.close()
    app.state.models_loaded = False
    logger.info("Shutdown complete.")


app = FastAPI(
    title="Garment-Prep Service",
    version="2.0.0",
    lifespan=lifespan,
)

# ── Test webapp (static) ─────────────────────────────────────────────
_WEBAPP_DIR = os.path.join(os.path.dirname(__file__), "webapp")
if os.path.isdir(_WEBAPP_DIR):
    app.mount("/ui", StaticFiles(directory=_WEBAPP_DIR, html=True), name="ui")

    @app.get("/", include_in_schema=False)
    async def _root_redirect():
        return RedirectResponse(url="/ui/")


# ── Shared helpers (used by /preprocess in TEST_MODE) ────────────────
_SRGB_ICC = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()

if TEST_MODE:
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

    def _process(image_rgb, pipeline_type, face, parser):
        if pipeline_type in ("full", "layered"):
            return full.run(image_rgb, face)
        return garment.run(image_rgb, pipeline_type, face, parser)


# ── POST /infer — async queue + callback ─────────────────────────────

class InferRequest(BaseModel):
    garment_id: str | None = None
    image_url: str
    pipeline_type: PipelineType
    callback_url: str | None = None


def _normalize_image_url(url: str) -> str:
    """Ensure the URL has an http(s) scheme, prepending IMAGE_URL_BASE if not."""
    if url.startswith(("http://", "https://")):
        return url
    base = IMAGE_URL_BASE or "https://"
    if base.endswith("://"):
        return base + url.lstrip("/")
    return base + "/" + url.lstrip("/")


@app.post("/infer", response_model=InferResponse)
async def infer(body: InferRequest):
    """Accept a garment image URL for async preprocessing.

    Returns immediately with job status.  The processed garment image is
    delivered to the callback URL as multipart/form-data upon completion.
    """
    image_url = _normalize_image_url(body.image_url)
    if image_url != body.image_url:
        logger.info(
            "Normalized image_url garment_id=%s: %s -> %s",
            body.garment_id, body.image_url, image_url,
        )

    # Download image from URL (shared pooled client)
    try:
        client = await _get_download_client()
        resp = await client.get(image_url)
        resp.raise_for_status()
        raw = resp.content
    except httpx.HTTPError as exc:
        logger.warning(
            "Download failed garment_id=%s url=%s type=%s: %s",
            body.garment_id, image_url, body.pipeline_type.value, exc,
        )
        raise HTTPException(
            status_code=400,
            detail=f"Failed to download image from URL: {exc}",
        )

    if not raw:
        logger.warning(
            "Empty image garment_id=%s url=%s",
            body.garment_id, image_url,
        )
        raise HTTPException(status_code=400, detail="Empty image at URL.")

    # Lightweight format check — full decode happens in the worker
    try:
        img = Image.open(io.BytesIO(raw))
        img.load()
    except Exception as exc:
        logger.warning(
            "Invalid image garment_id=%s url=%s bytes=%d: %s",
            body.garment_id, image_url, len(raw), exc,
        )
        raise HTTPException(status_code=400, detail="Invalid or corrupt image.")

    job = JobPayload(
        garment_id=body.garment_id,
        pipeline_type=body.pipeline_type.value,
        callback_url=body.callback_url or CALLBACK_URL,
        image_bytes=raw,
    )

    ok = await queue_worker.enqueue(job)
    if not ok:
        raise HTTPException(
            status_code=503,
            detail="Queue full — try again shortly.",
        )

    logger.info(
        "Job queued garment_id=%s type=%s",
        body.garment_id, body.pipeline_type.value,
    )

    return InferResponse(garment_id=body.garment_id)


# ── POST /preprocess — legacy sync endpoint (TEST_MODE only) ─────────
if TEST_MODE:
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
