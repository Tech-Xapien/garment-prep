"""Pydantic models for request / response validation."""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class PipelineType(str, Enum):
    full = "full"
    upper = "upper"
    lower = "lower"


# ── /infer immediate response ────────────────────────────────────────
class InferResponse(BaseModel):
    garment_id: str | None = None
    status: str = "queued"
    message: str = "Job accepted and queued for processing."


# ── /health response ─────────────────────────────────────────────────
class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    queue_depth: int
    inference_workers: int


# ── Internal: queued job payload ─────────────────────────────────────
class JobPayload(BaseModel):
    """Immutable snapshot of everything the worker needs to process a job."""

    garment_id: str | None = None
    pipeline_type: str
    callback_url: str
    image_bytes: bytes

    class Config:
        arbitrary_types_allowed = True
