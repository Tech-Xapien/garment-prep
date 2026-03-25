"""Pydantic models for request / response validation."""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class PipelineType(str, Enum):
    full = "full"
    upper = "upper"
    lower = "lower"


# ── /infer request (form fields alongside the image upload) ──────────
class InferRequest(BaseModel):
    """Validated view of the multipart form fields sent to /infer."""

    user_id: str = Field(..., min_length=1, description="Caller user ID")
    job_id: str = Field(..., min_length=1, description="Unique job identifier")
    provider: str = Field(..., min_length=1, description="Provider identifier")
    pipeline_type: PipelineType = Field(
        ..., description="Pipeline type: full | upper | lower"
    )
    callback_url: Optional[str] = Field(
        None,
        description="Override default callback URL. Falls back to CALLBACK_URL env.",
    )


# ── /infer immediate response ────────────────────────────────────────
class InferResponse(BaseModel):
    job_id: str
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

    user_id: str
    job_id: str
    provider: str
    pipeline_type: str
    callback_url: str
    image_bytes: bytes

    class Config:
        arbitrary_types_allowed = True
