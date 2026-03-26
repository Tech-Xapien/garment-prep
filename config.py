"""Central configuration for the preprocessor API."""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

# ── Models ───────────────────────────────────────────────────────────
HEAD_MODEL_PATH = MODEL_DIR / "crowdhuman_yolov8_head.pt"
SEGFORMER_MODEL_ID = "fashn-ai/fashn-human-parser"

# ── Cropping ─────────────────────────────────────────────────────────
PADDING_RATIO = 0.0

# Label IDs for garment types
UPPER_LABELS = [3, 4, 10]   # top, dress, scarf
LOWER_LABELS = [4, 5, 6]    # dress, skirt, pants

# Green-out colour (RGB)
GREEN_COLOR = (0, 255, 0)

# ── Server ───────────────────────────────────────────────────────────
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
WORKERS = int(os.getenv("WORKERS", "1"))

# ── Callback ─────────────────────────────────────────────────────────
CALLBACK_URL = os.getenv(
    "CALLBACK_URL",
    "https://apiprod.xapien.in:9002/v1/garment/upload",
)
CALLBACK_AUTH_TOKEN = os.getenv("CALLBACK_AUTH_TOKEN", "supersecret-internal-token")
CALLBACK_MAX_RETRIES = int(os.getenv("CALLBACK_MAX_RETRIES", "5"))
CALLBACK_BACKOFF_BASE = float(os.getenv("CALLBACK_BACKOFF_BASE", "1.0"))   # seconds
CALLBACK_BACKOFF_MAX = float(os.getenv("CALLBACK_BACKOFF_MAX", "16.0"))    # cap
CALLBACK_TIMEOUT = float(os.getenv("CALLBACK_TIMEOUT", "30.0"))            # per-request

# ── Canvas / Output ──────────────────────────────────────────────────
CANVAS_WIDTH = int(os.getenv("CANVAS_WIDTH", "768"))
CANVAS_HEIGHT = int(os.getenv("CANVAS_HEIGHT", "1024"))

# ── Image download ───────────────────────────────────────────────────
IMAGE_DOWNLOAD_TIMEOUT = float(os.getenv("IMAGE_DOWNLOAD_TIMEOUT", "30.0"))

# ── Queue / Workers ──────────────────────────────────────────────────
INFERENCE_WORKERS = int(os.getenv("INFERENCE_WORKERS", "2"))   # process pool size
QUEUE_WORKERS = int(os.getenv("QUEUE_WORKERS", "4"))           # async consumers
QUEUE_SIZE = int(os.getenv("QUEUE_SIZE", "100"))               # max pending jobs
