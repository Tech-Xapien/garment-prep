"""Gateway configuration — all knobs are env-driven for RunPod sweeps → EC2 pinning.

Two families of knobs:
  CPU  side  (gateway) : QUEUE_CONSUMERS, CPU_THREADS, GATEWAY_WORKERS(uvicorn procs)
  GPU  side  (triton)  : MAX_BATCH_SIZE, INSTANCES, MAX_QUEUE_DELAY_US (see config.pbtxt.template)
We measure throughput vs (cores, vram) on RunPod, then set the same env on EC2.
"""
import os

# ── Triton ───────────────────────────────────────────────────────────
TRITON_URL = os.getenv("TRITON_URL", "localhost:8001")           # in-container gRPC
TRITON_MODEL = os.getenv("TRITON_MODEL", "segformer_parser")
TRITON_TIMEOUT = float(os.getenv("TRITON_TIMEOUT", "30.0"))

# Engine input geometry (must match export_onnx.py / config.pbtxt).
IN_H, IN_W = 576, 384

# ── Segmentation labels (FASHN 18-class) ─────────────────────────────
FACE, HAIR, HAT, FEET = 1, 2, 9, 15
HEAD_LABELS = (FACE, HAIR, HAT)
UPPER_LABELS = (3, 4, 10)          # top, dress, scarf
LOWER_LABELS = (5, 6)              # skirt, pants
LR_MARGIN_RATIO = float(os.getenv("LR_MARGIN_RATIO", "0.01"))
BOTTOM_MARGIN_RATIO = 0.05

# ── Canvas / output ──────────────────────────────────────────────────
CANVAS_WIDTH = int(os.getenv("CANVAS_WIDTH", "928"))
CANVAS_HEIGHT = int(os.getenv("CANVAS_HEIGHT", "1664"))
CANVAS_MARGIN_RATIO = 0.05
# PNG is lossless, so level only trades encode-CPU for file size. Level 1 is ~2.5x
# faster than level 6 for ~13% larger files — a big throughput win on the CPU path.
PNG_COMPRESS_LEVEL = int(os.getenv("PNG_COMPRESS_LEVEL", "1"))

# ── Concurrency / CPU knobs ──────────────────────────────────────────
QUEUE_CONSUMERS = int(os.getenv("QUEUE_CONSUMERS", "8"))    # async pipelines in flight
CPU_THREADS = int(os.getenv("CPU_THREADS", "8"))            # threadpool for decode/encode
QUEUE_SIZE = int(os.getenv("QUEUE_SIZE", "256"))

# ── Image download ───────────────────────────────────────────────────
IMAGE_DOWNLOAD_TIMEOUT = float(os.getenv("IMAGE_DOWNLOAD_TIMEOUT", "30.0"))
IMAGE_URL_BASE = os.getenv("IMAGE_URL_BASE", "").rstrip("/")

# ── Callback ─────────────────────────────────────────────────────────
CALLBACK_URL = os.getenv("CALLBACK_URL", "http://3.110.84.73:9009/v1/garment/upload")
CALLBACK_AUTH_TOKEN = os.getenv("CALLBACK_AUTH_TOKEN", "supersecret-internal-token")
CALLBACK_MAX_RETRIES = int(os.getenv("CALLBACK_MAX_RETRIES", "5"))
CALLBACK_BACKOFF_BASE = float(os.getenv("CALLBACK_BACKOFF_BASE", "1.0"))
CALLBACK_BACKOFF_MAX = float(os.getenv("CALLBACK_BACKOFF_MAX", "16.0"))
CALLBACK_TIMEOUT = float(os.getenv("CALLBACK_TIMEOUT", "30.0"))
