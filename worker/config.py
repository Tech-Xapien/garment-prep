"""Worker-pull configuration (Redis Streams + cpu_bridge + asset-service).

Contract: /home/fashionx/projects/tuck_service/docs/GARMENT_PREP_UPDATE_GUIDE.md
No AWS credentials — S3 access is only via short-lived presigned URLs.
"""
import os
import socket

# ── Redis Streams ────────────────────────────────────────────────────
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")   # rediss:// + AUTH in prod
STREAM_KEY = os.getenv("GARMENT_STREAM_KEY", "garment:jobs")
CONSUMER_GROUP = os.getenv("GARMENT_CONSUMER_GROUP", "garment-workers")
CONSUMER_BASE = os.getenv("CONSUMER_NAME", socket.gethostname())
WORKER_INDEX = os.getenv("WORKER_INDEX", "0")                    # unique per process in a group

# ── Concurrency ──────────────────────────────────────────────────────
WORKER_CONCURRENCY = int(os.getenv("WORKER_CONCURRENCY", "8"))   # in-flight jobs per process
CPU_THREADS = int(os.getenv("CPU_THREADS", "4"))                 # threadpool for decode/encode
XREAD_BLOCK_MS = int(os.getenv("XREAD_BLOCK_MS", "5000"))

# ── At-least-once reclaim (crashed-worker recovery) ──────────────────
RECLAIM_MIN_IDLE_MS = int(os.getenv("RECLAIM_MIN_IDLE_MS", "60000"))   # PEL idle before reclaim
RECLAIM_INTERVAL_S = float(os.getenv("RECLAIM_INTERVAL_S", "30"))
RECLAIM_COUNT = int(os.getenv("RECLAIM_COUNT", "10"))

# ── Backend endpoints + auth ─────────────────────────────────────────
CPU_BRIDGE_URL = os.getenv("CPU_BRIDGE_URL", "").rstrip("/")
ASSET_SERVICE_URL = os.getenv("ASSET_SERVICE_URL", "").rstrip("/")
BRIDGE_TO_GPU_SECRET = os.getenv("BRIDGE_TO_GPU_SECRET", "")      # X-Internal-Auth → cpu_bridge
ASSET_INTERNAL_SECRET = os.getenv("ASSET_INTERNAL_SECRET", "")   # X-Internal-Auth → asset-service

HTTP_TIMEOUT = float(os.getenv("WORKER_HTTP_TIMEOUT", "30"))     # bridge/asset calls
S3_TIMEOUT = float(os.getenv("WORKER_S3_TIMEOUT", "60"))         # presigned GET/PUT (cross-region)
