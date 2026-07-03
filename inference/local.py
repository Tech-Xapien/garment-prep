"""Local inference backend — loads models in-process.

Each *process* gets its own model copies (no GIL contention).
This module is designed to be imported inside a worker process via
the process-pool initialiser.
"""

from __future__ import annotations

import logging
import sys

from services.face import FaceService
from services.parser import ParserService

logger = logging.getLogger("preprocessor.inference.local")

# ── Per-process singletons (set by init_worker) ─────────────────────
_backend: LocalBackend | None = None


class LocalBackend:
    """Concrete local backend wrapping YOLO head-detector + FASHN parser."""

    __slots__ = ("face", "parser")

    def __init__(self) -> None:
        logger.info("Loading models in worker process …")
        self.face = FaceService()
        self.parser = ParserService()
        logger.info("Models loaded.")

    def close(self) -> None:
        self.face.close()


# ── Process-pool helpers ─────────────────────────────────────────────

def init_worker() -> None:
    """Called once per spawned worker process to load models."""
    # Spawned workers never import main.py, so its basicConfig doesn't apply
    # here — without a handler every INFO log in the worker is dropped.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        stream=sys.stdout,
    )
    global _backend  # noqa: PLW0603
    _backend = LocalBackend()


def get_backend() -> LocalBackend:
    """Return the per-process backend (must call init_worker first)."""
    if _backend is None:
        raise RuntimeError("init_worker() was not called in this process")
    return _backend
