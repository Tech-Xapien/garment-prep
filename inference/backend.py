"""Abstract inference backend protocol.

Any backend (local, Triton, ONNX-RT, …) implements this protocol so
pipeline code never changes when the serving layer is swapped.
"""

from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class FaceDetector(Protocol):
    """Detects the most prominent head/face in an image."""

    def detect(self, image_rgb: np.ndarray) -> Optional[dict]:
        """Return {'x','y','width','height'} or None."""
        ...

    def close(self) -> None: ...


@runtime_checkable
class HumanParser(Protocol):
    """Semantic segmentation into garment/body classes."""

    def parse(self, image_rgb: np.ndarray) -> np.ndarray:
        """Return (H, W) segmentation map with class IDs 0-17."""
        ...


@runtime_checkable
class InferenceBackend(Protocol):
    """Unified handle that exposes both detectors."""

    face: FaceDetector
    parser: HumanParser

    def close(self) -> None: ...
