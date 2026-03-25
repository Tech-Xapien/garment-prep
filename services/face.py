"""YOLOv8 head detection service (CrowdHuman-trained)."""

import threading
from typing import Optional

import numpy as np
from ultralytics import YOLO

from config import HEAD_MODEL_PATH


class FaceService:
    """Thread-safe wrapper around YOLOv8 CrowdHuman head detector."""

    def __init__(self):
        self._lock = threading.Lock()
        self._model = YOLO(str(HEAD_MODEL_PATH))

    def detect(self, image_rgb: np.ndarray) -> Optional[dict]:
        """Detect the most prominent head in an RGB numpy array.

        Args:
            image_rgb: RGB uint8 numpy array (H, W, 3).

        Returns:
            Dict with 'x', 'y', 'width', 'height' (pixels, clipped to image)
            or None if no head found.
        """
        with self._lock:
            results = self._model(image_rgb, verbose=False, conf=0.3)
        boxes = results[0].boxes
        if len(boxes) == 0:
            return None

        # Pick highest-confidence detection
        best_idx = int(boxes.conf.argmax())
        x1, y1, x2, y2 = [int(v) for v in boxes.xyxy[best_idx].tolist()]

        return {
            "x": x1,
            "y": y1,
            "width": x2 - x1,
            "height": y2 - y1,
        }

    def close(self):
        pass
