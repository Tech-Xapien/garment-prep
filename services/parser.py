"""FASHN SegFormer Human Parser service."""

import threading

import numpy as np
from fashn_human_parser import FashnHumanParser

from config import SEGFORMER_MODEL_ID


class ParserService:
    """Thread-safe singleton wrapper around FASHN Human Parser."""

    def __init__(self):
        self._lock = threading.Lock()
        self._parser = FashnHumanParser(model_id=SEGFORMER_MODEL_ID)

    def parse(self, image_rgb: np.ndarray) -> np.ndarray:
        """Parse an RGB image into semantic segmentation map.

        Args:
            image_rgb: RGB uint8 numpy array (H, W, 3).

        Returns:
            Segmentation map (H, W) with class IDs 0-17.
        """
        with self._lock:
            return self._parser.predict(image_rgb)
