"""Fast-track pipeline: head exclusion crop only."""

import numpy as np

from services.crop import head_exclusion_crop
from services.face import FaceService


def run(image_rgb: np.ndarray, face_service: FaceService) -> np.ndarray:
    """Execute the full-body pipeline.

    Returns cropped image (below face) or original if no face detected.
    """
    face_bbox = face_service.detect(image_rgb)
    if face_bbox is None:
        return image_rgb
    return head_exclusion_crop(image_rgb, face_bbox)
