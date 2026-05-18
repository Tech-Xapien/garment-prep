"""Fast-track pipeline: head exclusion + feet crop."""

import numpy as np

from config import FEET_LABEL
from services.crop import head_exclusion_crop
from services.face import FaceService
from services.parser import ParserService


def run(image_rgb: np.ndarray, face_service: FaceService, parser_service: ParserService) -> np.ndarray:
    """Execute the full-body pipeline.

    Returns image cropped below the face and above the feet.
    """
    face_bbox = face_service.detect(image_rgb)
    base = head_exclusion_crop(image_rgb, face_bbox) if face_bbox else image_rgb

    seg_map = parser_service.parse(base)
    feet_ys = np.where(seg_map == FEET_LABEL)[0]
    if len(feet_ys) > 0:
        return base[: int(feet_ys.min()), :]
    return base
