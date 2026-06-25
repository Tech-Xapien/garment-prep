"""Fast-track pipeline: head exclusion + garment-width crop + feet crop."""

import numpy as np

from config import FEET_LABEL, UPPER_LABELS, LOWER_LABELS, LR_MARGIN_RATIO
from services.crop import head_exclusion_crop
from services.face import FaceService
from services.parser import ParserService


def run(image_rgb: np.ndarray, face_service: FaceService, parser_service: ParserService) -> np.ndarray:
    """Execute the full-body pipeline.

    Returns image cropped below the face and above the feet, and tightened
    horizontally to the garment bounding box with a left/right margin
    (same width logic as the upper-garment pipeline).
    """
    face_bbox = face_service.detect(image_rgb)
    base = head_exclusion_crop(image_rgb, face_bbox) if face_bbox else image_rgb

    seg_map = parser_service.parse(base)
    h, w = base.shape[:2]

    # Vertical: head bottom → top of feet region (exclude footwear).
    feet_ys = np.where(seg_map == FEET_LABEL)[0]
    y_max = int(feet_ys.min()) if len(feet_ys) > 0 else h

    # Horizontal: tight to garment pixels + left/right margin.
    garment_xs = np.where(np.isin(seg_map, UPPER_LABELS + LOWER_LABELS))[1]
    if len(garment_xs) > 0:
        x_min, x_max = int(garment_xs.min()), int(garment_xs.max())
        margin = int((x_max - x_min) * LR_MARGIN_RATIO)
        x_min = max(0, x_min - margin)
        x_max = min(w, x_max + margin)
    else:
        x_min, x_max = 0, w

    return base[:y_max, x_min:x_max]
