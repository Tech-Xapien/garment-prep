"""Precision pipeline: head exclusion + green-out + semantic garment crop."""

import numpy as np

from config import PADDING_RATIO, UPPER_LABELS, LOWER_LABELS
from services.crop import bbox_crop_with_padding, green_out_labels, head_exclusion_crop, upper_bbox_crop
from services.face import FaceService
from services.parser import ParserService


def run(
    image_rgb: np.ndarray,
    pipeline_type: str,
    face_service: FaceService,
    parser_service: ParserService,
) -> np.ndarray:
    """Execute the garment-isolation pipeline.

    Steps:
        1. Detect face and crop head out.
        2. Run semantic segmentation.
        3. Green-out opposite garment labels (upper→green lower, lower→green upper).
        4. Crop to target garment bounding box.
    """
    face_bbox = face_service.detect(image_rgb)
    base = head_exclusion_crop(image_rgb, face_bbox) if face_bbox else image_rgb

    # Semantic segmentation
    seg_map = parser_service.parse(base)

    if pipeline_type == "upper":
        cropped = upper_bbox_crop(base, seg_map, UPPER_LABELS, LOWER_LABELS)
    else:
        cropped = bbox_crop_with_padding(
            base, seg_map, LOWER_LABELS, PADDING_RATIO,
            extend_to_bottom=True,
        )
    return cropped if cropped is not None else base
