"""Cropping utilities for head exclusion, green-out, and bounding-box extraction."""

import numpy as np

from config import GREEN_COLOR


def head_exclusion_crop(image: np.ndarray, face_bbox: dict) -> np.ndarray:
    """Crop out the head region using face detection bounding box.

    Cuts at the bottom of the face bounding box, removing the head
    while preserving collars, necklines, and lapels.

    Args:
        image: RGB image (H, W, 3).
        face_bbox: Dict with 'x', 'y', 'width', 'height' in pixels.

    Returns:
        Cropped image from below face down.
    """
    cut_y = face_bbox["y"] + face_bbox["height"]
    cut_y = max(0, min(cut_y, image.shape[0]))
    return image[cut_y:, :]


def green_out_labels(
    image: np.ndarray,
    seg_map: np.ndarray,
    labels_to_green: list[int],
) -> np.ndarray:
    """Replace pixels belonging to specified labels with green.

    Args:
        image: RGB image (H, W, 3).
        seg_map: Segmentation map (H, W) with class IDs.
        labels_to_green: Label IDs whose pixels should be painted green.

    Returns:
        Copy of image with target label pixels set to green.
    """
    result = image.copy()
    mask = np.isin(seg_map, labels_to_green)
    result[mask] = GREEN_COLOR
    return result


def bbox_crop_with_padding(
    image: np.ndarray,
    mask: np.ndarray,
    target_labels: list[int],
    padding_ratio: float = 0.0,
    extend_to_bottom: bool = False,
    bottom_margin_ratio: float = 0.05,
) -> np.ndarray | None:
    """Crop to the bounding box of target label pixels with padding.

    Args:
        image: RGB image (H, W, 3).
        mask: Segmentation map (H, W) with class IDs.
        target_labels: List of label IDs to include.
        padding_ratio: Fraction of bbox dimensions to pad on each side.
        extend_to_bottom: If True, extend y_max to the bottom of the image
            minus bottom_margin_ratio (used for upper garments).
        bottom_margin_ratio: Fraction of image height to leave as margin
            at the bottom when extend_to_bottom is True.

    Returns:
        Cropped image or None if no target pixels found.
    """
    binary = np.isin(mask, target_labels)
    ys, xs = np.where(binary)

    if len(ys) == 0:
        return None

    y_min, y_max = int(ys.min()), int(ys.max())
    x_min, x_max = int(xs.min()), int(xs.max())

    h, w = image.shape[:2]
    pad_y = int((y_max - y_min) * padding_ratio)
    pad_x = int((x_max - x_min) * padding_ratio)

    y_min = max(0, y_min - pad_y)
    x_min = max(0, x_min - pad_x)
    x_max = min(w, x_max + pad_x)

    if extend_to_bottom:
        y_max = int(h * (1 - bottom_margin_ratio))
    else:
        y_max = min(h, y_max + pad_y)

    return image[y_min:y_max, x_min:x_max]


def upper_bbox_crop(
    image: np.ndarray,
    seg_map: np.ndarray,
    upper_labels: list[int],
    lower_labels: list[int],
    lr_margin_ratio: float = 0.10,
) -> np.ndarray | None:
    """Crop upper garment from top of upper bbox to top of lower bbox.

    Args:
        image: RGB image (H, W, 3).
        seg_map: Segmentation map (H, W) with class IDs.
        upper_labels: Label IDs for upper garment.
        lower_labels: Label IDs for lower garment.
        lr_margin_ratio: Fraction of crop width to add as margin on left and right.

    Returns:
        Cropped image or None if no upper pixels found.
    """
    upper_mask = np.isin(seg_map, upper_labels)
    upper_ys, upper_xs = np.where(upper_mask)

    if len(upper_ys) == 0:
        return None

    y_min = int(upper_ys.min())
    x_min = int(upper_xs.min())
    x_max = int(upper_xs.max())

    # y_max: top of lower bbox (excluding shared labels) if exists, else bottom of upper bbox
    exclusive_lower = [l for l in lower_labels if l not in upper_labels]
    lower_mask = np.isin(seg_map, exclusive_lower)
    lower_ys = np.where(lower_mask)[0]

    if len(lower_ys) > 0:
        y_max = int(lower_ys.min())
    else:
        y_max = int(upper_ys.max())

    # Apply left/right margin
    h, w = image.shape[:2]
    crop_w = x_max - x_min
    margin = int(crop_w * lr_margin_ratio)
    x_min = max(0, x_min - margin)
    x_max = min(w, x_max + margin)
    y_min = max(0, y_min)
    y_max = min(h, y_max)

    return image[y_min:y_max, x_min:x_max]
