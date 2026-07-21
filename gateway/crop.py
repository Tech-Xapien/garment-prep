"""Single-parse crop geometry.

One SegFormer pass yields the seg-map (576x384). From it we derive BOTH the head
cut and the garment bounding box — no YOLO. Validated against the old YOLO path:
face-bottom cut tracks YOLO within ~3px median and errs on the safe side.
All extents are computed in seg space then scaled back to original pixels.
"""
from __future__ import annotations

import numpy as np

from gateway.config import (
    BOTTOM_MARGIN_RATIO, FEET, HEAD_LABELS, IN_H, IN_W, LOWER_LABELS,
    LR_MARGIN_RATIO, UPPER_LABELS, FACE,
)

Box = tuple[int, int, int, int]  # y0, y1, x0, x1


def _head_cut_row(seg: np.ndarray) -> int:
    """Bottom row of the head region. face-label only; hair/hat is a back-view fallback.

    Never use the face∪hair union — long hair drags the cut into the garment.
    """
    face_rows = np.where(seg == FACE)[0]
    if face_rows.size:
        return int(face_rows.max())
    head_rows = np.where(np.isin(seg, HEAD_LABELS))[0]
    return int(head_rows.max()) if head_rows.size else 0


def _upper(sub: np.ndarray) -> Box | None:
    uy, ux = np.where(np.isin(sub, UPPER_LABELS))
    if uy.size == 0:
        return None
    y0, x0, x1 = int(uy.min()), int(ux.min()), int(ux.max())
    excl_lower = [l for l in LOWER_LABELS if l not in UPPER_LABELS]
    ly = np.where(np.isin(sub, excl_lower))[0]
    y1 = int(ly.min()) if ly.size else int(uy.max())
    m = int((x1 - x0) * LR_MARGIN_RATIO)
    return y0, y1, x0 - m, x1 + m


def _lower(sub: np.ndarray) -> Box | None:
    h = sub.shape[0]
    ly, lx = np.where(np.isin(sub, LOWER_LABELS))
    if ly.size == 0:
        return None
    y0, x0, x1 = int(ly.min()), int(lx.min()), int(lx.max())
    feet = np.where(sub == FEET)[0]
    natural_bottom = int(h * (1 - BOTTOM_MARGIN_RATIO))
    y1 = min(int(feet.min()), natural_bottom) if feet.size else natural_bottom
    return y0, y1, x0, x1


def _full(sub: np.ndarray) -> Box:
    h, w = sub.shape
    feet = np.where(sub == FEET)[0]
    y1 = int(feet.min()) if feet.size else h
    gx = np.where(np.isin(sub, UPPER_LABELS + LOWER_LABELS))[1]
    if gx.size:
        x0, x1 = int(gx.min()), int(gx.max())
        m = int((x1 - x0) * LR_MARGIN_RATIO)
        x0, x1 = x0 - m, x1 + m
    else:
        x0, x1 = 0, w
    return 0, y1, x0, x1


def compute_crop(seg: np.ndarray, orig_h: int, orig_w: int, ptype: str) -> Box:
    """Return crop box (y0,y1,x0,x1) in ORIGINAL image pixels."""
    sh, sw = seg.shape
    cut = _head_cut_row(seg)
    sub = seg[cut:, :]

    if ptype == "upper":
        box = _upper(sub)
    elif ptype == "lower":
        box = _lower(sub)
    else:  # full / layered
        box = _full(sub)

    if box is None:
        y0, y1, x0, x1 = 0, sub.shape[0], 0, sw
    else:
        y0, y1, x0, x1 = box
    y0 += cut
    y1 += cut

    fy, fx = orig_h / sh, orig_w / sw
    return (
        max(0, int(y0 * fy)), min(orig_h, int(y1 * fy)),
        max(0, int(x0 * fx)), min(orig_w, int(x1 * fx)),
    )
