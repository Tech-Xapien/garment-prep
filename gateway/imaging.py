"""CPU image ops: decode, preprocess (to engine input), canvas, encode.

These are the CPU-heavy stages we deliberately parallelise across cores. Decode,
INTER_AREA resize, Lanczos canvas and PNG encode all release the GIL in their C
extensions, so a threadpool gives real parallelism.
"""
from __future__ import annotations

import io

import cv2
import numpy as np
from PIL import Image, ImageCms, ImageOps

from gateway.config import (
    CANVAS_HEIGHT, CANVAS_MARGIN_RATIO, CANVAS_WIDTH, IN_H, IN_W, PNG_COMPRESS_LEVEL,
)

_SRGB_ICC = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()


def decode(raw: bytes) -> np.ndarray:
    """Bytes -> RGB uint8 (H, W, 3), EXIF-corrected."""
    img = Image.open(io.BytesIO(raw))
    img = ImageOps.exif_transpose(img)
    return np.asarray(img.convert("RGB"))


def preprocess(rgb: np.ndarray) -> np.ndarray:
    """RGB (H,W,3) -> engine input uint8 CHW (1,3,576,384).

    INTER_AREA + non-aspect-preserving resize matches parser training exactly;
    /255 + ImageNet norm happen on the GPU inside the engine.
    """
    resized = cv2.resize(rgb, (IN_W, IN_H), interpolation=cv2.INTER_AREA)  # -> (576,384,3)
    chw = np.ascontiguousarray(resized.transpose(2, 0, 1))                 # (3,576,384)
    return chw[None]                                                       # (1,3,576,384)


def crop(rgb: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    y0, y1, x0, x1 = box
    out = rgb[y0:y1, x0:x1]
    return out if out.size else rgb


def place_on_canvas(rgb: np.ndarray, cw: int | None = None, ch: int | None = None) -> np.ndarray:
    """Scale + center on a white, transparent-padded RGBA canvas.

    cv2 resize is ~9x faster than PIL Lanczos at equivalent quality: INTER_AREA
    for downscale (no ringing), INTER_LANCZOS4 for upscale. Padding is white with
    alpha=0 (RGB matches the old PIL canvas; only alpha marks garment vs padding).
    """
    cw = cw or CANVAS_WIDTH
    ch = ch or CANVAS_HEIGHT
    h, w = rgb.shape[:2]

    usable_w = int(cw * (1 - 2 * CANVAS_MARGIN_RATIO))
    usable_h = int(ch * (1 - 2 * CANVAS_MARGIN_RATIO))
    scale = min(usable_w / w, usable_h / h)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LANCZOS4
    resized = cv2.resize(rgb, (new_w, new_h), interpolation=interp)

    canvas = np.empty((ch, cw, 4), np.uint8)
    canvas[:, :, :3] = 255      # white padding (matches legacy output)
    canvas[:, :, 3] = 0         # transparent everywhere...
    x, y = (cw - new_w) // 2, (ch - new_h) // 2
    canvas[y:y + new_h, x:x + new_w, :3] = resized
    canvas[y:y + new_h, x:x + new_w, 3] = 255   # ...opaque over the garment
    return canvas


def encode_png(rgba: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rgba).save(
        buf, format="PNG", icc_profile=_SRGB_ICC, compress_level=PNG_COMPRESS_LEVEL,
    )
    return buf.getvalue()
