"""Canvas utility: scale and center image on a white 1024x1024 background."""

import numpy as np
from PIL import Image

CANVAS_SIZE = 1024
MARGIN_RATIO = 0.05  # 5% margin on each side -> image fits within 90% of canvas


def place_on_canvas(image_rgb: np.ndarray) -> np.ndarray:
    """Scale image to fit 1024x1024 white canvas with margin, preserving aspect ratio.

    Uses Lanczos (LANCZOS) resampling for highest quality downscaling.
    The image is centered on the canvas with equal margins.

    Args:
        image_rgb: RGB uint8 numpy array (H, W, 3).

    Returns:
        1024x1024 RGB uint8 numpy array with the image centered on white.
    """
    img = Image.fromarray(image_rgb)

    max_dim = int(CANVAS_SIZE * (1 - 2 * MARGIN_RATIO))  # usable area per side

    # Scale down preserving aspect ratio using Lanczos
    orig_w, orig_h = img.size
    scale = min(max_dim / orig_w, max_dim / orig_h)
    if scale < 1.0:
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)
    elif scale > 1.0:
        # Image is smaller than canvas area — scale up with Lanczos too
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)

    # Create white canvas
    canvas = Image.new("RGB", (CANVAS_SIZE, CANVAS_SIZE), (255, 255, 255))

    # Center paste
    x = (CANVAS_SIZE - img.width) // 2
    y = (CANVAS_SIZE - img.height) // 2
    canvas.paste(img, (x, y))

    return np.array(canvas)
