"""Canvas utility: scale and center image on a white background.

Default output: 768x1024 (3:4).  Configurable via CANVAS_WIDTH / CANVAS_HEIGHT env vars.
Set both to 1024 for square output.
"""

import numpy as np
from PIL import Image

from config import CANVAS_WIDTH, CANVAS_HEIGHT

MARGIN_RATIO = 0.05  # 5% margin on each side -> image fits within 90% of canvas


def place_on_canvas(image_rgb: np.ndarray) -> np.ndarray:
    """Scale image to fit canvas with margin, preserving aspect ratio.

    Uses Lanczos resampling for highest quality scaling.
    The image is centered on the canvas with equal margins.

    Args:
        image_rgb: RGB uint8 numpy array (H, W, 3).

    Returns:
        (CANVAS_HEIGHT, CANVAS_WIDTH, 3) RGB uint8 numpy array on white.
    """
    img = Image.fromarray(image_rgb)

    usable_w = int(CANVAS_WIDTH * (1 - 2 * MARGIN_RATIO))
    usable_h = int(CANVAS_HEIGHT * (1 - 2 * MARGIN_RATIO))

    orig_w, orig_h = img.size
    scale = min(usable_w / orig_w, usable_h / orig_h)

    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)
    if new_w != orig_w or new_h != orig_h:
        img = img.resize((new_w, new_h), Image.LANCZOS)

    canvas = Image.new("RGB", (CANVAS_WIDTH, CANVAS_HEIGHT), (255, 255, 255))

    x = (CANVAS_WIDTH - img.width) // 2
    y = (CANVAS_HEIGHT - img.height) // 2
    canvas.paste(img, (x, y))

    return np.array(canvas)
