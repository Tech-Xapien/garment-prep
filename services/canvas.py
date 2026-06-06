"""Canvas utility: scale and center image on a transparent-padded canvas.

Default output: 768x1024 (3:4).  Configurable via CANVAS_WIDTH / CANVAS_HEIGHT env vars.
Set both to 1024 for square output.

Output is RGBA — padding regions have alpha=0 so downstream attention models
do not attend to whitespace tokens.
"""

import numpy as np
from PIL import Image

from config import CANVAS_WIDTH, CANVAS_HEIGHT

MARGIN_RATIO = 0.05  # 5% margin on each side -> image fits within 90% of canvas


def place_on_canvas(image_rgb: np.ndarray, canvas_w: int | None = None, canvas_h: int | None = None) -> np.ndarray:
    """Scale image to fit canvas with margin, preserving aspect ratio.

    Uses Lanczos resampling for highest quality scaling.
    The image is centered on the canvas with equal margins.

    Padding regions carry alpha=0 so downstream models can mask them out;
    garment pixels carry alpha=255.

    Args:
        image_rgb: RGB uint8 numpy array (H, W, 3).

    Returns:
        (CANVAS_HEIGHT, CANVAS_WIDTH, 4) RGBA uint8 numpy array.
    """
    cw = canvas_w or CANVAS_WIDTH
    ch = canvas_h or CANVAS_HEIGHT
    img = Image.fromarray(image_rgb).convert("RGBA")

    usable_w = int(cw * (1 - 2 * MARGIN_RATIO))
    usable_h = int(ch * (1 - 2 * MARGIN_RATIO))

    orig_w, orig_h = img.size
    scale = min(usable_w / orig_w, usable_h / orig_h)

    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)
    if new_w != orig_w or new_h != orig_h:
        img = img.resize((new_w, new_h), Image.LANCZOS)

    # Alpha=0 background — padding is transparent so models skip those tokens
    canvas = Image.new("RGBA", (cw, ch), (255, 255, 255, 0))

    x = (cw - img.width) // 2
    y = (ch - img.height) // 2
    canvas.paste(img, (x, y), mask=img)

    return np.array(canvas)
