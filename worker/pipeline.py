"""run_preprocessing — the shared compute core, wired for the worker.

decode → Triton parse → crop → canvas → PNG, reusing gateway.imaging / gateway.crop /
the Triton client, so worker and HTTP modes emit byte-identical output. CPU stages run
in a threadpool; the GPU call is awaited. Errors are classified for the XACK policy
(see worker/redis_worker.py): TransientError is retried by reclaim, DefinitiveError is
reported failed.
"""
from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from tritonclient.utils import InferenceServerException

from gateway import crop, imaging
from gateway.triton_client import TritonParser
from worker.clients import DefinitiveError, TransientError


def _decode_prep(raw: bytes) -> tuple[np.ndarray, np.ndarray]:
    rgb = imaging.decode(raw)
    return rgb, imaging.preprocess(rgb)


def _postprocess(rgb: np.ndarray, seg: np.ndarray, ptype: str) -> bytes:
    box = crop.compute_crop(seg, rgb.shape[0], rgb.shape[1], ptype)
    return imaging.encode_png(imaging.place_on_canvas(imaging.crop(rgb, box)))


async def run_preprocessing(
    raw: bytes, ptype: str, triton: TritonParser, pool: ThreadPoolExecutor,
    timings: dict | None = None,
) -> bytes:
    """If `timings` is given, fills decode/infer/encode seconds (CPU vs GPU split)."""
    loop = asyncio.get_running_loop()
    t0 = time.perf_counter()
    try:
        rgb, x = await loop.run_in_executor(pool, _decode_prep, raw)
    except Exception as e:                                   # corrupt/unsupported image
        raise DefinitiveError(f"decode/preprocess: {e}")
    t1 = time.perf_counter()

    try:
        seg = await triton.infer(x)
    except InferenceServerException as e:
        msg = str(e).lower()
        if any(k in msg for k in ("unavailable", "connect", "timeout")):
            raise TransientError(f"triton unavailable: {e}")   # server down/restarting → retry
        raise DefinitiveError(f"triton inference: {e}")        # e.g. OOM on this input
    t2 = time.perf_counter()

    try:
        png = await loop.run_in_executor(pool, _postprocess, rgb, seg, ptype)
    except Exception as e:
        raise DefinitiveError(f"postprocess: {e}")
    t3 = time.perf_counter()

    if timings is not None:
        timings["decode"] = t1 - t0        # CPU: JPEG decode + resize
        timings["infer"] = t2 - t1         # GPU: the actual SegFormer parse (Triton)
        timings["encode"] = t3 - t2        # CPU: crop + canvas + PNG encode
    return png
