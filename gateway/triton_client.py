"""Async gRPC client to the co-located Triton server.

We send one image per request; Triton's dynamic batcher coalesces concurrent
requests into a batch server-side. So gateway concurrency (QUEUE_CONSUMERS ×
uvicorn workers) is what feeds the GPU batch — no client-side batching needed.
"""
from __future__ import annotations

import numpy as np
from tritonclient.grpc import aio as grpcclient

from gateway.config import TRITON_MODEL, TRITON_TIMEOUT, TRITON_URL


class TritonParser:
    def __init__(self) -> None:
        self._client = grpcclient.InferenceServerClient(url=TRITON_URL, verbose=False)

    async def ready(self) -> bool:
        return await self._client.is_model_ready(TRITON_MODEL)

    async def infer(self, pixel_values: np.ndarray) -> np.ndarray:
        """pixel_values uint8 (1,3,576,384) -> seg int32 (576,384)."""
        inp = grpcclient.InferInput("pixel_values", pixel_values.shape, "UINT8")
        inp.set_data_from_numpy(pixel_values)
        out = grpcclient.InferRequestedOutput("seg")
        res = await self._client.infer(
            model_name=TRITON_MODEL,
            inputs=[inp],
            outputs=[out],
            client_timeout=TRITON_TIMEOUT,
        )
        return res.as_numpy("seg")[0]

    async def close(self) -> None:
        await self._client.close()
