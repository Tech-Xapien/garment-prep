"""Async callback client — delivers the processed PNG to the upstream API.

Pooled httpx client, exponential back-off + jitter, dead-letter log on give-up.
"""
from __future__ import annotations

import asyncio
import logging
import random
from typing import Optional

import httpx

from gateway.config import (
    CALLBACK_AUTH_TOKEN, CALLBACK_BACKOFF_BASE, CALLBACK_BACKOFF_MAX,
    CALLBACK_MAX_RETRIES, CALLBACK_TIMEOUT,
)

logger = logging.getLogger("gateway.callback")

_client: Optional[httpx.AsyncClient] = None


def get_client() -> httpx.AsyncClient:
    global _client  # noqa: PLW0603
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(
            timeout=httpx.Timeout(CALLBACK_TIMEOUT, connect=10.0),
            limits=httpx.Limits(max_connections=40, max_keepalive_connections=20),
        )
    return _client


async def close_client() -> None:
    global _client  # noqa: PLW0603
    if _client and not _client.is_closed:
        await _client.aclose()
    _client = None


async def deliver(*, callback_url: str, garment_id: str | None, garment_png: bytes) -> bool:
    client = get_client()
    headers = {"X-Internal-Auth": CALLBACK_AUTH_TOKEN}

    for attempt in range(1, CALLBACK_MAX_RETRIES + 1):
        try:
            resp = await client.post(
                callback_url,
                headers=headers,
                files={"garment_file": (f"{garment_id or 'garment'}.png", garment_png, "image/png")},
                data={"garment_id": garment_id} if garment_id else {},
            )
            if resp.status_code == 200:
                try:
                    body = resp.json()
                except Exception:
                    body = {}
                if body.get("status") in ("ok", "success"):
                    logger.info("Callback ok garment_id=%s attempt=%d", garment_id, attempt)
                    return True
            logger.warning(
                "Callback rejected garment_id=%s status=%d attempt=%d/%d body=%s",
                garment_id, resp.status_code, attempt, CALLBACK_MAX_RETRIES, resp.text[:200],
            )
        except httpx.HTTPError as exc:
            logger.warning(
                "Callback net error garment_id=%s attempt=%d/%d: %s",
                garment_id, attempt, CALLBACK_MAX_RETRIES, exc,
            )

        if attempt < CALLBACK_MAX_RETRIES:
            delay = min(CALLBACK_BACKOFF_BASE * (2 ** (attempt - 1)), CALLBACK_BACKOFF_MAX)
            await asyncio.sleep(delay * random.uniform(0.5, 1.0))

    logger.error("DEAD_LETTER garment_id=%s url=%s — retries exhausted",
                 garment_id, callback_url)
    return False
