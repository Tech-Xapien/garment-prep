"""Async callback client — delivers processed garment images to the upstream API.

Features:
  - httpx.AsyncClient with connection pooling (reused across requests)
  - 5 retries with exponential back-off + jitter
  - Structured logging per attempt
  - Dead-letter logging on final failure
"""

from __future__ import annotations

import asyncio
import logging
import random
from typing import Optional

import httpx

from config import (
    CALLBACK_AUTH_TOKEN,
    CALLBACK_MAX_RETRIES,
    CALLBACK_BACKOFF_BASE,
    CALLBACK_BACKOFF_MAX,
    CALLBACK_TIMEOUT,
)

logger = logging.getLogger("preprocessor.callback")

# Module-level client — created once, reused across all callbacks.
_client: Optional[httpx.AsyncClient] = None


async def get_client() -> httpx.AsyncClient:
    global _client  # noqa: PLW0603
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(
            timeout=httpx.Timeout(CALLBACK_TIMEOUT, connect=10.0),
            limits=httpx.Limits(
                max_connections=20,
                max_keepalive_connections=10,
            ),
        )
    return _client


async def close_client() -> None:
    global _client  # noqa: PLW0603
    if _client and not _client.is_closed:
        await _client.aclose()
    _client = None


async def deliver(
    *,
    callback_url: str,
    garment_id: str | None,
    garment_png: bytes,
) -> bool:
    """POST the processed garment image to the upstream callback URL.

    Returns True on success, False after all retries are exhausted.
    """
    client = await get_client()

    headers = {"X-Internal-Auth": CALLBACK_AUTH_TOKEN}

    for attempt in range(1, CALLBACK_MAX_RETRIES + 1):
        try:
            resp = await client.post(
                callback_url,
                headers=headers,
                files={
                    "garment_file": (
                        f"{garment_id or 'garment'}.png",
                        garment_png,
                        "image/png",
                    ),
                },
                data={
                    k: v for k, v in {"garment_id": garment_id}.items() if v
                },
            )

            # Only 200 + {"status": "ok"} is treated as success
            if resp.status_code == 200:
                try:
                    body = resp.json()
                except Exception:
                    body = {}
                if body.get("status") in ("ok", "success"):
                    logger.info(
                        "Callback success garment_id=%s attempt=%d",
                        garment_id, attempt,
                    )
                    return True

            logger.warning(
                "Callback not accepted garment_id=%s status=%d attempt=%d/%d body=%s",
                garment_id, resp.status_code, attempt, CALLBACK_MAX_RETRIES,
                resp.text[:200],
            )

        except httpx.HTTPError as exc:
            logger.warning(
                "Callback network error garment_id=%s attempt=%d/%d error=%s",
                garment_id, attempt, CALLBACK_MAX_RETRIES, exc,
            )

        # Exponential back-off with full jitter
        if attempt < CALLBACK_MAX_RETRIES:
            delay = min(
                CALLBACK_BACKOFF_BASE * (2 ** (attempt - 1)),
                CALLBACK_BACKOFF_MAX,
            )
            delay *= random.uniform(0.5, 1.0)
            logger.info("Retrying garment_id=%s in %.1fs …", garment_id, delay)
            await asyncio.sleep(delay)

    # All retries exhausted — dead-letter log
    logger.error(
        "DEAD_LETTER garment_id=%s url=%s — all %d retries exhausted",
        garment_id, callback_url, CALLBACK_MAX_RETRIES,
    )
    return False
