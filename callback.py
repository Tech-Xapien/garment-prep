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
    user_id: str,
    job_id: str,
    provider: str,
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
                        f"{job_id}.png",
                        garment_png,
                        "image/png",
                    ),
                },
                data={
                    "user_id": user_id,
                    "job_id": job_id,
                    "provider": provider,
                },
            )

            if resp.status_code < 400:
                logger.info(
                    "Callback success job_id=%s status=%d attempt=%d",
                    job_id, resp.status_code, attempt,
                )
                return True

            # Server returned an error — log and retry
            logger.warning(
                "Callback HTTP %d job_id=%s attempt=%d/%d body=%s",
                resp.status_code, job_id, attempt, CALLBACK_MAX_RETRIES,
                resp.text[:200],
            )

        except httpx.HTTPError as exc:
            logger.warning(
                "Callback network error job_id=%s attempt=%d/%d error=%s",
                job_id, attempt, CALLBACK_MAX_RETRIES, exc,
            )

        # Exponential back-off with full jitter
        if attempt < CALLBACK_MAX_RETRIES:
            delay = min(
                CALLBACK_BACKOFF_BASE * (2 ** (attempt - 1)),
                CALLBACK_BACKOFF_MAX,
            )
            delay *= random.uniform(0.5, 1.0)
            logger.info("Retrying job_id=%s in %.1fs …", job_id, delay)
            await asyncio.sleep(delay)

    # All retries exhausted — dead-letter log
    logger.error(
        "DEAD_LETTER job_id=%s user_id=%s provider=%s url=%s — "
        "all %d retries exhausted",
        job_id, user_id, provider, callback_url, CALLBACK_MAX_RETRIES,
    )
    return False
