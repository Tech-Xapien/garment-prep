"""HTTP clients for the worker: cpu_bridge (mint URLs), asset-service (complete),
and plain S3 GET/PUT over presigned URLs.

Error taxonomy (drives XACK vs reclaim):
  DefinitiveError — bad input / rejected / 4xx (except handled cases): report failed, XACK.
  TransientError  — network blip / 5xx / timeout: do NOT XACK, let XAUTOCLAIM reclaim.
"""
from __future__ import annotations

import asyncio
import logging

import httpx

from worker import config as C

logger = logging.getLogger("worker.clients")

# Connection-level faults worth an immediate in-process retry on a fresh socket.
# NOT ReadTimeout: a slow-but-delivered request may already have had a side effect,
# so timeouts fall through to the reclaim path instead of being blindly re-sent.
_RETRYABLE = (
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.RemoteProtocolError,   # "Server disconnected" — stale pooled keepalive socket
    httpx.ReadError,
    httpx.WriteError,
    httpx.PoolTimeout,
)


class DefinitiveError(Exception):
    """Reprocessing won't help — report failed and XACK."""


class TransientError(Exception):
    """Might succeed later — leave unacked for reclaim."""


def _classify(status: int, ctx: str, body: str = "") -> None:
    if status >= 500:
        raise TransientError(f"{ctx} {status}")
    raise DefinitiveError(f"{ctx} {status}: {body[:200]}")


class Clients:
    def __init__(self, http: httpx.AsyncClient) -> None:
        self._http = http

    async def _send(self, ctx: str, method: str, url: str, **kw) -> httpx.Response:
        """One HTTP call with in-process retry on connection-level faults.

        httpx opens a NEW connection on each retry, so a dead pooled keepalive socket
        is sidestepped in ~ms rather than dropping the job to the 60s reclaim timer.
        Retryable faults exhausted, or any other httpx error → TransientError (reclaim).
        All worker calls are idempotent, so a retry can't double-produce a garment.
        """
        last: Exception | None = None
        for i in range(C.HTTP_RETRY_ATTEMPTS):
            try:
                return await self._http.request(method, url, **kw)
            except _RETRYABLE as e:
                last = e
                if i < C.HTTP_RETRY_ATTEMPTS - 1:
                    logger.info("%s retry %d/%d after %s: %s",
                                ctx, i + 1, C.HTTP_RETRY_ATTEMPTS - 1, type(e).__name__, e)
                    await asyncio.sleep(C.HTTP_RETRY_BACKOFF_S * (i + 1))
            except httpx.HTTPError as e:                 # non-retryable transport error
                raise TransientError(f"{ctx} network: {e}")
        raise TransientError(f"{ctx} network (after {C.HTTP_RETRY_ATTEMPTS}): {last}")

    # ── cpu_bridge: mint short-lived signed links (worker sends only garment_id) ──
    async def mint_urls(self, garment_id: str) -> dict:
        r = await self._send(
            "mint_urls", "POST", f"{C.CPU_BRIDGE_URL}/bridge/garment/urls",
            json={"garment_id": garment_id},
            headers={"X-Internal-Auth": C.BRIDGE_TO_GPU_SECRET},
        )
        if r.status_code == 200:
            return r.json()
        if r.status_code == 410:
            raise DefinitiveError("job metadata expired (410)")
        _classify(r.status_code, "mint_urls", r.text)

    # ── S3 over presigned URLs (no AWS creds) ──
    async def download(self, url: str) -> bytes:
        r = await self._send("download", "GET", url, timeout=C.S3_TIMEOUT)
        if r.status_code == 200:
            return r.content
        _classify(r.status_code, "download", r.text)

    async def upload(self, url: str, data: bytes) -> None:
        r = await self._send(
            "upload", "PUT", url, content=data,
            headers={"Content-Type": "image/png"}, timeout=C.S3_TIMEOUT,
        )
        if r.status_code in (200, 201, 204):
            return
        _classify(r.status_code, "upload", r.text)

    # ── asset-service: definitive completion signal (triggers Kafka event) ──
    async def complete(self, garment_id: str, status: str, error: str | None = None) -> None:
        payload = {"garment_id": garment_id, "status": status}
        if error:
            payload["error"] = error[:500]
        r = await self._send(
            "complete", "POST", f"{C.ASSET_SERVICE_URL}/v1/garment/complete",
            json=payload, headers={"X-Internal-Auth": C.ASSET_INTERNAL_SECRET},
        )
        if r.status_code == 200:
            return
        _classify(r.status_code, "complete", r.text)
