"""HTTP clients for the worker: cpu_bridge (mint URLs), asset-service (complete),
and plain S3 GET/PUT over presigned URLs.

Error taxonomy (drives XACK vs reclaim):
  DefinitiveError — bad input / rejected / 4xx (except handled cases): report failed, XACK.
  TransientError  — network blip / 5xx / timeout: do NOT XACK, let XAUTOCLAIM reclaim.
"""
from __future__ import annotations

import httpx

from worker import config as C


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

    # ── cpu_bridge: mint short-lived signed links (worker sends only garment_id) ──
    async def mint_urls(self, garment_id: str) -> dict:
        try:
            r = await self._http.post(
                f"{C.CPU_BRIDGE_URL}/bridge/garment/urls",
                json={"garment_id": garment_id},
                headers={"X-Internal-Auth": C.BRIDGE_TO_GPU_SECRET},
            )
        except httpx.HTTPError as e:
            raise TransientError(f"mint_urls network: {e}")
        if r.status_code == 200:
            return r.json()
        if r.status_code == 410:
            raise DefinitiveError("job metadata expired (410)")
        _classify(r.status_code, "mint_urls", r.text)

    # ── S3 over presigned URLs (no AWS creds) ──
    async def download(self, url: str) -> bytes:
        try:
            r = await self._http.get(url, timeout=C.S3_TIMEOUT)
        except httpx.HTTPError as e:
            raise TransientError(f"download network: {e}")
        if r.status_code == 200:
            return r.content
        _classify(r.status_code, "download", r.text)

    async def upload(self, url: str, data: bytes) -> None:
        try:
            r = await self._http.put(
                url, content=data, headers={"Content-Type": "image/png"}, timeout=C.S3_TIMEOUT,
            )
        except httpx.HTTPError as e:
            raise TransientError(f"upload network: {e}")
        if r.status_code in (200, 201, 204):
            return
        _classify(r.status_code, "upload", r.text)

    # ── asset-service: definitive completion signal (triggers Kafka event) ──
    async def complete(self, garment_id: str, status: str, error: str | None = None) -> None:
        payload = {"garment_id": garment_id, "status": status}
        if error:
            payload["error"] = error[:500]
        try:
            r = await self._http.post(
                f"{C.ASSET_SERVICE_URL}/v1/garment/complete",
                json=payload,
                headers={"X-Internal-Auth": C.ASSET_INTERNAL_SECRET},
            )
        except httpx.HTTPError as e:
            raise TransientError(f"complete network: {e}")
        if r.status_code == 200:
            return
        _classify(r.status_code, "complete", r.text)
