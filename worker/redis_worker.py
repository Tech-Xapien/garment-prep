"""Redis Streams consumer: XREADGROUP loop + XAUTOCLAIM reclaim.

Implements the Worker Contract owned by the tuck_service repo
(docs/GARMENT_PREP_UPDATE_GUIDE.md); see docs/PIPELINE.md section 4. XACK happens ONLY
on a definitive outcome (completed, or failed-and-reported); transient failures are
left in the PEL so XAUTOCLAIM hands them to another worker after the idle timeout.
"""
from __future__ import annotations

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor

import redis.asyncio as aioredis

from worker import config as C
from worker.clients import Clients, DefinitiveError, TransientError
from worker.pipeline import run_preprocessing

logger = logging.getLogger("worker")


class Worker:
    def __init__(self, redis, clients: Clients, triton, pool: ThreadPoolExecutor, consumer: str):
        self.redis = redis
        self.clients = clients
        self.triton = triton
        self.pool = pool
        self.consumer = consumer
        self._stop = asyncio.Event()

    async def ensure_group(self) -> None:
        # id="0": if the stream already holds un-trimmed (=unprocessed) entries — e.g.
        # after a Redis flush/restart re-created the stream without the group — the new
        # group picks them up instead of orphaning them. Processed entries are trimmed by
        # the backend, so "0" never reprocesses completed work.
        try:
            await self.redis.xgroup_create(C.STREAM_KEY, C.CONSUMER_GROUP, id="0", mkstream=True)
            logger.info("created group %s on %s (id=0)", C.CONSUMER_GROUP, C.STREAM_KEY)
        except aioredis.ResponseError as e:
            if "BUSYGROUP" not in str(e):
                raise

    async def _recreate_if_nogroup(self, e: Exception) -> bool:
        """Self-heal: if the group vanished at runtime (Redis flush/restart), re-create
        it and signal the caller to retry. Returns True if it was a NOGROUP error."""
        if "NOGROUP" not in str(e):
            return False
        logger.warning("consumer group missing — recreating %s/%s", C.STREAM_KEY, C.CONSUMER_GROUP)
        try:
            await self.ensure_group()
        except aioredis.RedisError as ce:
            logger.warning("group recreate failed: %s", ce)
        return True

    # ── one job, start to finish ──
    async def _handle(self, fields: dict) -> dict:
        """Run one job; return per-stage timings (seconds)."""
        garment_id = fields.get("garment_id", "?")
        t: dict = {}
        m0 = time.perf_counter()
        urls = await self.clients.mint_urls(garment_id)          # cpu_bridge
        m1 = time.perf_counter(); t["mint"] = m1 - m0
        raw = await self.clients.download(urls["download_url"])   # S3 GET (cross-region)
        m2 = time.perf_counter(); t["download"] = m2 - m1; t["src_kb"] = len(raw) / 1024
        png = await run_preprocessing(                            # decode/infer/encode → fills t
            raw, fields.get("pipeline_type", "full"), self.triton, self.pool, timings=t)
        m3 = time.perf_counter(); t["out_kb"] = len(png) / 1024
        await self.clients.upload(urls["upload_url"], png)        # S3 PUT (cross-region)
        m4 = time.perf_counter(); t["upload"] = m4 - m3
        await self.clients.complete(garment_id, "completed")      # asset-service → Kafka
        m5 = time.perf_counter(); t["complete"] = m5 - m4
        t["total"] = m5 - m0
        return t

    async def _process(self, entry_id: str, fields: dict) -> None:
        """Run one entry and apply the XACK / no-ack decision."""
        garment_id = fields.get("garment_id", "?")

        # Pre-inference failure entry from the backend: report failed, XACK, skip inference.
        if fields.get("error"):
            try:
                await self.clients.complete(garment_id, "failed", fields["error"])
                await self._ack(entry_id)
            except TransientError as e:
                logger.warning("failure-entry report transient garment_id=%s: %s", garment_id, e)
            return

        try:
            t = await self._handle(fields)
            await self._ack(entry_id)
            logger.info(
                "completed garment_id=%s | mint=%.0f dl=%.0f decode=%.0f infer=%.0f "
                "encode=%.0f up=%.0f complete=%.0f TOTAL=%.0fms | src=%.0fKB out=%.0fKB",
                garment_id, t["mint"] * 1e3, t["download"] * 1e3, t["decode"] * 1e3,
                t["infer"] * 1e3, t["encode"] * 1e3, t["upload"] * 1e3, t["complete"] * 1e3,
                t["total"] * 1e3, t["src_kb"], t["out_kb"],
            )
        except TransientError as e:
            logger.warning("transient garment_id=%s: %s (leaving for reclaim)", garment_id, e)
            # no ack
        except DefinitiveError as e:
            logger.error("definitive garment_id=%s: %s", garment_id, e)
            await self._report_failed_and_ack(entry_id, garment_id, str(e))
        except Exception as e:                                   # unexpected: don't poison the PEL
            logger.exception("unexpected garment_id=%s", garment_id)
            await self._report_failed_and_ack(entry_id, garment_id, f"unexpected: {e}")

    async def _report_failed_and_ack(self, entry_id: str, garment_id: str, err: str) -> None:
        try:
            await self.clients.complete(garment_id, "failed", err)
            await self._ack(entry_id)
        except TransientError as e:
            logger.warning("could not report failure garment_id=%s: %s (reclaim)", garment_id, e)

    async def _ack(self, entry_id: str) -> None:
        await self.redis.xack(C.STREAM_KEY, C.CONSUMER_GROUP, entry_id)

    # ── loops ──
    async def _consume_loop(self) -> None:
        while not self._stop.is_set():
            try:
                res = await self.redis.xreadgroup(
                    C.CONSUMER_GROUP, self.consumer, {C.STREAM_KEY: ">"},
                    count=1, block=C.XREAD_BLOCK_MS,
                )
            except aioredis.RedisError as e:
                if await self._recreate_if_nogroup(e):
                    continue                     # retry immediately against the fresh group
                logger.warning("xreadgroup error: %s", e)
                await asyncio.sleep(1)
                continue
            for _stream, entries in res or []:
                for entry_id, fields in entries:
                    await self._process(entry_id, fields)

    async def _reclaim_loop(self) -> None:
        cursor = "0-0"
        while not self._stop.is_set():
            await asyncio.sleep(C.RECLAIM_INTERVAL_S)
            try:
                cursor, entries, _ = await self.redis.xautoclaim(
                    C.STREAM_KEY, C.CONSUMER_GROUP, self.consumer,
                    min_idle_time=C.RECLAIM_MIN_IDLE_MS, start_id=cursor, count=C.RECLAIM_COUNT,
                )
            except aioredis.RedisError as e:
                if await self._recreate_if_nogroup(e):
                    cursor = "0-0"
                    continue
                logger.warning("xautoclaim error: %s", e)
                cursor = "0-0"
                continue
            if entries:
                logger.info("reclaimed %d idle entries", len(entries))
                for entry_id, fields in entries:
                    await self._process(entry_id, fields)
            if not cursor or cursor == "0-0":
                cursor = "0-0"

    async def run(self) -> None:
        await self.ensure_group()
        tasks = [asyncio.create_task(self._consume_loop()) for _ in range(C.WORKER_CONCURRENCY)]
        tasks.append(asyncio.create_task(self._reclaim_loop()))
        logger.info("worker %s up: %d consumers on %s/%s",
                    self.consumer, C.WORKER_CONCURRENCY, C.STREAM_KEY, C.CONSUMER_GROUP)
        await self._stop.wait()
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    def stop(self) -> None:
        self._stop.set()
