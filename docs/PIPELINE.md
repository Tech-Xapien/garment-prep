# Garment-Prep — Pipeline Reference

End-to-end walkthrough of how one garment image becomes a processed PNG, every
step with the real code. The service is **one Docker image** running two tiers:

- **App tier** — all CPU work + orchestration. Either `worker/` (Redis-pull, **production**)
  or `gateway/` (FastAPI, QA), selected by `RUN_MODE`.
- **Triton** (TensorRT backend) — the single SegFormer parse on the GPU.

They talk over **localhost gRPC** inside the container. The compute core is shared: both
tiers call the same `gateway.imaging` and `gateway.crop` functions, so output is identical.
§3 walks the pixel path using the HTTP framing; §4 covers the production framing.

```
POST /infer ─► download ─► queue ─► consumer ─┐
                                              │  (per job)
      ┌───────────────────────────────────────┘
      ▼   [thread] decode+preprocess        Triton (GPU, batched)
   rgb, uint8[1,3,576,384] ──gRPC──► segformer.plan ──► seg int32[576,384]
      │                                                     │
      └──────────────► [thread] crop → canvas → PNG ◄───────┘
                                              │
                                     callback POST (retries)
```

---

## 1. Code structure

```
gateway/                     CPU tier — runs as uvicorn (GATEWAY_WORKERS procs)
  app.py            FastAPI app: /infer, /preprocess, /health; queue + consumers + threadpool
  config.py         every env knob (CPU + GPU + canvas + callback)
  triton_client.py  async gRPC client to the co-located Triton
  imaging.py        decode / preprocess / crop / canvas / encode  (the CPU-heavy stages)
  crop.py           single-parse geometry: head-cut + garment bbox, seg-space → original px
  callback.py       pooled async delivery of the PNG, exp backoff + dead-letter
  bench.py          load-test harness (throughput × latency)

worker/                      production app tier — Redis Streams, wraps the same compute core
  main.py           entrypoint: one consumer process per WORKER_PROCESSES; waits for Triton
  redis_worker.py   XREADGROUP loop + XAUTOCLAIM reclaim; the XACK policy
  pipeline.py       run_preprocessing — decode → parse → crop → canvas → PNG, with timings
  clients.py        cpu_bridge URL minting, S3 GET/PUT, asset-service complete
  config.py         Redis + backend endpoint knobs (separate from gateway/config.py)

engine/                      build-time — produces the artifact Triton loads
  export_onnx.py    SegFormer → ONNX with normalize + argmax fused into the graph
  build_trt.sh      ONNX → segformer.plan (trtexec --fp16, dynamic batch)
  s3.py             tiny boto3 get/put helper (image has no aws CLI)

triton/models/segformer_parser/
  config.pbtxt.template   TensorRT model config; env knobs rendered at boot
  1/                       engine dir; model.plan fetched/mounted at boot (never committed)

docker/
  Dockerfile        FROM nvcr.io/nvidia/tritonserver:25.06-py3 (TRT 10.11 / CUDA 12.9 / sm_120)
  entrypoint.sh     render config → provide engine (mount/S3) → launch Triton + gateway
  requirements.txt  lean gateway deps (no torch)
  DEPLOY.md         EC2 env contract + the verified run commands
```

---

## 2. The engine — what the GPU actually runs

The `.plan` is not just the SegFormer network: **normalization and argmax are baked
into the graph** so the gateway ships raw `uint8` and receives a compact class map
(≈0.9 MB) instead of an 18-channel float tensor (≈16 MB). Built once from this wrapper
(`engine/export_onnx.py`):

```python
class ParserGraph(nn.Module):
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:   # uint8 [B,3,576,384]
        x = pixel_values.float() / 255.0
        x = (x - self.mean) / self.std                 # ImageNet norm, on GPU
        logits = self.model(pixel_values=x).logits     # (B,18,144,96)
        logits = nn.functional.interpolate(logits, size=(576, 384), mode="bilinear",
                                           align_corners=False)
        return logits.argmax(dim=1).to(torch.int32)     # (B,576,384) class ids 0..17
```

Triton serves it with a static spatial shape and a **dynamic batch** so concurrent
requests coalesce into one forward pass (`triton/models/segformer_parser/config.pbtxt.template`):

```
platform: "tensorrt_plan"
max_batch_size: ${MAX_BATCH_SIZE}
input  [ { name: "pixel_values" data_type: TYPE_UINT8 dims: [ 3, 576, 384 ] } ]
output [ { name: "seg"          data_type: TYPE_INT32 dims: [ 576, 384 ] } ]
dynamic_batching { preferred_batch_size: [ ${OPT_BATCH_SIZE} ]
                   max_queue_delay_microseconds: ${MAX_QUEUE_DELAY_US} }
instance_group [ { count: ${INSTANCES} kind: KIND_GPU } ]
```

**FASHN 18-class labels** the crop logic uses: `1 face, 2 hair, 9 hat` (head), `3 top,
4 dress, 10 scarf` (upper), `5 skirt, 6 pants` (lower), `15 feet`.

---

## 3. Job lifecycle — step by step (HTTP framing)

Steps 6–13 are the **shared compute core** and run identically in both modes. Steps 1–5
and 14 are the HTTP framing; for the production framing see §4.


### Step 1 — `POST /infer` accepts the job (`gateway/app.py`)
```python
@app.post("/infer", response_model=InferResponse)
async def infer(body: InferRequest):
    if body.pipeline_type not in _VALID_TYPES:            # full | upper | lower | layered
        raise HTTPException(422, ...)
    url = _normalize_url(body.image_url)                  # add scheme/IMAGE_URL_BASE if missing
```

### Step 2 — download the source image (pooled async client)
```python
    resp = await app.state.download.get(url)              # shared httpx.AsyncClient, keep-alive pool
    resp.raise_for_status()
    raw = resp.content                                    # bytes; no decode on the request path
```

### Step 3 — enqueue and return immediately (back-pressure)
```python
    job = _Job(garment_id=..., pipeline_type=..., callback_url=body.callback_url or CALLBACK_URL,
               image_bytes=raw)
    app.state.queue.put_nowait(job)                       # QueueFull -> HTTP 503
    return InferResponse(garment_id=body.garment_id)      # {"status":"queued"} — async contract
```
The bounded `asyncio.Queue(QUEUE_SIZE)` is the buffer; `QUEUE_CONSUMERS` coroutines drain it.

### Step 4 — a consumer picks the job (`_consumer`)
```python
async def _consumer(app, idx):
    while True:
        job = await app.state.queue.get()
        try:
            png = await _run(app, job.image_bytes, job.pipeline_type)
            await callback.deliver(callback_url=job.callback_url,
                                   garment_id=job.garment_id, garment_png=png)
        finally:
            app.state.queue.task_done()
```

### Step 5 — the core pipeline (`_run`): CPU stages off-loaded, GPU awaited
```python
async def _run(app, raw, ptype):
    loop = asyncio.get_running_loop()
    rgb, x = await loop.run_in_executor(app.state.pool, _decode_prep, raw)   # thread: CPU
    seg    = await app.state.triton.infer(x)                                 # GPU (batched)
    return   await loop.run_in_executor(app.state.pool, _postprocess, rgb, seg, ptype)  # thread: CPU
```
The CPU stages run in a `ThreadPoolExecutor(CPU_THREADS)`; decode/resize/encode release the
GIL, so threads give real parallelism while the GPU call is awaited.

### Step 6 — decode (`imaging.decode`)
```python
def decode(raw: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(raw))
    img = ImageOps.exif_transpose(img)          # honour camera orientation (product photos)
    return np.asarray(img.convert("RGB"))       # (H, W, 3) uint8
```

### Step 7 — preprocess to the engine's exact input (`imaging.preprocess`)
```python
def preprocess(rgb: np.ndarray) -> np.ndarray:
    resized = cv2.resize(rgb, (384, 576), interpolation=cv2.INTER_AREA)  # matches parser training
    chw = np.ascontiguousarray(resized.transpose(2, 0, 1))               # HWC -> CHW
    return chw[None]                                                     # (1,3,576,384) uint8
```
Only the resize is on the gateway; `/255` + ImageNet norm happen on the GPU (Step 9).

### Step 8 — GPU inference over gRPC (`gateway/triton_client.py`)
```python
async def infer(self, pixel_values):                    # (1,3,576,384) uint8
    inp = grpcclient.InferInput("pixel_values", pixel_values.shape, "UINT8")
    inp.set_data_from_numpy(pixel_values)
    res = await self._client.infer(model_name="segformer_parser",
                                   inputs=[inp], outputs=[grpcclient.InferRequestedOutput("seg")])
    return res.as_numpy("seg")[0]                        # (576,384) int32 class map
```
Triton's dynamic batcher merges this with other in-flight requests before the forward pass.

### Step 9 — inside the engine (GPU)
`uint8 → /255 → ImageNet norm → SegFormer B4 → bilinear upsample to 576×384 → argmax → int32`
(all fused, see §2). Output is the per-pixel class map.

### Step 10 — derive the crop geometry (`gateway/crop.py`)
One seg map yields **both** the head cut and the garment box — no YOLO. Head cut is the
**bottom of the `face` label** (falls back to hair/hat only for back views; never the
face∪hair union, which long hair would drag into the garment):
```python
def _head_cut_row(seg):
    face_rows = np.where(seg == FACE)[0]
    if face_rows.size: return int(face_rows.max())
    head_rows = np.where(np.isin(seg, HEAD_LABELS))[0]
    return int(head_rows.max()) if head_rows.size else 0
```
Then the per-type garment box is computed in seg space and scaled to original pixels:
```python
def compute_crop(seg, orig_h, orig_w, ptype):        # seg is 576x384
    cut = _head_cut_row(seg)
    sub = seg[cut:, :]
    box = _upper(sub) if ptype=="upper" else _lower(sub) if ptype=="lower" else _full(sub)
    y0,y1,x0,x1 = box or (0, sub.shape[0], 0, sub.shape[1])
    y0 += cut; y1 += cut                              # back to full-seg rows
    fy, fx = orig_h/seg.shape[0], orig_w/seg.shape[1] # seg-space → original px
    return (max(0,int(y0*fy)), min(orig_h,int(y1*fy)), max(0,int(x0*fx)), min(orig_w,int(x1*fx)))
```
- `_upper`: top of upper labels → top of lower labels (or bottom of upper), ± L/R margin.
- `_lower`: lower labels, extended down but capped at the top of the `feet` region.
- `_full`/`layered`: head-cut down to feet-top, tightened to garment width ± margin.

*(Validated against the old YOLO path: face-cut within ~3 px median of YOLO, final-crop
IoU 0.985 at engine resolution.)*

### Step 11 — crop the original (`imaging.crop`)
```python
def crop(rgb, box):
    y0, y1, x0, x1 = box
    out = rgb[y0:y1, x0:x1]
    return out if out.size else rgb
```

### Step 12 — place on the output canvas (`imaging.place_on_canvas`)
Scale to fit a `CANVAS_WIDTH×CANVAS_HEIGHT` (default 928×1664) canvas with a 5% margin,
centered, padding transparent (alpha 0) so downstream attention models skip whitespace.
Uses cv2 (≈9× faster than PIL Lanczos at equal quality):
```python
scale  = min(usable_w / w, usable_h / h)
interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LANCZOS4
resized = cv2.resize(rgb, (new_w, new_h), interpolation=interp)
canvas = np.empty((ch, cw, 4), np.uint8)
canvas[:, :, :3] = 255; canvas[:, :, 3] = 0                 # white, transparent padding
canvas[y:y+new_h, x:x+new_w, :3] = resized
canvas[y:y+new_h, x:x+new_w, 3]  = 255                      # opaque over the garment
```

### Step 13 — encode PNG (`imaging.encode_png`)
```python
Image.fromarray(rgba).save(buf, format="PNG",
                           icc_profile=_SRGB_ICC,           # embed sRGB
                           compress_level=PNG_COMPRESS_LEVEL)  # 1: lossless, ~2.5x faster than 6
```

### Step 14 — deliver via callback (`gateway/callback.py`)
```python
resp = await client.post(callback_url,
    headers={"X-Internal-Auth": CALLBACK_AUTH_TOKEN},
    files={"garment_file": (f"{garment_id}.png", garment_png, "image/png")},
    data={"garment_id": garment_id})
# success only on 200 + {"status": "ok"|"success"}; else exp backoff + jitter up to
# CALLBACK_MAX_RETRIES, then a DEAD_LETTER log.
```

---

## 4. Production framing — the Redis worker

`RUN_MODE=worker` (the default) replaces steps 1–5 and 14. There is no HTTP request and no
callback: jobs are pulled from a Redis stream and results are pushed to S3, with completion
reported to asset-service. Steps 6–13 above are unchanged.

`worker/main.py` launches `WORKER_PROCESSES` processes, each with a distinct
`WORKER_INDEX` → unique consumer name, all sharing the co-located Triton. Each process
waits for the model to report ready before consuming, and **exits non-zero if it never
does**, so the container's restart policy retries rather than idling with a dead GPU:

```python
async def _wait_for_triton(triton, tries=60):
    for _ in range(tries):
        try:
            if await triton.ready(): return True
        except Exception: pass
        await asyncio.sleep(1.0)
    return False
```

### One job, start to finish (`worker/redis_worker.py::_handle`)
```python
urls = await self.clients.mint_urls(garment_id)          # cpu_bridge → presigned GET+PUT
raw  = await self.clients.download(urls["download_url"]) # S3 GET (cross-region)
png  = await run_preprocessing(raw, fields["pipeline_type"], self.triton, self.pool, timings=t)
await self.clients.upload(urls["upload_url"], png)       # S3 PUT
await self.clients.complete(garment_id, "completed")     # asset-service → Kafka event
```
No AWS credentials live in the worker — S3 access is only ever via short-lived presigned
URLs minted per job by `cpu_bridge`.

### Delivery semantics: at-least-once, by way of the XACK policy
`XACK` fires **only on a definitive outcome**. That single rule is what makes the worker
crash-safe:

| outcome | classified as | action |
|---|---|---|
| completed | — | `XACK` |
| corrupt image, Triton OOM on this input | `DefinitiveError` | report `failed`, then `XACK` |
| Redis/Triton unreachable, S3 timeout | `TransientError` | **no ack** — left in the PEL |
| anything unexpected | `Exception` | report `failed`, `XACK` — never poison the PEL |

Unacked entries stay in the pending-entries list; a `_reclaim_loop` runs `XAUTOCLAIM`
every `RECLAIM_INTERVAL_S` and hands anything idle longer than `RECLAIM_MIN_IDLE_MS`
(default 60 s) to a live worker. A worker killed mid-job therefore loses nothing.

Classification happens in `worker/pipeline.py`, by inspecting the Triton error:
```python
except InferenceServerException as e:
    msg = str(e).lower()
    if any(k in msg for k in ("unavailable", "connect", "timeout")):
        raise TransientError(f"triton unavailable: {e}")   # server down/restarting → retry
    raise DefinitiveError(f"triton inference: {e}")        # e.g. OOM on this input
```

### Self-healing on a vanished group
If Redis is flushed or restarted the consumer group can disappear under a running worker.
Rather than spinning on `NOGROUP`, the worker recreates the group at `id=0` and retries —
so un-trimmed (unprocessed) entries are picked up instead of orphaned. Completed entries
are trimmed by the backend, so `id=0` never reprocesses finished work.

### Observability
One line per job, with the CPU/GPU split broken out — this is the primary signal for where
time is going in production:
```
completed garment_id=ext_abc | mint=12 dl=143 decode=26 infer=31 encode=48 up=87
  complete=19 TOTAL=366ms | src=284KB out=1782KB
```
While the stream is empty the worker logs **nothing**. That is idle, not stuck — confirm
with consumer idle time (`docker/DEPLOY.md` → Verify).

---

## 5. `/preprocess` — synchronous path (QA + benchmarking)
Same compute as `/infer` but the client uploads the bytes and gets the PNG back inline
(no queue, no callback). Used by `gateway/bench.py` and for visual QA:
```python
@app.post("/preprocess")
async def preprocess(image: UploadFile, type: str = Form(...)):
    raw = await image.read()
    png = await _run(app, raw, type)
    return Response(content=png, media_type="image/png")
```

---

## 6. Concurrency & scaling model

| Layer | Knob | Mode | Effect |
|-------|------|------|--------|
| app processes | `WORKER_PROCESSES` | worker | **primary CPU-parallelism** (throughput scales ~linearly with cores) |
| app processes | `GATEWAY_WORKERS` | http | same role for uvicorn |
| in-flight jobs / process | `WORKER_CONCURRENCY` | worker | how many jobs feed the GPU batch |
| in-flight jobs / process | `QUEUE_CONSUMERS` | http | same role, draining the bounded queue |
| CPU threadpool / process | `CPU_THREADS` | both | parallelism for decode/resize/encode (GIL-releasing) |
| GPU batch | `MAX_BATCH_SIZE` / `OPT_BATCH_SIZE` | both | Triton coalesces concurrent requests; also sets VRAM. `MAX_BATCH_SIZE` must be ≤ the engine's built `MAX_BATCH` |
| batch fill wait | `MAX_QUEUE_DELAY_US` | both | latency ↔ batch-fill tradeoff |
| GPU copies | `INSTANCES` | both | parallel engine copies (more VRAM, more throughput) |

**Measured profile (per image, CPU):** decode ~26 ms, resize small, canvas ~8 ms,
PNG encode ~48 ms → ~100 ms total. The pipeline is **CPU-bound** — on an 8-vCPU box the
GPU sits ~30–50% utilized, so throughput tracks CPU cores, not VRAM. Latest numbers:
**~37 img/s on 4 cores**, ~6.8 GB VRAM (at batch 32).
