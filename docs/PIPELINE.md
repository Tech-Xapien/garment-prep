# Garment-Prep — Pipeline Reference

End-to-end walkthrough of how one garment image becomes a processed PNG, every
step with the real code. The service is **one Docker image** running two processes:

- **Gateway** (FastAPI/uvicorn) — all CPU work + orchestration.
- **Triton** (TensorRT backend) — the single SegFormer parse on the GPU.

They talk over **localhost gRPC** inside the container.

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
  DEPLOY.md         RunPod/EC2 env contract + commands
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

## 3. Request lifecycle — step by step

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

## 4. `/preprocess` — synchronous path (QA + benchmarking)
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

## 5. Concurrency & scaling model

| Layer | Knob | Effect |
|-------|------|--------|
| uvicorn processes | `GATEWAY_WORKERS` | **primary CPU-parallelism** (throughput scales ~linearly with cores) |
| async pipelines / worker | `QUEUE_CONSUMERS` | how many jobs are in flight feeding the GPU batch |
| CPU threadpool / worker | `CPU_THREADS` | parallelism for decode/resize/encode (GIL-releasing) |
| GPU batch | `MAX_BATCH_SIZE` / `OPT_BATCH_SIZE` | Triton coalesces concurrent requests; also sets VRAM |
| batch fill wait | `MAX_QUEUE_DELAY_US` | latency ↔ batch-fill tradeoff |
| GPU copies | `INSTANCES` | parallel engine copies (more VRAM, more throughput) |

**Measured profile (per image, CPU):** decode ~26 ms, resize small, canvas ~8 ms,
PNG encode ~48 ms → ~100 ms total. The pipeline is **CPU-bound** — on an 8-vCPU box the
GPU sits ~30–50% utilized, so throughput tracks CPU cores, not VRAM. Latest numbers:
**~37 img/s on 4 cores**, ~6.8 GB VRAM (at batch 32).
