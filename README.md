# Garment-Prep Service

FastAPI service that preprocesses garment images for downstream virtual try-on pipelines. It accepts a garment image (URL or upload), isolates the requested garment region (upper / lower / full / layered), places the result on a fixed-size RGBA canvas, and either returns the PNG synchronously or delivers it via async callback.

---

## Table of Contents

1. [Endpoints](#endpoints)
2. [Quick Start](#quick-start)
3. [Architecture](#architecture)
4. [Repository Layout](#repository-layout)
5. [Request Lifecycle](#request-lifecycle)
6. [Configuration Reference](#configuration-reference)
7. [Models & Inference Backend](#models--inference-backend)
8. [Pipelines](#pipelines)
9. [Output Contract](#output-contract)
10. [Operations](#operations)
11. [Extending the Service](#extending-the-service)
12. [Troubleshooting](#troubleshooting)

---

## Endpoints

| Method | Path          | Mode  | Purpose                                                                |
|--------|---------------|-------|------------------------------------------------------------------------|
| POST   | `/infer`      | Async | Enqueue a job; result delivered to `CALLBACK_URL` as multipart PNG     |
| POST   | `/preprocess` | Sync  | Returns processed PNG inline. **Only enabled when `TEST_MODE=1`**      |
| GET    | `/health`     | —     | Liveness, model-load state, queue depth, worker count                  |
| GET    | `/ui/`        | —     | Static webapp shipped under `webapp/` (test harness)                   |

### `POST /infer`

```json
{
  "garment_id": "abc123",
  "image_url": "https://cdn.example.com/garment.jpg",
  "pipeline_type": "upper",
  "callback_url": "http://..."
}
```

- `pipeline_type` is one of `full`, `upper`, `lower`, `layered`.
- `callback_url` is optional; falls back to `CALLBACK_URL`.
- `image_url` is normalized: if it lacks `http(s)://`, `IMAGE_URL_BASE` is prepended.
- The image is downloaded inside the request, lightly validated, then enqueued.
- `503` is returned when the queue is full. `400` is returned for download or decode failures.
- On success the body is `{"garment_id": ..., "status": "queued", "message": ...}` — the actual PNG is POSTed to `callback_url` later (see [Callback Delivery](#callback-delivery)).

### `POST /preprocess` (TEST_MODE only)

Multipart form: `image=<file>`, `type=<pipeline>`. Returns `image/png` in the body. Useful for local iteration; not enabled in production builds.

### `GET /health`

```json
{
  "status": "ok",
  "models_loaded": true,
  "queue_depth": 3,
  "inference_workers": 2
}
```

---

## Quick Start

### One-shot setup

```bash
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
./setup.sh                # install deps + pull model weights from S3
./setup.sh --start        # production: /infer only
./setup.sh --test         # adds TEST_MODE: /preprocess + /ui
```

`setup.sh` creates a Python 3.10 venv at `.venv/`, installs dependencies, syncs model weights from `s3://xapienappassets/garment-prep/models/` into `models/`, and pre-caches the FASHN human-parser checkpoint.

### Manual launch

```bash
source .venv/bin/activate
INFERENCE_WORKERS=2 QUEUE_WORKERS=4 \
  uvicorn main:app --host 0.0.0.0 --port 8000
```

---

## Architecture

```
                ┌──────────────────────────────────────────────┐
client ──POST──►│ /infer (FastAPI)                             │
                │  • httpx download (pooled, follow_redirects) │
                │  • PIL sanity-check                          │
                │  • enqueue JobPayload                        │
                └─────────────┬────────────────────────────────┘
                              │ asyncio.Queue (bounded, QUEUE_SIZE)
                              ▼
                ┌──────────────────────────┐
                │  QUEUE_WORKERS consumers │   ← async tasks in event loop
                └────────────┬─────────────┘
                             │ run_in_executor
                             ▼
                ┌────────────────────────────────────────────┐
                │  ProcessPoolExecutor (INFERENCE_WORKERS)   │
                │   spawn ctx; each proc loads its own       │
                │   YOLO head model + FASHN SegFormer parser │
                │   warmup pings on startup keep models hot  │
                └────────────┬───────────────────────────────┘
                             │ PNG bytes
                             ▼
                ┌──────────────────────────────────────────┐
                │ callback.deliver (httpx AsyncClient)     │
                │  multipart POST + retries w/ jitter      │
                │  DEAD_LETTER log after final failure     │
                └──────────────────────────────────────────┘
```

Key invariants:
- The event loop is never blocked: heavy CPU/GPU work runs in the process pool.
- Each worker process holds its **own** model copy — no shared GIL, no cross-process IPC during inference.
- Worker processes are warmed up at startup so the first real request doesn't pay the cold-load tax.
- The queue is bounded; back-pressure is surfaced to the client as `503`.

---

## Repository Layout

```
preprocessor/
├── main.py              FastAPI app: /infer, /preprocess (TEST_MODE), /health
├── queue_worker.py      asyncio.Queue + ProcessPoolExecutor lifecycle
├── callback.py          Async callback POSTer (retries, jitter, dead-letter log)
├── config.py            All tunables (env-driven)
├── schemas.py           Pydantic models — request/response/JobPayload
├── setup.sh             Venv + deps + S3 model sync + optional launcher
├── inference/
│   ├── backend.py       Protocol: FaceDetector, HumanParser, InferenceBackend
│   └── local.py         LocalBackend — loads models per-process; pool initializer
├── pipelines/
│   ├── full.py          Head-exclusion crop (with feet cap on alpha-padding)
│   └── garment.py       Upper / lower / layered crops with green-out
├── services/
│   ├── face.py          YOLOv8 CrowdHuman head detector (thread-locked)
│   ├── parser.py        FASHN SegFormer wrapper (thread-locked)
│   ├── crop.py          bbox / head-exclusion / green-out helpers
│   └── canvas.py        Final RGBA canvas placement (Lanczos, transparent pad)
├── models/              Local model weights (git-ignored, populated by setup.sh)
└── webapp/index.html    Static test harness mounted at /ui/
```

---

## Request Lifecycle

1. **Receive.** `POST /infer` resolves `image_url` against `IMAGE_URL_BASE`, downloads via a shared `httpx.AsyncClient` (pooled), and runs a fast `Image.open(...).load()` to reject obviously bad payloads.
2. **Enqueue.** Constructs a `JobPayload` (image bytes + pipeline type + callback URL) and `put_nowait`s onto the bounded queue. Returns `503` immediately if full.
3. **Consume.** One of `QUEUE_WORKERS` async tasks picks the job and submits it to the process pool via `loop.run_in_executor`.
4. **Process.** Inside a worker process:
    - Decode bytes → `np.ndarray` with EXIF transpose.
    - Run pipeline (`pipelines/full.py` or `pipelines/garment.py`) using that process's hot `FaceService` + `ParserService`.
    - Pass result through `services/canvas.place_on_canvas` → fixed-size RGBA.
    - Encode to PNG with embedded sRGB ICC profile.
5. **Deliver.** Back on the event loop, the consumer awaits `callback.deliver(...)`. See [Callback Delivery](#callback-delivery).
6. **Shutdown.** On SIGTERM/CTRL-C, `queue_worker.shutdown` drains in-flight jobs (≤ 120s), cancels consumers, tears down the pool, and closes the callback client.

### Callback Delivery

- `POST <callback_url>` multipart with `garment_file` + optional `garment_id`, header `X-Internal-Auth: <CALLBACK_AUTH_TOKEN>`.
- Success = HTTP 200 **and** JSON body `{"status": "ok"}` or `{"status": "success"}`.
- Up to `CALLBACK_MAX_RETRIES` attempts; backoff = `min(BASE * 2^(n-1), MAX) * uniform(0.5, 1.0)` (full jitter).
- Final failure emits a `DEAD_LETTER` log line — no other persistence.

---

## Configuration Reference

All settings are environment variables (see `config.py` for current defaults).

### Server
| Var       | Default   | Notes                              |
|-----------|-----------|------------------------------------|
| `HOST`    | `0.0.0.0` | Uvicorn bind host                  |
| `PORT`    | `8000`    | Uvicorn bind port                  |
| `WORKERS` | `1`       | Uvicorn worker count (keep at 1)   |

### Queue & inference workers
| Var                  | Default | Notes                                                     |
|----------------------|---------|-----------------------------------------------------------|
| `INFERENCE_WORKERS`  | `1`     | Size of `ProcessPoolExecutor`. Each loads its own models. |
| `QUEUE_WORKERS`      | `4`     | Async consumers draining the queue                        |
| `QUEUE_SIZE`         | `100`   | Max pending jobs before `/infer` returns `503`            |

### Callback
| Var                      | Default                                       | Notes                                              |
|--------------------------|-----------------------------------------------|----------------------------------------------------|
| `CALLBACK_URL`           | `http://3.110.153.25:9009/v1/garment/upload`  | Default destination if request omits `callback_url` |
| `CALLBACK_AUTH_TOKEN`    | `supersecret-internal-token`                  | Sent as `X-Internal-Auth`                          |
| `CALLBACK_MAX_RETRIES`   | `5`                                           | Total attempts including the first                 |
| `CALLBACK_BACKOFF_BASE`  | `1.0`                                         | Seconds; doubled each attempt                      |
| `CALLBACK_BACKOFF_MAX`   | `16.0`                                        | Hard cap on per-attempt sleep                      |
| `CALLBACK_TIMEOUT`       | `30.0`                                        | Per-request HTTP timeout                           |

### Image input
| Var                       | Default | Notes                                                      |
|---------------------------|---------|------------------------------------------------------------|
| `IMAGE_DOWNLOAD_TIMEOUT`  | `30.0`  | Per-request timeout for `image_url` fetch                  |
| `IMAGE_URL_BASE`          | `""`    | Prepended to relative `image_url` values                   |

### Output canvas
| Var             | Default | Notes                                       |
|-----------------|---------|---------------------------------------------|
| `CANVAS_WIDTH`  | `768`   | Final image width                           |
| `CANVAS_HEIGHT` | `1024`  | Final image height. Set both to 1024 for square |

### Mode
| Var         | Default | Notes                                                                                |
|-------------|---------|--------------------------------------------------------------------------------------|
| `TEST_MODE` | `0`     | When `1`, loads models in the API process and enables `POST /preprocess` (sync mode) |

---

## Models & Inference Backend

The service uses two models, both loaded in every worker process:

1. **Head detector** — `models/crowdhuman_yolov8_head.pt`, served via `ultralytics.YOLO`. Picks the highest-confidence box at `conf=0.3`. Wrapped by `services/face.py:FaceService` with a `threading.Lock`.
2. **Human parser** — `fashn-ai/fashn-human-parser` (SegFormer), loaded via the `fashn-human-parser` package. Returns an `(H, W)` array of class IDs 0–17. Wrapped by `services/parser.py:ParserService`.

### Backend abstraction

`inference/backend.py` defines three `Protocol` types — `FaceDetector`, `HumanParser`, `InferenceBackend`. Pipelines depend only on this protocol; the concrete `inference/local.py:LocalBackend` is the only current implementation. A remote backend (Triton, ONNX-RT, etc.) would implement the same protocol and replace the import in `queue_worker.py:_run_inference`.

### Worker initialization

`queue_worker.start()`:
1. Creates the `ProcessPoolExecutor` with `mp_context="spawn"` and `initializer=init_worker`.
2. Submits one `_warmup_noop` task per worker so every process forks, runs `init_worker`, and loads its models before any real request arrives.
3. Spawns `QUEUE_WORKERS` async consumer tasks.

---

## Pipelines

All pipelines start with **head exclusion**: detect the head bbox and cut the image at its bottom edge. This preserves collars and necklines while removing facial identity.

| `pipeline_type`    | Module                | Behavior                                                                                       |
|--------------------|-----------------------|------------------------------------------------------------------------------------------------|
| `full` / `layered` | `pipelines/full.py`   | Head exclusion; on `alpha-padding`, crop also stops above the feet region                      |
| `upper`            | `pipelines/garment.py`| Crop between top of upper-garment bbox and top of lower-garment bbox; ±10% LR margin           |
| `lower`            | `pipelines/garment.py`| Crop the lower-garment bbox; extend downward, on `alpha-padding` capped above feet (footwear) |

Label IDs (`UPPER_LABELS`, `LOWER_LABELS`, and `FEET_LABEL` where applicable) live in `config.py` and follow the FASHN parser class scheme.

After pipeline output, `services/canvas.place_on_canvas`:
- Scales the crop to fit inside `(CANVAS_WIDTH × CANVAS_HEIGHT)` with a 5% margin, preserving aspect ratio via Lanczos.
- Centers it on an RGBA canvas where padding pixels carry `alpha=0` so downstream attention models can mask them out.

---

## Output Contract

- **Format:** PNG, RGBA, embedded sRGB ICC profile.
- **Dimensions:** `CANVAS_WIDTH × CANVAS_HEIGHT` (default 768×1024).
- **Padding pixels:** `alpha=0`. Garment pixels: `alpha=255`.
- **Delivery:** multipart `garment_file` field (filename `<garment_id>.png`), optional `garment_id` form field, `X-Internal-Auth` header.
- **Success criterion:** HTTP 200 + JSON body containing `"status": "ok"` or `"status": "success"`. Any other response is treated as a failure and retried.

---

## Operations

### Health checks
`GET /health` returns `queue_depth` and `inference_workers`. Use it for both liveness and basic capacity inspection.

### Logging
- Structured stdlib logging at INFO. Loggers: `preprocessor`, `preprocessor.worker`, `preprocessor.callback`, `preprocessor.inference.local`.
- Per-request log lines include `garment_id`, `pipeline_type`, and `attempt` counts.
- A failed callback emits `DEAD_LETTER garment_id=... url=... — all N retries exhausted` — this is the signal to investigate upstream availability.

### Scaling
- **CPU/GPU-bound throughput:** raise `INFERENCE_WORKERS`. Each new worker adds one full model copy in memory.
- **I/O parallelism:** raise `QUEUE_WORKERS`. Cheap — they're just async tasks.
- **Burst headroom:** raise `QUEUE_SIZE`. Be aware: enqueued image bytes sit in process RAM.

### Shutdown semantics
On shutdown the queue is drained for up to 120s before consumers are cancelled and the pool is torn down. Long-running pipeline calls inside worker processes are not interrupted gracefully — set deployment grace periods accordingly.

---

## Extending the Service

### Add a new pipeline type
1. Add the value to `schemas.PipelineType`.
2. Implement a `run(image_rgb, ...)` function in `pipelines/`.
3. Dispatch to it in `queue_worker._run_inference` (and `main._process` if you want it under TEST_MODE).

### Swap the inference backend
1. Implement a class matching the `InferenceBackend` protocol in `inference/`.
2. Replace `from inference.local import get_backend` in `queue_worker._run_inference`.
3. Update `init_worker` registration if the new backend needs per-process setup.

### Change the output canvas
Set `CANVAS_WIDTH` / `CANVAS_HEIGHT`. The canvas placement logic in `services/canvas.py` is the single source of truth for downstream consumers.

---

## Troubleshooting

| Symptom                                       | Likely cause / check                                                                  |
|-----------------------------------------------|---------------------------------------------------------------------------------------|
| `/infer` returns `503`                        | Queue full. Inspect `queue_depth` via `/health`; raise `QUEUE_SIZE` or scale workers. |
| `/infer` returns `400 Failed to download`     | `image_url` unreachable / wrong scheme. Check `IMAGE_URL_BASE` if URL is relative.    |
| `/infer` returns `400 Invalid or corrupt image` | PIL couldn't decode the payload; verify content-type and bytes at source.            |
| `DEAD_LETTER` in logs                         | Callback upstream rejected the PNG after all retries. Check `CALLBACK_URL` reachability and that the upstream returns `{"status":"ok"}`. |
| Cold-start latency on first request           | Workers should warm up at boot — check for `All N inference workers warmed up.` log.  |
| OOM on startup                                | Each `INFERENCE_WORKERS` process loads its own models. Lower the count or upgrade RAM. |
| `RuntimeError: init_worker() was not called`  | A code path is reaching `get_backend()` outside the pool. Pool initialization changed — re-check `queue_worker.start()`. |

---

## Branches

- **`main`** — stable. Pipelines do head exclusion + semantic crops without footwear-cap.
- **`alpha-padding`** — adds `FEET_LABEL`-based capping so `full`/`lower` crops stop above the feet region (excludes footwear). The README on both branches is otherwise identical.
