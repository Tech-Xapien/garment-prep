# Garment-Prep — Single-Parse TensorRT Preprocessor

Turns a garment photo into the fixed **928×1664 RGBA PNG** the virtual try-on pipeline
consumes: one SegFormer parse on the GPU yields both the head cut and the garment bounding
box, the crop is placed on a transparent-padded canvas, and the result is delivered to the
platform.

**One parse per image, no second model.** The head cut is derived from the parser's own
`face` label rather than a separate YOLO detector — validated against the detector it
replaced at ~3 px median agreement on the face-bottom row and 0.985 crop IoU.

> **Deploying:** [`docker/DEPLOY.md`](docker/DEPLOY.md) — the verified run commands.
> **How it works, step by step:** [`docs/PIPELINE.md`](docs/PIPELINE.md).
> **Clone to running service:** [`docs/SETUP.md`](docs/SETUP.md).

## Layout

```
gateway/      CPU tier — FastAPI + all pixel work + the Triton client
worker/       production tier — Redis Streams consumer wrapping gateway's compute core
engine/       BUILD-TIME ONLY — SegFormer → ONNX → .plan, plus an S3 get/put helper
triton/       Triton model repo; config.pbtxt.template rendered from env at boot
docker/       Dockerfile · entrypoint.sh · pinned requirements.txt · DEPLOY.md
docs/         PIPELINE.md (internals) · SETUP.md (clone→run) · DEFERRED_OPTIMIZATIONS.md
```

`engine/` runs only when an engine is being built and is never imported by serving code.
`worker/` and `gateway/` share one compute core: `worker/pipeline.py` calls the same
`gateway.imaging` and `gateway.crop` functions the HTTP path uses, so both modes produce
byte-identical output.

## Two run modes, one image

`RUN_MODE` selects the app tier. Triton starts first either way, so the GPU path is
identical; only the job source differs.

| `RUN_MODE` | app tier | job source | used for |
|---|---|---|---|
| `worker` (default) | `WORKER_PROCESSES` × `python -m worker.main` | Redis Streams | **production** |
| `http` | `uvicorn gateway.app:app` on `:8000` | HTTP request | QA, benchmarking, smoke tests |

### Production — Redis-pull worker

Per job: mint presigned URLs from `cpu_bridge` → S3 `GET` the source → decode → Triton
parse → crop → canvas → PNG → S3 `PUT` → report `completed` to asset-service, which emits
the Kafka event. Every stage is timed and logged on one line.

Delivery is **at-least-once**. `XACK` fires only on a definitive outcome — completed, or
failed-and-reported. A transient fault (Redis or Triton unreachable, S3 timeout) is left
in the pending-entries list so `XAUTOCLAIM` hands it to another worker after
`RECLAIM_MIN_IDLE_MS`. The group self-heals: `NOGROUP` after a Redis flush recreates the
group at `id=0` so un-trimmed entries are picked up rather than orphaned.

### QA — HTTP gateway

| method | path | purpose |
|---|---|---|
| `POST` | `/preprocess` | multipart `image` + `type`; returns the PNG inline. **Use this for smoke tests.** |
| `POST` | `/infer` | JSON; queues the job and returns `{"status":"queued"}`, PNG delivered to `CALLBACK_URL` |
| `GET` | `/health` | `{"status":"ok","queue_depth":N,"consumers":N}` |

`pipeline_type` is one of `upper` (tops/coats/blazers), `lower` (skirts), `full` (dresses),
`layered` (kaftans).

## The engine

`segformer_fp16_576x384_sm120.plan` — SegFormer human parser, fp16, static 576×384 input,
dynamic batch 1..N. Normalization **and** argmax are fused into the graph, so the gateway
sends raw `uint8` and gets back a compact `int32` class map (~0.9 MB) instead of an
18-channel float tensor (~16 MB).

FASHN 18-class labels the crop logic reads: `1 face`, `2 hair`, `9 hat` (head);
`3 top`, `4 dress`, `10 scarf` (upper); `5 skirt`, `6 pants` (lower); `15 feet`.

A `.plan` is tied to its exact TensorRT version **and** GPU architecture, so it is **never
baked into the image**. It is bind-mounted from the host, or fetched from S3 at boot — see
[`docker/DEPLOY.md`](docker/DEPLOY.md), which also records that this host's IAM role
currently cannot read the bucket, so the bind mount is what production relies on.

## Output contract

PNG, RGBA, embedded sRGB ICC profile, `CANVAS_WIDTH`×`CANVAS_HEIGHT` (**928×1664**).
Padding pixels carry `alpha=0` and garment pixels `alpha=255`, so downstream attention
models can mask the padding out. Canvas margin is 5%.

## Performance

**CPU-bound by design.** Per image: decode ~26 ms, canvas ~8 ms, PNG encode ~48 ms —
~100 ms total CPU against a GPU parse that leaves the card ~30–50% utilized. Throughput
tracks cores, not VRAM: **~37 img/s on 4 cores**, ~6.8 GB VRAM at batch 32.

That is deliberate. This box is co-tenant with the live vton backend on one Blackwell
card, so keeping pixel work on the CPU acts as a GPU-usage cap. Moving it onto the GPU
would contend for the tensor cores vton needs — see
[`docs/DEFERRED_OPTIMIZATIONS.md`](docs/DEFERRED_OPTIMIZATIONS.md).

## Hardware / image

- **GPU:** RTX PRO 6000 Blackwell (sm_120). The engine only loads on the arch it was built for.
- **Base:** `nvcr.io/nvidia/tritonserver:25.06-py3` — TensorRT 10.11, CUDA 12.9. Chosen to
  share the NGC release train, and therefore the host driver floor, with vton.
- **No torch in the runtime image.** `docker/requirements.txt` is gateway deps only;
  torch and transformers are needed solely to regenerate the ONNX.
- Triton HTTP is disabled so the gateway owns `:8000`; Triton serves gRPC on `:8001`
  in-container and metrics on `:8002`.

Deployed: `fashionx/garment-prep:0.2.3`.
