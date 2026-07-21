# Deploy — RunPod (dev/tune) → EC2 (prod)

**Portability principle:** the *same image* and the *same env-var names* run in both
places. Only three things differ between RunPod and EC2:
1. **Where the engine comes from** — a mounted file (RunPod) vs S3 (EC2).
2. **AWS auth** — access keys (RunPod) vs instance IAM role (EC2).
3. **The tuned numbers** — discovered by the benchmark on RunPod, pinned on EC2.

Everything else is identical, so a config that works on RunPod works on EC2.

Image: `fashionx/garment-prep:0.1.0`  ·  Ports: `8000` gateway, `8002` Triton metrics.

---

## Build & push the image (from a Docker-enabled box with a fast uplink)

The image is **architecture-independent** — build it anywhere with a Docker daemon
(a RunPod pod, a build server) and it runs on both the Ada dev pod and the Blackwell
EC2 box. Only the TRT `.plan` is arch-specific (built later, on Blackwell).

```bash
cd /workspace
git clone -b update/efficient https://<user>:<gh-pat>@github.com/Tech-Xapien/garment-prep.git
cd garment-prep
docker version --format '{{.Server.Version}}'    # confirm a daemon exists first
docker build -f docker/Dockerfile -t fashionx/garment-prep:0.1.0 -t fashionx/garment-prep:latest .
docker login -u fashionx                          # paste a Docker Hub PAT
docker push fashionx/garment-prep:0.1.0 && docker push fashionx/garment-prep:latest
```
Do **not** originate this push from a slow/home uplink — the NGC base has ~14 GB of
single layers that Docker cannot resume if the connection drops.

---

## Env vars

### A. Engine source + AWS  (the only *structural* difference between boxes)
| Var | RunPod (dev) | EC2 (prod) | Notes |
|-----|--------------|------------|-------|
| `ENGINE_LOCAL_PATH` | `/artifacts/segformer.plan` | — | mounted file; **wins if set** |
| `ENGINE_S3_URI` | (optional) | `s3://xapien-vton-engines/garment-prep/segformer_fp16_576x384_sm120.plan` | fetched at boot via boto3 |
| `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` | set (for S3) | **omit** | EC2 uses the instance IAM role |
| `AWS_DEFAULT_REGION` | `us-east-1` | `us-east-1` | bucket `xapien-vton-engines` lives in us-east-1 (S3 access is global) |

> Artifacts already in S3: `s3://xapien-vton-engines/garment-prep/segformer_576x384.onnx` (ONNX, 245 MB).
> The `.plan` is built on RunPod from that ONNX and uploaded next to it.

### B. GPU / Triton throughput knobs  (tune on RunPod, pin on EC2)
| Var | Default | Meaning |
|-----|---------|---------|
| `MAX_BATCH_SIZE` | `16` | max coalesced batch — **must be ≤ the `MAX_BATCH` used in `build_trt.sh`** |
| `OPT_BATCH_SIZE` | `8` | dynamic-batcher preferred size (match `OPT_BATCH` from the build) |
| `INSTANCES` | `1` | parallel engine copies on the GPU — VRAM multiplier, throughput multiplier |
| `MAX_QUEUE_DELAY_US` | `2000` | how long the batcher waits to fill a batch (latency ↔ batch-fill) |

### C. CPU / gateway knobs  (tune on RunPod, pin on EC2)
| Var | Default | Meaning |
|-----|---------|---------|
| `GATEWAY_WORKERS` | `1` | uvicorn processes — **primary CPU-parallelism knob** |
| `QUEUE_CONSUMERS` | `8` | async pipelines in flight per worker (feeds the GPU batch) |
| `CPU_THREADS` | `8` | threadpool per worker for decode/resize/encode |

### D. Output + callback  (same on both boxes)
| Var | Default |
|-----|---------|
| `CANVAS_WIDTH` / `CANVAS_HEIGHT` | `928` / `1664` |
| `LR_MARGIN_RATIO` | `0.01` |
| `CALLBACK_URL` | prod callback endpoint |
| `CALLBACK_AUTH_TOKEN` | internal token |
| `IMAGE_URL_BASE` | prefix for scheme-less image URLs (optional) |
| `IMAGE_DOWNLOAD_TIMEOUT` | `30` |

---

## Build the sm_120 engine (one-time, on ANY Blackwell box)

A TRT `.plan` is tied to the exact TensorRT version **and** GPU arch, so it MUST be
built on Blackwell (sm_120) **inside our image** (guarantees TRT 10.11 == the runtime).
Build it once, push to S3, and every EC2 boot just fetches it.

```bash
# on a Blackwell pod (or the EC2 box). Needs a Docker daemon + nvidia-container-toolkit.
nvidia-smi -L                              # confirm: "NVIDIA RTX PRO 6000 Blackwell" (sm_120)
docker pull fashionx/garment-prep:0.1.0
mkdir -p /workspace/artifacts

export AWS_ACCESS_KEY_ID=...  AWS_SECRET_ACCESS_KEY=...  AWS_DEFAULT_REGION=us-east-1
docker run --rm --gpus all \
  -e AWS_ACCESS_KEY_ID -e AWS_SECRET_ACCESS_KEY -e AWS_DEFAULT_REGION \
  -v /workspace/artifacts:/artifacts \
  --entrypoint bash fashionx/garment-prep:0.1.0 -c '
    set -e
    python engine/s3.py get s3://xapien-vton-engines/garment-prep/segformer_576x384.onnx /artifacts/segformer.onnx
    ONNX=/artifacts/segformer.onnx PLAN=/artifacts/segformer.plan MAX_BATCH=32 OPT_BATCH=16 bash engine/build_trt.sh
    trtexec --loadEngine=/artifacts/segformer.plan --shapes=pixel_values:8x3x576x384 2>&1 | tail -15   # verify it loads on THIS gpu
    python engine/s3.py put /artifacts/segformer.plan s3://xapien-vton-engines/garment-prep/segformer_fp16_576x384_sm120.plan
  '
```
Built with `MAX_BATCH=32` so you can sweep runtime `MAX_BATCH_SIZE` up to 32 during the
bench **without rebuilding the engine** (rebuilds are cheap, ~1-2 min, but this saves the loop).

## RunPod template

- **Image:** `fashionx/garment-prep:0.1.0`
- **GPU:** 1× RTX PRO 6000 (Blackwell) — same arch as EC2 so the engine is portable
- **Container disk:** 40 GB · **Volume:** 30 GB mounted at `/artifacts` (persists engine/onnx across restarts)
- **Expose HTTP ports:** `8000` (+ `8002`)
- **Container start command:** `sleep infinity`  ← dev pods only; the default entrypoint
  exits without an engine, so we override it, build the engine, then launch the server.

### Phase 1 — build the engine (no torch needed; ONNX already in S3)
```bash
cd /workspace
python engine/s3.py get s3://xapien-vton-engines/garment-prep/segformer_576x384.onnx /artifacts/segformer.onnx
ONNX=/artifacts/segformer.onnx PLAN=/artifacts/segformer.plan \
  MAX_BATCH=16 OPT_BATCH=8 bash engine/build_trt.sh
# verify binding shapes trtexec prints: in uint8 [*,3,576,384], out int32 [*,576,384]
```

### Phase 2 — serve + benchmark
```bash
ENGINE_LOCAL_PATH=/artifacts/segformer.plan GATEWAY_WORKERS=4 QUEUE_CONSUMERS=8 \
  MAX_BATCH_SIZE=16 INSTANCES=1 /entrypoint.sh &
curl -s localhost:8000/health
python -m gateway.bench --dir "/workspace/Lafayette Final" --type upper --concurrency 32 --n 500
# sweep GATEWAY_WORKERS / MAX_BATCH_SIZE / INSTANCES; record img/s vs cpu-cores vs vram.
```
Then upload the engine so EC2 can pull it:
```bash
python engine/s3.py put /artifacts/segformer.plan \
  s3://xapien-vton-engines/garment-prep/segformer_fp16_576x384_sm120.plan
```

---

## EC2 prod run  (coexist with vton — cap CPU, bound VRAM)
```bash
docker run -d --restart unless-stopped --name garment-prep \
  --gpus '"device=0"' \
  --cpus=<C_from_bench> --cpuset-cpus=<pinned-core-range> \   # CPU ceiling → won't starve vton
  -p 8000:8000 \
  -e ENGINE_S3_URI=s3://<bucket>/garment-prep/segformer-trt10.11-sm120.plan \
  -e AWS_DEFAULT_REGION=us-east-1 \                            # creds from IAM role
  -e MAX_BATCH_SIZE=<B> -e OPT_BATCH_SIZE=<O> -e INSTANCES=<N> -e MAX_QUEUE_DELAY_US=2000 \
  -e GATEWAY_WORKERS=<W> -e QUEUE_CONSUMERS=<Q> -e CPU_THREADS=<T> \
  -e CALLBACK_URL=<prod-callback> \
  fashionx/garment-prep:0.1.0
```
- **CPU coexistence** is enforced by `--cpus` / `--cpuset-cpus` (values from the bench).
- **VRAM** is bounded by `INSTANCES × batch` — tiny against 96 GB, so vton is never squeezed.
- **GPU compute** is time-shared; keep `INSTANCES` modest. If vton latency ever regresses,
  gate the share with CUDA MPS (`CUDA_MPS_ACTIVE_THREAD_PERCENTAGE`).
