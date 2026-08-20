# Deploy — EC2 (Blackwell, sm_120)

One image serves two roles, selected by `RUN_MODE`:

| `RUN_MODE` | app tier | used for |
|---|---|---|
| `worker` (default) | `WORKER_PROCESSES` × `python -m worker.main`, Redis Streams pull | production |
| `http` | `uvicorn gateway.app:app` on `:8000` | QA, benchmarking, smoke tests |

Both roles start Triton first (gRPC `:8001` in-container, metrics `:8002`) with the TensorRT
SegFormer engine, then the app tier. Triton HTTP is off so the gateway owns `:8000`.

Image: `fashionx/garment-prep:0.2.3` · Base: `nvcr.io/nvidia/tritonserver:25.06-py3`
(TensorRT 10.11, CUDA 12.9, sm_120).

> ⚠️ **TEMPORARY — secrets are inline below.** The real tokens are written into the `docker run`
> commands in this file so a deploy is copy-pasteable today. This file is committed, so these
> values are in git history. They are to be moved into an env file held outside the repo
> (`--env-file`), and this warning removed, once a location for it is chosen.

---

## Verified live deployment

Both containers run on `3.219.50.111` and were verified healthy on 2026-08-19 (Triton
`segformer_parser` READY in ~1 s, 0 errors after restart, `/preprocess` returning correct
928×1664 RGBA for `upper` / `lower` / `full`).

### Production — Redis-pull worker

```bash
docker run -d --name garment-prep-worker \
  --restart unless-stopped \
  --gpus all \
  -v /opt/dlami/nvme/gp/segformer.plan:/models/segformer_parser/1/model.plan:ro \
  -e REDIS_URL=redis://13.201.242.226:6379/0 \
  -e CPU_BRIDGE_URL=http://13.201.242.226:8080 \
  -e ASSET_SERVICE_URL=http://13.201.242.226:9009 \
  -e BRIDGE_TO_GPU_SECRET=123 \
  -e ASSET_INTERNAL_SECRET=supersecret-internal-token \
  -e WORKER_PROCESSES=4 \
  -e WORKER_CONCURRENCY=8 \
  -e CPU_THREADS=4 \
  fashionx/garment-prep:0.2.3
```

No published ports: production work arrives by pulling from Redis, not over HTTP.
`RUN_MODE` is unset, so it defaults to `worker`.

### QA / benchmark — HTTP gateway on host `:8002`

```bash
docker run -d --name garment-prep-test \
  --restart unless-stopped \
  --gpus all \
  -p 8002:8000 \
  -v /opt/dlami/nvme/gp/segformer.plan:/models/segformer_parser/1/model.plan:ro \
  -e RUN_MODE=http \
  -e CALLBACK_URL=http://127.0.0.1:1/disabled \
  -e GATEWAY_WORKERS=1 \
  -e QUEUE_CONSUMERS=4 \
  -e CPU_THREADS=2 \
  -e MAX_BATCH_SIZE=8 \
  -e OPT_BATCH_SIZE=4 \
  -e INSTANCES=1 \
  fashionx/garment-prep:0.2.3
```

`CALLBACK_URL` is deliberately pointed at a dead address so QA traffic through `/infer`
can never deliver into a real asset-service. Use `/preprocess` (synchronous, returns the
PNG) for smoke tests.

---

## How the engine is provided

`entrypoint.sh` resolves the engine in this order:

1. **A file already at `/models/segformer_parser/1/model.plan`** — nothing to do.
2. `ENGINE_LOCAL_PATH` — copied into place.
3. `ENGINE_S3_URI` — fetched with boto3.
4. Otherwise it exits 1.

Both live containers hit case 1: the `-v` bind mount lands directly on the destination path,
which is why neither `ENGINE_LOCAL_PATH` nor `ENGINE_S3_URI` is set. The plan is **never baked
into the image** — it is TensorRT-version and GPU-architecture specific.

Host artifacts:

| path | size | what |
|---|---|---|
| `/opt/dlami/nvme/gp/segformer.plan` | 132 MiB | fp16 sm_120 engine, 576×384, batch 1..16 |
| `/opt/dlami/nvme/gp/segformer.onnx` | 245 MiB | portable ONNX the plan is built from |

Also published at `s3://xapien-vton-engines/garment-prep/` as `segformer_576x384.onnx` and
`segformer_fp16_576x384_sm120.plan`.

> **Known gap:** the box's instance role (`gpu-node-role`) gets `403` on that bucket — both
> `s3:ListBucket` and `GetObject`. The `ENGINE_S3_URI` path therefore needs explicit
> `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`, or an IAM policy fix, before it will work
> on this host. The bind mount is what production actually relies on.

---

## Env reference

### Engine source
| var | value in use | notes |
|---|---|---|
| `ENGINE_LOCAL_PATH` | unset | copy a mounted file into the model repo |
| `ENGINE_S3_URI` | unset | boto3 fetch; needs credentials (see gap above) |
| `AWS_DEFAULT_REGION` | unset | `us-east-1` for `xapien-vton-engines` |

### Triton / GPU knobs — rendered into `config.pbtxt` at boot by `envsubst`
| var | default | worker | test | meaning |
|---|---|---|---|---|
| `MAX_BATCH_SIZE` | `16` | `16` | `8` | max coalesced batch; **must be ≤ the `MAX_BATCH` used in `engine/build_trt.sh`** |
| `OPT_BATCH_SIZE` | `8` | `8` | `4` | batcher's preferred size |
| `INSTANCES` | `1` | `1` | `1` | engine copies on the GPU — multiplies VRAM and throughput |
| `MAX_QUEUE_DELAY_US` | `2000` | default | default | how long the batcher waits to fill a batch |

### App tier
| var | default | worker | test | notes |
|---|---|---|---|---|
| `RUN_MODE` | `worker` | default | `http` | selects the app tier |
| `WORKER_PROCESSES` | `4` | `4` | — | Redis consumer processes |
| `WORKER_CONCURRENCY` | `8` | `8` | — | in-flight jobs per process |
| `GATEWAY_WORKERS` | `1` | — | `1` | uvicorn processes |
| `QUEUE_CONSUMERS` | `8` | — | `4` | async pipelines in flight |
| `CPU_THREADS` | `4` worker / `8` http | `4` | `2` | read from `worker/config.py` in worker mode, `gateway/config.py` in http mode — the two defaults differ |

### Worker mode — Redis + backends  *(contains secrets)*
| var | value |
|---|---|
| `REDIS_URL` | `redis://13.201.242.226:6379/0` |
| `CPU_BRIDGE_URL` | `http://13.201.242.226:8080` |
| `ASSET_SERVICE_URL` | `http://13.201.242.226:9009` |
| `BRIDGE_TO_GPU_SECRET` | `123` |
| `ASSET_INTERNAL_SECRET` | `supersecret-internal-token` |
| `GARMENT_STREAM_KEY` | default `garment:jobs` |
| `GARMENT_CONSUMER_GROUP` | default `garment-workers` |

These endpoint IPs have changed repeatedly, and `gateway/config.py` still carries a stale
`CALLBACK_URL` default (`3.110.84.73`) plus a real `CALLBACK_AUTH_TOKEN` default
(`supersecret-internal-token`). **Always set them explicitly at `docker run`** — do not rely
on the committed defaults.

### HTTP mode — output + callback
| var | default |
|---|---|
| `CALLBACK_URL` | `http://3.110.84.73:9009/v1/garment/upload` — **stale, always override** |
| `CALLBACK_AUTH_TOKEN` | `supersecret-internal-token` — **committed, override** |
| `CANVAS_WIDTH` / `CANVAS_HEIGHT` | `928` / `1664` |
| `LR_MARGIN_RATIO` | `0.01` |
| `PNG_COMPRESS_LEVEL` | `1` |
| `IMAGE_URL_BASE` | `""` — prefix for scheme-less image URLs |
| `IMAGE_DOWNLOAD_TIMEOUT` | `30` |

---

## Build & push the image

The image is architecture-independent; only the TRT plan is not. Build on a box with a fast
uplink — the NGC base has ~14 GB single layers that Docker cannot resume on a dropped
connection, so do not push from a home network. The EC2 box is a good choice: Docker's
data-root is already on the 1.7 TB NVMe.

```bash
git clone -b update/efficient https://github.com/Tech-Xapien/garment-prep.git
cd garment-prep
docker build -f docker/Dockerfile -t fashionx/garment-prep:<version> .
docker login -u fashionx          # paste a Docker Hub PAT
docker push fashionx/garment-prep:<version>
```

`triton/models/` must be present in the checkout — `docker/Dockerfile` copies it and
`entrypoint.sh` reads `config.pbtxt.template` out of it at boot. It went missing from git
once because `.gitignore` had an unanchored `models/` pattern; the pattern is now `/models/`.

## Rebuild the sm_120 engine

Only when the parser model, input geometry, TensorRT version, or GPU architecture changes.
A `.plan` is tied to the exact TRT version **and** arch, so build it on Blackwell **inside
this image** to guarantee TRT 10.11 matches the runtime.

```bash
nvidia-smi -L        # expect: NVIDIA RTX PRO 6000 Blackwell  (sm_120)
mkdir -p /opt/dlami/nvme/gp

docker run --rm --gpus all \
  -e AWS_ACCESS_KEY_ID -e AWS_SECRET_ACCESS_KEY -e AWS_DEFAULT_REGION=us-east-1 \
  -v /opt/dlami/nvme/gp:/artifacts \
  --entrypoint bash fashionx/garment-prep:0.2.3 -c '
    set -e
    python engine/s3.py get s3://xapien-vton-engines/garment-prep/segformer_576x384.onnx \
      /artifacts/segformer.onnx
    ONNX=/artifacts/segformer.onnx PLAN=/artifacts/segformer.plan \
      MAX_BATCH=32 OPT_BATCH=16 bash engine/build_trt.sh
    trtexec --loadEngine=/artifacts/segformer.plan \
      --shapes=pixel_values:8x3x576x384 2>&1 | tail -15
    python engine/s3.py put /artifacts/segformer.plan \
      s3://xapien-vton-engines/garment-prep/segformer_fp16_576x384_sm120.plan
  '
```

Build with `MAX_BATCH=32` so runtime `MAX_BATCH_SIZE` can be swept up to 32 without a
rebuild. Regenerating the ONNX itself needs torch + transformers: `engine/export_onnx.py`.

## Verify

```bash
docker logs --since "$(docker inspect garment-prep-worker --format '{{.State.StartedAt}}')" \
  garment-prep-worker 2>&1 | grep -Ei 'entrypoint|READY|worker .* up|error'
```

Expect `segformer_parser | 1 | READY`, then one `worker <host>-N up: ... on
garment:jobs/garment-workers` line per `WORKER_PROCESSES`, and no errors.

End-to-end through the HTTP container — a real GPU parse:

```bash
curl -s localhost:8002/health          # {"status":"ok","queue_depth":0,"consumers":4}
curl -s -o /tmp/out.png -w '%{http_code}\n' \
  -F image=@person.jpg -F type=upper localhost:8002/preprocess
python3 -c "from PIL import Image; im=Image.open('/tmp/out.png'); print(im.size, im.mode)"
# expect: 200, then (928, 1664) RGBA
```

Worker mode logs nothing while the stream is empty — that is idle, not stuck. Confirm by
checking consumer idle time is small:

```bash
docker exec garment-prep-worker python3 -c "
import redis, os
r = redis.Redis.from_url(os.environ['REDIS_URL'], socket_timeout=6)
for c in r.xinfo_consumers('garment:jobs', 'garment-workers'):
    print(c['name'].decode(), 'idle_ms', c['idle'], 'pending', c['pending'])"
```

Consumers belonging to the current container should show single-digit-to-low-thousands
`idle_ms`. Large values (days) are leftover registrations from dead containers and are
harmless; clear them with `XGROUP DELCONSUMER` if they get noisy.

## Changing any of this

A container's environment is fixed when it is **created** — `docker restart` reuses the
old env and will silently appear to do nothing. To change a variable you must recreate
the container; the step-by-step is
[`../docs/SETUP.md`](../docs/SETUP.md) Branch C. The entrypoint re-renders
`config.pbtxt` on every start, so the Triton batching knobs retune on a recreate with no
image rebuild and no engine re-fetch.

Recreating a Redis-pull worker under load is safe: an in-flight job is simply not acked,
and `XAUTOCLAIM` hands it to another worker after 60 s.

## Rollback

`docker inspect` the container before changing it, then recreate from the saved JSON:

```bash
docker inspect garment-prep-worker > /opt/dlami/nvme/rollback-garment-prep-worker-$(date -u +%Y%m%dT%H%M%SZ).json
docker stop garment-prep-worker && docker rm garment-prep-worker
# re-run the previous `docker run` with the prior image tag
```

Published tags: `0.1.0`, `0.1.1`, `0.2.0`, `0.2.1`, `0.2.2`, `0.2.3`, `latest`.

## Coexistence on the shared box

This host also runs `vton-worker` (~42 GB VRAM) and `measurement-pipeline`. GPU compute is
time-shared and VRAM use here is bounded by `INSTANCES × MAX_BATCH_SIZE` — small against
96 GB, so vton is not squeezed. Neither garment-prep container currently sets a CPU ceiling;
if CPU contention ever shows up, add `--cpus` / `--cpuset-cpus` (size them with
`python -m gateway.bench`), and if GPU latency regresses, gate the share with CUDA MPS
(`CUDA_MPS_ACTIVE_THREAD_PERCENTAGE`).
