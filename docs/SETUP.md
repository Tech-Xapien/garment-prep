# Garment-Prep — Setup

Standing the service up on a GPU box. Pick one branch:

| | when | how long |
|---|---|---|
| **[Branch A](#branch-a--pull-and-run-normal-path)** | the engine already exists for your GPU arch | ~10 min |
| **[Branch B](#branch-b--build-the-engine-from-scratch)** | new GPU arch, new TensorRT, or the parser model changed | ~30 min |
| **[Branch C](#branch-c--change-env-and-relaunch)** | retuning throughput, repointing Redis, rotating a token | ~1 min |

Verbatim production commands with the env values actually in use live in
[`../docker/DEPLOY.md`](../docker/DEPLOY.md). Pipeline internals are in
[`PIPELINE.md`](PIPELINE.md).

The system is **one image** plus **one external artifact** (the TensorRT engine). The
engine is never baked into the image: a `.plan` is tied to a specific TensorRT version
*and* GPU architecture, so baking it would make the image non-portable.

---

## 0. Prerequisites

| To… | You need |
|-----|----------|
| Run the service | NVIDIA driver ≥ 570 + `nvidia-container-toolkit`; Blackwell (sm_120) for the published engine |
| Pull the image | ~31 GB of disk for the image alone |
| Build the engine (Branch B) | a **Blackwell** GPU — the engine only loads on the arch it was built for |
| Export the ONNX (Branch B) | `torch` + `transformers`; neither is in the runtime image |

Published artifacts (`us-east-1`):
```
s3://xapien-vton-engines/garment-prep/segformer_576x384.onnx              # portable ONNX, 245 MB
s3://xapien-vton-engines/garment-prep/segformer_fp16_576x384_sm120.plan   # Blackwell engine, 132 MB
```

> **Known gap:** on the current prod host the instance role (`gpu-node-role`) gets `403`
> on this bucket for both `ListBucket` and `GetObject`. Until that IAM policy is fixed,
> any S3 step below needs explicit `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`.

## 1. Host prep (common to all branches)

```bash
nvidia-smi -L                       # expect: NVIDIA RTX PRO 6000 Blackwell  (sm_120)
docker info --format '{{.DockerRootDir}}'
```

**Put Docker's storage on the big disk.** The image is ~31 GB and the root volume on this
box is 73 GB with ~17 GB free — pulling to `/` will fail with "no space left on device".
On the prod host `data-root` is already `/opt/dlami/nvme/docker`; if you are provisioning
a new box, move it (and containerd's snapshotter) before pulling anything.

```bash
mkdir -p /opt/dlami/nvme/gp        # where the engine lives on the host
```

---

## Branch A — pull and run (normal path)

Nothing is built. You pull the image, put the engine on the host, and start a container.

### A.1 — get the image
```bash
docker pull fashionx/garment-prep:0.2.3
```

### A.2 — get the engine onto the host
```bash
aws s3 cp s3://xapien-vton-engines/garment-prep/segformer_fp16_576x384_sm120.plan \
  /opt/dlami/nvme/gp/segformer.plan
ls -l /opt/dlami/nvme/gp/segformer.plan     # expect ~132 MB
```
On the prod host this file already exists — check before re-downloading. If your creds
can't reach the bucket (see the gap above), copy it from another box that has it; the
`.plan` is portable across machines with the **same GPU arch and TensorRT version**.

### A.3 — launch

The engine is bind-mounted **straight onto its destination path**, so the entrypoint's
"already present" branch wins and neither `ENGINE_LOCAL_PATH` nor `ENGINE_S3_URI` is
needed. Production (`RUN_MODE` defaults to `worker`):

```bash
docker run -d --name garment-prep-worker \
  --restart unless-stopped --gpus all \
  -v /opt/dlami/nvme/gp/segformer.plan:/models/segformer_parser/1/model.plan:ro \
  --env-file /opt/dlami/nvme/garment-prep.env \
  fashionx/garment-prep:0.2.3
```

No published port: production work is pulled from Redis, not served over HTTP.

> **`garment-prep.env` does not exist on the host yet.** The live containers were started
> with individual `-e` flags, which is why the secrets are currently written into
> [`../docker/DEPLOY.md`](../docker/DEPLOY.md) — a committed file. Creating the env file
> is what gets them out of git:
>
> ```bash
> # values are in docker/DEPLOY.md; this file is NOT in the repo
> cat > /opt/dlami/nvme/garment-prep.env <<'EOF'
> REDIS_URL=redis://<host>:6379/0
> CPU_BRIDGE_URL=http://<host>:8080
> ASSET_SERVICE_URL=http://<host>:9009
> BRIDGE_TO_GPU_SECRET=<secret>
> ASSET_INTERNAL_SECRET=<secret>
> WORKER_PROCESSES=4
> WORKER_CONCURRENCY=8
> CPU_THREADS=4
> EOF
> chmod 600 /opt/dlami/nvme/garment-prep.env
> ```
>
> Until then, substitute the `-e` flags from `docker/DEPLOY.md` for `--env-file` above.
> Note the env file is read **at container create**, not on restart — see
> [Branch C](#branch-c--change-env-and-relaunch).

For a QA container instead, add `-p 8002:8000 -e RUN_MODE=http` and point `CALLBACK_URL`
at a dead address so `/infer` can never deliver into a real asset-service.

### A.4 — verify
Jump to [§2 Verify](#2-verify).

---

## Branch B — build the engine from scratch

Only when the GPU architecture changed, TensorRT changed, or the parser model / input
geometry changed. Otherwise reuse the published `.plan` via Branch A.

### B.0 — clone and build the image
```bash
git clone https://github.com/Tech-Xapien/garment-prep.git
cd garment-prep
docker build -f docker/Dockerfile -t fashionx/garment-prep:<version> .
```
`triton/models/` must be present in the checkout — `docker/Dockerfile` copies it and
`entrypoint.sh` reads `config.pbtxt.template` out of it at boot.

Push from a datacenter uplink, never a home connection: the NGC base has ~14 GB single
layers that Docker cannot resume if the connection drops.
```bash
docker login -u fashionx && docker push fashionx/garment-prep:<version>
```

### B.1 — build the `.plan` on Blackwell
Built **inside the image** so its TensorRT (10.11) is byte-for-byte the runtime's.
```bash
docker run --rm --gpus all \
  -e AWS_ACCESS_KEY_ID -e AWS_SECRET_ACCESS_KEY -e AWS_DEFAULT_REGION=us-east-1 \
  -v /opt/dlami/nvme/gp:/artifacts \
  --entrypoint bash fashionx/garment-prep:<version> -c '
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
Verify `trtexec` prints bindings `in uint8 [*,3,576,384]`, `out int32 [*,576,384]`.

Build with `MAX_BATCH=32` even if you intend to run at 16 — runtime `MAX_BATCH_SIZE` can
then be swept up to 32 without rebuilding. **Runtime `MAX_BATCH_SIZE` must never exceed
the `MAX_BATCH` the engine was built with.**

### B.2 — regenerate the ONNX (rarely needed)
Only if the parser model or input geometry changed. Needs torch, which the runtime image
does not carry:
```bash
pip install torch transformers
python engine/export_onnx.py --out artifacts/segformer.onnx
python engine/s3.py put artifacts/segformer.onnx \
  s3://xapien-vton-engines/garment-prep/segformer_576x384.onnx
```

### B.3 — launch
Continue at [A.3](#a3--launch), then verify.

---

## Branch C — change env and relaunch

For retuning throughput, repointing Redis, or rotating a token. **No image rebuild and no
engine re-fetch.**

> **`docker restart` will not do this.** A container's environment is fixed when it is
> *created*; restarting reuses the old env. To change any variable you must **recreate**
> the container.

```bash
# 1. snapshot the current config so you can roll back
docker inspect garment-prep-worker \
  > /opt/dlami/nvme/rollback-garment-prep-worker-$(date -u +%Y%m%dT%H%M%SZ).json

# 2. edit the env file
vi /opt/dlami/nvme/garment-prep.env

# 3. recreate
docker stop garment-prep-worker && docker rm garment-prep-worker
docker run -d --name garment-prep-worker \
  --restart unless-stopped --gpus all \
  -v /opt/dlami/nvme/gp/segformer.plan:/models/segformer_parser/1/model.plan:ro \
  --env-file /opt/dlami/nvme/garment-prep.env \
  fashionx/garment-prep:0.2.3
```

What a recreate picks up for free: `entrypoint.sh` re-renders `config.pbtxt` from
`MAX_BATCH_SIZE` / `OPT_BATCH_SIZE` / `INSTANCES` / `MAX_QUEUE_DELAY_US` on every start,
so Triton batching is retuned **without touching the image or the engine**.

| you changed | what's required |
|---|---|
| any env var (Redis, secrets, concurrency, batching) | recreate the container — this branch |
| the `.plan` on the host | recreate (the mount is read at boot) |
| Python code | rebuild the image, re-pull, recreate |
| `MAX_BATCH` beyond what the engine was built for | rebuild the engine — Branch B |

Redis-pull workers are safe to recreate under load: an in-flight job is simply not acked,
and `XAUTOCLAIM` hands it to another worker after 60 s. Nothing is lost.

---

## 2. Verify

```bash
START=$(docker inspect garment-prep-worker --format '{{.State.StartedAt}}')
docker logs --since "$START" garment-prep-worker 2>&1 \
  | grep -Ei 'entrypoint|READY|worker .* up|error'
```

Expect the entrypoint's config line, `segformer_parser | 1 | READY`, then one
`worker <host>-N up: ... on garment:jobs/garment-workers` per `WORKER_PROCESSES`, and no
errors.

**A worker-mode container logs nothing while the Redis stream is empty. That is idle, not
stuck.** Confirm it is actually consuming by checking consumer idle time:
```bash
docker exec garment-prep-worker python3 -c "
import redis, os
r = redis.Redis.from_url(os.environ['REDIS_URL'], socket_timeout=6)
for c in r.xinfo_consumers('garment:jobs', 'garment-workers'):
    print(c['name'].decode(), 'idle_ms', c['idle'], 'pending', c['pending'])"
```
Consumers from the current container should show single-digit to low-thousands `idle_ms`.
Values in the millions are leftover registrations from dead containers — harmless; clear
them with `XGROUP DELCONSUMER`.

**End-to-end GPU check** needs a `RUN_MODE=http` container (the worker has no HTTP port):
```bash
curl -s localhost:8002/health          # {"status":"ok","queue_depth":0,"consumers":4}
curl -s -o /tmp/out.png -w '%{http_code}\n' \
  -F image=@person.jpg -F type=upper localhost:8002/preprocess
python3 -c "from PIL import Image; im=Image.open('/tmp/out.png'); print(im.size, im.mode)"
# expect: 200, then (928, 1664) RGBA
```
Run it for `upper`, `lower` and `full` — they exercise different crop paths.

**Load test** (`gateway/bench.py`, bundled in the image):
```bash
docker run --rm --network host -v /path/to/images:/thumbs \
  --entrypoint python3 fashionx/garment-prep:0.2.3 \
  -m gateway.bench --host http://localhost:8002 --dir /thumbs --type upper \
  --concurrency 64 --n 300
```

## 3. Local development (no GPU)

The pixel code is plain numpy/cv2/PIL and needs neither a GPU nor Triton:
```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r docker/requirements.txt
```
- `gateway/imaging.py` and `gateway/crop.py` are directly unit-testable and profilable
  (see [`PIPELINE.md`](PIPELINE.md) §6).
- `gateway/app.py` and `worker/main.py` both need a reachable Triton (`TRITON_URL`).
  Point them at a Triton serving the engine on a GPU box, or run the whole image locally
  if you have an NVIDIA GPU.

## 4. Where things live

| Concern | File |
|---------|------|
| Production run commands + full env contract | `docker/DEPLOY.md` |
| Gateway/HTTP + canvas + callback knobs | `gateway/config.py` |
| Redis/worker + backend endpoint knobs | `worker/config.py` |
| Pipeline internals, step by step | `docs/PIPELINE.md` |
| Redis Streams contract, XACK policy | `worker/redis_worker.py` |
| Engine build | `engine/export_onnx.py`, `engine/build_trt.sh` |
| Triton model config | `triton/models/segformer_parser/config.pbtxt.template` |
| S3 get/put helper | `engine/s3.py` |
| Throughput ideas deliberately not shipped | `docs/DEFERRED_OPTIMIZATIONS.md` |
