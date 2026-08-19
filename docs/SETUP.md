# Garment-Prep — Setup

Clone to running service. For the production env contract and the verified deploy
commands, see [`../docker/DEPLOY.md`](../docker/DEPLOY.md). For how the pipeline works
internally, see [`PIPELINE.md`](PIPELINE.md).

The system is **one image** (gateway/worker + Triton) plus **one external artifact** (the
TensorRT engine, kept out of the image because a `.plan` is tied to a specific TensorRT
version + GPU architecture).

---

## 0. Prerequisites

| To… | You need |
|-----|----------|
| Build/run the **image** | Docker with a fast uplink (image ≈ 31 GB; the NGC base has ~14 GB layers) |
| Run the **service** | NVIDIA driver ≥ 570 + `nvidia-container-toolkit`; a Blackwell (sm_120) GPU for the prod engine |
| Build the **TRT engine** | a **Blackwell** GPU — the engine only loads on the arch it was built for |
| Export the **ONNX** | Python with `torch` + `transformers` (any machine; ONNX is portable) |

Published artifacts (`us-east-1`):
```
s3://xapien-vton-engines/garment-prep/segformer_576x384.onnx              # portable ONNX
s3://xapien-vton-engines/garment-prep/segformer_fp16_576x384_sm120.plan   # Blackwell engine
```
On the current prod host the instance role cannot read this bucket — see the known gap in
[`../docker/DEPLOY.md`](../docker/DEPLOY.md). The engine is bind-mounted from
`/opt/dlami/nvme/gp/segformer.plan` instead.

---

## 1. Clone

```bash
git clone -b update/efficient https://github.com/Tech-Xapien/garment-prep.git
cd garment-prep
```

`triton/models/` must be present after cloning — `docker/Dockerfile` copies it and
`entrypoint.sh` reads `config.pbtxt.template` from it at boot.

## 2. Build the image

Architecture-independent; build anywhere with Docker.
```bash
docker build -f docker/Dockerfile -t fashionx/garment-prep:<version> .
docker login -u fashionx && docker push fashionx/garment-prep:<version>
```
Push from a datacenter uplink, not a home connection — Docker cannot resume the NGC base's
~14 GB layers if the connection drops.

## 3. Build the TRT engine (one-time, on Blackwell)

Built **inside the image** so its TensorRT (10.11) matches the runtime exactly. Full
command in [`../docker/DEPLOY.md`](../docker/DEPLOY.md) — summary:
```bash
docker run --rm --gpus all -v $PWD/artifacts:/artifacts \
  -e AWS_ACCESS_KEY_ID -e AWS_SECRET_ACCESS_KEY -e AWS_DEFAULT_REGION=us-east-1 \
  --entrypoint bash fashionx/garment-prep:<version> -c '
    python engine/s3.py get s3://xapien-vton-engines/garment-prep/segformer_576x384.onnx /artifacts/segformer.onnx
    ONNX=/artifacts/segformer.onnx PLAN=/artifacts/segformer.plan MAX_BATCH=32 OPT_BATCH=16 bash engine/build_trt.sh
    python engine/s3.py put /artifacts/segformer.plan s3://xapien-vton-engines/garment-prep/segformer_fp16_576x384_sm120.plan'
```
To regenerate the ONNX itself (rarely needed — torch is not in the runtime image):
```bash
pip install torch transformers
python engine/export_onnx.py --out artifacts/segformer.onnx
```

## 4. Run it

`RUN_MODE` picks the app tier: `worker` (default, Redis-pull — production) or `http`
(FastAPI gateway — QA). **Production commands live in
[`../docker/DEPLOY.md`](../docker/DEPLOY.md)**; both are reproduced there verbatim with the
env values actually in use.

For local QA, the HTTP mode needs only the engine and a port:
```bash
docker run -d --rm --name garment-prep-qa --gpus all -p 8080:8000 \
  -v $PWD/artifacts/segformer.plan:/models/segformer_parser/1/model.plan:ro \
  -e RUN_MODE=http \
  -e CALLBACK_URL=http://127.0.0.1:1/disabled \
  -e GATEWAY_WORKERS=4 -e QUEUE_CONSUMERS=8 -e CPU_THREADS=4 \
  -e MAX_BATCH_SIZE=32 -e OPT_BATCH_SIZE=16 \
  fashionx/garment-prep:<version>

curl -s localhost:8080/health     # {"status":"ok","queue_depth":0,"consumers":8}
```
Point `CALLBACK_URL` at a dead address so QA traffic through `/infer` can never deliver
into a real asset-service. `MAX_BATCH_SIZE` must be ≤ the `MAX_BATCH` the engine was built
with. Full knob list: `docker/DEPLOY.md` and `gateway/config.py`.

## 5. Test it

**Synchronous — returns the PNG. The smoke test to use:**
```bash
curl -s -o out.png -w '%{http_code}\n' \
  -F image=@sample.jpg -F type=upper localhost:8080/preprocess
python3 -c "from PIL import Image; im=Image.open('out.png'); print(im.size, im.mode)"
# expect: 200, then (928, 1664) RGBA
```

**Async — queued, delivered by callback:**
```bash
curl -X POST localhost:8080/infer -H 'Content-Type: application/json' -d '{
  "garment_id":"t1", "image_url":"https://…/thumb.jpg",
  "pipeline_type":"upper", "callback_url":"http://…/cb"}'
```

**Load test** (`gateway/bench.py`, bundled in the image):
```bash
docker run --rm --network host -v /path/to/images:/thumbs \
  --entrypoint python3 fashionx/garment-prep:<version> \
  -m gateway.bench --host http://localhost:8080 --dir /thumbs --type upper --concurrency 64 --n 300
```

Verifying a **worker-mode** container is different: it logs nothing while the Redis stream
is empty, which is idle, not stuck. See the verify section of
[`../docker/DEPLOY.md`](../docker/DEPLOY.md).

## 6. Local development (no GPU)

The pixel code is plain numpy/cv2/PIL and needs neither a GPU nor Triton:
```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r docker/requirements.txt
```
- `gateway/imaging.py` and `gateway/crop.py` are directly unit-testable and profilable
  (see the profiling approach in [`PIPELINE.md`](PIPELINE.md) §6).
- `gateway/app.py` and `worker/main.py` both need a reachable Triton (`TRITON_URL`). For
  logic work, point them at a Triton serving the engine on a GPU box, or run the whole
  image locally if you have an NVIDIA GPU.

## 7. Where things live

| Concern | File |
|---------|------|
| Deploy commands + full env contract | `docker/DEPLOY.md` |
| Gateway/HTTP + canvas + callback knobs | `gateway/config.py` |
| Redis/worker + backend endpoint knobs | `worker/config.py` |
| Pipeline internals, step by step | `docs/PIPELINE.md` |
| Redis Streams contract, XACK policy | `worker/redis_worker.py` |
| Engine build | `engine/export_onnx.py`, `engine/build_trt.sh` |
| Triton model config | `triton/models/segformer_parser/config.pbtxt.template` |
| S3 get/put helper | `engine/s3.py` |
