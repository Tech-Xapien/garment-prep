# Garment-Prep — Setup

How to go from a clone to a running service. For the production env contract and the
RunPod/EC2 runbooks, see [`../docker/DEPLOY.md`](../docker/DEPLOY.md). For how the
pipeline works internally, see [`PIPELINE.md`](PIPELINE.md).

The system is **one image** (gateway + Triton) plus **one external artifact** (the TRT
engine, kept in S3 — never baked into the image, because a `.plan` is tied to a specific
TensorRT version + GPU arch).

---

## 0. Prerequisites

| To… | You need |
|-----|----------|
| Build/run the **image** | Docker with a fast uplink (image ≈ 20 GB; NGC base has ~14 GB layers) |
| Run the **service** | NVIDIA driver ≥ 570 + `nvidia-container-toolkit`; a Blackwell (sm_120) GPU for the prod engine |
| Build the **TRT engine** | a **Blackwell** GPU (the engine only loads on the arch it was built for) |
| Export the **ONNX** | Python with `torch` + `transformers` (any machine; ONNX is portable) |

Artifacts already published (US S3, `us-east-1`):
```
s3://xapien-vton-engines/garment-prep/segformer_576x384.onnx              # portable ONNX
s3://xapien-vton-engines/garment-prep/segformer_fp16_576x384_sm120.plan   # Blackwell engine
```

---

## 1. Clone

```bash
git clone -b update/efficient https://github.com/Tech-Xapien/garment-prep.git
cd garment-prep
```

---

## 2. Build the image

Architecture-independent — build anywhere with Docker; it runs on any NVIDIA GPU.
```bash
docker build -f docker/Dockerfile -t fashionx/garment-prep:latest .
# push (do it from a fast/datacenter uplink — see docker/DEPLOY.md):
docker login -u fashionx && docker push fashionx/garment-prep:latest
```

## 3. Build the TRT engine (one-time, on Blackwell)

The engine is built **inside the image** so its TensorRT (10.11) matches the runtime.
Full command in [`../docker/DEPLOY.md`](../docker/DEPLOY.md) — summary:
```bash
docker run --rm --gpus all -v $PWD/artifacts:/artifacts \
  -e AWS_ACCESS_KEY_ID -e AWS_SECRET_ACCESS_KEY -e AWS_DEFAULT_REGION=us-east-1 \
  --entrypoint bash fashionx/garment-prep:latest -c '
    python engine/s3.py get s3://xapien-vton-engines/garment-prep/segformer_576x384.onnx /artifacts/segformer.onnx
    ONNX=/artifacts/segformer.onnx PLAN=/artifacts/segformer.plan MAX_BATCH=32 OPT_BATCH=16 bash engine/build_trt.sh
    python engine/s3.py put /artifacts/segformer.plan s3://xapien-vton-engines/garment-prep/segformer_fp16_576x384_sm120.plan'
```
To (re)generate the ONNX itself (rarely needed):
```bash
pip install torch transformers    # not in the runtime image
python engine/export_onnx.py --out artifacts/segformer.onnx
```

## 4. Run the service

Mount the engine (no runtime S3/keys needed) and launch:
```bash
docker run -d --restart unless-stopped --name garment-prep --gpus all \
  --cpuset-cpus=4-7 --cpus=4 -p 8080:8000 \
  -v $PWD/artifacts/segformer.plan:/models/segformer_parser/1/model.plan:ro \
  -e MAX_BATCH_SIZE=32 -e GATEWAY_WORKERS=4 -e QUEUE_CONSUMERS=8 -e CPU_THREADS=4 \
  -e CALLBACK_URL=<your-callback> \
  fashionx/garment-prep:latest

curl -s localhost:8080/health          # {"status":"ok",...}
```
(Or set `-e ENGINE_S3_URI=s3://…/segformer_fp16_576x384_sm120.plan` + AWS creds to fetch
at boot instead of mounting.) Full knob list: `docker/DEPLOY.md` and `gateway/config.py`.

---

## 5. Test it

**Synchronous (returns the PNG):**
```bash
curl -F image=@sample_thumb.jpg -F type=upper localhost:8080/preprocess -o out.png
```

**Async (queued → callback):**
```bash
curl -X POST localhost:8080/infer -H 'Content-Type: application/json' -d '{
  "garment_id":"t1", "image_url":"https://…/thumb.jpg",
  "pipeline_type":"upper", "callback_url":"http://…/cb"}'
```

**Load test / find the throughput ceiling** (`gateway/bench.py`, bundled in the image):
```bash
docker run --rm --network host -v /path/to/images:/thumbs \
  --entrypoint python3 fashionx/garment-prep:latest \
  -m gateway.bench --host http://localhost:8080 --dir /thumbs --type upper --concurrency 64 --n 300
```

---

## 6. Local development (no GPU)

You can iterate on the pure-CPU code without a GPU or Triton:
```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r docker/requirements.txt
```
- `gateway/imaging.py`, `gateway/crop.py` are plain numpy/cv2/PIL — unit-test and profile
  them directly (see the profiling approach in `PIPELINE.md` §5).
- `gateway/app.py` needs a reachable Triton (`TRITON_URL`) to serve `/infer`; for logic
  work, point it at a Triton running the engine on a GPU box, or run the whole image locally
  if you have an NVIDIA GPU.

`pipeline_type` values: `upper` (tops/coats/blazers), `lower` (skirts), `full` (dresses),
`layered` (kaftans) — see the category map in the dataset's `type.txt`.

---

## 7. Where things live

| Concern | File |
|---------|------|
| All tunable knobs | `gateway/config.py` (env-driven) |
| Prod env contract + RunPod/EC2 runbooks | `docker/DEPLOY.md` |
| Pipeline internals, step by step | `docs/PIPELINE.md` |
| Engine build | `engine/export_onnx.py`, `engine/build_trt.sh` |
| S3 get/put helper | `engine/s3.py` |
