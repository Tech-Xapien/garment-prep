#!/usr/bin/env bash
# Boot: render Triton config from env knobs → fetch engine → launch Triton + gateway.
set -euo pipefail

MODEL_DIR=/models/segformer_parser
PLAN_DST="${MODEL_DIR}/1/model.plan"

# 1) Render Triton config (VRAM/throughput knobs) — no image rebuild to retune.
export MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-16}"
export OPT_BATCH_SIZE="${OPT_BATCH_SIZE:-8}"
export INSTANCES="${INSTANCES:-1}"
export MAX_QUEUE_DELAY_US="${MAX_QUEUE_DELAY_US:-2000}"
envsubst '${MAX_BATCH_SIZE} ${OPT_BATCH_SIZE} ${INSTANCES} ${MAX_QUEUE_DELAY_US}' \
    < "${MODEL_DIR}/config.pbtxt.template" > "${MODEL_DIR}/config.pbtxt"
echo "[entrypoint] triton config: batch<=${MAX_BATCH_SIZE} instances=${INSTANCES} delay=${MAX_QUEUE_DELAY_US}us"

# 2) Provide the TRT engine (mount wins; else S3; else must be pre-baked).
mkdir -p "${MODEL_DIR}/1"
if [ ! -f "${PLAN_DST}" ]; then
  if [ -n "${ENGINE_LOCAL_PATH:-}" ]; then
    cp "${ENGINE_LOCAL_PATH}" "${PLAN_DST}"
    echo "[entrypoint] engine <- ${ENGINE_LOCAL_PATH}"
  elif [ -n "${ENGINE_S3_URI:-}" ]; then
    PLAN_DST="${PLAN_DST}" python3 - <<'PY'
import os, urllib.parse, boto3
u = urllib.parse.urlparse(os.environ["ENGINE_S3_URI"])
boto3.client("s3").download_file(u.netloc, u.path.lstrip("/"), os.environ["PLAN_DST"])
print("[entrypoint] engine <-", os.environ["ENGINE_S3_URI"])
PY
  else
    echo "[entrypoint] ERROR: no engine. Set ENGINE_S3_URI or ENGINE_LOCAL_PATH." >&2
    exit 1
  fi
fi

# 3) Triton — gRPC only (HTTP off so the gateway owns :8000), metrics on :8002.
tritonserver \
  --model-repository=/models \
  --allow-http=false \
  --grpc-port=8001 \
  --allow-metrics=true --metrics-port=8002 \
  --log-verbose=0 &
TRITON_PID=$!

# 4) Gateway — CPU scale via GATEWAY_WORKERS (uvicorn procs).
uvicorn gateway.app:app --host 0.0.0.0 --port 8000 \
  --workers "${GATEWAY_WORKERS:-1}" --no-access-log &
UVICORN_PID=$!

trap 'kill ${TRITON_PID} ${UVICORN_PID} 2>/dev/null || true' TERM INT
wait -n ${TRITON_PID} ${UVICORN_PID}
echo "[entrypoint] a process exited — shutting down."
kill ${TRITON_PID} ${UVICORN_PID} 2>/dev/null || true
