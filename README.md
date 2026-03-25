# Garment-Prep Service

Image preprocessing module for virtual try-on pipelines. Accepts garment images, isolates the target garment region (upper/lower/full), and delivers the processed result via async callback.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/infer` | Async — queues job, returns immediately, delivers result via callback |
| POST | `/preprocess` | Sync — returns processed PNG directly (legacy) |
| GET | `/health` | Liveness check + queue depth |

## Quick Start

```bash
# Setup
./setup.sh

# Run
CALLBACK_AUTH_TOKEN=supersecret-internal-token \
  uvicorn main:app --host 0.0.0.0 --port 8000
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CALLBACK_URL` | `https://apiprod.xapien.in:9002/v1/garment/upload` | Result delivery URL |
| `CALLBACK_AUTH_TOKEN` | `supersecret-internal-token` | X-Internal-Auth header value |
| `INFERENCE_WORKERS` | `2` | Process pool size (each loads own models) |
| `QUEUE_WORKERS` | `4` | Async queue consumers |
| `QUEUE_SIZE` | `100` | Max pending jobs before 503 |

## Architecture

```
/infer request
    -> asyncio.Queue (bounded, backpressure)
        -> N async consumers
            -> ProcessPoolExecutor (hot models per process)
                -> pipeline (face detect -> segment -> crop -> canvas)
                    -> callback POST with retry (5x exponential backoff)
```

### Triton Migration

Inference is abstracted behind `inference/backend.py` Protocol. Current implementation is `inference/local.py`. To migrate to Triton, implement `inference/triton.py` matching the same protocol and swap the import in `queue_worker.py`.
