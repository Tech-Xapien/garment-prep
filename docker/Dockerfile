# Garment-Prep — CPU gateway + Triton (TensorRT) in one image.
#
# Base = the Triton image from the SAME NGC release train as the vton foundation
# (pytorch:25.06 → TensorRT 10.11, CUDA 12.9, sm_120 Blackwell). This ships Triton,
# the TensorRT backend and trtexec, and shares the host driver floor with vton so
# both coexist on the prod box.
#
# The TRT engine is NOT baked in (it is TRT-version + GPU-arch specific): the
# entrypoint fetches segformer.plan from S3 (or a mount) into the model repo at
# boot. That is what makes the image portable across boxes and GPU generations.
#
# Build: docker build -f docker/Dockerfile -t fashionx/garment-prep:latest .
# syntax=docker/dockerfile:1.7
ARG NGC_VERSION=25.06
FROM nvcr.io/nvidia/tritonserver:${NGC_VERSION}-py3

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/usr/src/tensorrt/bin:${PATH}"
WORKDIR /workspace

RUN apt-get update -qq \
    && apt-get install -y -qq --no-install-recommends gettext-base \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf "$(command -v python3)" /usr/local/bin/python

# Pure-Python / ABI-flexible wheels only — no torch, keeps the image lean.
COPY docker/requirements.txt /tmp/requirements.txt
RUN PIP_CONSTRAINT= python3 -m pip install --no-cache-dir -r /tmp/requirements.txt

COPY gateway/ /workspace/gateway/
COPY worker/ /workspace/worker/
COPY engine/ /workspace/engine/
COPY triton/models/ /models/
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# 8000 gateway (Triton HTTP is disabled so the gateway owns it), 8002 Triton metrics
EXPOSE 8000 8002
ENTRYPOINT ["/entrypoint.sh"]
