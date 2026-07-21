#!/usr/bin/env bash
# Build the TensorRT engine from the exported ONNX.
#
# Runs INSIDE the runtime container (trtexec ships in the Triton NGC image) on the
# RunPod RTX PRO 6000 box, so the .plan matches the exact TRT version + Blackwell
# arch used in prod. The resulting .plan is uploaded to S3; it is NOT baked into
# the image and is NOT portable across TRT versions or GPU architectures.
#
# Knobs (env): MAX_BATCH throughput/VRAM ceiling, OPT_BATCH the shape TRT tunes for.
set -euo pipefail

ONNX="${ONNX:-engine/artifacts/segformer.onnx}"
PLAN="${PLAN:-engine/artifacts/segformer.plan}"
MAX_BATCH="${MAX_BATCH:-16}"
OPT_BATCH="${OPT_BATCH:-8}"
WORKSPACE_MB="${WORKSPACE_MB:-4096}"

H=576; W=384

trtexec \
  --onnx="${ONNX}" \
  --saveEngine="${PLAN}" \
  --fp16 \
  --minShapes="pixel_values:1x3x${H}x${W}" \
  --optShapes="pixel_values:${OPT_BATCH}x3x${H}x${W}" \
  --maxShapes="pixel_values:${MAX_BATCH}x3x${H}x${W}" \
  --memPoolSize="workspace:${WORKSPACE_MB}" \
  --builderOptimizationLevel=5 \
  --skipInference

echo "Built engine -> ${PLAN}  (fp16, batch 1..${MAX_BATCH}, opt=${OPT_BATCH})"
