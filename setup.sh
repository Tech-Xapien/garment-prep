#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="$SCRIPT_DIR/models"
VENV_DIR="$SCRIPT_DIR/.venv"
PYTHON_BASE="python3.10"

S3_BUCKET="xapienappassets"
S3_PREFIX="garment-prep/models"
AWS_REGION="ap-south-1"

echo "=== Garment-Prep Service Setup ==="

# ── 0. Verify AWS credentials are exported ───────────────────────────
if [[ -z "${AWS_ACCESS_KEY_ID:-}" || -z "${AWS_SECRET_ACCESS_KEY:-}" ]]; then
    echo "ERROR: Export AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY before running."
    echo "  export AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=..."
    exit 1
fi

# ── 1. Verify Python 3.10 ───────────────────────────────────────────
if ! command -v "$PYTHON_BASE" &>/dev/null; then
    echo "ERROR: $PYTHON_BASE not found. Install Python 3.10 first."
    exit 1
fi
echo "[1/5] Python: $($PYTHON_BASE --version)"

# ── 2. Create virtual environment ───────────────────────────────────
if [ -d "$VENV_DIR" ]; then
    echo "[2/5] Virtual environment already exists."
else
    echo "[2/5] Creating virtual environment..."
    $PYTHON_BASE -m venv "$VENV_DIR"
fi
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"
PYTHON="$VENV_DIR/bin/python"

# ── 3. Install Python dependencies ──────────────────────────────────
echo "[3/5] Installing Python dependencies..."
$PYTHON -m pip install --quiet --upgrade pip
$PYTHON -m pip install --quiet --upgrade \
    fastapi \
    "uvicorn[standard]" \
    python-multipart \
    mediapipe \
    fashn-human-parser \
    pillow \
    numpy \
    httpx \
    ultralytics \
    awscli

AWS_BIN="$VENV_DIR/bin/aws"
if ! "$AWS_BIN" --version &>/dev/null; then
    echo "ERROR: awscli failed to install into venv at $AWS_BIN"
    exit 1
fi

# ── 4. Download models from S3 ──────────────────────────────────────
mkdir -p "$MODEL_DIR"
echo "[4/5] Downloading models from s3://$S3_BUCKET/$S3_PREFIX/ ..."
AWS_DEFAULT_REGION="$AWS_REGION" "$AWS_BIN" s3 sync \
    "s3://$S3_BUCKET/$S3_PREFIX/" "$MODEL_DIR/" \
    --no-progress
echo "  Models downloaded to $MODEL_DIR"

# ── 5. Pre-cache FASHN SegFormer weights ─────────────────────────────
echo "[5/5] Pre-caching FASHN Human Parser model weights..."
$PYTHON -c "
from fashn_human_parser import FashnHumanParser
FashnHumanParser()
print('  FASHN model cached.')
"

echo ""
echo "=== Setup complete ==="
echo ""

# ── Optional: start server ──────────────────────────────────────────
START=false
TEST=false
for arg in "$@"; do
    case "$arg" in
        --start) START=true ;;
        --test)  TEST=true ;;
    esac
done

if $START || $TEST; then
    HOST="${HOST:-0.0.0.0}"
    PORT="${PORT:-8000}"

    if $TEST; then
        export TEST_MODE=1
        echo "Starting Garment-Prep Service on $HOST:$PORT (TEST_MODE — /preprocess enabled) ..."
    else
        echo "Starting Garment-Prep Service on $HOST:$PORT ..."
    fi

    exec "$PYTHON" -m uvicorn main:app \
        --host "$HOST" \
        --port "$PORT" \
        --app-dir "$SCRIPT_DIR"
else
    echo "To start the server:"
    echo "  source $VENV_DIR/bin/activate && cd $SCRIPT_DIR && python -m uvicorn main:app --host 0.0.0.0 --port 8000"
    echo ""
    echo "Or re-run with:"
    echo "  ./setup.sh --start        # /infer only (production)"
    echo "  ./setup.sh --test         # /infer + /preprocess (testing)"
fi
