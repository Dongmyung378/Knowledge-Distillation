#!/usr/bin/env bash
set -euo pipefail

echo "=== [1/4] Checking NVIDIA GPU (expecting L4 24GB) ==="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
else
  echo "WARNING: nvidia-smi not found. Continuing with CPU-only environment."
fi

echo "=== [2/4] Installing PyTorch CUDA (if missing) and L4 KD dependencies ==="
python3 - <<'PY'
import importlib.util
print("torch installed:", importlib.util.find_spec("torch") is not None)
PY

pip install --upgrade pip
pip install -r requirements-l4.txt
pip install -e . --no-deps

echo "=== [3/4] Running seg_kd doctor ==="
python3 -m seg_kd doctor

echo "=== [4/4] Running unit tests ==="
python3 -m unittest discover -s tests -v
echo "L4 environment ready."
