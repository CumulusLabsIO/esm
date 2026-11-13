#!/usr/bin/env bash
set -Eeuo pipefail
TASK_ROOT=/home/ubuntu/esm-ci
REPO="$TASK_ROOT/repo"
RESULTS="$TASK_ROOT/results/run-3"
mkdir -p "$RESULTS"
cd "$REPO"
trap 'rc=$?; echo "$rc" > "$RESULTS/gpu-tests.exit"; date -u +%FT%TZ > "$RESULTS/finished.txt"' EXIT
date -u +%FT%TZ > "$RESULTS/started.txt"
unset SKIP PYTEST_ADDOPTS ESM_API_KEY INFRA_PROVIDER
export HF_HOME="$TASK_ROOT/hf-cache" HF_HUB_OFFLINE=1 ESM_TESTS_STRICT_ARTIFACTS=1
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 PYTHONPATH="$REPO" PYTHONUNBUFFERED=1
export LD_PRELOAD="$REPO/.pixi/envs/dev/lib/libstdc++.so.6"
GPU_PYTHON="$TASK_ROOT/blackwell-venv/bin/python"
"$GPU_PYTHON" - <<'PY' > "$RESULTS/environment.json"
import json
import hashlib
import importlib.metadata
from pathlib import Path
import subprocess
import torch
import flash_attn
import flash_attn_2_cuda
assert torch.__version__ == '2.11.0+cu130', torch.__version__
assert Path(flash_attn.__file__).is_relative_to('/home/ubuntu/esm-ci/blackwell-venv')
assert torch.cuda.get_device_capability(0) == (10, 0)
print(json.dumps({
    'head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
    'torch': torch.__version__, 'cuda': torch.version.cuda,
    'gpu': torch.cuda.get_device_name(0),
    'flash_attn': importlib.metadata.version('flash-attn'),
    'flash_attn_file': flash_attn.__file__,
    'extension': flash_attn_2_cuda.__file__,
    'extension_sha256': hashlib.sha256(Path(flash_attn_2_cuda.__file__).read_bytes()).hexdigest(),
    'model_sha256': hashlib.sha256(Path('esm/models/esm3.py').read_bytes()).hexdigest(),
    'scope': 'All tests/models tests marked gpu and not manual; no fixture exclusion plugin; checkpoint cache required offline.',
}, indent=2))
PY
timeout --signal=TERM --kill-after=15s 900s "$GPU_PYTHON" -m pytest \
    -o addopts='' -n 0 -m 'gpu and not manual' -ra \
    --junitxml="$RESULTS/gpu-tests.xml" tests/models
git diff --stat > "$RESULTS/tracked-diff-after.txt"
test ! -s "$RESULTS/tracked-diff-after.txt"
