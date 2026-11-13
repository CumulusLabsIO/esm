#!/usr/bin/env bash
set -Eeuo pipefail
cd /home/ubuntu/esm3-cluster
trap 'code=$?; echo "$code" > setup-exit-code.txt; date -u +%FT%TZ > setup-finished-at.txt' EXIT
date -u +%FT%TZ > setup-started-at.txt
nvidia-smi -q > nvidia-smi-before.txt
nvidia-smi topo -m > nvidia-topology.txt
python3 -m venv venv
venv/bin/python -m pip install --disable-pip-version-check uv
venv/bin/uv pip install --python venv/bin/python 'torch==2.15.0' --index-url https://download.pytorch.org/whl/cu132
venv/bin/uv pip install --python venv/bin/python 'esm==3.4.3'
venv/bin/uv pip freeze --python venv/bin/python > requirements-installed.txt
venv/bin/python - <<'PY'
import json
import torch
from huggingface_hub import snapshot_download

assert torch.__version__.startswith('2.15.0+cu132'), torch.__version__
assert torch.cuda.device_count() == 8
gpu_names = [torch.cuda.get_device_name(i) for i in range(8)]
assert all('B200' in name for name in gpu_names), gpu_names
print(json.dumps({'torch': torch.__version__, 'cuda_runtime': torch.version.cuda,
                  'nccl': torch.cuda.nccl.version(), 'gpus': gpu_names}), flush=True)
snapshot = snapshot_download('biohub/esm3-sm-open-v1', revision='47f0545b2b6daf26a93439a3cd610f4f7f3d5478')
with open('snapshot-path.txt', 'w') as file:
    file.write(snapshot + '\n')
PY
