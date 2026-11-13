#!/usr/bin/env python3
"""Check the exact environment and perform a real BF16 CUDA operation."""

import argparse
import hashlib
from importlib import metadata
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys

import torch
from esm.models.esm3 import ESM3

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--expected-torch", required=True)
parser.add_argument("--expected-cuda")
parser.add_argument("--require-strict-numerics", action="store_true")
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
assert torch.__version__.split("+")[0] == args.expected_torch, torch.__version__
if args.expected_cuda:
    assert torch.version.cuda == args.expected_cuda, torch.version.cuda
assert torch.cuda.is_available(), "CUDA runtime is unavailable"
assert torch.cuda.is_bf16_supported(), "BF16 is unavailable"
torch.cuda.set_device(0)
assert torch.cuda.get_device_capability(0)[0] >= 10, "Expected Blackwell GPU"
left = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)
right = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)
product = left @ right
torch.cuda.synchronize()
assert torch.isfinite(product).all().item()
if args.require_strict_numerics:
    from torch import _inductor

    assert "numerics" in _inductor.list_options(), (
        "Strict numerics compiler option is unavailable"
    )
source = Path(inspect.getfile(ESM3))
report = {
    "python": sys.version,
    "executable": sys.executable,
    "torch": torch.__version__,
    "cuda_runtime": torch.version.cuda,
    "cudnn": torch.backends.cudnn.version(),
    "gpu": torch.cuda.get_device_name(0),
    "capability": torch.cuda.get_device_capability(0),
    "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
    "esm3_source": str(source),
    "esm3_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    "bf16_matmul": "passed",
    "packages": {},
}
for name in (
    "numpy",
    "transformers",
    "tokenizers",
    "triton",
    "ty",
    "DockQ",
    "rdkit",
    "pytest",
    "pytest-xdist",
    "flash-attn",
):
    try:
        report["packages"][name] = metadata.version(name)
    except metadata.PackageNotFoundError:
        report["packages"][name] = None
report["nvidia_smi"] = subprocess.check_output(
    [
        "nvidia-smi",
        "--query-gpu=index,name,driver_version,memory.total",
        "--format=csv,noheader",
    ],
    text=True,
).strip()
args.output.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
