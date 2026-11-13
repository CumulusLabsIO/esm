#!/usr/bin/env bash
set -Eeuo pipefail
cd /home/ubuntu/esm3-test
trap 'code=$?; echo "$code" > exit-code.txt; date -u +%FT%TZ > finished-at.txt' EXIT
date -u +%FT%TZ > started-at.txt
nvidia-smi -q > nvidia-smi-before.txt
python3 -m venv venv
venv/bin/python -m pip install --disable-pip-version-check uv
venv/bin/uv pip install --python venv/bin/python 'torch==2.11.0' --index-url https://download.pytorch.org/whl/cu130
venv/bin/uv pip install --python venv/bin/python 'esm==3.4.3'
venv/bin/uv pip freeze --python venv/bin/python > requirements-installed.txt
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
timeout --signal=TERM --kill-after=30s 3600s venv/bin/python esm3_blackwell_test.py
nvidia-smi -q > nvidia-smi-after.txt
