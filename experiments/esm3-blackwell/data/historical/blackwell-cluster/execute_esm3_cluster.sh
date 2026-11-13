#!/usr/bin/env bash
set -uo pipefail
cd /home/ubuntu/esm3-cluster
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=4
unset CUDA_VISIBLE_DEVICES
date -u +%FT%TZ > workload-started-at.txt

timeout --signal=TERM --kill-after=30s 600s venv/bin/torchrun --standalone --nproc-per-node=8 \
    esm3_blackwell_cluster_test.py inference --output cluster-results/inference > inference.log 2>&1
echo "$?" > inference-exit-code.txt
timeout --signal=TERM --kill-after=30s 900s venv/bin/torchrun --standalone --nproc-per-node=8 \
    esm3_blackwell_cluster_test.py training --output cluster-results/training --steps 3 > training.log 2>&1
echo "$?" > training-exit-code.txt
timeout --signal=TERM --kill-after=30s 300s venv/bin/python esm3_blackwell_test.py > single-gpu-recheck.log 2>&1
echo "$?" > single-gpu-recheck-exit-code.txt
nvidia-smi -q > nvidia-smi-after.txt
date -u +%FT%TZ > workload-finished-at.txt
