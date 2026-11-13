#!/usr/bin/env bash
set -Eeuo pipefail
cd /home/ubuntu/esmopt
trap 'echo "$?" > graph-profile-exit.txt' EXIT
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTHONPATH=/home/ubuntu/esmopt/baseline
mkdir -p results
nsys_path=$(cat nsys-path.txt)
timeout --signal=TERM --kill-after=15s 600s "$nsys_path" profile --force-overwrite=true --trace=cuda,nvtx --sample=none --cpuctxsw=none --cuda-graph-trace=node --capture-range=cudaProfilerApi --capture-range-end=stop --output=results/graph-baseline venv/bin/python graph_profile.py --output /home/ubuntu/esmopt/results/graph-baseline.json --warmup 20 --repeats 100
"$nsys_path" export --type=sqlite --force-overwrite=true --output=results/graph-baseline.sqlite results/graph-baseline.nsys-rep
"$nsys_path" stats --report=cuda_gpu_kern_sum --format=csv results/graph-baseline.nsys-rep > results/graph-baseline-kernels.csv
