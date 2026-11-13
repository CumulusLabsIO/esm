#!/usr/bin/env bash
set -u
cd /home/ubuntu/esmopt
export PYTHONPATH=/home/ubuntu/esmopt/baseline HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=3
export TORCHINDUCTOR_CACHE_DIR=/home/ubuntu/esmopt/swiglu-strict-cache
nsys=$(cat nsys-path.txt)
timeout 300 "$nsys" profile --trace=cuda,nvtx --sample=none --cpuctxsw=none --capture-range=cudaProfilerApi --capture-range-end=stop --cuda-graph-trace=node --force-overwrite=true --output=ffn-accuracy-results/strict-swiglu venv/bin/python swiglu_nsys_probe.py --warmup 5 --output /home/ubuntu/esmopt/ffn-accuracy-results/strict-swiglu-nsys.json
status=$?
echo "$status" > ffn-accuracy-results/strict-swiglu-nsys.exit
test "$status" = 0 || exit "$status"
"$nsys" export --type=sqlite --force-overwrite=true --output=ffn-accuracy-results/strict-swiglu.sqlite ffn-accuracy-results/strict-swiglu.nsys-rep
