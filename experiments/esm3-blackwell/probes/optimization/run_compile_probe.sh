#!/usr/bin/env bash
set -uo pipefail
cd /home/ubuntu/esmopt
until [[ -f setup-exit.txt ]]; do sleep 2; done
if [[ $(cat setup-exit.txt) != 0 ]]; then
  echo 'Setup failed; compilation not attempted.'
  exit 1
fi
export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH=/home/ubuntu/esmopt/baseline
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4
export TORCHINDUCTOR_CACHE_DIR=/home/ubuntu/esmopt/compile-cache-first
export TORCHINDUCTOR_COMPILE_THREADS=8
export TORCH_LOGS=graph_breaks,recompiles,perf_hints
timeout --signal=TERM 1800 /home/ubuntu/esmopt/venv/bin/python -u /home/ubuntu/esmopt/compile_probe.py --output /home/ubuntu/esmopt/compile-probe.json --repeats 60
result=$?
echo "$result" > /home/ubuntu/esmopt/compile-probe-exit.txt
exit "$result"
