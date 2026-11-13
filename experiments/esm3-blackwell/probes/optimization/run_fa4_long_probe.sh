#!/usr/bin/env bash
set -u
cd /home/ubuntu/esmopt
test "$(cat setup-exit.txt)" = 0 || exit 90
for attempt in $(seq 1 120); do
  test -f attention-results-clean/finished.txt && break
  sleep 5
done
test -f attention-results-clean/finished.txt || exit 91
mkdir -p attention-results-long
export PYTHONPATH=/home/ubuntu/esmopt/baseline HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=2
timeout 900 venv/bin/python attention_probe.py --backend fa4 --length 1024 --warmup 20 --repeats 100 --output attention-results-long/fa4-l1024-b1.json >attention-results-long/fa4-l1024-b1.log 2>&1
echo "$?" >attention-results-long/fa4-l1024-b1.exit
date -u +%FT%TZ > attention-results-long/finished.txt
