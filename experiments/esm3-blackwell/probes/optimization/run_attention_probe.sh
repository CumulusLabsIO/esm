#!/usr/bin/env bash
set -u
cd /home/ubuntu/esmopt
test "$(cat setup-exit.txt)" = 0 || exit 90
export PYTHONPATH=/home/ubuntu/esmopt/baseline HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=2
for batch in 1 16; do
  for backend in cudnn fa4; do
    name="${backend}-b${batch}"
    timeout 900 venv/bin/python attention_probe.py --backend "$backend" --batch "$batch" --warmup 20 --repeats 100 --output "attention-results-clean/$name.json" >"attention-results-clean/$name.log" 2>&1
    echo "$?" >"attention-results-clean/$name.exit"
  done
done
date -u +%FT%TZ > attention-results-clean/finished.txt
