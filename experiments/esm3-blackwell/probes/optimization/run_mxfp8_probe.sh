#!/usr/bin/env bash
set -u
cd /home/ubuntu/esmopt
test "$(cat setup-exit.txt)" = 0 || exit 90
export PYTHONPATH=/home/ubuntu/esmopt/baseline HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=3
for batch in 1 16; do
  name="mxfp8-b${batch}"
  timeout 1200 venv/bin/python mxfp8_probe.py --batch "$batch" --allow-small-m --warmup 20 --repeats 100 --output "quant-results-clean/$name.json" >"quant-results-clean/$name.log" 2>&1
  echo "$?" >"quant-results-clean/$name.exit"
done
date -u +%FT%TZ > quant-results-clean/finished.txt
