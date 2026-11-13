#!/usr/bin/env bash
set -u
cd /home/ubuntu/esmopt
test "$(cat setup-exit.txt)" = 0 || exit 90
for attempt in $(seq 1 120); do
  test -f quant-results-clean/finished.txt && break
  sleep 5
done
test -f quant-results-clean/finished.txt || exit 91
mkdir -p quant-results-compiled
export PYTHONPATH=/home/ubuntu/esmopt/baseline HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=3
export TORCHINDUCTOR_CACHE_DIR=/home/ubuntu/esmopt/quant-inductor-cache
for batch in 1 16; do
  name="mxfp8-compiled-b${batch}"
  timeout 1200 venv/bin/python mxfp8_probe.py --batch "$batch" --allow-small-m --compile-ffn --warmup 20 --repeats 100 --output "quant-results-compiled/$name.json" >"quant-results-compiled/$name.log" 2>&1
  echo "$?" >"quant-results-compiled/$name.exit"
done
date -u +%FT%TZ > quant-results-compiled/finished.txt
