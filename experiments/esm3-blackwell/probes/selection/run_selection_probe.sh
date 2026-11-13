#!/usr/bin/env bash
set -u
cd /home/ubuntu/esmopt/repo || exit 1
export CUDA_VISIBLE_DEVICES=4 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 PYTHONPATH=/home/ubuntu/esmopt/repo
mkdir -p /home/ubuntu/esmopt/selection-results
/home/ubuntu/esmopt/venv/bin/python -m pytest -o addopts='' tests/models/esm3_cuda_graph_test.py tests/models/esm3_logits_selection_test.py -q --tb=short > /home/ubuntu/esmopt/selection-results/tests.log 2>&1
tests_status=$?
echo "$tests_status" > /home/ubuntu/esmopt/selection-results/tests-exit.txt
if [ "$tests_status" -ne 0 ]; then exit "$tests_status"; fi
for shape in 1:76 8:76 1:512; do
    batch=${shape%:*}
    length=${shape#*:}
    /home/ubuntu/esmopt/venv/bin/python /home/ubuntu/esmopt/selection_probe.py --baseline-source /home/ubuntu/esmopt/baseline/esm/models/esm3.py --output "/home/ubuntu/esmopt/selection-results/b${batch}-l${length}.json" --batch "$batch" --length "$length" --warmup 5 --repeats 30 > "/home/ubuntu/esmopt/selection-results/b${batch}-l${length}.log" 2>&1
    echo "$?" > "/home/ubuntu/esmopt/selection-results/b${batch}-l${length}-exit.txt"
done
date -u +%FT%TZ > /home/ubuntu/esmopt/selection-results/finished.txt
