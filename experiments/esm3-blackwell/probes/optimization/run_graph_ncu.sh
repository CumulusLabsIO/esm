#!/usr/bin/env bash
set -Eeuo pipefail
cd /home/ubuntu/esmopt
trap 'echo "$?" > graph-ncu-exit.txt' EXIT
ncu_path=$(cat ncu-path.txt)
sudo -E env CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 HF_HOME=/home/ubuntu/.cache/huggingface PYTHONPATH=/home/ubuntu/esmopt/baseline timeout --signal=TERM --kill-after=15s 600s "$ncu_path" --target-processes all --profile-from-start off --graph-profiling node --set basic --kernel-name 'regex:nvjet_sm100_tst_128x40' --launch-count 1 --force-overwrite --export /home/ubuntu/esmopt/results/graph-gemm /home/ubuntu/esmopt/venv/bin/python /home/ubuntu/esmopt/graph_profile.py --output /home/ubuntu/esmopt/results/graph-ncu.json --warmup 5 --repeats 10
sudo "$ncu_path" --import results/graph-gemm.ncu-rep --csv > results/graph-gemm.csv
sudo chown ubuntu:ubuntu results/graph-gemm.ncu-rep results/graph-ncu.json
