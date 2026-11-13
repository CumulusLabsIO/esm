# Evidence index

All files below are listed in [MANIFEST.json](MANIFEST.json) with original and published SHA-256, sizes, and any text sanitization. Run `python3 verify.py` to verify the bundle. Source labels locate files within the archived experiment families; they are not public download URLs.

| File | Purpose | Bytes | Transformation |
|---|---|---:|---|
| [data/attention/cudnn-b1.json](data/attention/cudnn-b1.json) | O6/O7 experiment; superseded directory is excluded from headline results | 229683 | None |
| [data/attention/cudnn-b16.json](data/attention/cudnn-b16.json) | O6/O7 experiment; superseded directory is excluded from headline results | 232525 | None |
| [data/attention/fa4-b1.json](data/attention/fa4-b1.json) | O6/O7 experiment; superseded directory is excluded from headline results | 233411 | None |
| [data/attention/fa4-b16.json](data/attention/fa4-b16.json) | O6/O7 experiment; superseded directory is excluded from headline results | 236270 | None |
| [data/attention/fa4-l1024-b1.json](data/attention/fa4-l1024-b1.json) | O6/O7 experiment; superseded directory is excluded from headline results | 234440 | None |
| [data/capture/patched.json](data/capture/patched.json) | O1 capture and changing-input experiment | 63337 | None |
| [data/capture/stock-after.json](data/capture/stock-after.json) | O1 capture and changing-input experiment | 25441 | None |
| [data/capture/stock-before.json](data/capture/stock-before.json) | O1 capture and changing-input experiment | 22866 | None |
| [data/capture/stock-capture.json](data/capture/stock-capture.json) | O1 capture and changing-input experiment | 1400 | None |
| [data/compiler/compile-fullgraph-failed.json](data/compiler/compile-fullgraph-failed.json) | O4-O5 compiler experiments including failures | 41770 | None |
| [data/compiler/compile-profile.json](data/compiler/compile-profile.json) | O4-O5 compiler experiments including failures | 72088 | None |
| [data/compiler/compile-transformer-strict-aten.json](data/compiler/compile-transformer-strict-aten.json) | O4-O5 compiler experiments including failures | 85007 | None |
| [data/compiler/compile-transformer-strict.json](data/compiler/compile-transformer-strict.json) | O4-O5 compiler experiments including failures | 85049 | None |
| [data/compiler/compile-transformer.json](data/compiler/compile-transformer.json) | O4-O5 compiler experiments including failures | 85060 | None |
| [data/compiler/strict-ffn-b1.json](data/compiler/strict-ffn-b1.json) | O4-O5 compiler experiments including failures | 71630 | None |
| [data/compiler/strict-swiglu-b1.json](data/compiler/strict-swiglu-b1.json) | O4-O5 compiler experiments including failures | 62902 | None |
| [data/historical/blackwell-cluster/cluster-results/inference/rank-0.json](data/historical/blackwell-cluster/cluster-results/inference/rank-0.json) | H1-H5 recovered historical artifact | 870 | removed private host label; removed bare GPU UUID field |
| [data/historical/blackwell-cluster/cluster-results/training/summary.json](data/historical/blackwell-cluster/cluster-results/training/summary.json) | H1-H5 recovered historical artifact | 18772 | removed private host label; removed bare GPU UUID field |
| [data/historical/blackwell-cluster/execute_esm3_cluster.sh](data/historical/blackwell-cluster/execute_esm3_cluster.sh) | H1-H5 recovered historical artifact | 921 | None |
| [data/historical/blackwell-cluster/run_esm3_cluster.sh](data/historical/blackwell-cluster/run_esm3_cluster.sh) | H1-H5 recovered historical artifact | 1305 | None |
| [data/historical/blackwell-cluster/summary.json](data/historical/blackwell-cluster/summary.json) | H1-H5 recovered historical artifact | 1209 | removed infrastructure identifier |
| [data/historical/blackwell-profile/perf-results/logits-cudnn.json](data/historical/blackwell-profile/perf-results/logits-cudnn.json) | H1-H5 recovered historical artifact | 3655 | None |
| [data/historical/blackwell-profile/perf-results/logits-eight-batched.json](data/historical/blackwell-profile/perf-results/logits-eight-batched.json) | H1-H5 recovered historical artifact | 2407 | None |
| [data/historical/blackwell-profile/perf-results/logits-eight-serial-after.json](data/historical/blackwell-profile/perf-results/logits-eight-serial-after.json) | H1-H5 recovered historical artifact | 2395 | None |
| [data/historical/blackwell-profile/perf-results/logits-eight-serial-before.json](data/historical/blackwell-profile/perf-results/logits-eight-serial-before.json) | H1-H5 recovered historical artifact | 2123 | None |
| [data/historical/blackwell-profile/perf-results/logits-followup-eager-repeat.json](data/historical/blackwell-profile/perf-results/logits-followup-eager-repeat.json) | H1-H5 recovered historical artifact | 3669 | None |
| [data/historical/blackwell-profile/perf-results/logits-followup-raw-graph.log](data/historical/blackwell-profile/perf-results/logits-followup-raw-graph.log) | H1-H5 recovered historical artifact | 1404 | None |
| [data/historical/blackwell-profile/perf-results/structure-eight-batch.json](data/historical/blackwell-profile/perf-results/structure-eight-batch.json) | H1-H5 recovered historical artifact | 1708 | None |
| [data/historical/blackwell-profile/perf-results/structure-eight-serial-after.json](data/historical/blackwell-profile/perf-results/structure-eight-serial-after.json) | H1-H5 recovered historical artifact | 1690 | None |
| [data/historical/blackwell-profile/perf-results/structure-eight-serial-before.json](data/historical/blackwell-profile/perf-results/structure-eight-serial-before.json) | H1-H5 recovered historical artifact | 1370 | None |
| [data/historical/blackwell-profile/traces/logits-default.log](data/historical/blackwell-profile/traces/logits-default.log) | H1-H5 recovered historical artifact | 10062 | None |
| [data/historical/blackwell-profile/traces/logits-raw-graph.log](data/historical/blackwell-profile/traces/logits-raw-graph.log) | H1-H5 recovered historical artifact | 9895 | None |
| [data/historical/blackwell-profile/traces/structure-default.log](data/historical/blackwell-profile/traces/structure-default.log) | H1-H5 recovered historical artifact | 11759 | None |
| [data/historical/blackwell-profile/train-profile/adamw-after/summary.json](data/historical/blackwell-profile/train-profile/adamw-after/summary.json) | H1-H5 recovered historical artifact | 15531 | removed private host label; removed bare GPU UUID field |
| [data/historical/blackwell-profile/train-profile/adamw-before/summary.json](data/historical/blackwell-profile/train-profile/adamw-before/summary.json) | H1-H5 recovered historical artifact | 15510 | removed private host label; removed bare GPU UUID field |
| [data/historical/blackwell-profile/train-profile/adamw-fused/summary.json](data/historical/blackwell-profile/train-profile/adamw-fused/summary.json) | H1-H5 recovered historical artifact | 15495 | removed private host label; removed bare GPU UUID field |
| [data/historical/blackwell-profile/train-profile/trace/summary.json](data/historical/blackwell-profile/train-profile/trace/summary.json) | H1-H5 recovered historical artifact | 14238 | removed private host label; removed bare GPU UUID field |
| [data/historical/initial/esm3_blackwell_test.py](data/historical/initial/esm3_blackwell_test.py) | H1 initial released model smoke test | 6248 | None |
| [data/historical/initial/exit-code.txt](data/historical/initial/exit-code.txt) | H1 initial released model smoke test | 2 | None |
| [data/historical/initial/finished-at.txt](data/historical/initial/finished-at.txt) | H1 initial released model smoke test | 21 | None |
| [data/historical/initial/requirements-installed.txt](data/historical/initial/requirements-installed.txt) | H1 initial released model smoke test | 2240 | None |
| [data/historical/initial/results/generated-sequence.txt](data/historical/initial/results/generated-sequence.txt) | H1 initial released model smoke test | 77 | None |
| [data/historical/initial/results/report.json](data/historical/initial/results/report.json) | H1 initial released model smoke test | 7398 | None |
| [data/historical/initial/results/ubiquitin-generated.pdb](data/historical/initial/results/ubiquitin-generated.pdb) | H1 initial released model smoke test | 24542 | None |
| [data/historical/initial/run.log](data/historical/initial/run.log) | H1 initial released model smoke test | 14870 | removed GPU UUID |
| [data/historical/initial/run_esm3_test.sh](data/historical/initial/run_esm3_test.sh) | H1 initial released model smoke test | 736 | None |
| [data/historical/initial/started-at.txt](data/historical/initial/started-at.txt) | H1 initial released model smoke test | 21 | None |
| [data/profiling/compile-profile-compiled-kernels.csv](data/profiling/compile-profile-compiled-kernels.csv) | O3/O4/O5 profiler export; not unprofiled wall time | 19462 | None |
| [data/profiling/compile-profile-compiled-summary.json](data/profiling/compile-profile-compiled-summary.json) | O3/O4/O5 profiler export; not unprofiled wall time | 1505 | None |
| [data/profiling/compile-profile-eager-kernels.csv](data/profiling/compile-profile-eager-kernels.csv) | O3/O4/O5 profiler export; not unprofiled wall time | 22871 | None |
| [data/profiling/compile-profile-eager-summary.json](data/profiling/compile-profile-eager-summary.json) | O3/O4/O5 profiler export; not unprofiled wall time | 1285 | None |
| [data/profiling/graph-baseline-kernel-summary.json](data/profiling/graph-baseline-kernel-summary.json) | O3/O4/O5 profiler export; not unprofiled wall time | 1301 | None |
| [data/profiling/graph-baseline-kernels.csv](data/profiling/graph-baseline-kernels.csv) | O3/O4/O5 profiler export; not unprofiled wall time | 22964 | None |
| [data/profiling/graph-baseline.json](data/profiling/graph-baseline.json) | O3/O4/O5 profiler export; not unprofiled wall time | 10848 | None |
| [data/profiling/graph-gemm.csv](data/profiling/graph-gemm.csv) | O3/O4/O5 profiler export; not unprofiled wall time | 13196 | None |
| [data/profiling/strict-swiglu-nsys-summary.json](data/profiling/strict-swiglu-nsys-summary.json) | O3/O4/O5 profiler export; not unprofiled wall time | 54352 | None |
| [data/quantization/mxfp8-b1.json](data/quantization/mxfp8-b1.json) | O6/O7 experiment; superseded directory is excluded from headline results | 267842 | None |
| [data/quantization/mxfp8-b16.json](data/quantization/mxfp8-b16.json) | O6/O7 experiment; superseded directory is excluded from headline results | 269075 | None |
| [data/quantization/mxfp8-compiled-b1.json](data/quantization/mxfp8-compiled-b1.json) | O6/O7 experiment; superseded directory is excluded from headline results | 254485 | None |
| [data/quantization/mxfp8-compiled-b16.json](data/quantization/mxfp8-compiled-b16.json) | O6/O7 experiment; superseded directory is excluded from headline results | 256101 | None |
| [data/recipe-validation/compile-transformer.json](data/recipe-validation/compile-transformer.json) | Public recipe result including numerical rejections | 31777 | None |
| [data/recipe-validation/cudnn.json](data/recipe-validation/cudnn.json) | Public recipe result including numerical rejections | 30447 | None |
| [data/recipe-validation/default.json](data/recipe-validation/default.json) | Public recipe result including numerical rejections | 30438 | None |
| [data/recipe-validation/embeddings.json](data/recipe-validation/embeddings.json) | Public recipe result including numerical rejections | 11710 | None |
| [data/recipe-validation/fa4.json](data/recipe-validation/fa4.json) | Public recipe result including numerical rejections | 31754 | None |
| [data/recipe-validation/mxfp8-b1.json](data/recipe-validation/mxfp8-b1.json) | Public recipe result including numerical rejections | 31931 | None |
| [data/recipe-validation/mxfp8-compiled-b16-cyclic.json](data/recipe-validation/mxfp8-compiled-b16-cyclic.json) | Public recipe result including numerical rejections | 32322 | None |
| [data/recipe-validation/mxfp8-compiled-b16.json](data/recipe-validation/mxfp8-compiled-b16.json) | Public recipe result including numerical rejections | 32234 | None |
| [data/recipe-validation/sequence.json](data/recipe-validation/sequence.json) | Public recipe result including numerical rejections | 11683 | None |
| [data/selection/b1-l512.json](data/selection/b1-l512.json) | O2 selective output conversion | 573095 | None |
| [data/selection/b1-l76.json](data/selection/b1-l76.json) | O2 selective output conversion | 572227 | None |
| [data/selection/b8-l76.json](data/selection/b8-l76.json) | O2 selective output conversion | 577254 | None |
| [data/selection/memory-b1-l76.json](data/selection/memory-b1-l76.json) | O2 selective output conversion | 2783 | None |
| [data/superseded/attention/cudnn-b1.json](data/superseded/attention/cudnn-b1.json) | O6/O7 experiment; superseded directory is excluded from headline results | 208834 | None |
| [data/superseded/attention/cudnn-b16.json](data/superseded/attention/cudnn-b16.json) | O6/O7 experiment; superseded directory is excluded from headline results | 211209 | None |
| [data/superseded/attention/fa4-b1.json](data/superseded/attention/fa4-b1.json) | O6/O7 experiment; superseded directory is excluded from headline results | 212519 | None |
| [data/superseded/attention/fa4-b16.json](data/superseded/attention/fa4-b16.json) | O6/O7 experiment; superseded directory is excluded from headline results | 215012 | None |
| [data/superseded/quantization/mxfp8-b1.json](data/superseded/quantization/mxfp8-b1.json) | O6/O7 experiment; superseded directory is excluded from headline results | 246922 | None |
| [data/superseded/quantization/mxfp8-b16.json](data/superseded/quantization/mxfp8-b16.json) | O6/O7 experiment; superseded directory is excluded from headline results | 248037 | None |
| [data/swiglu-original/swiglu-b1.json](data/swiglu-original/swiglu-b1.json) | Public recipe result including numerical rejections | 30478 | None |
| [data/swiglu-original/swiglu-b8.json](data/swiglu-original/swiglu-b8.json) | Public recipe result including numerical rejections | 30469 | None |
| [data/swiglu-original/swiglu-l512.json](data/swiglu-original/swiglu-l512.json) | Public recipe result including numerical rejections | 30798 | None |
| [data/swiglu-rerun/default-b1.json](data/swiglu-rerun/default-b1.json) | Fresh-host B200 public-recipe rerun | 30387 | None |
| [data/swiglu-rerun/embeddings-b1.json](data/swiglu-rerun/embeddings-b1.json) | Fresh-host B200 public-recipe rerun | 11663 | None |
| [data/swiglu-rerun/sequence-b1.json](data/swiglu-rerun/sequence-b1.json) | Fresh-host B200 public-recipe rerun | 11636 | None |
| [data/swiglu-rerun/swiglu-b1.json](data/swiglu-rerun/swiglu-b1.json) | Fresh-host B200 public-recipe rerun | 30394 | None |
| [data/swiglu-rerun/swiglu-b8.json](data/swiglu-rerun/swiglu-b8.json) | Fresh-host B200 public-recipe rerun | 30404 | None |
| [data/swiglu-rerun/swiglu-l512.json](data/swiglu-rerun/swiglu-l512.json) | Fresh-host B200 public-recipe rerun | 30705 | None |
| [data/validation/fa2/build-command.txt](data/validation/fa2/build-command.txt) | FA2 source/build/native SM100 evidence | 207 | None |
| [data/validation/fa2/build.exit](data/validation/fa2/build.exit) | FA2 source/build/native SM100 evidence | 2 | None |
| [data/validation/fa2/build.log](data/validation/fa2/build.log) | FA2 source/build/native SM100 evidence | 415865 | None |
| [data/validation/fa2/compiled-architectures.txt](data/validation/fa2/compiled-architectures.txt) | FA2 source/build/native SM100 evidence | 5607 | None |
| [data/validation/fa2/finished.txt](data/validation/fa2/finished.txt) | FA2 source/build/native SM100 evidence | 21 | None |
| [data/validation/fa2/locked-extension-before.sha256](data/validation/fa2/locked-extension-before.sha256) | FA2 source/build/native SM100 evidence | 185 | None |
| [data/validation/fa2/sdist.sha256](data/validation/fa2/sdist.sha256) | FA2 source/build/native SM100 evidence | 96 | None |
| [data/validation/fa2/sm100-probe-import-order.log](data/validation/fa2/sm100-probe-import-order.log) | FA2 source/build/native SM100 evidence | 223 | None |
| [data/validation/fa2/sm100-probe.exit](data/validation/fa2/sm100-probe.exit) | FA2 source/build/native SM100 evidence | 2 | None |
| [data/validation/fa2/sm100-probe.json](data/validation/fa2/sm100-probe.json) | FA2 source/build/native SM100 evidence | 509 | None |
| [data/validation/fa2/sm100-probe.log](data/validation/fa2/sm100-probe.log) | FA2 source/build/native SM100 evidence | 509 | None |
| [data/validation/fa2/started.txt](data/validation/fa2/started.txt) | FA2 source/build/native SM100 evidence | 21 | None |
| [data/validation/fa2/toolchain.txt](data/validation/fa2/toolchain.txt) | FA2 source/build/native SM100 evidence | 1091 | None |
| [data/validation/fa2/wheel.sha256](data/validation/fa2/wheel.sha256) | FA2 source/build/native SM100 evidence | 125 | None |
| [data/validation/hosted-ci-summary.json](data/validation/hosted-ci-summary.json) | Hosted CI result snapshot; public Actions links are authoritative | 2658 | None |
| [data/validation/invocations/default-b1.json](data/validation/invocations/default-b1.json) | Unmodified contribution script or benchmark invocation | 431 | None |
| [data/validation/invocations/embeddings-b1.json](data/validation/invocations/embeddings-b1.json) | Unmodified contribution script or benchmark invocation | 469 | None |
| [data/validation/invocations/sequence-b1.json](data/validation/invocations/sequence-b1.json) | Unmodified contribution script or benchmark invocation | 465 | None |
| [data/validation/invocations/smoke-esm3.json](data/validation/invocations/smoke-esm3.json) | Unmodified contribution script or benchmark invocation | 294 | None |
| [data/validation/invocations/smoke-esmc.json](data/validation/invocations/smoke-esmc.json) | Unmodified contribution script or benchmark invocation | 294 | None |
| [data/validation/invocations/smoke-raw-forwards.json](data/validation/invocations/smoke-raw-forwards.json) | Unmodified contribution script or benchmark invocation | 299 | None |
| [data/validation/invocations/swiglu-b1.json](data/validation/invocations/swiglu-b1.json) | Unmodified contribution script or benchmark invocation | 454 | None |
| [data/validation/invocations/swiglu-b8.json](data/validation/invocations/swiglu-b8.json) | Unmodified contribution script or benchmark invocation | 483 | None |
| [data/validation/invocations/swiglu-l512.json](data/validation/invocations/swiglu-l512.json) | Unmodified contribution script or benchmark invocation | 483 | None |
| [data/validation/requirements-performance.txt](data/validation/requirements-performance.txt) | Observed package freeze; not a hash-locked environment | 2438 | None |
| [data/validation/run-1/failure-summary.json](data/validation/run-1/failure-summary.json) | V1 exact validation record | 5245 | None |
| [data/validation/run-1/locked-environment.json](data/validation/run-1/locked-environment.json) | V1 exact validation record | 1147 | None |
| [data/validation/run-1/locked-esm3-regression.xml](data/validation/run-1/locked-esm3-regression.xml) | V1 exact validation record | 3815 | removed JUnit hostname attribute |
| [data/validation/run-1/locked-lint.log](data/validation/run-1/locked-lint.log) | V1 exact validation record | 2202 | None |
| [data/validation/run-1/locked-pytest.xml](data/validation/run-1/locked-pytest.xml) | V1 exact validation record | 81668 | removed JUnit hostname attribute |
| [data/validation/run-1/locked-small-gpu.xml](data/validation/run-1/locked-small-gpu.xml) | V1 exact validation record | 90931 | removed JUnit hostname attribute |
| [data/validation/run-1/performance-environment.log](data/validation/run-1/performance-environment.log) | V1 exact validation record | 177 | None |
| [data/validation/run-1/small-gpu-selection.json](data/validation/run-1/small-gpu-selection.json) | V1 exact validation record | 6268 | None |
| [data/validation/run-1/source.json](data/validation/run-1/source.json) | V1 exact validation record | 592 | None |
| [data/validation/run-1/summary.json](data/validation/run-1/summary.json) | V1 exact validation record | 3633 | None |
| [data/validation/run-2/checkpoints.json](data/validation/run-2/checkpoints.json) | V2 exact validation record | 3382 | None |
| [data/validation/run-2/performance-environment.json](data/validation/run-2/performance-environment.json) | V2 exact validation record | 1092 | None |
| [data/validation/run-2/performance-esm3-regression.xml](data/validation/run-2/performance-esm3-regression.xml) | V2 exact validation record | 3815 | removed JUnit hostname attribute |
| [data/validation/run-2/smoke-esm3.exit](data/validation/run-2/smoke-esm3.exit) | V2 exact validation record | 2 | None |
| [data/validation/run-2/smoke-esm3.log](data/validation/run-2/smoke-esm3.log) | V2 exact validation record | 5155 | None |
| [data/validation/run-2/smoke-esmc.exit](data/validation/run-2/smoke-esmc.exit) | V2 exact validation record | 2 | None |
| [data/validation/run-2/smoke-esmc.log](data/validation/run-2/smoke-esmc.log) | V2 exact validation record | 894 | None |
| [data/validation/run-2/smoke-raw-forwards.exit](data/validation/run-2/smoke-raw-forwards.exit) | V2 exact validation record | 2 | None |
| [data/validation/run-2/smoke-raw-forwards.log](data/validation/run-2/smoke-raw-forwards.log) | V2 exact validation record | 3899 | None |
| [data/validation/run-2/source.json](data/validation/run-2/source.json) | V2 exact validation record | 598 | None |
| [data/validation/run-2/summary.json](data/validation/run-2/summary.json) | V2 exact validation record | 7557 | None |
| [data/validation/run-2/validation-summary.json](data/validation/run-2/validation-summary.json) | V2 exact validation record | 2740 | None |
| [data/validation/run-3/environment.json](data/validation/run-3/environment.json) | V3 exact validation record | 704 | None |
| [data/validation/run-3/finished.txt](data/validation/run-3/finished.txt) | V3 exact validation record | 21 | None |
| [data/validation/run-3/gpu-tests.exit](data/validation/run-3/gpu-tests.exit) | V3 exact validation record | 2 | None |
| [data/validation/run-3/gpu-tests.log](data/validation/run-3/gpu-tests.log) | V3 exact validation record | 1866 | None |
| [data/validation/run-3/gpu-tests.xml](data/validation/run-3/gpu-tests.xml) | V3 exact validation record | 7433 | removed JUnit hostname attribute |
| [data/validation/run-3/started.txt](data/validation/run-3/started.txt) | V3 exact validation record | 21 | None |
| [data/validation/run-3/tracked-diff-after.txt](data/validation/run-3/tracked-diff-after.txt) | V3 exact validation record | 0 | None |
| [probes/capture/validation.py](probes/capture/validation.py) | O1 historical validation probe | 10426 | None |
| [probes/historical/recompute_historical.py](probes/historical/recompute_historical.py) | Recompute H3 and H5 from raw samples | 3320 | adjusted historical verifier data path for public layout |
| [probes/optimization/attention_probe.py](probes/optimization/attention_probe.py) | Archived experiment source; historical paths may require adaptation | 4255 | None |
| [probes/optimization/compile_probe.py](probes/optimization/compile_probe.py) | Archived experiment source; historical paths may require adaptation | 15098 | None |
| [probes/optimization/ffn_accuracy_probe.py](probes/optimization/ffn_accuracy_probe.py) | Archived experiment source; historical paths may require adaptation | 2704 | None |
| [probes/optimization/graph_profile.py](probes/optimization/graph_profile.py) | Archived experiment source; historical paths may require adaptation | 983 | None |
| [probes/optimization/mxfp8_probe.py](probes/optimization/mxfp8_probe.py) | Archived experiment source; historical paths may require adaptation | 4356 | None |
| [probes/optimization/probe_common.py](probes/optimization/probe_common.py) | Archived experiment source; historical paths may require adaptation | 9691 | None |
| [probes/optimization/requirements-attention-mxfp8.txt](probes/optimization/requirements-attention-mxfp8.txt) | Archived experiment source; historical paths may require adaptation | 316 | None |
| [probes/optimization/run_attention_probe.sh](probes/optimization/run_attention_probe.sh) | Archived experiment source; historical paths may require adaptation | 606 | None |
| [probes/optimization/run_compile_probe.sh](probes/optimization/run_compile_probe.sh) | Archived experiment source; historical paths may require adaptation | 743 | None |
| [probes/optimization/run_fa4_long_probe.sh](probes/optimization/run_fa4_long_probe.sh) | Archived experiment source; historical paths may require adaptation | 693 | None |
| [probes/optimization/run_graph_ncu.sh](probes/optimization/run_graph_ncu.sh) | Archived experiment source; historical paths may require adaptation | 859 | None |
| [probes/optimization/run_graph_profile.sh](probes/optimization/run_graph_profile.sh) | Archived experiment source; historical paths may require adaptation | 883 | None |
| [probes/optimization/run_mxfp8_compiled_probe.sh](probes/optimization/run_mxfp8_compiled_probe.sh) | Archived experiment source; historical paths may require adaptation | 823 | None |
| [probes/optimization/run_mxfp8_probe.sh](probes/optimization/run_mxfp8_probe.sh) | Archived experiment source; historical paths may require adaptation | 533 | None |
| [probes/optimization/run_swiglu_nsys.sh](probes/optimization/run_swiglu_nsys.sh) | Archived experiment source; historical paths may require adaptation | 854 | None |
| [probes/optimization/strict-swiglu-generated.py](probes/optimization/strict-swiglu-generated.py) | Archived experiment source; historical paths may require adaptation | 8326 | None |
| [probes/optimization/summarize_kernels.py](probes/optimization/summarize_kernels.py) | Archived experiment source; historical paths may require adaptation | 1914 | None |
| [probes/optimization/swiglu_nsys_probe.py](probes/optimization/swiglu_nsys_probe.py) | Archived experiment source; historical paths may require adaptation | 1943 | None |
| [probes/selection/baseline/esm/models/esm3.py](probes/selection/baseline/esm/models/esm3.py) | O2 exact compared model source | 23414 | None |
| [probes/selection/probe_common.py](probes/selection/probe_common.py) | O2 exact archived probe source | 9691 | None |
| [probes/selection/repo/esm/models/esm3.py](probes/selection/repo/esm/models/esm3.py) | O2 exact compared model source | 23706 | None |
| [probes/selection/run_selection_probe.sh](probes/selection/run_selection_probe.sh) | O2 exact archived probe source | 1164 | None |
| [probes/selection/selection_memory_probe.py](probes/selection/selection_memory_probe.py) | O2 exact archived probe source | 2335 | None |
| [probes/selection/selection_probe.py](probes/selection/selection_probe.py) | O2 exact archived probe source | 5258 | None |
| [probes/validation/build_fa2_sm100.sh](probes/validation/build_fa2_sm100.sh) | Archived validation helper; review historical workspace paths | 2974 | None |
| [probes/validation/environment_probe.py](probes/validation/environment_probe.py) | Archived validation helper; review historical workspace paths | 2633 | None |
| [probes/validation/probe_fa2_sm100.py](probes/validation/probe_fa2_sm100.py) | Archived validation helper; review historical workspace paths | 1685 | None |
| [probes/validation/run_gpu_repaired.sh](probes/validation/run_gpu_repaired.sh) | Archived validation helper; review historical workspace paths | 2002 | None |
| [probes/validation/run_local_example.py](probes/validation/run_local_example.py) | Archived validation helper; review historical workspace paths | 1579 | None |
| [probes/validation/run_validation.py](probes/validation/run_validation.py) | Archived validation helper; review historical workspace paths | 13903 | None |
| [probes/validation/small_gpu_selection.py](probes/validation/small_gpu_selection.py) | Archived validation helper; review historical workspace paths | 1391 | None |
| [probes/validation/stage_checkpoints.py](probes/validation/stage_checkpoints.py) | Archived validation helper; review historical workspace paths | 1296 | None |
