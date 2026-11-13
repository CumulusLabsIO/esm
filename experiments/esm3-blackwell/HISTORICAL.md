# Historical ESM3 experiments on B200

These experiments established that the released small ESM3 checkpoint could run on B200, reproduced the raw-forward dtype failure, and measured the benefits of existing batching and optimizer settings. They also motivated the CUDA Graph work. The results below distinguish execution checks, timing measurements, numerical checks, and evidence that could not be recovered.

The audit was performed on 2026-11-13 using saved artifacts. Recomputing statistics did not run another GPU experiment. The final PR's source changes and benchmark are documented separately in [the CUDA Graph recipe](../../cookbook/local/esm3_cuda_graph.md) and [its recorded results](../../cookbook/local/esm3_cuda_graph_results.json).

## Source, environment, and interpretation

All inspected historical model records pin checkpoint `biohub/esm3-sm-open-v1` at revision `47f0545b2b6daf26a93439a3cd610f4f7f3d5478` and package `esm==3.4.3`. They concern the small released checkpoint, not every ESM3 model size.

| Experiment | Model implementation | PyTorch / CUDA runtime | Hardware and execution |
| --- | --- | --- | --- |
| H1: initial smoke / BF16 | Released `esm==3.4.3`; no ESM/PyTorch source patch in the recovered harness | **2.11.0+cu130 / 13.0**; Python 3.12.3 | One visible B200, compute capability 10.0; BF16 parameters; `MultiHeadAttention`; no FlashAttention installed |
| H2: replicated inference / DDP | Released `esm==3.4.3`; aggregate reports zero ESM/PyTorch source patches | **2.15.0+cu132 / 13.2**; NCCL 2.30.7 | Eight B200s in one host; no FlashAttention installed |
| H3–H5: batching, prototype graph, AdamW | Released `esm==3.4.3`; H4 is a separate raw-forward graph probe | **2.15.0+cu132 / 13.2** | One B200 for inference; eight B200s for DDP |

The H2 aggregate additionally records driver 595.112.07 and CUDA toolkit wheel dependency 13.2.2. The driver, toolkit dependency, and PyTorch CUDA runtime are different version fields. These are historical tested versions, not a claim about the latest software available today.

The original package version is not an immutable source commit identifier. The later PR validation target was commit `d285ddeead83631bbe0bc6aa7c879eb5a8946ccb`; the [final result record](../../cookbook/local/esm3_cuda_graph_results.json) separately identifies its tested ESM model source SHA-256 and environment, including Triton. Its multiple-input, copy-inclusive benchmark protocol must not be attributed to the earlier H4 probe.

“Pass” below means that the listed assertions passed. It does not establish biological quality, long-run training convergence, production serving performance, or superiority to Hopper. No historical experiment here compares B200 with H100/H200 or tests B300.

## H1 — Public checkpoint, BF16 failure, and initial inference

**Question.** Can the released checkpoint execute inference on B200, and does the reported raw-forward BF16 problem reproduce?

**Evidence.** [Result JSON](data/historical/initial/results/report.json), [test source](data/historical/initial/esm3_blackwell_test.py), [environment runner](data/historical/initial/run_esm3_test.sh), [installed packages](data/historical/initial/requirements-installed.txt), and [run log](data/historical/initial/run.log).

The run log explicitly records an unauthenticated Hugging Face Hub request warning followed by a successful download. This establishes public downloadability of the pinned small checkpoint at the time of the run. It does not establish an open-source license or availability of larger ESM3 variants.

The recovered test source does the following:

1. Exposes one GPU and asserts that it is a B200.
2. Downloads the pinned checkpoint revision, uses the snapshot directory for local assets, and loads `ESM3.from_pretrained("esm3-open", device=...)` in evaluation mode.
3. Encodes the 76-residue ubiquitin fixture below.
4. Calls raw `forward()` with and without BF16 autocast, then exercises public logits, generation, and structure encoding/decoding.
5. Checks finite tensors, output dimensions, preservation of fixed sequence residues, and completion of masked sequence positions.

```text
MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG
```

The report has **11 passing cases and one deliberately expected failure**. Its aggregate `passed: true` means that there was no unexpected failure.

| Case | Observed result | Exact scope of the check |
| --- | --- | --- |
| Raw forward, autocast disabled | `RuntimeError: mat1 and mat2 must have the same dtype, but got Float and BFloat16` | Traceback reaches `plddt_projection(rbf_16_fn(average_plddt))` in the input encoder |
| Same forward, BF16 CUDA autocast enabled | Finite BF16 output `[1,78,64]` | The wrapper changes autocast, not model source |
| Public sequence logits, lengths 76/256/512/1024 | Finite FP32 output `[1,L+2,64]` for each length | Longer fixtures repeat and truncate ubiquitin |
| Sequence generation | Eight masked positions completed; fixed positions retained; output length 76 | Positions 20–27 are masked using zero-based indexing; seed 2026; eight steps; temperature 0.7 |
| Structure generation, decode, re-encode | Finite backbone `[76,3,3]`; 78 re-encoded structure tokens; PDB output 24,542 bytes | Seed 2026; eight generation steps; first three atom coordinates per residue checked |
| Batch sequence generation | Two complete 76-residue outputs | Two prompts with the same eight masked positions; eight steps; default generation settings otherwise |
| Three additional 512-residue logits calls | All finite | Repeated execution check |

The raw-forward error concerns a Float activation entering a BF16 linear projection. It is not an error reporting an unsupported Blackwell kernel. The recovered workaround is:

```python
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    output = model(sequence_tokens=encoded.sequence.unsqueeze(0))
```

The later H2 aggregate reports the same failure without autocast and a pass with autocast on PyTorch 2.15, with a reference to [upstream PR #417](https://github.com/Biohub/esm/pull/417). The detailed later recheck log could not be recovered, so that later reproduction is supported by the aggregate rather than an independently reopened traceback. This report makes no claim about that PR's present merge status.

**Timing boundary.** The H1 source synchronizes before a case and after a successful operation, then records elapsed wall time. It does not warm each case or reset peak memory counters between cases. Consequently, its seconds include first-use effects and its memory maxima accumulate across earlier operations. Those values are not used as warmed latency or per-operation memory benchmarks.

**Warnings retained in the evidence.** The run warns about optional ESMC FlashAttention import/fallback, missing entity metadata with a `None` fallback, and deprecated non-tuple multidimensional indexing during structure processing. The relevant cases still pass. No fix for the latter two warnings was evaluated here. The optional-wheel warning describes a wheel for compute capabilities 8.0–9.0; it is not evidence that every FlashAttention implementation fails on Blackwell.

The initial setup and test source are available for inspection and reproduction. Reproducing the historical environment requires the pinned torch/esm versions and dependency record; running only the current PR's recipe would be a different experiment.

## H2 — Eight-GPU inference and DDP training smoke test

**Question.** Does the released package execute on every GPU and perform synchronized optimization across eight B200s?

**Evidence.** [Aggregate summary](data/historical/blackwell-cluster/summary.json), [inference rank 0](data/historical/blackwell-cluster/cluster-results/inference/rank-0.json), [all training ranks](data/historical/blackwell-cluster/cluster-results/training/summary.json), [setup runner](data/historical/blackwell-cluster/run_esm3_cluster.sh), and [workload runner](data/historical/blackwell-cluster/execute_esm3_cluster.sh).

The recovered runner sets `OMP_NUM_THREADS=4`, enables offline checkpoint access after setup, and uses `torchrun --standalone --nproc-per-node=8`. Inference and training are separate invocations. This is **eight GPUs on one machine**, with replicated inference and one DDP process per GPU for training. It is not eight nodes or model-parallel inference.

The recorded launch operations, with machine-specific paths omitted, are:

```bash
torchrun --standalone --nproc-per-node=8 \
    esm3_blackwell_cluster_test.py inference --output cluster-results/inference

torchrun --standalone --nproc-per-node=8 \
    esm3_blackwell_cluster_test.py training --output cluster-results/training --steps 3
```

The workload source file itself remains unavailable, so these commands document the original launch; they are not a complete runnable reproduction bundle.

| Workload | Result | Evidence boundary |
| --- | --- | --- |
| Replicated inference | Aggregate reports eight successful ranks | The reopened rank-0 record reports successful inference, logits shape `[1,78,64]`, structure backbone shape `[76,3,3]`, eight structure-generation steps, and peak allocated memory 5.063070 GiB. It does not contain tensors or a separate finiteness flag. The other inference rank records were not individually recovered. |
| DDP training | All eight recovered rank records report success for three steps | Local batch 1, global batch 8, 76 residues, eight masked residues per sequence, FP32 parameters, BF16 autocast, AdamW, learning rate `1e-5` |

Training records contain 1,378,105,920 trainable parameters out of 1,401,735,748 total. The structure, SS8, SASA, function, and residue output heads are inactive. Each step records 509 gradient tensors. First-transformer QKV, last-transformer FFN, and sequence-head parameter probes all change. All eight final parameter hashes match and all records report `replicas_identical: true`.

The global mean loss decreases across the three steps:

| Step | Global mean loss |
| --- | ---: |
| 1 | 0.02736555226147175 |
| 2 | 0.022470152005553246 |
| 3 | 0.015547005459666252 |

Peak allocated memory is 28.926699 GiB per GPU; peak reserved memory is 30.486328 GiB. These are smoke-test results, not the later warmed optimizer comparison in H5.

**Limitations.** This is a short training execution check with inactive output heads, not full multimodal training or a convergence study. Without the original workload source, exact mask construction, gradient-finiteness assertions, loss-reduction code, and timing boundaries cannot be independently re-inspected. The matching hashes support synchronized replicas within the observed run; they do not establish scaling efficiency or training quality.

## H3 — Existing batching APIs

**Question.** How much throughput does batching recover for eight equal-length requests on one B200?

The records identify 76-residue workloads, default attention backend, eager execution, batch size eight, and `profile_capture: false`. The source of the original profiling harness could not be recovered. Therefore the exact original sequence construction and timer implementation remain unavailable; the JSON's workload descriptors, raw samples, and stored numerical comparisons are the evidence for these results.

Each inference median below is recomputed from the saved wall-time samples, in milliseconds. The speedup uses the arithmetic mean of the before/after control medians divided by the candidate median. The before/after range measures drift between these controls, not a confidence interval.

| Workload | Serial before | Batched candidate | Serial after | Speedup vs mean controls | Throughput |
| --- | ---: | ---: | ---: | ---: | --- |
| Eight logits requests | 269.119319 ms | 34.110756 ms | 263.335569 ms | **7.804795×** | 29.73–30.38 → 234.53 requests/s |
| Eight structure generations | 2442.699997 ms | 572.858467 ms | 2437.091538 ms | **4.259160×** | 3.275–3.283 → 13.965 structures/s |

The logits measurements use **eight warmups and 25 timed repetitions** per path. The structure measurements use **three warmups and 10 timed repetitions** per path. All six JSON records preserve every timed wall-time and CUDA-stream sample:

| Workload | Before | Candidate | After |
| --- | --- | --- | --- |
| Logits | [Serial before](data/historical/blackwell-profile/perf-results/logits-eight-serial-before.json) | [Batch eight](data/historical/blackwell-profile/perf-results/logits-eight-batched.json) | [Serial after](data/historical/blackwell-profile/perf-results/logits-eight-serial-after.json) |
| Structure | [Serial before](data/historical/blackwell-profile/perf-results/structure-eight-serial-before.json) | [Batch eight](data/historical/blackwell-profile/perf-results/structure-eight-batch.json) | [Serial after](data/historical/blackwell-profile/perf-results/structure-eight-serial-after.json) |

**Logits correctness.** The candidate output `[8,78,64]` is finite and exactly matches the saved serial reference: maximum absolute error zero, mean absolute error zero, and top-1 agreement 1.0. The control median changes by −2.149140%; candidate speedup is 7.720016× to 7.889574× against the individual controls.

**Structure correctness boundary.** The candidate outputs `[8,76,3,3]` are finite, but their coordinates differ from the serial reference: maximum absolute difference **10.24785327911377**, mean absolute difference **0.457936555147171**. The saved record notes that generation is stochastic and batching can alter the random draw. The serial after-control matches the serial before-control exactly. Its timing drift is −0.229601%. No structural-quality metric was evaluated, so the throughput result does not establish numerical or biological equivalence of the batched outputs.

These gains came from existing batching APIs. They do not measure a new kernel, mixed-length padding correctness, or Blackwell's advantage over another GPU architecture. Batching documentation is a useful contribution, but its speedup should not be represented as a new implementation in this PR.

## H4 — Prototype CUDA Graph replay

**Question.** Could graph replay reduce the latency of a short, repeatedly executed forward pass enough to justify a supported public-API implementation?

**Evidence.** [Raw-graph summary log](data/historical/blackwell-profile/perf-results/logits-followup-raw-graph.log) and [repeated eager control JSON](data/historical/blackwell-profile/perf-results/logits-followup-eager-repeat.json).

| Path | Recorded unprofiled wall median | Protocol |
| --- | ---: | --- |
| Repeated eager control | 33.9561195 ms | Batch 1, 76 residues, eight warmups, 50 timed repetitions |
| `raw-graph` prototype | 6.4115735 ms | Batch 1, 76 residues, eight warmups, 50 timed repetitions |

The graph log reports finite output `[1,78,64]`, exact agreement with its stored eager reference, maximum error zero, and top-1 agreement 1.0. The ratio against this recovered eager control is **5.296066×**.

The earlier discussion quoted a 32.58–33.96 ms eager bracket and a 5.19× speedup against mean controls. The **32.58 ms control could not be recovered**, so that aggregate is not independently verified in this bundle. The 5.296066× single-control ratio above is a different calculation and should not be substituted silently for it. The graph log contains summary values; its JSON with all timing samples also remains unavailable.

This historical result is a raw-forward prototype. It does not establish stock public `logits()` capture, correctness with multiple changing inputs, or inclusion of input-copy overhead. The original probe source and exact graph-capture failure log could not be recovered. The later PR benchmark addresses those questions with its own source and evidence; its stronger checks do not retroactively apply to the prototype.

### Separate profiler observations

Saved Nsight Systems text logs show profile capture for the same short workloads:

| Profiled path | Wall median | Protocol | Evidence |
| --- | ---: | --- | --- |
| Eager logits | 47.017315 ms | Five warmups / five measured profiled calls | [Log](data/historical/blackwell-profile/traces/logits-default.log) |
| Raw graph | 6.445893 ms | Five warmups / five measured profiled calls | [Log](data/historical/blackwell-profile/traces/logits-raw-graph.log) |
| Structure | 441.555845 ms | Five warmups / five measured profiled calls | [Log](data/historical/blackwell-profile/traces/structure-default.log) |

The profiled eager result is substantially slower than the unprofiled control. These runs demonstrate that profiling can perturb this workload; their timings are excluded from the performance headline. Historical Nsight Compute attempt logs could not be recovered, so this appendix does not claim successful hardware-counter profiling from those attempts.

## H5 — Fused AdamW

**Question.** Does PyTorch's fused AdamW implementation reduce step time and memory for the eight-GPU training workload?

**Evidence.** [Foreach before](data/historical/blackwell-profile/train-profile/adamw-before/summary.json), [fused candidate](data/historical/blackwell-profile/train-profile/adamw-fused/summary.json), and [foreach after](data/historical/blackwell-profile/train-profile/adamw-after/summary.json).

Each result contains eight successful ranks, **three warmup steps and five measured steps**, local batch 1/global batch 8, 1,378,105,920 trainable parameters, and the same five inactive output heads as H2. The JSON identifies `optimizer_implementation` as `foreach` or `fused`. The original training harness source is unavailable, so the exact constructor arguments and synchronization placement cannot be independently re-read.

The historical latency statistic can be reproduced as follows:

1. For measured step `i`, take the maximum elapsed seconds across the eight ranks.
2. Take the median of those five step maxima.
3. Multiply by 1,000 to express milliseconds.

In these records, that statistic also equals the maximum of the per-rank medians. A median over all 40 rank-step observations gives a different result and is not the historical headline statistic.

| Optimizer | Warm step median | Maximum allocated per GPU | Maximum reserved per GPU |
| --- | ---: | ---: | ---: |
| Foreach before | 130.036929 ms | 25.978272 GiB | 27.466797 GiB |
| Fused | 121.573611 ms | 23.791707 GiB | 24.802734 GiB |
| Foreach after | 130.233107 ms | 25.978272 GiB | 27.466797 GiB |

Relative to the mean of the control medians, fused AdamW reduces latency **6.578865%**. Peak allocated memory falls **2.186565 GiB per GPU**, or **8.416899%**. Control latency drift is 0.150863%. These are short warmed runs; there is no independent-run confidence interval or long-run quality evaluation.

**Numerical results.** Every run has matching final parameter hashes across its eight ranks and nonzero updates in the monitored parameters. Both foreach runs also match each other. The fused run has a different final parameter hash and a slightly different loss trajectory:

| Step, including warmup | Foreach global mean loss | Fused global mean loss |
| --- | ---: | ---: |
| 1 | 0.02736555226147175 | 0.02736555226147175 |
| 2 | 0.022470152005553246 | 0.022336922585964203 |
| 3 | 0.015547005459666252 | 0.015636123716831207 |
| 4 | 0.014258335344493389 | 0.014161637984216213 |
| 5 | 0.004379061982035637 | 0.004413703456521034 |
| 6 | 0.0038611104246228933 | 0.003802554216235876 |
| 7 | 0.0031585642136633396 | 0.003106222953647375 |
| 8 | 0.0006040996522642672 | 0.0006117185694165528 |

The result supports a speed/memory improvement for this workload; it does not establish bitwise or training-quality equivalence of the optimizer implementations. Fused AdamW is an existing PyTorch setting suitable for a training recipe, not a new Blackwell-specific optimizer contributed by this PR.

A separate [instrumented training record](data/historical/blackwell-profile/train-profile/trace/summary.json) has two warmups and three measured steps, with rank-0 times around 175–178 ms. It is excluded from this optimizer comparison because its instrumentation and protocol differ.

## Recompute the recoverable historical statistics

From the repository root:

```bash
python3 experiments/esm3-blackwell/probes/historical/recompute_historical.py
```

The script uses only the Python standard library. It does not import torch, request a GPU, or contact a service. It verifies every saved H3 median against its samples, repeat counts, eight-rank H5 success, and within-run replica hashes, then prints the reconstructed statistics. The source arrays are retained so reviewers can compute other descriptive statistics without treating consecutive timing samples as independent experimental replications.

## Recovery gaps and conclusions the evidence does not support

The original archive contained cloud-file placeholders whose contents were not locally available. Targeted download requests and bounded reads recovered the evidence linked above, but not every original artifact. The following gaps remain explicit:

| Missing artifact | Consequence |
| --- | --- |
| `esm3_profile.py`, `esm3_training_profile.py`, `esm3_blackwell_cluster_test.py` | Exact historical profiling/training code, input construction and timing boundaries cannot be fully reconstructed |
| Original comparison/orchestration shell scripts for profiling | Full historical command sequence cannot be independently re-inspected |
| `logits-followup-eager.json` | Earlier 32.58 ms eager control is unverified here |
| `logits-followup-raw-graph.json` | Graph summary is available, but its per-call sample array is not |
| `logits-graph.log` | Original stock-API capture error cannot be verified from that historical log |
| Full inference rank summary | Eight-rank inference success is supported by the aggregate plus one recovered rank; the other seven rank records were not individually audited |
| `single-gpu-recheck.log` | Later PyTorch 2.15 autocast reproduction is aggregate-supported rather than traceback-verified |
| Historical Nsight Compute logs and complete profile summary | No complete inventory or counter-based conclusion is claimed for those attempts |

No failures are reported by the recovered batching records or eight-rank training summaries. That does not certify every unavailable setup/profiling attempt as successful. The confirmed initial failure is the BF16/FP32 raw-forward mismatch, resolved for the tested call by autocast. The source files and evidence for later fixes belong to the separate final PR validation, rather than being inferred from these older summaries.

These measurements support B200 execution, substantial throughput gains from equal-length batching, promising raw graph replay latency, and a smaller fused-optimizer speed/memory improvement. They do not establish mixed-length batching correctness, complete multimodal training, long-run fine-tuning quality, FP8/FP4 accuracy, production service latency, B300 compatibility, or a Blackwell-versus-Hopper performance comparison.
