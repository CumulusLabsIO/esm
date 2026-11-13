# Optimization experiments

The capture repair enables the largest measured improvement: approximately a 5.23× speedup over eager execution on the tested B200 workload, including a fresh input copy. Strict compilation of only SwiGLU then reduces graph latency by 3.64–5.67% on the original three fixtures while preserving their returned values. Conditional output conversion removes unnecessary work but has little measured latency effect. Broader compilation, FA4, and MXFP8 did not produce an accepted equivalent improvement.

This document records both successful and rejected candidates. The [dossier overview](README.md) covers implementation, test status, provenance, and subsequent validation. [REPRODUCE.md](REPRODUCE.md) provides commands and environment setup. These are B200 observations; there is no B300 measurement or Hopper comparison here.

## Objective, method, and constraints

The objective was to reduce synchronized inference latency for repeated, fixed-shape ESM3-open requests on one B200, including input transfer, while retaining original BF16 output values. This differs from maximizing asynchronous throughput, training convergence, or end-to-end protein-design quality.

The original optimization environment was NVIDIA B200 / SM100, driver 595.112.07, PyTorch 2.15.0+cu132, CUDA runtime 13.2, Triton 3.9.0, cuDNN 9.26.0.51, and `OMP_NUM_THREADS=4`. The checkpoint was `biohub/esm3-sm-open-v1@47f0545b2b6daf26a93439a3cd610f4f7f3d5478`. Attention experiments used visible GPU 2, quantization GPU 3, final SwiGLU timing GPU 0, and its separate profile GPU 3. Compare each candidate with its own controls; absolute timings across GPUs/processes are not a controlled comparison.

The standard fixture is 76-residue ubiquitin, encoded as 78 tokens. Four inputs comprise the unmasked sequence and three masks at offsets 0, 1, and 2 modulo 7. Batch rows cycle through these variants. The longer fixtures repeat ubiquitin synthetically. Unless an experiment explicitly selects outputs, it requests all six logits heads plus embeddings.

Each timed request includes a fresh preallocated pinned-host input copy, computation or graph replay, and completion synchronization. Timing excludes tokenization, host allocation/pinning, model loading, compilation, graph capture, and output transfer to CPU. Before/after controls and the candidate run in one process in A/B/A order. Raw samples are retained, including outliers.

| Experiment | Warmup calls per path | Measured calls per path |
|---|---:|---:|
| O1 capture/copy | 20 | 100 |
| O2 selective-output timing | 5 | 30 |
| O2 separate peak-memory check | — | 5 allocation measurements per mode/path |
| O4 full/transformer compilation sweep | 15 | 60 |
| O4 strict FFN follow-up | 20 | 100 |
| O5 strict SwiGLU, dedicated and public recipe | 20 | 100 |
| O6/O7 final attention and quantization | 20 | 100 |

Profiler captures are separate experiments with their own replay counts; they do not supply the uninstrumented wall timings in the performance tables. Wall-time and CUDA-event measurements in the private probes are separate sampling passes, not per-request paired observations.

Two distinct correctness checks matter:

1. **Replay fidelity:** a captured graph matches eager execution of that same configuration on changing inputs.
2. **Reference equivalence:** the candidate matches the original BF16 reference. Exact equality uses `torch.equal` on returned FP32 tensors. Approximate experiments also record a declared `rtol=atol=1e-2` gate where applicable.

`torch.equal` establishes exact elementwise values, not raw byte equality; older files label this “bitwise.” A successful replay-fidelity check does not establish reference equivalence. Neither fixture-level equality nor output-error statistics establish biological quality.

Source and dispatch inspection constrained the candidates before interpreting measurements:

| Candidate | Constraint that matters |
|---|---|
| CUDA Graphs | Captured addresses/shapes stay fixed; new input values must be copied into retained storage. CPU-backed scalar transfers can prevent capture. |
| Requested outputs | Output selection occurs after the model computes its heads. Avoiding unwanted FP32 conversion is possible without changing `forward()` or hooks; skipping head computation is a different change. |
| Broad compilation | Whole-forward tracing reaches tokenizer methods even when tokens are supplied. Regional compilation avoids that call, but may change numerical behavior. |
| Strict SwiGLU | The original SiLU result rounds to BF16 before multiplication. A fused kernel must preserve that intermediate rounding for exact agreement. |
| Attention | The installed default already dispatches to cuDNN for these fixtures. FA4 must be confirmed in the executed kernel, not inferred from an accepted option. |
| MXFP8 | Quantization targets 96 FFN linears; conversion/scaling overhead and matrix shape can outweigh lower-precision GEMM savings. A compiled candidate needs a correspondingly compiled BF16 control. |

## O1. Repair CUDA Graph capture and measure fresh-input overhead

**Problem.** The released public logits path creates a CPU-backed confidence scalar and transfers it to CUDA during capture. [The stock capture probe](data/capture/stock-capture.json) fails with:

```text
Cannot copy between CPU and CUDA tensors during CUDA graph capture unless the CPU tensor is pinned.
```

**Change.** Create the scalar on the active device with `torch.ones((), device=...)`. The library change adds no model-global graph cache or mutable request buffer. The separate raw-`forward()` BF16/autocast issue is not fixed by this scalar change.

**Verification.** [The validation probe](probes/capture/validation.py) compares released and patched outputs for four changing inputs and all seven returned tensors. Device-copy and pinned-host-copy graphs also match their references over repeated input cycles. [Stock before](data/capture/stock-before.json), [patched](data/capture/patched.json), and [stock after](data/capture/stock-after.json) records preserve source hashes, numerical checks, and raw samples.

The same-process patched measurements separate replay from the new-input copy:

| Path | Before / candidate median (ms) | After-control median (ms) |
|---|---:|---:|
| Eager, existing device input | 32.8079315 | 32.8011630 |
| Eager, fresh pinned H2D input | 32.8113860 | 32.8209470 |
| Fixed-input graph replay | 6.2617780 | 6.2620405 |
| Fresh device-input copy + graph | 6.2709435 | — |
| Fresh pinned H2D input + graph | 6.2694705 | — |

The fresh-H2D graph path is approximately **5.23× faster** than the average eager-host control. Relative to the mean of the fixed-replay medians, its increment is **7.56125 microseconds**, approximately **0.12%** of the complete copied-input call. This is a difference of medians, not an isolated bandwidth measurement. The input is only 624 bytes; it cannot predict costs for large structure or multimodal inputs. The H2D candidate's p90 is 6.273935 ms under the original probe's quantile convention.

Applications own the graph's retained buffers. They must consume or clone captured outputs before the next replay overwrites them, and recapture when shapes, input tracks, output selection, model state, or backend configuration change. The measured result does not establish a general dynamic-shape serving system.

## O2. Convert only requested outputs

**Problem and change.** The previous logits path converts every returned track to FP32 before discarding unrequested tracks. The patch conditions conversion on `LogitsConfig`. All model heads are still computed; raw `forward()` and its hooks retain the full output contract. This is conversion avoidance, not head pruning.

[The selection probe](probes/selection/selection_probe.py) checks requested values, captured replay, and empty configurations. All requested outputs match across the four inputs for B1/L76, B8/L76, and B1/L512. Three-call profile samples remove 18 `_to_copy` operations, from 2,187 to 2,169: six unnecessary conversions per request. All-output mode is a control.

Graph wall medians, including fresh H2D, are:

| Fixture | Outputs | Before (ms) | Candidate (ms) | After (ms) |
|---|---|---:|---:|---:|
| B1/L76 | Sequence only | 6.2890265 | 6.2795925 | 6.2891575 |
| B1/L76 | Embeddings only | 6.5453295 | 6.5362085 | 6.5544530 |
| B1/L76 | All outputs | 6.5552490 | 6.5497930 | 6.5552310 |
| B8/L76 | Sequence only | 9.2604385 | 9.2578305 | 9.2587040 |
| B8/L76 | Embeddings only | 9.7824180 | 9.7774090 | 9.7724370 |
| B8/L76 | All outputs | 9.7864445 | 9.8009375 | 9.7772460 |
| B1/L512 | Sequence only | 11.0672920 | 11.0572800 | 11.0696200 |
| B1/L512 | Embeddings only | 11.5793155 | 11.5683390 | 11.5798155 |
| B1/L512 | All outputs | 11.5865750 | 11.5802465 | 11.5857630 |

Raw records: [B1/L76](data/selection/b1-l76.json), [B8/L76](data/selection/b8-l76.json), [B1/L512](data/selection/b1-l512.json). Some differences are comparable with control variation. The observed improvement is approximately 0–0.2%; it is not the source of the large graph speedup. Compare candidates within a row: profile collection between different output-mode experiments makes cross-row absolute timing a weaker comparison.

[The separate memory probe](probes/selection/selection_memory_probe.py) found the same whole-call peak allocation, **56,651,264 bytes**, for all three output modes and all five before/candidate/after measurements. [The memory record](data/selection/memory-b1-l76.json) also records tensor sizes. The avoided conversions occur after the peak: **there is no measured peak-allocation reduction** here.

## O3. Profile work inside the graph

The hypothesis was that once replay removes host launch overhead, smaller kernels and memory operations may matter more than switching the attention backend.

A five-replay Nsight Systems node capture found **2,386 kernels per public logits replay**. Kernel-name grouping gives the following instrumented duration shares:

| Group | Kernels per replay | Share of summed kernel duration |
|---|---:|---:|
| GEMMs | 213 | 22.04% |
| Attention | 48 | 3.73% |
| Normalization | 200 | 14.38% |
| Casts | 727 | 16.75% |
| Copy/concatenation | 392 | 14.94% |
| Other elementwise work | 806 | 28.16% |

The total instrumented duration sum is 5.8397754 ms/replay. Classification is heuristic, especially for fused kernels. This is neither unprofiled wall time nor device utilization. [Summary](data/profiling/graph-baseline-kernel-summary.json), [kernel CSV](data/profiling/graph-baseline-kernels.csv), [probe](probes/optimization/graph_profile.py), and [summarizer](probes/optimization/summarize_kernels.py) preserve the measurement and method.

The 3.73% attention share constrains the attention-only opportunity on this fixture: if all other work stayed constant and this traced decomposition predicted wall time, even removing attention entirely would reduce the sum by only 3.73% (about 1.039× speedup). That conditional bound is not a promise about other lengths, batches, profiles, or wall-time behavior.

Nsight Compute then sampled one `nvjet_sm100_tst_128x40_64x10_2x2_h_bz_TNT` GEMM with `--graph-profiling node --launch-count 1`. It reported 10.11 microseconds, 17.47% SM throughput, 32.84% DRAM throughput, and a grid of 128 blocks / 0.86 waves. This kernel's grid does not fill all SMs. [Metrics](data/profiling/graph-gemm.csv) and [invocation](probes/optimization/run_graph_ncu.sh) document the sample. These metrics apply to that one kernel, not model-wide utilization.

Source inspection also corrected an early theoretical estimate: the real FFN intermediate width is **4096**, because `TransformerStack` passes `expansion_ratio=8/3` and overrides the block's isolated 4× default. Estimates based on width 6144 are invalid. The actual loaded modules and transformed weight shapes, not one constructor default, determine the work.

## O4. Explore broader compiler fusion and reject numerical regressions

**Whole-forward attempt.** `fullgraph=True` cannot trace `ESM3.forward` because evaluating `t.sequence.mask_token_id` reaches the Rust tokenizer's `token_to_id`, including when sequence tokens are already supplied. [The failure record](data/compiler/compile-fullgraph-failed.json) preserves the error. Regional transformer compilation avoids that call without caching tokenizer state or changing its semantics.

**Regional variants.** The [compiler probe](probes/optimization/compile_probe.py) compares each compiled candidate against the original BF16 reference with predeclared `rtol=atol=1e-2`. Controls repeat after timing; those gates were not relaxed.

| Candidate | Control before (ms) | Candidate (ms) | Control after (ms) | First compiled call (s) | Original BF16 gate |
|---|---:|---:|---:|---:|---|
| Transformer, max-autotune-no-cudagraphs | 6.4383745 | 2.4888670 | 6.4362645 | 103.6911066 | Fail |
| Transformer, strict numerics | 6.4352685 | 2.5660355 | 6.4351900 | 95.8868093 | Fail |
| Transformer, strict + ATEN GEMMs + forced cuDNN | 6.4342685 | 2.3952635 | 6.4338625 | 74.6566607 | Fail |
| Strict FFN only | 6.2516310 | 5.6928820 | 6.2514025 | See record | Fail on every output head |

Records: [ordinary transformer](data/compiler/compile-transformer.json), [strict transformer](data/compiler/compile-transformer-strict.json), [strict + ATEN](data/compiler/compile-transformer-strict-aten.json), [strict FFN](data/compiler/strict-ffn-b1.json).

The compiled graphs match their own compiled eager outputs, but they do not meet the original-reference gate. The large latency reductions are therefore **rejected as equivalent replacements**. Error and argmax statistics in the JSON do not substitute for downstream biological validation.

The installed strict Inductor settings emulate eager precision casts, preserve non-flushing pointwise behavior, and use division rounding. Generated strict kernels include `enable_fp_fusion=False` and `disable_ftz=True`. Forcing cuDNN on both sides and restricting autotuned GEMMs to ATEN did not resolve end-to-end drift. The later exact SwiGLU result must not be used to validate these broader compiler variants.

A separate three-replay Nsight comparison shows where broad fusion saves work:

| Measurement | Raw wrapper graph | Strict-ATEN compiled graph |
|---|---:|---:|
| Kernels/replay | 2,385 | 628 |
| Cast kernels/replay | 727 | 40 |
| Copy/concat kernels/replay | 392 | 15 |
| GEMM kernels/replay | 213 | 213 |
| Attention kernels/replay | 48 | 48 |
| Summed traced kernel duration (ms/replay) | 5.983601 | 2.241375 |
| GEMM duration sum (ms/replay) | 1.264501 | 1.247220 |
| Attention duration sum (ms/replay) | 0.226304 | 0.233141 |

The wrapper preallocates confidence, explaining its one fewer baseline kernel than the public API. [Eager profile summary](data/profiling/compile-profile-eager-summary.json) and [compiled summary](data/profiling/compile-profile-compiled-summary.json) support these counts. They are instrumented sums, not the wall timings above, and belong to a numerically rejected candidate. This profile motivated narrower fusion rather than a default whole-transformer compilation change.

## O5. Preserve rounding while fusing only SwiGLU

The accepted narrow candidate compiles each `block.ffn[2]` with `fullgraph=True`, `dynamic=False`, `numerics="strict"`, and `triton.cudagraphs=False`. Native GEMMs, attention, normalization, and other modules remain unchanged.

The [generated kernel](probes/optimization/strict-swiglu-generated.py) explicitly rounds the SiLU intermediate to BF16 before multiplying. It uses libdevice exp, correctly rounded division, `enable_fp_fusion=False`, and `disable_ftz=True`. That intermediate rounding is material to preserving the eager result.

The [initial dedicated probe](probes/optimization/ffn_accuracy_probe.py) produced controls **6.2512125 / 6.2510020 ms** and candidate **6.0235920 ms**, a **3.64%** median reduction. All seven returned outputs match the original BF16 reference for four inputs and repeated fresh-input replay. [Original record](data/compiler/strict-swiglu-b1.json).

The public recipe then repeated the experiment across three workloads:

| Workload | Control before (ms) | Candidate (ms) | Control after (ms) | Lower median latency | First candidate call (s) |
|---|---:|---:|---:|---:|---:|
| B1/L76 | 6.2902075 | 6.0611490 | 6.2899740 | 3.63972% | 2.287874 |
| B8/L76 | 9.2313095 | 8.7075770 | 9.2313695 | 5.67374% | 2.303544 |
| B1/L512 synthetic | 11.0850135 | 10.6292095 | 11.0852855 | 4.11307% | 2.318572 |

Records: [B1](data/swiglu-original/swiglu-b1.json), [B8](data/swiglu-original/swiglu-b8.json), [L512](data/swiglu-original/swiglu-l512.json). All returned values match exactly in each workload. The [public sample dataset](../../cookbook/local/esm3_cuda_graph_results.json) contains 900 timing samples, source-record hashes, and recomputed inclusive linearly interpolated p10/p90. The [README](README.md) separates these original measurements from subsequent reruns.

Amortizing the complete first candidate call against the observed steady-state saving takes approximately 10,000 / 4,400 / 5,100 graph calls, respectively; B8 processes eight proteins per call. These estimates depend on compile/cache conditions and are not guaranteed startup costs. The B1 dataset retains a **14.481934 ms** candidate outlier: the median improves, but these records do not establish a tail-latency gain.

A separate Nsight Systems 2026.1.1 node capture records five control and five candidate replays. Kernels fall from **2,386 to 2,338 per replay**: 48 SiLU kernels plus 48 multiply kernels become 48 fused SwiGLU kernels, with all other counts unchanged. Summed traced duration falls from 5.7983132 to 5.5906832 ms/replay. [Summary](data/profiling/strict-swiglu-nsys-summary.json), [probe](probes/optimization/swiglu_nsys_probe.py), and [launcher](probes/optimization/run_swiglu_nsys.sh) document this separate measurement. The public timing recipe alone does not create these probe-specific NVTX ranges.

Exact agreement is bounded by the tested fixtures, compiler, device, and settings. It is not a mathematical guarantee for every shape or future compiler release. The recipe checks outputs before treating this candidate as accepted.

## O6. Test cuDNN selection and FlashAttention 4

The question was whether an attention backend advertised for newer NVIDIA architectures improved this ESM3 workload. The tested default already selected cuDNN. The [attention probe](probes/optimization/attention_probe.py) uses PyTorch's SDPA backend registry and confirms executed operators/kernel names while preserving ESM's QK normalization and rotary path.

| Candidate | Batch / residues | Control before (ms) | Candidate (ms) | Control after (ms) | Finding |
|---|---|---:|---:|---:|---|
| Forced cuDNN | 1 / 76 | 6.386692 | 6.388626 | 6.386394 | No gain |
| Forced cuDNN | 16 / 76 | 13.378214 | 13.383814 | 13.380272 | No gain |
| FA4 | 1 / 76 | 6.386741 | 6.667877 | 6.387776 | 4.4% slower |
| FA4 | 16 / 76 | 13.379774 | 13.590107 | 13.380025 | 1.6% slower |
| FA4 | 1 / 1024 synthetic | 22.339721 | 22.654333 | 22.335786 | 1.4% slower |

Records: [cuDNN B1](data/attention/cudnn-b1.json), [cuDNN B16](data/attention/cudnn-b16.json), [FA4 B1](data/attention/fa4-b1.json), [FA4 B16](data/attention/fa4-b16.json), [FA4 L1024](data/attention/fa4-l1024-b1.json).

Packages were flash-attn-4 4.0.0b39, nvidia-cutlass-dsl 4.8.1, and quack-kernels 0.6.5. The profile confirms an SM100 CuTe kernel, so the negative result is not merely an option silently falling back. The tested PyTorch FA4 dispatcher requires BF16/FP16, compatible SM9x/10x, zero dropout, and no arbitrary attention mask or requested attention weights. It is separate from ESMC's legacy FlashAttention installation path.

Forced cuDNN matches the default exactly. FA4 changes outputs; the final public recipe records a numerical mismatch against the original reference. Each private configuration's graph nevertheless matches its own eager outputs. No default backend change is justified by these latency and correctness results.

## O7. Test MXFP8 FFNs with matched compiled controls

The hypothesis was that Blackwell's block-scaled GEMMs could accelerate the FFN. The [quantization probe](probes/optimization/mxfp8_probe.py) uses TorchAO 0.18.0, `MXDynamicActivationMXWeightConfig`, E4M3 activations and weights, and `KernelPreference.AUTO`. It transforms 96 FFN linears; normalization, embeddings, and output heads remain BF16. Profiles confirm `aten::_scaled_mm` and SM100 block-scaled `Avec32UE8M0_Bvec32UE8M0` kernels.

| Candidate | Batch / residues | Control before (ms) | Candidate (ms) | Control after (ms) | Finding |
|---|---|---:|---:|---:|---|
| MXFP8, uncompiled | 1 / 76 | 6.250072 | 11.361639 | 6.250922 | 81.8% slower |
| MXFP8, uncompiled | 16 / 76 | 13.315788 | 22.058418 | 13.309633 | 65.7% slower |
| MXFP8, both FFNs compiled | 1 / 76 | 5.679132 | 6.083058 | 5.679070 | 7.1% slower |
| MXFP8, both FFNs compiled | 16 / 76 | 11.946815 | 11.951037 | 11.933649 | No meaningful gain |

Records: [uncompiled B1](data/quantization/mxfp8-b1.json), [uncompiled B16](data/quantization/mxfp8-b16.json), [compiled B1](data/quantization/mxfp8-compiled-b1.json), [compiled B16](data/quantization/mxfp8-compiled-b16.json).

The matched compiled experiments compile each block's FFN on **both** the BF16 control and quantized candidate with `fullgraph=True`, `dynamic=False`, and `triton.cudagraphs=False`. They isolate quantization's incremental effect; they do not validate that the compiled control matches raw BF16. O4 independently found that strict FFN compilation fails the original-reference gate.

Compiled B16 CUDA-event medians are 11.925792 / 11.926720 / 11.925152 ms, also providing no meaningful gain. Minimum sequence argmax agreement is 99.6795%, minimum structure argmax agreement 86.8590%, and maximum embedding normalized RMSE 5.3267% across the four inputs. These errors are not a validation of downstream biological quality. Private graph/eager fidelity passes, while the final public recipe rejects original-reference equivalence. No MXFP8 default is justified by these experiments.

## Negative findings, exclusions, and limits

- Initial attention and quantization attempts inserted profiling between timing phases, perturbing later controls. Those records are retained under [superseded attention](data/superseded/attention/) and [superseded quantization](data/superseded/quantization/), but excluded from final tables. Final O6/O7 runs profile only after all timing phases.
- Private probe exit success means the configured experiment completed and replay matched its own eager outputs. The public recipe additionally rejects numerical mismatches against the labeled reference. An expected rejection of FA4/MXFP8/broad compilation is distinct from a broken unit test or CI failure.
- Calls within one process are not independent experimental runs. A/B/A constrains within-process drift, but order is fixed; there is no cross-host replication or independent-run confidence interval. Sample quantiles are not confidence intervals.
- B1 uses a 624-byte sequence copy. Mixed lengths, large structure inputs, general graph caching, concurrent serving, and asynchronous throughput need separate tests.
- Public profiler summaries preserve observed counts and the probes permit rerunning them. Without the optional large traces, a reader cannot independently reconstruct the exact original trace aggregation. Instrumented kernel sums must remain separate from wall latency.
- These are inference optimization experiments. Historical eight-GPU training/optimizer results have separate provenance and do not validate this patch's training behavior. There is no B300 result, Hopper comparison, biological benchmark, or evidence here establishing ESM3's original architecture target.

## Research basis and reproduction

The four research directions were output-selection work, compiler fusion, attention dispatch, and lower precision. Their applicability was checked against the installed artifact and actual dispatched operations. Useful primary references include [PyTorch 2.15 compilation](https://docs.pytorch.org/docs/2.15/generated/torch.compile.html), [regional compilation](https://docs.pytorch.org/tutorials/recipes/regional_compilation.html), [the pinned FA4 dispatcher source](https://github.com/pytorch/pytorch/blob/v2.15.0/torch/nn/attention/_fa4.py), [SDPA backend selection](https://github.com/pytorch/pytorch/blob/v2.15.0/aten/src/ATen/native/transformers/cuda/sdp_utils.cpp), and [TorchAO's Blackwell diffusion study](https://pytorch.org/blog/faster-diffusion-on-blackwell-mxfp8-and-nvfp4-with-diffusers-and-torchao/). The diffusion result motivated a matched compiled-control experiment; it was not assumed to transfer to ESM3.

Use [REPRODUCE.md](REPRODUCE.md) for portable commands. The [public recipe](../../cookbook/local/esm3_cuda_graph.py) is the canonical application example; archived launch scripts preserve the exact historical methods and may contain environment-specific paths or GPU assignments. Obtain the pinned checkpoint separately under its license. Raw JSON and source-record hashes allow arithmetic and provenance checks without downloading model weights or running a GPU.
