# ESM3 on B200: changes, experiments, and validation

This is the evidence record for [Biohub/esm PR #431](https://github.com/Biohub/esm/pull/431), contributed by [@CumulusLabsIO](https://github.com/CumulusLabsIO). It documents the issues encountered, fixes, successful and rejected experiments, validation failures, and remaining gaps. Measurements were collected on November 12–13, 2026; timestamps in artifacts use UTC unless the field says otherwise. This publication assembles existing measurements; it does not represent another GPU run.

**The supported contribution is a CUDA Graph capture repair, conditional conversion of requested outputs, and a measured optional SwiGLU fusion recipe.** On one B200, fresh-input graph execution took about 6.27 ms versus 32.81–32.82 ms eager for the 76-residue fixture. Strict SwiGLU fusion reduced graph median latency by a further 3.6–5.7% in the tested shapes, with exact returned values. Wider compilation failed numerical checks; FA4 and MXFP8 did not produce an accepted performance gain. The work establishes B200 execution, not a B300 result or a speed advantage over Hopper.

## How to review this record

1. Read the issue ledger and result tables below for scope and decisions.
2. [Historical experiments](HISTORICAL.md) cover initial model access, BF16, eight-GPU inference/training, batching, the first graph prototype, and fused AdamW.
3. [Optimization experiments O1–O7](OPTIMIZATIONS.md) give the controls, raw-file links, correctness gates, compiler failures, profiler findings, and negative results.
4. [Reproduction instructions](REPRODUCE.md) separate the locked test environment from the performance environment and describe the SM100 dependency rebuild.
5. [Evidence index](EVIDENCE_INDEX.md) links every archived artifact. [MANIFEST.json](MANIFEST.json) records original and published hashes, sizes, and sanitization. Run `python3 verify.py` in this directory for offline integrity, timing arithmetic, and test-outcome verification; no GPU or network is needed.

The evidence branch is separate from the PR implementation branch. Its parent is the tested implementation commit. This keeps the implementation diff focused and makes the full experiment record public without treating megabytes of historical logs as library changes.

## 1. Objective and acceptance criteria

The primary optimization objective was lower synchronized latency for repeated, fixed-shape ESM3-open inference on one B200, including the copy of a fresh sequence input. The accepted exact path had to retain all requested output values against the original BF16 reference on changing inputs. Historical batching optimized aggregate throughput; the optimizer experiment measured warm distributed training-step latency and allocation. Those are separate objectives and cannot be multiplied into one end-to-end speedup.

Most inference experiments use 76-residue ubiquitin, encoded as 78 tokens. Later shape tests use batches of repeated input variants and synthetic 512/1024-residue repeats. These are controlled fixtures, not a representative protein benchmark. Exact comparison means `torch.equal` on returned tensors. Earlier raw fields named `bitwise_*` refer to this value comparison, not a byte-level comparison. Approximate experiments retain their declared `rtol=atol=1e-2` gate; it was not widened to turn a failure into a pass.

For an A/B/A timing comparison, the reference is the mean of the before/after control medians. Latency reduction is `100 * (1 - candidate / reference)`; speedup is `reference / candidate`. These are different quantities. The report retains raw samples and outliers. Consecutive calls are not independent trials; sample quantiles are not confidence intervals. Protocol counts differ between experiment families and are stated in the appendices.

## 2. Issue and decision ledger

| ID | Observed problem or hypothesis | Solution tried | Evidence and decision | In PR #431? |
|---|---|---|---|---|
| H1 | Released BF16 model's raw `forward()` meets an FP32/BF16 mismatch | Use BF16 autocast; compare with public `logits()`, which already uses autocast | Raw call fails; autocast executes. A dtype boundary, not proof that Blackwell is unsupported | No duplicate fix; [#417](https://github.com/Biohub/esm/pull/417) addresses raw BF16 forward |
| H2 | Does the model execute across eight B200s? | Independent inference ranks and short NCCL/DDP training | Successful recorded ranks, finite losses, nonzero monitored updates, matching replicas; full inference aggregate is not recovered | No cluster/runtime patch |
| H3 | Serial small requests underuse one GPU | Batch eight equal-length requests using existing APIs | Logits 7.80×; structure throughput 4.26×, with stochastic coordinates differing | Documented usage, not a new batching implementation |
| H4/O1 | Public `logits()` cannot be captured because of a pageable CPU scalar transfer | Prototype preallocated scalar, then device-native scalar creation in `logits()` | Capture succeeds; changing inputs and all seven returned tensors match eager, including fresh-copy timing | **Yes: device-native confidence scalar and regression tests** |
| O2 | Every output was converted to FP32 before unrequested tracks were discarded | Convert only requested outputs | Six conversions removed for one-output requests; ~0–0.2% latency effect; measured peak allocation unchanged | **Yes: selective conversion and contract/hook tests** |
| O3 | Is replay mostly attention or matmul work? | Nsight Systems node traces and one Nsight Compute GEMM sample | Many small cast/copy/elementwise kernels remain; sampled small GEMM underfills SMs | Evidence informing recipes; no device-wide utilization claim |
| O4 | Broader compilation might remove most small kernels | Full-forward, transformer, strict transformer, ATEN-constrained transformer, strict FFN | Fullgraph tokenizer trace fails; regional variants are faster but fail original-BF16 tolerance | Opt-in experiments only; rejected as equivalent replacements |
| O5 | Can fusion be narrowed enough to retain values? | Compile only SwiGLU with strict numerics, outside graph capture | Exact outputs on tested inputs; 48 fewer kernels; 3.6–5.7% lower graph medians, repeated on a fresh host | **Yes: optional recipe, not a library default** |
| O6 | Would forced cuDNN or FA4 improve attention? | Force cuDNN; register tested FA4 SDPA route | Default already uses cuDNN; FA4 is 1.4–4.4% slower in measured cases and changes outputs | Diagnostic options and negative results; no backend default change |
| O7 | Would Blackwell MXFP8 FFN kernels help? | Quantize 96 FFN linears; compare eager and matched compiled FFNs | Active SM100 scaled matmul verified; slower or no gain, with output drift | Opt-in experiment; no default quantization |
| H5 | Optimizer launch/allocation overhead in eight-GPU training | PyTorch `AdamW(fused=True)` versus foreach | 6.58% lower warm-step latency; 8.42% lower peak allocation; different losses/final weights | Training-setting evidence, not ESM optimizer code |
| V1/V3 | Nine GPU tests fail with `no kernel image` | Replace SM80/90-only FA2 binary in a separate overlay with an official-source SM100 build | All original cases retained; final 59 GPU cases pass; source/assertions unchanged | Environment repair; no wheel or dependency-default change in this PR |
| V1 | Earlier type diagnostics came from a different environment | Run full native Linux locked typing/lint checks | Full-project `ty` passes. Optional TorchAO imports use three narrow unresolved-import annotations | Yes: targeted cookbook typing annotations, not blanket suppression |
| V2 | First performance probe used the wrong Python after resolving a venv symlink | Preserve the venv interpreter path with `.absolute()` | Missing-Torch harness failure retained; corrected run passes | Validation harness only |
| V3 | First raw FA2 diagnostic import cannot find `libc10.so` | Import Torch before the extension | Forward/backward probe and GPU suite pass | Probe correction only |
| V4 | Forge integration cannot construct a client without credentials | Run native CI and retain failures | Four API tests remain credential-blocked; no fake key or skipped assertions | Unresolved external integration gate |

See [HISTORICAL.md](HISTORICAL.md) for H1–H5, [OPTIMIZATIONS.md](OPTIMIZATIONS.md) for O1–O7, and sections 5–6 below for V1–V4. Changes to the local cloud launcher and cleanup helper are operational repairs, not additional ESM fixes.

## 3. Exact source and environment identities

| Stage | Source / checkpoint | Execution environment |
|---|---|---|
| Initial smoke | Released ESM 3.4.3; checkpoint recorded in [initial report](data/historical/initial/results/report.json) | PyTorch 2.11.0+cu130; one B200 |
| Historical cluster, batching, AdamW | Released model; source/harness recovery limits in [historical record](HISTORICAL.md) | PyTorch 2.15.0+cu132; one or eight B200s as specified |
| Capture fix | Base `fddac936fc921445da2d099a3e4f7ae928717292`; original patch `e4ada24f895e35c454f9ddcaed0b3fa9d1f8b0a2` | PyTorch 2.15.0+cu132; B200 |
| Optimization recipe publication | `6da7422bd5fd031a2aa7faaf0720898fa6ef8602`; recorded model SHA below | PyTorch 2.15.0+cu132, Triton 3.9.0, cuDNN 9.26.0.51 |
| Final test and performance rerun | **`d285ddeead83631bbe0bc6aa7c879eb5a8946ccb`**, unchanged through all three runs | Locked: Torch 2.11.0+cu130/Triton 3.6; performance: Torch 2.15.0+cu132/Triton 3.9 |
| Final GPU dependency repair | Same `d285dde`; official FA2 2.8.3.post2, source commit `49774b31c8d1172a5b28f72e9aed0f322bf4ccba` | Overlay against locked Torch 2.11/CUDA 13.0; native SM100 cubins |

Hardware was AWS `p6-b200.48xlarge` (eight B200s), compute capability 10.0; inference optimization comparisons each use one GPU. Driver: 595.112.07. A CUDA runtime version reported by Torch is distinct from the installed driver and from the NVCC version used to build an extension. These are recorded tested versions, not a claim that every component was the latest available release.

Pinned ESM3 checkpoint: `biohub/esm3-sm-open-v1@47f0545b2b6daf26a93439a3cd610f4f7f3d5478`. ESMC validation cache: `biohub/ESMC-300M@96a2d8d61654ac0c08881be44da6c867686449b9`. Final `esm/models/esm3.py` SHA-256: `913e6fad41315e24b5575982c3c54d1ebed8dc6fe10fb8442bc0a6ac3c5ab5ab`. Final `pixi.lock` SHA-256: `be02a79f924abae4a9eef714c302beb8e0fea8aa8681f6d229f7bcfb36f90409`.

Full version/source records: [locked environment](data/validation/run-1/locked-environment.json), [performance environment](data/validation/run-2/performance-environment.json), [source](data/validation/run-2/source.json), [FA2 overlay](data/validation/run-3/environment.json), [observed performance package freeze](data/validation/requirements-performance.txt). The freeze is not a hash-locked dependency specification.

## 4. Results reviewers can recompute

### Historical throughput and training

| Experiment | Control before / after | Candidate | Result | Qualification |
|---|---:|---:|---|---|
| Eight logits requests, one B200 | 269.119 / 263.336 ms serial | 34.111 ms batched | 7.805× throughput | Equal-length fixture; compared sequence logits exact |
| Eight structure requests, one B200 | 2442.700 / 2437.092 ms serial | 572.858 ms batched | 4.259× throughput | Different stochastic coordinates; biological quality not tested |
| Eight-GPU warm training step | 130.037 / 130.233 ms foreach AdamW | 121.574 ms fused AdamW | 6.579% lower latency | Median over five measured steps of the slowest rank per step; three warmups |
| Peak training allocation per GPU | 25.978 GiB foreach | 23.792 GiB fused | 8.417% lower allocation | Allocation, not total physical GPU memory; optimizer trajectories differ |

These measurements use existing batching/optimizer facilities. [HISTORICAL.md](HISTORICAL.md) links samples and numerical comparisons. The old 6.412 ms graph prototype was preliminary; the changing-input capture test below is the stronger result. The earlier quoted lower eager bound of 32.58 ms could not be recovered and is not used to recompute a headline claim here.

### Capture repair, including input copy

| Path, B1 / 76 residues | Median before / after or candidate |
|---|---:|
| Eager with fresh pinned-host input | 32.811386 / 32.820947 ms |
| Graph replay with fixed input | 6.261778 / 6.262041 ms |
| Graph replay with fresh device-input copy | 6.270944 ms |
| Graph replay with fresh pinned-host input copy | **6.269471 ms** |

The fresh-host-input path is **5.234×** faster than the average eager control. Its increment relative to the average fixed-input graph control is **7.561 microseconds**, about 0.12% of the complete copied-input call. This is a difference of measured medians for a 624-byte input, not a general transfer-bandwidth benchmark. Four changing inputs and all six heads plus embeddings are checked. [Raw timing and correctness record](data/capture/patched.json); [original capture failure](data/capture/stock-capture.json).

### Strict SwiGLU inside the graph

| Shape | Original graph controls → candidate | Original reduction | Fresh-host controls → candidate | Fresh reduction |
|---|---:|---:|---:|---:|
| B1 / 76 residues | 6.290208 / 6.289974 → 6.061149 ms | 3.640% | 6.226198 / 6.227283 → 5.997263 ms | 3.685% |
| B8 / 76 residues | 9.231310 / 9.231370 → 8.707577 ms | 5.674% | 9.203096 / 9.203771 → 8.687181 ms | 5.609% |
| B1 / 512 synthetic residues | 11.085014 / 11.085286 → 10.629210 ms | 4.113% | 11.072819 / 11.072446 → 10.619242 ms | 4.095% |

All three shapes pass exact returned-value checks in both series. Each path has 20 warmups and 100 consecutive timed calls. The fresh run is a same-team replication on another host, not an independent study. First candidate calls cost roughly 2.3 seconds in the original series and 1.96–4.26 seconds in the rerun; this setup cost must be amortized. The original B1 candidate includes a 14.482 ms outlier, retained in the data, so no tail-latency improvement is claimed. [Original samples](data/swiglu-original/) and [fresh samples](data/swiglu-rerun/).

A separate node-level profile measured **2386 → 2338 kernels/replay**: 48 SiLU/multiply pairs become 48 fused SwiGLU kernels. The larger **2385 → 628** reduction belongs to a different, numerically rejected broad-compiler experiment. Those two results must not be combined.

### Negative and negligible results

- Selective FP32 conversion removes six copies for a one-output request, but measured graph latency changes only ~0–0.2%; B1 peak allocation stays 56,651,264 bytes.
- Broad transformer compilation reaches about 2.40–2.57 ms, but fails the original-BF16 numerical gate even with strict numerics/ATEN restrictions. Strict FFN-only compilation also fails.
- Forced cuDNN provides no gain because the default already selects it.
- Tested FA4 is 1.4–4.4% slower across the tested shapes and changes outputs.
- Uncompiled MXFP8 is 65.7–81.8% slower; matched compiled comparisons are slower or effectively unchanged. Approximate output agreement does not establish biological quality.

[OPTIMIZATIONS.md](OPTIMIZATIONS.md) gives every candidate's controls, timing counts, correctness outcome, source, and raw-file link. Private probes that returned success after checking replay against their *own* eager configuration do not certify agreement with the original model. The final public recipe records the failed original-reference gates as `numerical_mismatch`.

## 5. Validation, including the failures

The final source commit stayed fixed throughout the fresh validation. No ESM source change or test-assertion relaxation was used to turn the GPU suite green.

| Validation | Environment | Outcome | Evidence |
|---|---|---|---|
| Hosted CPU tests | Locked Ubuntu, Python 3.12 / 3.13 / 3.14 | Each: 592 passed, 3 skipped, 7 xfailed | [Native Actions run](https://github.com/CumulusLabsIO/esm/actions/runs/43403400414), [snapshot](data/validation/hosted-ci-summary.json) |
| Full native lint/typing | Locked Linux, including whole-project `ty` | Pass | [Hosted job](https://github.com/CumulusLabsIO/esm/actions/runs/43403400414/job/131811633226), [B200 lint log](data/validation/run-1/locked-lint.log) |
| B200-host CPU suite | Locked Torch 2.11 | 592 passed, 3 skipped, 7 xfailed | [JUnit](data/validation/run-1/locked-pytest.xml) |
| ESM3 capture/selection regression modules | Torch 2.11 and Torch 2.15 | 24 passed in each environment | [2.11 JUnit](data/validation/run-1/locked-esm3-regression.xml), [2.15 JUnit](data/validation/run-2/performance-esm3-regression.xml) |
| Initial GPU selection | Locked SM80/90 FA2 wheel | **40 passed, 9 failed**; 10 checkpoint cases deliberately excluded initially | [Failure JUnit](data/validation/run-1/locked-small-gpu.xml), [selection record](data/validation/run-1/small-gpu-selection.json) |
| Final GPU selection | Same source, FA2 SM100 overlay; offline cached checkpoints | **59 passed, 0 failed, 0 errors, 0 skipped**; 352 deselected by marker selection | [Final JUnit](data/validation/run-3/gpu-tests.xml), [log](data/validation/run-3/gpu-tests.log), [empty tracked diff](data/validation/run-3/tracked-diff-after.txt) |
| Public benchmark recipe | Torch 2.15 B200 | Default, sequence-only, embeddings-only, and three SwiGLU shapes: exact equality and passing status | [Summary](data/validation/run-2/validation-summary.json), [six records](data/swiglu-rerun/) |
| Required local contribution scripts | Unmodified checkout, local checkpoint mode | `raw_forwards.py`, `snippets/esm3.py`, `snippets/esmc.py` passed on B200 host | [Invocation records](data/validation/invocations/), [run summary](data/validation/run-2/summary.json) |
| Docker/Forge API tests | Hosted fork workflow; Docker installs published `esm>=3.4.1.post1,<4.0.0`, not this checkout | **Four failures: missing API credential before inference**; no integration execution of `d285dde` established | [Native Actions run](https://github.com/CumulusLabsIO/esm/actions/runs/43403400414) |
| Four tutorial notebooks | Not run | Open validation gap | [Contribution requirements](../../CONTRIBUTIONS.md) |

The 24 regression cases include CPU cases and four CUDA cases; they are not 24 separate GPU tests. The final 59-case GPU set contains **every original 49-case identity**, including all nine repaired failures, plus ten cached-checkpoint cases. `verify.py` checks this set inclusion and each outcome directly from the XML. JUnit represents xfails using skipped elements; the verifier reports the seven xfails separately.

### Why the nine GPU tests failed, and what fixed them

The repository's dev dependency points to a wheel labeled `py312-pt211-cu13-sm80-90`; B200 is SM100. All nine failures entered FA2 `fwd` or `varlen_fwd` with `CUDA error: no kernel image is available for execution on the device`. They were ESMC/ESMFold2 paths, separate from ESM3's BF16/autocast issue and the graph scalar-copy issue.

We built official **FA2 2.8.3.post2** source for SM100 against locked Torch 2.11/CUDA 13.0 in a separate overlay. Critical switches were `FLASH_ATTENTION_FORCE_BUILD=TRUE` and `FLASH_ATTN_CUDA_ARCHS=100`. The package supplies explicit architecture flags; `TORCH_CUDA_ARCH_LIST` alone does not control that build. The force-build flag prevents a prebuilt-wheel shortcut from silently replacing the intended compile.

The [build record](data/validation/fa2/build-command.txt), [toolchain](data/validation/fa2/toolchain.txt), [native cubin listing](data/validation/fa2/compiled-architectures.txt), and [direct forward/backward probe](data/validation/fa2/sm100-probe.json) precede the final suite. The BF16 attention probe's maximum absolute error against SDPA was 0.001953125 at `rtol=atol=1e-2`; Q/K/V gradients were finite. This probe checks tolerance, not exact equality. Build/install took about five minutes and did not replace the driver.

The source archive SHA-256 is `d0f9f0b08a986cdeadbffcf5ec8b3a8f514366c68ad1853d3ab061bc83cc340b`; the locally built wheel SHA-256 is `648bf33445c4f6fd45368bf1f35e5f05a2a9a85f42164fe62f4b88740342eda6`. The wheel is not redistributed here. [REPRODUCE.md](REPRODUCE.md) documents the source build and checks.

This was a binary-target repair, not evidence that FA2 only gained Blackwell support in this version. The repair changed both version and binary architecture, so it does not isolate architecture flags alone. Rebuilding the pinned version is an alternative that was not tested. ESM3 performance measurements used a separate Torch 2.15 environment without FA2 installed; their speedups must not be attributed to this rebuild.

## 6. Failed attempts and methodological corrections

| Attempt or earlier interpretation | What the record supports now |
|---|---|
| Fixed-input, replay-only graph prototype | Kept as exploratory history; superseded for the headline by four changing inputs, all outputs, and included H2D copy |
| Profile between A/B/A timing phases | This perturbed later controls. Initial attention/quantization records remain under [superseded](data/superseded/), excluded from headline comparisons; final runs profile after timing |
| Use instrumented timings as normal latency | Rejected. Nsight duration sums and sampled-kernel metrics are labeled separately from unprofiled synchronized wall time |
| Estimate FFN work from the block's default expansion | Incorrect: the stack passes `8/3`, giving width 4096. Earlier width-6144 FLOP/efficiency estimates are withdrawn |
| Broad compile graph matches its compiled eager outputs | Necessary for replay fidelity, insufficient for original-model agreement; original-reference failures remain failures |
| Treat fewer casts as a major speed/memory gain | Six copies disappear, but measured latency/peak-memory effects are negligible; claim narrowed |
| Dereference the performance venv Python symlink | Reached system Python and failed to import Torch. The corrected harness preserves the interpreter path; [failure retained](data/validation/run-1/performance-environment.log) |
| Import the FA2 extension before Torch in the direct probe | Failed to locate `libc10.so`; importing Torch first repaired the diagnostic. No model patch was required |
| Compare typing failures from mismatched environments to CI | Replaced with full checks under the exact Linux lock. A prior macOS native pytest abort remains unexplained; Linux/B200 results do not establish macOS support |
| EC2 launcher double-encoded user-data; cleanup wait expired early | Local operational fixes supplied raw user-data to AWS CLI and lengthened the cleanup wait. These are not library changes or performance results; no public standalone before/after launcher test is supplied |

The final saved cleanup receipt records termination and removal of temporary cloud resources at **2026-11-13 10:21:43 UTC**. Infrastructure identifiers and credentials are intentionally outside this public bundle. This is a recorded cleanup outcome, not a live cloud audit performed by the offline verifier.

## 7. Contribution scope and reviewer decisions

The implementation diff contains seven files: the scalar/conditional-conversion changes in `esm/models/esm3.py`, two regression test modules, the cookbook benchmark/guide/results, and a README link. It does not add custom CUDA kernels, alter weights, introduce a global graph cache, prune output-head computation, or change default precision/attention/compilation settings. Model and output-head hooks continue to receive the complete raw-forward outputs.

The scalar fix is the most direct upstream defect repair. Conditional output conversion is a small efficiency/contract-preserving change whose modest measurement is stated plainly. Strict SwiGLU is an optional recipe because compilation costs, compiler version, shapes, and equivalence checks matter. The diagnostic compiler/FA4/MXFP8 switches help others reproduce negative results; they are not advertised as faster safe defaults.

Related work remains separate: [#417](https://github.com/Biohub/esm/pull/417) covers raw BF16 forward; [#430](https://github.com/Biohub/esm/pull/430) covers padding when `sequence_id` is absent. Both were open when checked for this record. Equal-length batching here does not validate mixed-length padding, and #430's proposed `if is_pad.any()` would require another capture review. These results use the stated base and this PR, not those unmerged changes.

[CONTRIBUTIONS.md](../../CONTRIBUTIONS.md) was reviewed. Its Python 3.10 example conflicts with this checkout's `requires-python >=3.12`; validation followed the actual manifest and native CI lock. Required local scripts passed, hosted API validation remains credential-blocked, and notebooks remain unrun. The [upstream workflow](https://github.com/Biohub/esm/actions/runs/43402656061) was `action_required` with no jobs executed at publication review: maintainer approval is distinct from a failed test. The PR remains a draft.

## 8. Limits and next experiments

There is no Hopper head-to-head, B300 run, representative biological evaluation, mixed-length/multimodal benchmark, randomized independent-run confidence interval, or end-to-end service tail-latency study. The final B200 rerun did not repeat distributed training. Historical training is a short fixture-level execution/performance check, not convergence evidence. The precise operator responsible for broader compiler drift is not isolated.

The next useful steps are to reproduce on a representative sequence/structure set; isolate compiler drift one region at a time without relaxing correctness; evaluate captured graphs under real service concurrency; test the dependency packaging on another SM100 system; and compare B200/Hopper with the same pinned software, workload, correctness criterion, and power/performance conditions. These are proposals, not completed work.

## 9. Artifact coverage and omissions

The bundle publishes an explicit allowlist of small JSON, JUnit XML, profiler CSV summaries, text logs, generated-kernel text, and experiment source. Original failures, rejected candidates, outliers, and superseded runs are retained and labeled. Some historical source/sample files could not be recovered from local storage; [HISTORICAL.md](HISTORICAL.md) lists them. That prevents a claim of complete reproduction for the early private harness.

Weights, output tensor dumps, binary wheels, full Nsight binary/SQLite traces, credentials, SSH keys, token caches, and cloud resource state are not included. Profiler exports support count/timing review, but the complete original aggregation cannot be independently regenerated without the omitted traces. Probe sources and commands allow a fresh profile. Historical helpers retain generic `/home/ubuntu/...` workspace assumptions; use the portable public recipe and reproduction guide first.

Personal home paths, private host labels, resource identifiers, GPU UUIDs, and ANSI escapes were removed where present. Each transformed file has both original and public hashes in the manifest. Numerical samples, test identities/outcomes, and failure assertions are preserved. Hashes establish file identity and internal consistency; they do not independently prove a measurement happened, reproduce a binary build bit-for-bit, or substitute for rerunning the experiments.
