# Reproduce the B200 validation

Start with the [experiment report](README.md) for results and claim boundaries. [MANIFEST.json](MANIFEST.json) identifies the published artifacts and their hashes. This guide reproduces the validation at **`d285ddeead83631bbe0bc6aa7c879eb5a8946ccb`**, not whatever is currently on the branch.

There are three environments: the repository's unchanged CI lock; a separate PyTorch 2.15 performance environment; and a FlashAttention2 overlay on the locked environment. Keep them separate. The recorded versions are tested versions, not claims about the latest releases.

The commands below are portable recipes for a Linux x86-64 B200 machine. They have been derived from the executed scripts, with paths parameterized; the complete portable sequence has not itself been rerun on another GPU host. The files under `probes/validation/` are historical run scripts. Some contain the original `/home/ubuntu/esm-ci` paths, including the FA2 probe's output path. Do not run those path-dependent scripts unchanged on another machine. Portable equivalents appear below.

## Prerequisites and isolated workspace

Use a B200 with a working Blackwell-capable NVIDIA driver, Python 3.12 with `venv`, Git, curl, and standard Linux build tools. The recorded host used driver **595.112.07**. The graph tests and performance runs select one card; having eight cards installed does not make this a distributed test. The FA2 probe below requires compute capability **10.0** and does not validate B300.

Run this section in Bash from the repository containing this report. All later blocks use the same shell variables. Choose an empty workspace with room for two Python environments, checkpoints, and the FA2 compiler/build files. If model access requires authentication, use your own authorized Hugging Face credentials; no token belongs in logs or this evidence bundle.

```bash
set -euo pipefail
export ESM_EVIDENCE_DIR="$PWD/experiments/esm3-blackwell"
test -f "$ESM_EVIDENCE_DIR/MANIFEST.json"
export ESM_WORK="$(mktemp -d -t esm3-review.XXXXXXXX)"
export ESM_REPO="$ESM_WORK/repo"
export ESM_RESULTS="$ESM_WORK/results"
export ESM_PROBES="$ESM_EVIDENCE_DIR/probes/validation"
export HF_HOME="$ESM_WORK/hf-cache"
export HF_HUB_CACHE="$HF_HOME/hub"
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4
unset PYTHONPATH PYTEST_ADDOPTS SKIP INFRA_PROVIDER ESM_API_KEY
mkdir -p "$ESM_RESULTS" "$ESM_WORK/bin"

git clone https://github.com/CumulusLabsIO/esm.git "$ESM_REPO"
git -C "$ESM_REPO" checkout --detach d285ddeead83631bbe0bc6aa7c879eb5a8946ccb
test "$(git -C "$ESM_REPO" rev-parse HEAD)" = d285ddeead83631bbe0bc6aa7c879eb5a8946ccb
git -C "$ESM_REPO" diff --exit-code
(
  cd "$ESM_REPO"
  printf '%s\n' \
    '913e6fad41315e24b5575982c3c54d1ebed8dc6fe10fb8442bc0a6ac3c5ab5ab  esm/models/esm3.py' \
    'be02a79f924abae4a9eef714c302beb8e0fea8aa8681f6d229f7bcfb36f90409  pixi.lock' \
    | sha256sum --check -
)
```

The separate checkout prevents a later evidence-only commit or locally edited source from being mistaken for the tested model implementation. Do not test an installed release in place of this checkout.

## 1. Exact locked CI checks

The repository CI used **pixi 0.82.1**. The following downloads that release and verifies the archive checksum recorded by the original bootstrap:

```bash
curl --fail --location --retry 3 \
  https://github.com/prefix-dev/pixi/releases/download/v0.82.1/pixi-x86_64-unknown-linux-musl.tar.gz \
  --output "$ESM_WORK/pixi.tar.gz"
(
  cd "$ESM_WORK"
  printf '%s\n' '7ac72af79c50653bfcfddb2473b6e9eff2654affe0057d19fe21b0d8556a7fe2  pixi.tar.gz' \
    | sha256sum --check -
)
tar -xzf "$ESM_WORK/pixi.tar.gz" -C "$ESM_WORK/bin"
export ESM_PIXI="$ESM_WORK/bin/pixi"
"$ESM_PIXI" --version
cd "$ESM_REPO"
"$ESM_PIXI" install --locked -e dev
"$ESM_PIXI" run --locked -e dev lint-all \
  2>&1 | tee "$ESM_RESULTS/locked-lint.log"
CUDA_VISIBLE_DEVICES='' PYTEST_XDIST_AUTO_NUM_WORKERS=8 \
  "$ESM_PIXI" run --locked -e dev cov-test \
  2>&1 | tee "$ESM_RESULTS/locked-cpu.log"

"$ESM_PIXI" run --locked -e dev python "$ESM_PROBES/environment_probe.py" \
  --expected-torch 2.11.0 --expected-cuda 13.0 \
  --output "$ESM_RESULTS/locked-environment.json"
"$ESM_PIXI" run --locked -e dev python -m pytest -o addopts='' -n 0 -ra \
  --junitxml="$ESM_RESULTS/locked-esm3-regression.xml" \
  tests/models/esm3_cuda_graph_test.py tests/models/esm3_logits_selection_test.py \
  2>&1 | tee "$ESM_RESULTS/locked-esm3-regression.log"
```

The environment probe imports ESM3 from the checkout, records its hash and package versions, checks the GPU, and executes a finite BF16 matrix multiplication. It therefore tests a real CUDA operation, not only device enumeration.

Expected recorded results: full lint passed; CPU selection **592 passed, 3 skipped, 7 xfailed**; ESM3 modules **24 passed**, including four CUDA capture cases and CPU selective-output cases. CPU CI intentionally excludes GPU/manual markers. Clearing pytest `addopts` and running serially for the regression modules is necessary to include their GPU cases. The new modules are not 24 GPU tests.

Saved evidence: [`data/validation/run-1/`](data/validation/run-1/). The original fork's Python 3.12, 3.13, and 3.14 CPU jobs each reported the same counts; see the report for public Actions links. This local recipe reproduces the dev/Python 3.12 environment, not all three CI jobs at once.

## 2. Separate performance environment

The performance environment used **PyTorch 2.15.0+cu132**, CUDA runtime **13.2**, Triton **3.9.0**, and NumPy **2.5.4**. The strict SwiGLU experiment requires Inductor's `numerics` option. Do not upgrade the locked environment to obtain it.

[`requirements-performance.txt`](data/validation/requirements-performance.txt) is the captured installation freeze, not a hash-locked dependency specification. The recipe installs the exact CUDA Torch wheel first, uses the freeze for the other packages, then installs the separate tested checkout editable. The editable VCS line is replaced deliberately; it must not install another source checkout behind the reader's back.

```bash
python3.12 -m venv "$ESM_WORK/perf-venv"
export ESM_PERF_PYTHON="$ESM_WORK/perf-venv/bin/python"
"$ESM_PERF_PYTHON" -m pip install 'torch==2.15.0+cu132' \
  --index-url https://download.pytorch.org/whl/cu132
"$ESM_PERF_PYTHON" - <<'PY'
import os
from pathlib import Path
src = Path(os.environ['ESM_EVIDENCE_DIR']) / 'data/validation/requirements-performance.txt'
dst = Path(os.environ['ESM_WORK']) / 'requirements-performance-dependencies.txt'
lines = [line for line in src.read_text().splitlines()
         if line and not line.startswith('-e ') and not line.startswith('torch==')]
dst.write_text('\n'.join(lines) + '\n')
PY
"$ESM_PERF_PYTHON" -m pip install \
  -r "$ESM_WORK/requirements-performance-dependencies.txt"
"$ESM_PERF_PYTHON" -m pip install --no-deps -e "$ESM_REPO"
"$ESM_PERF_PYTHON" -m pip check
export PYTHONPATH="$ESM_REPO"
"$ESM_PERF_PYTHON" "$ESM_PROBES/environment_probe.py" \
  --expected-torch 2.15.0 --expected-cuda 13.2 --require-strict-numerics \
  --output "$ESM_RESULTS/performance-environment.json"
cd "$ESM_REPO"
"$ESM_PERF_PYTHON" -m pytest -o addopts='' -n 0 -ra \
  --junitxml="$ESM_RESULTS/performance-esm3-regression.xml" \
  tests/models/esm3_cuda_graph_test.py tests/models/esm3_logits_selection_test.py
```

Retain the venv interpreter path as written. Resolving its symlink with `Path.resolve()` can execute system Python outside the venv; that caused an initial harness failure and was corrected to `Path.absolute()`. `data/validation/run-1/performance-environment.log` preserves that failure. The rerun is in [`data/validation/run-2/`](data/validation/run-2/).

## 3. Pinned checkpoints and unchanged local scripts

Stage the recorded revisions before setting offline mode:

```bash
unset HF_HUB_OFFLINE
export ESMC_REVISION=96a2d8d61654ac0c08881be44da6c867686449b9
"$ESM_PERF_PYTHON" "$ESM_PROBES/stage_checkpoints.py" \
  --output "$ESM_RESULTS/checkpoints.json"
"$ESM_PERF_PYTHON" - <<'PY'
import json
import os
from pathlib import Path

records = json.loads((Path(os.environ['ESM_RESULTS']) / 'checkpoints.json').read_text())
assert records['esm3']['resolved_revision'] == '47f0545b2b6daf26a93439a3cd610f4f7f3d5478'
assert records['esmc']['resolved_revision'] == '96a2d8d61654ac0c08881be44da6c867686449b9'
# The unchanged ESMC snippet/fixture asks for the default revision. A download
# by commit alone need not create refs/main. Pin this isolated local cache alias
# to the recorded commit before offline execution; no remote ref is modified.
snapshot = Path(records['esmc']['directory'])
assert snapshot.parent.name == 'snapshots'
ref = snapshot.parent.parent / 'refs' / 'main'
ref.parent.mkdir(parents=True, exist_ok=True)
ref.write_text(records['esmc']['resolved_revision'])
PY
export HF_HUB_OFFLINE=1
```

The historical ESMC download requested `main` and recorded the resolved commit. This fresh-cache recipe explicitly pins that commit. The local `refs/main` assignment preserves the unchanged examples' resolution behavior while preventing model drift. Use the isolated cache created above, not a shared cache whose default ref other applications depend on. ESM3 is explicitly pinned by `stage_checkpoints.py`; the wrapper supplies its local `data` directory.

The wrapper records each script's SHA256 and arguments in `invocation.json`, changes only the working directory/environment, and invokes the script unchanged with `runpy`. It unsets `ESM_API_KEY` and selects local model loading; it does not patch model code, precision, or assertions.

```bash
for entry in \
  'smoke-raw-forwards:cookbook/local/raw_forwards.py' \
  'smoke-esm3:cookbook/snippets/esm3.py' \
  'smoke-esmc:cookbook/snippets/esmc.py'
do
  label="${entry%%:*}"
  script="${entry#*:}"
  "$ESM_PERF_PYTHON" "$ESM_PROBES/run_local_example.py" \
    --checkpoints "$ESM_RESULTS/checkpoints.json" \
    --workdir "$ESM_RESULTS/work/$label" -- "$ESM_REPO/$script" \
    2>&1 | tee "$ESM_RESULTS/$label.log"
done
```

All three scripts exited zero on the recorded B200 host. The ESMC loader defaults to CUDA if available, but the snippet's device was not separately instrumented; the explicit GPU suite below is the device-specific validation. Hugging Face offline mode prevents checkpoint refresh; the ESM3 examples can still fetch RCSB structures over the network.

## 4. Graph and strict SwiGLU measurements

Keep other GPU work idle. These are steady-state, same-shape inference measurements; they do not include model loading, compilation, graph capture, tokenization, or output D2H. They do include pinned H2D input copying, use four changing inputs, and compare control/candidate/control. Each timed path has 20 warmups and 100 samples. The optional compiler path must pass exact equality; a numerically different result is a failed candidate, not a speedup.

```bash
run_esm3_benchmark() {
  local label="$1"
  shift
  "$ESM_PERF_PYTHON" "$ESM_PROBES/run_local_example.py" \
    --checkpoints "$ESM_RESULTS/checkpoints.json" \
    --workdir "$ESM_RESULTS/work/$label" -- \
    "$ESM_REPO/cookbook/local/esm3_cuda_graph.py" \
    --warmup 20 --repeats 100 --output "$ESM_RESULTS/$label.json" "$@" \
    2>&1 | tee "$ESM_RESULTS/$label.log"
}
run_esm3_benchmark default-b1
run_esm3_benchmark sequence-b1 --outputs sequence
run_esm3_benchmark embeddings-b1 --outputs embeddings
run_esm3_benchmark swiglu-b1 --compile-swiglu
run_esm3_benchmark swiglu-b8 --compile-swiglu --batch-size 8
run_esm3_benchmark swiglu-l512 --compile-swiglu --length 512
```

Compare with [`data/swiglu-rerun/`](data/swiglu-rerun/). On the recorded host, strict SwiGLU reduced graph-plus-copy median latency by **3.685%** at B1/L76, **5.609%** at B8/L76, and **4.095%** at B1/L512. Compute the reduction as `1 - candidate / mean(control_before, control_after)`. The corresponding throughput increase is a different percentage. First-call costs and amortization estimates are in the JSON and must remain visible when evaluating whether compilation helps a real workload.

The default/sequence/embeddings runs validate those configurations; their candidate and control use the same output selection. They do not isolate the selective-output patch's performance effect. No FA2, TorchAO, or FA4 package was used by this six-case performance rerun. Its timings must not be attributed to the FA2 repair below.

## 5. Repair the locked FA2 wheel for the extended GPU suite

The locked manifest installs an FA2 2.7.4.post1 wheel from a `py312-pt211-cu13-sm80-90` release. Nine original tests failed with `no kernel image is available for execution on the device` on B200. The repair builds **FA2 2.8.3.post2 for SM100** in a separate overlay. It leaves the repository lock, original wheel, model code, and test assertions unchanged. This is a known working repair, not proof that a version upgrade was necessary: earlier FA2 source already had an SM100 build path, but rebuilding the earlier version was not tested here.

The recorded build used CUDA compiler **13.0.88**, g++ **13.3.0**, Ninja **1.13.2**, and the locked PyTorch **2.11.0+cu130**, C++11 ABI true. Have that toolchain installed before proceeding. On the original Ubuntu 24.04 host, with NVIDIA's CUDA package repository already configured, the compiler packages were installed using:

```bash
# Ubuntu 24.04 / configured NVIDIA APT repository only. No driver package.
sudo apt-get -o DPkg::Lock::Timeout=120 --no-install-recommends -y install \
  cuda-nvcc-13-0=13.0.88-1 cuda-cudart-dev-13-0=13.0.88-1 \
  cuda-cccl-13-0=13.0.85-1 cuda-cuobjdump-13-0=13.0.85-1
```

Other Linux distributions need equivalent compiler/runtime development packages. Do not assume these APT package names are portable. A wheel-provided CUDA runtime alone does not supply the compiler required for this source build.

```bash
export ESM_LOCKED="$ESM_REPO/.pixi/envs/dev"
export ESM_OVERLAY="$ESM_WORK/blackwell-venv"
export ESM_FA2_BUILD="$ESM_WORK/fa2-build"
export CUDA_HOME=/usr/local/cuda-13.0
export LD_PRELOAD="$ESM_LOCKED/lib/libstdc++.so.6"
mkdir -p "$ESM_FA2_BUILD/wheels"
"$ESM_LOCKED/bin/python" -m venv --system-site-packages "$ESM_OVERLAY"
"$ESM_OVERLAY/bin/python" -m pip install --no-deps ninja==1.13.2
export PATH="$ESM_OVERLAY/bin:$CUDA_HOME/bin:$PATH"
"$ESM_LOCKED/bin/python" - <<'PY' > "$ESM_FA2_BUILD/locked-extension.sha256"
import hashlib
import importlib.util
from pathlib import Path
p = Path(importlib.util.find_spec('flash_attn_2_cuda').origin)
print(hashlib.sha256(p.read_bytes()).hexdigest(), p)
PY
(
  cd "$ESM_FA2_BUILD"
  curl --fail --location --retry 3 \
    https://files.pythonhosted.org/packages/6c/b1/b77625e3d11260d2bd6a4fc249a4be9a3da97ff848e2222e810c4f0e0211/flash_attn-2.8.3.post2.tar.gz \
    --output flash_attn-2.8.3.post2.tar.gz
  printf '%s\n' \
    'd0f9f0b08a986cdeadbffcf5ec8b3a8f514366c68ad1853d3ab061bc83cc340b  flash_attn-2.8.3.post2.tar.gz' \
    | sha256sum --check -
  tar -xzf flash_attn-2.8.3.post2.tar.gz
  export CUDA_VISIBLE_DEVICES=''
  export FLASH_ATTENTION_FORCE_BUILD=TRUE FLASH_ATTENTION_SKIP_CUDA_BUILD=FALSE
  export FLASH_ATTN_CUDA_ARCHS=100 MAX_JOBS=48 NVCC_THREADS=2
  unset TORCH_CUDA_ARCH_LIST FLASH_ATTENTION_FORCE_CXX11_ABI
  nvcc --version
  g++ --version
  ninja --version
  "$ESM_OVERLAY/bin/python" - <<'PY'
import torch
assert torch.__version__ == '2.11.0+cu130', torch.__version__
assert torch.version.cuda == '13.0', torch.version.cuda
assert torch._C._GLIBCXX_USE_CXX11_ABI is True
print(torch.__version__, torch.version.cuda, torch._C._GLIBCXX_USE_CXX11_ABI)
PY
  "$ESM_OVERLAY/bin/python" -m pip wheel --no-build-isolation --no-deps --no-cache-dir \
    --wheel-dir "$ESM_FA2_BUILD/wheels" "$ESM_FA2_BUILD/flash_attn-2.8.3.post2" -v \
    2>&1 | tee "$ESM_FA2_BUILD/build.log"
  sha256sum wheels/*.whl > wheel.sha256
  "$ESM_OVERLAY/bin/python" -m pip install --no-deps --no-index wheels/*.whl
)
export CUDA_VISIBLE_DEVICES=0
ESM_EXTENSION="$("$ESM_OVERLAY/bin/python" -c \
  'import importlib.util; print(importlib.util.find_spec("flash_attn_2_cuda").origin)')"
"$CUDA_HOME/bin/cuobjdump" --list-elf "$ESM_EXTENSION" \
  | tee "$ESM_FA2_BUILD/compiled-architectures.txt"
grep -q sm_100 "$ESM_FA2_BUILD/compiled-architectures.txt"
sha256sum --check "$ESM_FA2_BUILD/locked-extension.sha256"
```

`MAX_JOBS=48` is the original host's build parallelism; reduce it on a smaller machine. Source controls CUDA architecture through `FLASH_ATTN_CUDA_ARCHS` and explicit compiler flags. A generic `TORCH_CUDA_ARCH_LIST` setting is not enough. `FLASH_ATTENTION_FORCE_BUILD` prevents a prebuilt-wheel shortcut from selecting an unsuitable binary. Use the overlay interpreter directly after installation; invoking pixi can reconcile the locked environment back to its pinned wheel.

Recorded build evidence is in [`data/validation/fa2/`](data/validation/fa2/). Original wheel SHA256 was `648bf33445c4f6fd45368bf1f35e5f05a2a9a85f42164fe62f4b88740342eda6`; installed extension SHA256 was `9c52426af8b1d6fda21ea7da4c58bf7e9606f63cfd7136d87e53c0084f98bc65`. Those identify the recorded artifacts; do not require a new build to match them bit-for-bit, because compilation paths/toolchain details can affect binary reproducibility. Verify your own architecture list, probe, and suite results.

This portable probe uses the same input, assertions, and import order as the historical `probe_fa2_sm100.py`, but writes into the chosen results directory:

```bash
"$ESM_OVERLAY/bin/python" - <<'PY'
import json
import os
from importlib import metadata
from pathlib import Path
import torch  # Must load Torch's shared libraries before the raw extension.
import torch.nn.functional as F
import flash_attn_2_cuda
from flash_attn import flash_attn_func

assert torch.cuda.get_device_capability(0) == (10, 0)
assert Path(flash_attn_2_cuda.__file__).is_relative_to(os.environ['ESM_OVERLAY'])
torch.manual_seed(100)
q, k, v = [torch.randn(2, 64, 4, 64, device='cuda', dtype=torch.bfloat16,
                       requires_grad=True) for _ in range(3)]
output = flash_attn_func(q, k, v, dropout_p=0.0, causal=False)
reference = F.scaled_dot_product_attention(
    q.detach().transpose(1, 2), k.detach().transpose(1, 2), v.detach().transpose(1, 2)
).transpose(1, 2)
torch.testing.assert_close(output, reference, rtol=1e-2, atol=1e-2)
output.float().square().mean().backward()
torch.cuda.synchronize()
assert torch.isfinite(output).all().item()
assert all(x.grad is not None and torch.isfinite(x.grad).all().item() for x in (q, k, v))
result = dict(status='passed', torch=torch.__version__, cuda=torch.version.cuda,
              flash_attn=metadata.version('flash-attn'),
              extension=flash_attn_2_cuda.__file__, gpu=torch.cuda.get_device_name(0),
              shape=list(q.shape), dtype=str(q.dtype),
              forward_vs_sdpa_max_abs=(output-reference).abs().max().item(),
              backward='finite q/k/v gradients')
(Path(os.environ['ESM_RESULTS']) / 'sm100-probe.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2))
PY
```

The recorded maximum absolute difference was **0.001953125**. This is tolerance-based agreement, not bitwise equality. Importing the raw extension before Torch originally produced a `libc10.so` error; the failed probe log is retained separately.

Now run every non-manual GPU test under `tests/models`, with no fixture-exclusion plugin:

```bash
cd "$ESM_REPO"
unset SKIP PYTEST_ADDOPTS ESM_API_KEY INFRA_PROVIDER
export HF_HUB_OFFLINE=1 ESM_TESTS_STRICT_ARTIFACTS=1
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 PYTHONPATH="$ESM_REPO"
"$ESM_OVERLAY/bin/python" -m pytest -o addopts='' -n 0 \
  -m 'gpu and not manual' -ra --junitxml="$ESM_RESULTS/gpu-tests.xml" tests/models \
  2>&1 | tee "$ESM_RESULTS/gpu-tests.log"
git diff --exit-code
```

The recorded final result is **59 passed, no failures/errors/skips**. Its JUnit includes every original 49 case and the nine previously failing cases now pass; ten checkpoint-dependent cases were added after staging. `ESM_TESTS_STRICT_ARTIFACTS=1` makes a missing checkpoint fail instead of hiding missing coverage behind an artifact skip. Compare [`run-1`](data/validation/run-1/) with [`run-3`](data/validation/run-3/). The original selection plugin is published for historical verification only; do not use it to reproduce the final 59-case claim.

## What these steps do not establish

The repository's [contribution instructions](../../CONTRIBUTIONS.md) were read. Their Python 3.10 setup example conflicts with the tested manifest's `requires-python >=3.12`; this guide follows the pinned source/CI environment. The three local contribution scripts ran, but the four `treon` notebook checks named in the contribution document were not completed. Notebook names in that document also need reconciliation with the pinned checkout before claiming compliance.

The hosted Forge/Biohub integration remains unvalidated: the fork's four Docker API tests fail during client construction because no API key is supplied. They are not reported as passing or silently skipped. A maintainer or authorized user with a valid key must run that hosted check and the applicable notebooks separately. The Docker integration installs a published package rather than this checkout, which also limits what that result can say about this PR.

The final GPU selection excludes `manual` tests and covers `tests/models`, not full published ESMFold2 validation or every repository test. This rerun does not establish B300 behavior, a B200/Hopper comparison, distributed training gains, biological task quality, or performance on variable-length/multimodal workloads. It also does not include a new Nsight session; historical profiler experiments have their own artifacts in the main report.

Keep new logs, JUnit, environment JSON, source hashes, checkpoint revisions, and raw timing samples. JUnit counts should distinguish `pytest.xfail` from ordinary skips. Before publishing, remove hostnames and private paths from copies while preserving outcomes, assertions, timings, and failure text; record all transformations and both original/published hashes. Do not include credentials or downloaded model weights. Use [MANIFEST.json](MANIFEST.json) to distinguish published artifacts from private operational files.
