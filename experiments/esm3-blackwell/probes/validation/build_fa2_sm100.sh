#!/usr/bin/env bash
set -euo pipefail
build_root=/home/ubuntu/esm-ci/fa2-build
locked=/home/ubuntu/esm-ci/repo/.pixi/envs/dev
overlay=/home/ubuntu/esm-ci/blackwell-venv
mkdir -p "$build_root/wheels"
trap 'code=$?; printf "%s\n" "$code" > "$build_root/build.exit"; date -u +%FT%TZ > "$build_root/finished.txt"' EXIT
date -u +%FT%TZ > "$build_root/started.txt"
export CUDA_VISIBLE_DEVICES=''
export LD_PRELOAD="$locked/lib/libstdc++.so.6"
export DEBIAN_FRONTEND=noninteractive
sudo apt-get -o DPkg::Lock::Timeout=120 --no-install-recommends -y install \
    cuda-nvcc-13-0=13.0.88-1 cuda-cudart-dev-13-0=13.0.88-1 \
    cuda-cccl-13-0=13.0.85-1 cuda-cuobjdump-13-0=13.0.85-1
"$locked/bin/python" -m venv --system-site-packages "$overlay"
"$overlay/bin/python" -m pip install --no-deps ninja==1.13.2
export CUDA_HOME=/usr/local/cuda-13.0
export PATH="$overlay/bin:$CUDA_HOME/bin:$PATH"
export FLASH_ATTENTION_FORCE_BUILD=TRUE FLASH_ATTENTION_SKIP_CUDA_BUILD=FALSE
export FLASH_ATTN_CUDA_ARCHS=100 MAX_JOBS=48 NVCC_THREADS=2
unset TORCH_CUDA_ARCH_LIST FLASH_ATTENTION_FORCE_CXX11_ABI
cd "$build_root"
curl -fsSL https://files.pythonhosted.org/packages/6c/b1/b77625e3d11260d2bd6a4fc249a4be9a3da97ff848e2222e810c4f0e0211/flash_attn-2.8.3.post2.tar.gz -o flash_attn-2.8.3.post2.tar.gz
printf '%s\n' 'd0f9f0b08a986cdeadbffcf5ec8b3a8f514366c68ad1853d3ab061bc83cc340b  flash_attn-2.8.3.post2.tar.gz' > sdist.sha256
sha256sum -c sdist.sha256
tar -xzf flash_attn-2.8.3.post2.tar.gz
{
    nvcc --version
    g++ --version
    ninja --version
    "$overlay/bin/python" -c 'import sys,torch; print(sys.version); print("torch",torch.__version__,"cuda",torch.version.cuda,"cxx11abi",torch._C._GLIBCXX_USE_CXX11_ABI)'
    printf '%s\n' 'source_commit=49774b31c8d1172a5b28f72e9aed0f322bf4ccba' 'FLASH_ATTENTION_FORCE_BUILD=TRUE' 'FLASH_ATTN_CUDA_ARCHS=100' 'MAX_JOBS=48' 'NVCC_THREADS=2' 'CUDA_VISIBLE_DEVICES='
    dpkg-query -W 'cuda-*-13-0' 'libnvvm-13-0' 'libnvptxcompiler-13-0'
} > toolchain.txt
"$locked/bin/python" -c 'import importlib.util; print(importlib.util.find_spec("flash_attn_2_cuda").origin)' > locked-extension-path.txt
sha256sum "$(cat locked-extension-path.txt)" > locked-extension-before.sha256
printf '%s\n' "$overlay/bin/python -m pip wheel --no-build-isolation --no-deps --no-cache-dir --wheel-dir $build_root/wheels $build_root/flash_attn-2.8.3.post2" > build-command.txt
"$overlay/bin/python" -m pip wheel --no-build-isolation --no-deps --no-cache-dir \
    --wheel-dir "$build_root/wheels" "$build_root/flash_attn-2.8.3.post2" -v
sha256sum wheels/*.whl > wheel.sha256
"$overlay/bin/python" -m pip install --no-deps --no-index wheels/*.whl
"$overlay/bin/python" -c 'import importlib.util; print(importlib.util.find_spec("flash_attn_2_cuda").origin)' > overlay-extension-path.txt
cuobjdump --list-elf "$(cat overlay-extension-path.txt)" > compiled-architectures.txt
grep -q sm_100 compiled-architectures.txt
sha256sum -c locked-extension-before.sha256
date -u +%FT%TZ
