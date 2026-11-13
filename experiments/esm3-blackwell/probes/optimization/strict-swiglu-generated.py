# AOT ID: ['0_inference']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
from torch._C._dynamo.guards import copy_if_misaligned
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
assert_size_stride_grouped = torch._C._dynamo.guards.assert_size_stride_grouped
assert_alignment = torch._C._dynamo.guards.assert_alignment
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cpu_pinned = torch._C._dynamo.guards._empty_strided_cpu_pinned
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
empty_strided_mtia = torch._C._dynamo.guards._empty_strided_mtia
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p


# kernel path: /home/ubuntu/esmopt/swiglu-strict-cache/f6/cf6emuez6uk7wjnsqum45czefgmfusox4ax4inhgniowttdoijnk.py
# Topologically Sorted Source Nodes: [chunk, silu, mul], Original ATen: [aten.split, aten.silu, aten.mul]
# Source node to ATen node mapping:
#   chunk => split
#   mul => mul
#   silu => add, convert_element_type, convert_element_type_1, div, exp, neg
# Graph fragment:
#   %arg0_1 : Tensor "bf16[1, 78, 8192][638976, 8192, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %split : [num_users=2] = call_function[target=torch.ops.aten.split.Tensor](args = (%arg0_1, 4096, -1), kwargs = {})
#   %convert_element_type : Tensor "f32[1, 78, 4096][319488, 4096, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%getitem, torch.float32), kwargs = {})
#   %neg : Tensor "f32[1, 78, 4096][319488, 4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%convert_element_type,), kwargs = {})
#   %exp : Tensor "f32[1, 78, 4096][319488, 4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%neg,), kwargs = {})
#   %add : Tensor "f32[1, 78, 4096][319488, 4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%exp, 1), kwargs = {})
#   %div : Tensor "f32[1, 78, 4096][319488, 4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%convert_element_type, %add), kwargs = {})
#   %convert_element_type_1 : Tensor "bf16[1, 78, 4096][319488, 4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div, torch.bfloat16), kwargs = {})
#   %mul : Tensor "bf16[1, 78, 4096][319488, 4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_1, %getitem_1), kwargs = {})
#   return %mul
triton_poi_fused_mul_silu_split_0 = async_compile.triton('triton_poi_fused_mul_silu_split_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': False, 'launch_pdl': False, 'disable_ftz': True, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_mul_silu_split_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': (), 'tiling_scores': {'x': 2555904}, 'backend_hash': '99E9D6A98A618436B20C92DD6DBC71A1485013C9E23985AAA309B1A75439FF4A', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_silu_split_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 319488
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = (xindex % 4096)
    x1 = xindex // 4096
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 8192*x1), None).to(tl.float32)
    tmp13 = tl.load(in_ptr0 + (4096 + x0 + 8192*x1), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = -tmp1
    tmp3 = libdevice.exp(tmp2)
    tmp4 = tl.full([1], 1.0, tl.float32)
    tmp5 = tmp3 + tmp4
    tmp6 = triton.language.div_rn(tmp1, tmp5)
    tmp7 = tmp6.to(tl.bfloat16)
    tmp8 = tmp7.to(tl.float32)
    tmp9 = tmp8.to(tl.bfloat16)
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tmp10.to(tl.bfloat16)
    tmp12 = tmp11.to(tl.float32)
    tmp14 = tmp13.to(tl.bfloat16)
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp12 * tmp15
    tmp17 = tmp16.to(tl.bfloat16)
    tmp18 = tmp17.to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp18, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

class Runner:
    def __init__(self, partitions):
        self.partitions = partitions

    def recursively_apply_fns(self, fns):
        new_callables = []
        for fn, c in zip(fns, self.partitions):
            new_callables.append(fn(c))
        self.partitions = new_callables

    def call(self, args):
        arg0_1, = args
        args.clear()
        assert_size_stride(arg0_1, (1, 78, 8192), (638976, 8192, 1), 'input')
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            arg0_1 = copy_if_misaligned(arg0_1)
            buf0 = empty_strided_cuda((1, 78, 4096), (319488, 4096, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [chunk, silu, mul], Original ATen: [aten.split, aten.silu, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_mul_silu_split_0.run(arg0_1, buf0, 319488, stream=raw_stream0)
            del arg0_1
        return (buf0, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((1, 78, 8192), (638976, 8192, 1), device='cuda:0', dtype=torch.bfloat16)
    return [arg0_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='cuda')


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
