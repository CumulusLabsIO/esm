"""One explicit SM100 FlashAttention2 forward/backward after the overlay build."""

from importlib import metadata
import json
from pathlib import Path
import time

import torch
import torch.nn.functional as F
import flash_attn_2_cuda
from flash_attn import flash_attn_func

started = time.time()
assert torch.cuda.is_available()
assert torch.cuda.get_device_capability(0) == (10, 0)
torch.manual_seed(100)
q, k, v = [
    torch.randn(2, 64, 4, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    for _ in range(3)
]
output = flash_attn_func(q, k, v, dropout_p=0.0, causal=False)
reference = F.scaled_dot_product_attention(
    q.detach().transpose(1, 2), k.detach().transpose(1, 2), v.detach().transpose(1, 2)
).transpose(1, 2)
torch.testing.assert_close(output, reference, rtol=1e-2, atol=1e-2)
output.float().square().mean().backward()
torch.cuda.synchronize()
assert torch.isfinite(output).all().item()
for value in (q, k, v):
    assert value.grad is not None and torch.isfinite(value.grad).all().item()
result = {
    "status": "passed",
    "torch": torch.__version__,
    "cuda_runtime": torch.version.cuda,
    "flash_attn": metadata.version("flash-attn"),
    "extension": flash_attn_2_cuda.__file__,
    "gpu": torch.cuda.get_device_name(0),
    "compute_capability": torch.cuda.get_device_capability(0),
    "shape": list(q.shape),
    "dtype": str(q.dtype),
    "forward_vs_sdpa_max_abs": (output - reference).abs().max().item(),
    "backward": "finite q/k/v gradients",
    "elapsed_seconds": time.time() - started,
}
Path("/home/ubuntu/esm-ci/fa2-build/sm100-probe.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
