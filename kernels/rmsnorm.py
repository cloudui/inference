"""
RMSNorm — PyTorch reference + Triton kernel

RMSNorm(x) = x / sqrt(mean(x^2) + eps) * weight
"""

import torch
import triton
import triton.language as tl


def rmsnorm_pytorch(
    x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    rms = torch.sqrt(x.pow(2).mean(dim=1, keepdim=True) + eps)
    x = (x / rms) * weight
    return x


def rmsnorm_native(
    x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    return torch.nn.functional.rms_norm(x, (x.shape[-1],), weight, eps)


def early_config_prune(configs, named_args, **kwargs):
    N = named_args["N"]
    # We must ensure BLOCK_SIZE is at least N, otherwise the thread block 
    # will not cover the entire dimension of the row.
    pruned = [c for c in configs if c.kwargs["BLOCK_SIZE"] >= N]
    if not pruned:
        # Fallback config when N is larger than our pre-defined block sizes
        fallback_block_size = triton.next_power_of_2(N)
        num_warps = 16 if fallback_block_size >= 4096 else 8
        return [triton.Config({"BLOCK_SIZE": fallback_block_size}, num_warps=num_warps)]
    return pruned


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=16),
        triton.Config({"BLOCK_SIZE": 8192}, num_warps=16),
    ],
    key=["N"],
    prune_configs_by={"early_config_prune": early_config_prune},
)
@triton.jit
def rmsnorm_kernel(
    x_ptr,
    weight_ptr,
    out_ptr,
    stride,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    block_start = pid * stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    offsets = block_start + col_offsets

    mask = col_offsets < N

    x = tl.load(x_ptr + offsets, mask=mask)
    weight = tl.load(weight_ptr + col_offsets, mask=mask)

    # Compute RMS norm
    rms = tl.sqrt(tl.sum(x * x) / N + eps)
    output = x / rms * weight

    tl.store(out_ptr + offsets, output, mask=mask)


def rmsnorm_triton(
    x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    N = x.shape[-1]
    stride = x.stride()[0]
    output = torch.empty_like(x)

    # Grid runs over row dimension
    grid = (x.shape[0],)
    
    # Launch kernel; BLOCK_SIZE is omitted as it will be selected by the autotuner
    rmsnorm_kernel[grid](x, weight, output, stride, N, eps)

    return output
