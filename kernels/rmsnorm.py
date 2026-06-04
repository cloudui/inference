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
    stride_batch,
    stride_row,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid_batch = tl.program_id(axis=0)
    pid_row = tl.program_id(axis=1)

    # Create block pointers for 1D row slices
    x_block_ptr = tl.make_block_ptr(
        base=x_ptr + pid_batch * stride_batch + pid_row * stride_row,
        shape=(N,),
        strides=(1,),
        offsets=(0,),
        block_shape=(BLOCK_SIZE,),
        order=(0,),
    )
    weight_block_ptr = tl.make_block_ptr(
        base=weight_ptr,
        shape=(N,),
        strides=(1,),
        offsets=(0,),
        block_shape=(BLOCK_SIZE,),
        order=(0,),
    )

    # Load block pointer elements (with boundary checking to handle trailing elements safely)
    x = tl.load(x_block_ptr, boundary_check=(0,))
    weight = tl.load(weight_block_ptr, boundary_check=(0,))

    # Compute RMS norm
    rms = tl.sqrt(tl.sum(x * x) / N + eps)
    output = x / rms * weight

    # Create block pointer for output
    out_block_ptr = tl.make_block_ptr(
        base=out_ptr + pid_batch * stride_batch + pid_row * stride_row,
        shape=(N,),
        strides=(1,),
        offsets=(0,),
        block_shape=(BLOCK_SIZE,),
        order=(0,),
    )

    # Store computed values using block pointer
    tl.store(out_block_ptr, output, boundary_check=(0,))


def rmsnorm(
    x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    N = x.shape[-1]
    stride_batch, stride_row, _ = x.stride()
    output = torch.empty_like(x)

    # Grid runs over row dimension
    grid = (x.shape[0], x.shape[1])
    
    # Launch kernel; BLOCK_SIZE is omitted as it will be selected by the autotuner
    rmsnorm_kernel[grid](x, weight, output, stride_batch, stride_row, N, eps)

    return output
