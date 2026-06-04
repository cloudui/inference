"""
Fused RMSNorm + SwiGLU — single Triton kernel

Fuses normalization and activation into one kernel to halve HBM accesses:
  Separate: load x -> normalize -> store | load normalized x -> SwiGLU -> store
  Fused:    load x -> normalize -> SwiGLU -> store
"""

import torch
import triton
import triton.language as tl


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
def fused_rmsnorm_swiglu_kernel(
    x_ptr,
    gate_ptr,
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

    # Create block pointers for input rows, weight, and gate
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
    gate_block_ptr = tl.make_block_ptr(
        base=gate_ptr + pid_batch * stride_batch + pid_row * stride_row,
        shape=(N,),
        strides=(1,),
        offsets=(0,),
        block_shape=(BLOCK_SIZE,),
        order=(0,),
    )

    # Load block pointer elements with boundary checking
    x = tl.load(x_block_ptr, boundary_check=(0,))
    weight = tl.load(weight_block_ptr, boundary_check=(0,))
    gate = tl.load(gate_block_ptr, boundary_check=(0,))

    # Compute RMSNorm
    rms = tl.sqrt(tl.sum(x * x) / N + eps)
    x_norm = x / rms * weight

    # Compute SwiGLU activation on normalized input
    output = x_norm * (gate * tl.sigmoid(gate.to(tl.float32)))

    # Create block pointer for output
    out_block_ptr = tl.make_block_ptr(
        base=out_ptr + pid_batch * stride_batch + pid_row * stride_row,
        shape=(N,),
        strides=(1,),
        offsets=(0,),
        block_shape=(BLOCK_SIZE,),
        order=(0,),
    )

    # Store using block pointer with boundary checking
    tl.store(out_block_ptr, output.to(tl.float16), boundary_check=(0,))


def fused_rmsnorm_swiglu(
    x: torch.Tensor, gate: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    N = x.shape[-1]
    stride_batch, stride_row, _ = x.stride()
    output = torch.empty_like(x)

    # Grid runs over row dimension
    grid = (x.shape[0], x.shape[1])
    
    # Launch kernel; BLOCK_SIZE is selected by the autotuner
    fused_rmsnorm_swiglu_kernel[grid](
        x, gate, weight, output, stride_batch, stride_row, N, eps
    )

    return output
