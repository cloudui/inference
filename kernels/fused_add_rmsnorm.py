"""
RMSNorm — PyTorch reference + Triton kernel

RMSNorm(x) = x / sqrt(mean(x^2) + eps) * weight
"""

import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
        triton.Config({}, num_warps=16),
    ],
    key=["N"],
)
@triton.jit
def fused_add_rmsnorm_kernel(
    x_ptr,
    weight_ptr,
    hidden_states_ptr, # save residual post-add for next res add
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
    hidden_states_block_ptr = tl.make_block_ptr(
        base=hidden_states_ptr + pid_batch * stride_batch + pid_row * stride_row,
        shape=(N,),
        strides=(1,),
        offsets=(0,),
        block_shape=(BLOCK_SIZE,),
        order=(0,),
    )

    # Load block pointer elements (with boundary checking to handle trailing elements safely)
    x = tl.load(x_block_ptr, boundary_check=(0,))
    weight = tl.load(weight_block_ptr, boundary_check=(0,))
    hidden_states = tl.load(hidden_states_block_ptr, boundary_check=(0,))

    x = hidden_states + x
    tl.store(hidden_states_block_ptr, x, boundary_check=(0,))

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
    tl.store(out_block_ptr, output.to(tl.float16), boundary_check=(0,))

def fused_add_rmsnorm_out(
    x: torch.Tensor, 
    weight: torch.Tensor, 
    hidden_states: torch.Tensor,
    output: torch.Tensor, 
    eps: float = 1e-6
) -> None:
    N = x.shape[-1]
    stride_batch, stride_row, _ = x.stride()
    grid = (x.shape[0], x.shape[1])
    BLOCK_SIZE = triton.next_power_of_2(N)
    fused_add_rmsnorm_kernel[grid](
        x, weight, hidden_states, output, stride_batch, stride_row, N, eps, BLOCK_SIZE=BLOCK_SIZE
    )

def fused_add_rmsnorm(
    x: torch.Tensor, weight: torch.Tensor, hidden_states: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    output = torch.empty_like(x)
    fused_add_rmsnorm_out(x, weight, hidden_states, output, eps)
    return output
