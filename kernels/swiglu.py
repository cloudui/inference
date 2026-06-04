"""
SwiGLU — PyTorch reference + Triton kernel

SwiGLU(x, gate) = x * silu(gate)
where silu(x) = x * sigmoid(x)

Used in Llama, Mistral, etc. as the FFN activation.
"""

import torch
import triton
import triton.language as tl


def swiglu_pytorch(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    return x * (gate * torch.nn.functional.sigmoid(gate))


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8),
    ],
    key=["n_elements"],
)
@triton.jit
def swiglu_kernel(
    x_ptr,
    gate_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    block_start = pid * BLOCK_SIZE

    # Create block pointers for input/gate/output tensors
    x_block_ptr = tl.make_block_ptr(
        base=x_ptr,
        shape=(n_elements,),
        strides=(1,),
        offsets=(block_start,),
        block_shape=(BLOCK_SIZE,),
        order=(0,),
    )
    gate_block_ptr = tl.make_block_ptr(
        base=gate_ptr,
        shape=(n_elements,),
        strides=(1,),
        offsets=(block_start,),
        block_shape=(BLOCK_SIZE,),
        order=(0,),
    )

    # Load using block pointer with boundary checking
    x = tl.load(x_block_ptr, boundary_check=(0,))
    gate = tl.load(gate_block_ptr, boundary_check=(0,))

    output = x * (gate * tl.sigmoid(gate.to(tl.float32)))

    out_block_ptr = tl.make_block_ptr(
        base=out_ptr,
        shape=(n_elements,),
        strides=(1,),
        offsets=(block_start,),
        block_shape=(BLOCK_SIZE,),
        order=(0,),
    )

    # Store using block pointer with boundary checking
    tl.store(out_block_ptr, output.to(tl.float16), boundary_check=(0,))


def swiglu_native(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    return x * torch.nn.functional.silu(gate)


def swiglu(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    x_flatten = x.view(-1)
    gate_flatten = gate.view(-1)
    output = torch.empty_like(x_flatten)

    n_elements = output.numel()

    # The grid is specified as a function of meta-parameters to adapt to the autotuned BLOCK_SIZE
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    swiglu_kernel[grid](x_flatten, gate_flatten, output, n_elements)

    return output.view(x.shape)
