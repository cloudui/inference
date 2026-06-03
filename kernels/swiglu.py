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
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    gate = tl.load(gate_ptr + offsets, mask=mask)

    output = x * (gate * tl.sigmoid(gate.to(tl.float32)))

    tl.store(out_ptr + offsets, output.to(tl.float16), mask=mask)


def swiglu_native(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    return x * torch.nn.functional.silu(gate)


def swiglu_triton(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    x_flatten = x.view(-1)
    gate_flatten = gate.view(-1)
    output = torch.empty_like(x_flatten)

    n_elements = output.numel()

    # The grid is specified as a function of meta-parameters to adapt to the autotuned BLOCK_SIZE
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    swiglu_kernel[grid](x_flatten, gate_flatten, output, n_elements)

    return output.view(x.shape)
