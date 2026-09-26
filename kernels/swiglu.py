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
    key=["n_cols"],
)
@triton.jit
def swiglu_kernel(
    x_ptr,
    gate_ptr,
    out_ptr,
    n_cols,
    stride_x_row,
    stride_gate_row,
    stride_out_row,
    BLOCK_SIZE: tl.constexpr,
):
    # Rows can be strided: in the MLP, x and gate are the two halves of each row of the fused
    # gate_up output, so a row's x starts 2 * n_cols after the previous row's
    pid_col = tl.program_id(axis=0)
    row = tl.program_id(axis=1)

    offsets = pid_col * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols

    x = tl.load(x_ptr + row * stride_x_row + offsets, mask=mask)
    gate = tl.load(gate_ptr + row * stride_gate_row + offsets, mask=mask)

    output = x * (gate * tl.sigmoid(gate.to(tl.float32)))

    tl.store(out_ptr + row * stride_out_row + offsets, output.to(tl.float16), mask=mask)


def swiglu_native(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    return x * torch.nn.functional.silu(gate)


def _rows(t: torch.Tensor) -> torch.Tensor:
    """(..., n) -> (rows, n) without copying. Raises if the leading dims can't be merged
    or the last dim isn't contiguous, rather than silently copying."""
    assert t.stride(-1) == 1, "swiglu needs a contiguous last dimension"
    return t.view(-1, t.shape[-1])


def swiglu_out(x: torch.Tensor, gate: torch.Tensor, output: torch.Tensor) -> None:
    """output = x * silu(gate). Inputs may be row-strided views (e.g. the gate/up halves
    of a fused projection); only the last dim needs to be contiguous."""
    x2, gate2, out2 = _rows(x), _rows(gate), _rows(output)
    n_rows, n_cols = out2.shape
    assert x2.shape == gate2.shape == out2.shape, (x.shape, gate.shape, output.shape)

    grid = lambda meta: (triton.cdiv(n_cols, meta["BLOCK_SIZE"]), n_rows)
    swiglu_kernel[grid](
        x2, gate2, out2, n_cols,
        x2.stride(0), gate2.stride(0), out2.stride(0),
    )


def swiglu(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(x)
    swiglu_out(x, gate, output)
    return output
