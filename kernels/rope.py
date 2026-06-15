import torch
import triton
import triton.language as tl

@triton.jit
def rope_decode_kernel(
    x_ptr,
    cos_ptr,
    sin_ptr,
    out_ptr,
    stride_batch,
    stride_head,
    stride_row,
    N_HEADS, 
    HEAD_DIM: tl.constexpr
):
    pid_bh = tl.program_id(axis=0)
    pid_row = tl.program_id(axis=1)

    batch_idx = pid_bh // N_HEADS
    head_idx = pid_bh % N_HEADS

    # Create block pointers for 1D row slices
    offsets = tl.arange(0, HEAD_DIM//2)

    x_start = batch_idx * stride_batch + head_idx * stride_head + pid_row * stride_row

    x_offsets_top = x_start + offsets
    x_offsets_bottom = x_offsets_top + HEAD_DIM // 2

    x_top = tl.load(x_ptr + x_offsets_top)
    x_bottom = tl.load(x_ptr + x_offsets_bottom)

    cos = tl.load(cos_ptr + offsets)
    sin = tl.load(sin_ptr + offsets)

    # ROPE
    output_top = (x_top * cos) + (-x_bottom * sin)
    output_bottom = (x_bottom * cos) + (x_top * sin)

    # Store computed values using block pointer
    tl.store(out_ptr + x_offsets_top, output_top)
    tl.store(out_ptr + x_offsets_bottom, output_bottom.to(tl.float16))

def apply_rope_decode(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    output = torch.zeros_like(x)

    n_batches, n_heads, seqlen, head_dim = x.shape
    grid = (n_batches*n_heads, seqlen)

    stride_batch, stride_head, stride_row, _ = x.stride()

    rope_decode_kernel[grid](
        x,
        cos,
        sin,
        output,
        stride_batch,
        stride_head,
        stride_row,
        n_heads,
        head_dim, # almost always 64, 128 BLOCK_SIZE
    )

    return output
