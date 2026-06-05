import torch
import triton
import triton.language as tl
import math


@triton.jit
def flash_decode_generation_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    mid_o_ptr,
    mid_lse_ptr,
    seq_len,
    scale,
    head_dim,
    stride_q_head,
    stride_k_head,
    stride_k_seq,
    stride_mid_o_head,
    stride_mid_o_block,
    stride_mid_o_gqa,
    stride_mid_lse_head,
    stride_mid_lse_block,
    stride_mid_lse_gqa,
    BLOCK_SEQ_KV: tl.constexpr,
    BLOCK_HEAD_DIM: tl.constexpr,
    gqa_ratio: tl.constexpr,
):
    # BLOCK_GQA represents the maximum query heads per KV head grouped in one thread block.
    # It is hardcoded to 16 to match the GQA blocking size.
    BLOCK_GQA: tl.constexpr = 16

    # Program axes
    pid_head = tl.program_id(axis=0)  # Index of the KV head
    pid_kv = tl.program_id(axis=1)    # Index of the KV block

    # Offsets for dimensions
    offs_d = tl.arange(0, BLOCK_HEAD_DIM)
    offs_gqa = tl.arange(0, BLOCK_GQA)
    offs_kv = tl.arange(0, BLOCK_SEQ_KV)

    # Compute pointers for Q
    # Q shape: (k_heads * gqa_ratio, 1, head_dim) -> accessed as GQA-blocked rows
    q_start_offset = pid_head * gqa_ratio * stride_q_head
    q_row_offsets = q_start_offset + offs_gqa * stride_q_head
    q_offsets = q_row_offsets[:, None] + offs_d[None, :]
    q_mask = (offs_gqa < gqa_ratio)[:, None] & (offs_d < BLOCK_HEAD_DIM)[None, :]

    # Compute pointers for K and V
    # K/V shape: (k_heads, seq_len, head_dim)
    k_start_offset = pid_head * stride_k_head + pid_kv * BLOCK_SEQ_KV * stride_k_seq
    k_row_offsets = k_start_offset + offs_kv * stride_k_seq
    k_offsets = k_row_offsets[:, None] + offs_d[None, :]
    k_mask = (pid_kv * BLOCK_SEQ_KV + offs_kv < seq_len)[:, None] & (offs_d < head_dim)[None, :]

    # Load Q, K, V
    q = tl.load(q_ptr + q_offsets, mask=q_mask)
    k = tl.load(k_ptr + k_offsets, mask=k_mask)
    v = tl.load(v_ptr + k_offsets, mask=k_mask)

    # Compute attention scores
    acc = tl.zeros((BLOCK_GQA, BLOCK_HEAD_DIM), dtype=tl.float32)
    attn_scores = tl.dot(q, tl.trans(k)) * scale

    # Softmax calculation
    max_scores = tl.max(attn_scores, axis=-1)
    exp_scores = tl.exp(attn_scores - max_scores[:, None])
    sum_exp = tl.sum(exp_scores, axis=-1)
    lse = max_scores + tl.log(sum_exp)

    # Compute weighted sum
    acc = tl.dot(exp_scores.to(tl.float16), v, acc) / sum_exp[:, None]

    # Store intermediate accumulator (Mid_O)
    # Shape: (k_heads, n_blocks, gqa_ratio, head_dim)
    mid_o_start_offset = pid_head * stride_mid_o_head + pid_kv * stride_mid_o_block
    mid_o_rows = mid_o_start_offset + offs_gqa * stride_mid_o_gqa
    mid_o_offsets = mid_o_rows[:, None] + offs_d[None, :]
    mid_o_mask = (offs_gqa < gqa_ratio)[:, None] & (offs_d < BLOCK_HEAD_DIM)[None, :]
    tl.store(mid_o_ptr + mid_o_offsets, acc.to(tl.float16), mask=mid_o_mask)

    # Store intermediate LSE (Mid_LSE)
    # Shape: (k_heads, n_blocks, gqa_ratio)
    mid_lse_start_offset = pid_head * stride_mid_lse_head + pid_kv * stride_mid_lse_block
    mid_lse_offsets = mid_lse_start_offset + offs_gqa * stride_mid_lse_gqa
    mid_lse_mask = offs_gqa < gqa_ratio
    tl.store(mid_lse_ptr + mid_lse_offsets, lse, mask=mid_lse_mask)


@triton.jit
def flash_decode_reduce_kernel(
    mid_o_ptr,            # Intermediate output tensor from split-KV steps (Mid_O)
    mid_lse_ptr,          # Intermediate LSE tensor from split-KV steps (Mid_LSE)
    out_ptr,              # Final output tensor (O)
    gqa_ratio,            # Number of Q heads per KV head (for Grouped-Query Attention)
    n_blocks,             # Number of split-KV blocks that were reduced over
    stride_mid_o_head,    # Strides for the intermediate Output tensor
    stride_mid_o_gqa,
    stride_mid_o_block,
    stride_mid_lse_head,  # Strides for the intermediate LSE tensor
    stride_mid_lse_gqa,
    stride_mid_lse_block,
    stride_out_head,      # Stride for the final Output tensor
    BLOCK_HEAD_DIM: tl.constexpr,
):
    pid_q_head = tl.program_id(axis=0)

    # Determine which KV head and query subhead within the GQA group this thread processes
    kv_head_idx = pid_q_head // gqa_ratio
    q_subhead_idx = pid_q_head % gqa_ratio
    
    # Compute base pointer offset for reading intermediate output blocks
    mid_o_start_offset = kv_head_idx * stride_mid_o_head + q_subhead_idx * stride_mid_o_gqa
    offs_d = tl.arange(0, BLOCK_HEAD_DIM)

    # Initialize running softmax statistics (LSE accumulator and output accumulator)
    lse_accum = tl.full([1], float("-inf"), dtype=tl.float32)
    mid_lse_start_offset = kv_head_idx * stride_mid_lse_head + q_subhead_idx * stride_mid_lse_gqa
    out_accum = tl.zeros([BLOCK_HEAD_DIM], dtype=tl.float32)

    # Loop over all split-KV blocks to reduce them online
    for block_idx in tl.range(n_blocks):
        # Compute pointer offsets for the current block
        mid_o_row_offsets = mid_o_start_offset + block_idx * stride_mid_o_block
        mid_o_offsets = mid_o_row_offsets + offs_d

        # Load intermediate values for the current block
        block_acc = tl.load(mid_o_ptr + mid_o_offsets)
        block_lse = tl.load(mid_lse_ptr + mid_lse_start_offset + block_idx * stride_mid_lse_block)

        # Update Log-Sum-Exp
        max_lse = tl.maximum(lse_accum, block_lse)
        new_lse = max_lse + tl.log(tl.exp(lse_accum - max_lse) + tl.exp(block_lse - max_lse))

        # Re-scale running output accumulator and accumulate block output
        scale_accum = tl.exp(lse_accum - new_lse)
        scale_block = tl.exp(block_lse - new_lse)
        out_accum = scale_accum * out_accum + scale_block * block_acc

        lse_accum = new_lse

    # Write out final reduced values
    out_offsets = pid_q_head * stride_out_head + offs_d
    tl.store(out_ptr + out_offsets, out_accum.to(tl.float16))
    

def flash_decode(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Computes Grouped-Query Attention (GQA) using a Split-KV flash decoding approach.
    
    Args:
        q: Query tensor of shape (q_heads, 1, head_dim)
        k: Key tensor of shape (k_heads, seq_len, head_dim)
        v: Value tensor of shape (k_heads, seq_len, head_dim)
        
    Returns:
        A tuple of (mid_o, out):
            mid_o: Intermediate accumulator tensor of shape (k_heads, n_blocks, gqa_ratio, head_dim)
            out: Final reduced output tensor of shape (q_heads, 1, head_dim)
    """
    q_heads = q.shape[0]
    k_heads, seq_len, head_dim = k.shape
    
    BLOCK_SEQ_KV = 64
    n_blocks = triton.cdiv(seq_len, BLOCK_SEQ_KV)
    gqa_ratio = q_heads // k_heads
    
    # Allocate intermediate output and LSE buffers
    mid_o = torch.zeros(
        (k_heads, n_blocks, gqa_ratio, head_dim), 
        device=q.device, 
        dtype=torch.float16
    )
    mid_lse = torch.zeros(
        (k_heads, n_blocks, gqa_ratio), 
        device=q.device, 
        dtype=torch.float16
    )
    
    scale = 1 / math.sqrt(head_dim)
    
    # Launch Generation Kernel
    grid_gen = (k_heads, n_blocks)
    stride_mid_o_head, stride_mid_o_block, stride_mid_o_gqa, _ = mid_o.stride()
    stride_mid_lse_head, stride_mid_lse_block, stride_mid_lse_gqa = mid_lse.stride()
    
    flash_decode_generation_kernel[grid_gen](
        q,
        k,
        v,
        mid_o,
        mid_lse,
        seq_len,
        scale,
        head_dim,
        q.stride()[0],
        k.stride()[0],
        k.stride()[1],
        stride_mid_o_head,
        stride_mid_o_block,
        stride_mid_o_gqa,
        stride_mid_lse_head,
        stride_mid_lse_block,
        stride_mid_lse_gqa,
        BLOCK_SEQ_KV,
        head_dim,
        gqa_ratio,
    )

    torch.cuda.synchronize()
    
    # Launch Reduction Kernel
    out = torch.zeros((q_heads, 1, head_dim), device=q.device, dtype=torch.float16)
    grid_reduce = (q_heads,)
    
    flash_decode_reduce_kernel[grid_reduce](
        mid_o,
        mid_lse,
        out,
        gqa_ratio,
        n_blocks,
        stride_mid_o_head,
        stride_mid_o_gqa,
        stride_mid_o_block,
        stride_mid_lse_head,
        stride_mid_lse_gqa,
        stride_mid_lse_block,
        out.stride()[0],
        head_dim,
    )
    
    # return mid_o
    return out