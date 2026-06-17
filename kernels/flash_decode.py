import torch
import triton
import triton.language as tl
import math


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SEQ_KV": 32}, num_warps=4),
        triton.Config({"BLOCK_SEQ_KV": 64}, num_warps=4),
        triton.Config({"BLOCK_SEQ_KV": 128}, num_warps=4),
        triton.Config({"BLOCK_SEQ_KV": 256}, num_warps=4),
    ],
    key=["head_dim"],
)
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
    stride_q_batch,
    stride_q_head,
    stride_k_batch,
    stride_k_head,
    stride_k_seq,
    stride_mid_o_batch,
    stride_mid_o_head,
    stride_mid_o_block,
    stride_mid_o_gqa,
    stride_mid_lse_batch,
    stride_mid_lse_head,
    stride_mid_lse_block,
    stride_mid_lse_gqa,
    BLOCK_SEQ_KV: tl.constexpr,
    BLOCK_HEAD_DIM: tl.constexpr,
    gqa_ratio: tl.constexpr,
):
    # BLOCK_GQA represents the GQA dimension size for tl.dot tensor core requirements.
    BLOCK_GQA: tl.constexpr = 16

    # Program axes
    pid_batch = tl.program_id(axis=0)  # Index of the Batch
    pid_head = tl.program_id(axis=1)   # Index of the KV head
    pid_kv = tl.program_id(axis=2)     # Index of the KV block

    # Base pointers with batch offsets
    q_batch_ptr = q_ptr + pid_batch * stride_q_batch
    k_batch_ptr = k_ptr + pid_batch * stride_k_batch
    v_batch_ptr = v_ptr + pid_batch * stride_k_batch
    mid_o_batch_ptr = mid_o_ptr + pid_batch * stride_mid_o_batch
    mid_lse_batch_ptr = mid_lse_ptr + pid_batch * stride_mid_lse_batch

    # Create block pointer for Q
    q_start_ptr = q_batch_ptr + pid_head * gqa_ratio * stride_q_head
    q_block_ptr = tl.make_block_ptr(
        base=q_start_ptr,
        shape=(gqa_ratio, head_dim),
        strides=(stride_q_head, 1),
        offsets=(0, 0),
        block_shape=(BLOCK_GQA, BLOCK_HEAD_DIM),
        order=(1, 0)
    )

    # Create block pointers for K and V
    k_start_ptr = k_batch_ptr + pid_head * stride_k_head
    v_start_ptr = v_batch_ptr + pid_head * stride_k_head

    k_block_ptr = tl.make_block_ptr(
        base=k_start_ptr,
        shape=(seq_len, head_dim),
        strides=(stride_k_seq, 1),
        offsets=(pid_kv * BLOCK_SEQ_KV, 0),
        block_shape=(BLOCK_SEQ_KV, BLOCK_HEAD_DIM),
        order=(1, 0)
    )
    v_block_ptr = tl.make_block_ptr(
        base=v_start_ptr,
        shape=(seq_len, head_dim),
        strides=(stride_k_seq, 1),
        offsets=(pid_kv * BLOCK_SEQ_KV, 0),
        block_shape=(BLOCK_SEQ_KV, BLOCK_HEAD_DIM),
        order=(1, 0)
    )

    # Load blocks using boundary checks
    q = tl.load(q_block_ptr, boundary_check=(0, 1))
    k = tl.load(k_block_ptr, boundary_check=(0, 1))
    v = tl.load(v_block_ptr, boundary_check=(0, 1))

    # Compute attention scores
    acc = tl.zeros((BLOCK_GQA, BLOCK_HEAD_DIM), dtype=tl.float32)
    attn_scores = tl.dot(q, tl.trans(k)) * scale

    # Mask rows attention to -inf (seqlen mod BLOCK_SEQ_KV != 0)
    # for out-of-bounds rows. exp(-inf) = 0
    kv_indices = pid_kv * BLOCK_SEQ_KV + tl.arange(0, BLOCK_SEQ_KV)
    kv_mask = kv_indices[None, :] < seq_len
    attn_scores = tl.where(kv_mask, attn_scores, float("-inf"))

    # Softmax calculation
    max_scores = tl.max(attn_scores, axis=-1)
    exp_scores = tl.exp(attn_scores - max_scores[:, None])
    sum_exp = tl.sum(exp_scores, axis=-1)
    lse = max_scores + tl.log(sum_exp)

    # Compute weighted sum
    acc = tl.dot(exp_scores.to(tl.float16), v, acc) / sum_exp[:, None]

    # Store intermediate accumulator (Mid_O)
    mid_o_start_ptr = mid_o_batch_ptr + pid_head * stride_mid_o_head + pid_kv * stride_mid_o_block
    mid_o_block_ptr = tl.make_block_ptr(
        base=mid_o_start_ptr,
        shape=(gqa_ratio, BLOCK_HEAD_DIM),
        strides=(stride_mid_o_gqa, 1),
        offsets=(0, 0),
        block_shape=(BLOCK_GQA, BLOCK_HEAD_DIM),
        order=(1, 0)
    )
    tl.store(mid_o_block_ptr, acc.to(tl.float16), boundary_check=(0, 1))

    # Store intermediate LSE (Mid_LSE)
    mid_lse_start_ptr = mid_lse_batch_ptr + pid_head * stride_mid_lse_head + pid_kv * stride_mid_lse_block
    mid_lse_block_ptr = tl.make_block_ptr(
        base=mid_lse_start_ptr,
        shape=(gqa_ratio,),
        strides=(stride_mid_lse_gqa,),
        offsets=(0,),
        block_shape=(BLOCK_GQA,),
        order=(0,)
    )
    tl.store(mid_lse_block_ptr, lse, boundary_check=(0,))


@triton.jit
def flash_decode_reduce_kernel(
    mid_o_ptr,            # Intermediate output tensor from split-KV steps (Mid_O)
    mid_lse_ptr,          # Intermediate LSE tensor from split-KV steps (Mid_LSE)
    out_ptr,              # Final output tensor (O)
    gqa_ratio,            # Number of Q heads per KV head (for Grouped-Query Attention)
    n_blocks,             # Number of split-KV blocks that were reduced over
    stride_mid_o_batch,   # Strides for intermediate Output tensor
    stride_mid_o_head,    
    stride_mid_o_gqa,
    stride_mid_o_block,
    stride_mid_lse_batch, # Strides for intermediate LSE tensor
    stride_mid_lse_head,  
    stride_mid_lse_gqa,
    stride_mid_lse_block,
    stride_out_batch,     # Strides for final Output tensor
    stride_out_head,      
    BLOCK_HEAD_DIM: tl.constexpr,
):
    pid_batch = tl.program_id(axis=0)
    pid_q_head = tl.program_id(axis=1)

    # Base pointers with batch offsets
    mid_o_batch_ptr = mid_o_ptr + pid_batch * stride_mid_o_batch
    mid_lse_batch_ptr = mid_lse_ptr + pid_batch * stride_mid_lse_batch
    out_batch_ptr = out_ptr + pid_batch * stride_out_batch

    # Determine which KV head and query subhead within the GQA group this thread processes
    kv_head_idx = pid_q_head // gqa_ratio
    q_subhead_idx = pid_q_head % gqa_ratio
    
    # Base pointers for this GQA head
    mid_o_start_ptr = mid_o_batch_ptr + kv_head_idx * stride_mid_o_head + q_subhead_idx * stride_mid_o_gqa
    mid_lse_start_ptr = mid_lse_batch_ptr + kv_head_idx * stride_mid_lse_head + q_subhead_idx * stride_mid_lse_gqa

    # Initialize running softmax statistics (LSE accumulator and output accumulator)
    lse_accum = tl.full([1], float("-inf"), dtype=tl.float32)
    out_accum = tl.zeros([BLOCK_HEAD_DIM], dtype=tl.float32)

    # Loop over all split-KV blocks to reduce them online
    for block_idx in tl.range(n_blocks):
        # Load block intermediate output using block pointer (exact size BLOCK_HEAD_DIM)
        mid_o_block_ptr = tl.make_block_ptr(
            base=mid_o_start_ptr + block_idx * stride_mid_o_block,
            shape=(BLOCK_HEAD_DIM,),
            strides=(1,),
            offsets=(0,),
            block_shape=(BLOCK_HEAD_DIM,),
            order=(0,)
        )
        block_acc = tl.load(mid_o_block_ptr)

        # Load scalar block LSE
        block_lse = tl.load(mid_lse_start_ptr + block_idx * stride_mid_lse_block)

        # Update Log-Sum-Exp
        max_lse = tl.maximum(lse_accum, block_lse)
        new_lse = max_lse + tl.log(tl.exp(lse_accum - max_lse) + tl.exp(block_lse - max_lse))

        # Re-scale running output accumulator and accumulate block output
        scale_accum = tl.exp(lse_accum - new_lse)
        scale_block = tl.exp(block_lse - new_lse)
        out_accum = scale_accum * out_accum + scale_block * block_acc

        lse_accum = new_lse

    # Write out final reduced values
    out_block_ptr = tl.make_block_ptr(
        base=out_batch_ptr + pid_q_head * stride_out_head,
        shape=(BLOCK_HEAD_DIM,),
        strides=(1,),
        offsets=(0,),
        block_shape=(BLOCK_HEAD_DIM,),
        order=(0,)
    )
    tl.store(out_block_ptr, out_accum.to(tl.float16))
    

def flash_decode_out(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, seq_len: int,
    mid_o: torch.Tensor, mid_lse: torch.Tensor, out: torch.Tensor
) -> None:
    """Computes Grouped-Query Attention (GQA) using a Split-KV flash decoding approach.
    Uses pre-allocated intermediate tensors (mid_o, mid_lse) and output tensor (out).
    """
    batch_size = q.shape[0]
    q_heads = q.shape[1]
    _, k_heads, _, head_dim = k.shape
    
    gqa_ratio = q_heads // k_heads
    scale = 1 / math.sqrt(head_dim)
    
    # Stride calculation for input and intermediate tensors
    stride_q_batch, stride_q_head, _, _ = q.stride()
    stride_k_batch, stride_k_head, stride_k_seq, _ = k.stride()
    
    stride_mid_o_batch, stride_mid_o_head, stride_mid_o_block, stride_mid_o_gqa, _ = mid_o.stride()
    stride_mid_lse_batch, stride_mid_lse_head, stride_mid_lse_block, stride_mid_lse_gqa = mid_lse.stride()
    
    # # Reset intermediate lse buffer to float("-inf") for reduction kernel
    # mid_lse.fill_(float("-inf"))
    
    # The generation grid is a lambda that dynamically reads the current BLOCK_SEQ_KV
    # being benchmarked by the autotuner.
    grid_gen = lambda meta: (batch_size, k_heads, triton.cdiv(seq_len, meta["BLOCK_SEQ_KV"]))
    
    flash_decode_generation_kernel[grid_gen](
        q,
        k,
        v,
        mid_o,
        mid_lse,
        seq_len,
        scale,
        head_dim,
        stride_q_batch,
        stride_q_head,
        stride_k_batch,
        stride_k_head,
        stride_k_seq,
        stride_mid_o_batch,
        stride_mid_o_head,
        stride_mid_o_block,
        stride_mid_o_gqa,
        stride_mid_lse_batch,
        stride_mid_lse_head,
        stride_mid_lse_block,
        stride_mid_lse_gqa,
        BLOCK_HEAD_DIM=head_dim,
        gqa_ratio=gqa_ratio,
    )
    
    # Retrieve the best selected configuration and compute actual block count for reduction
    best_config = flash_decode_generation_kernel.best_config
    if best_config is not None:
        best_block_seq_kv = best_config.kwargs["BLOCK_SEQ_KV"]
    else:
        # Fallback when compilation/tuning hasn't finished or is in a non-autotuned path
        best_block_seq_kv = 64
        
    n_blocks_actual = triton.cdiv(seq_len, best_block_seq_kv)
    
    # Launch Reduction Kernel
    grid_reduce = (batch_size, q_heads)
    stride_out_batch, stride_out_head, _, _ = out.stride()
    
    flash_decode_reduce_kernel[grid_reduce](
        mid_o,
        mid_lse,
        out,
        gqa_ratio,
        n_blocks_actual,
        stride_mid_o_batch,
        stride_mid_o_head,
        stride_mid_o_gqa,
        stride_mid_o_block,
        stride_mid_lse_batch,
        stride_mid_lse_head,
        stride_mid_lse_gqa,
        stride_mid_lse_block,
        stride_out_batch,
        stride_out_head,
        head_dim,
    )


def flash_decode(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, seq_len: int,
) -> torch.Tensor:
    batch_size = q.shape[0]
    q_heads = q.shape[1]
    _, k_heads, _, head_dim = k.shape
    
    min_block_seq_kv = 32
    n_blocks_max = triton.cdiv(seq_len, min_block_seq_kv)
    gqa_ratio = q_heads // k_heads
    
    mid_o = torch.empty(
        (batch_size, k_heads, n_blocks_max, gqa_ratio, head_dim), 
        device=q.device, 
        dtype=torch.float16
    )
    mid_lse = torch.empty(
        (batch_size, k_heads, n_blocks_max, gqa_ratio), 
        device=q.device, 
        dtype=torch.float32
    )
    out = torch.empty((batch_size, q_heads, 1, head_dim), device=q.device, dtype=torch.float16)
    
    flash_decode_out(q, k, v, seq_len, mid_o, mid_lse, out)
    return out