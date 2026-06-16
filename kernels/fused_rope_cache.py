import torch
import triton
import triton.language as tl

@triton.jit
def fused_rope_cache_kernel(
    qkv_ptr, # (batch_size, 1, qkv_concat_dim_size)
    cos_ptr,
    sin_ptr,
    q_out_ptr, # (batch_size, num_heads, 1, head_dim)
    k_out_ptr, 
    v_out_ptr,
    stride_batch_qkv,
    stride_batch_out_q,
    stride_batch_out_kv,
    stride_head_out_kv,
    cache_pos,
    N_HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr
):
    pid_bh = tl.program_id(axis=0)
    GQA_RATIO = N_HEADS // KV_HEADS

    # q
    batch_idx_q = pid_bh // N_HEADS
    head_idx_q = pid_bh % N_HEADS
    Q_DIM = N_HEADS * HEAD_DIM

    # Create block pointers for 1D row slices
    offsets = tl.arange(0, HEAD_DIM//2)

    batch_start = batch_idx_q * stride_batch_qkv
    q_start = batch_start + head_idx_q * HEAD_DIM

    q_offsets_top = q_start + offsets
    q_offsets_bottom = q_offsets_top + HEAD_DIM // 2

    q_top = tl.load(qkv_ptr + q_offsets_top)
    q_bottom = tl.load(qkv_ptr + q_offsets_bottom)

    cos = tl.load(cos_ptr + offsets)
    sin = tl.load(sin_ptr + offsets)

    # Q 
    q_output_top = (q_top * cos) + (-q_bottom * sin)
    q_output_bottom = (q_bottom * cos) + (q_top * sin)

    # KV 
    if head_idx_q % GQA_RATIO == 0:
        head_idx_kv = head_idx_q // GQA_RATIO
        KV_DIM = KV_HEADS * HEAD_DIM

        k_start = batch_start + Q_DIM + head_idx_kv * HEAD_DIM
        k_offsets_top = offsets + k_start
        k_offsets_bottom = k_offsets_top + HEAD_DIM // 2
        v_offsets = tl.arange(0, HEAD_DIM) + batch_start + Q_DIM + KV_DIM + head_idx_kv * HEAD_DIM
        
        k_top = tl.load(qkv_ptr + k_offsets_top)
        k_bottom = tl.load(qkv_ptr + k_offsets_bottom)
        v = tl.load(qkv_ptr + v_offsets)

        k_output_top = (k_top * cos) + (-k_bottom * sin)
        k_output_bottom = (k_bottom * cos) + (k_top * sin)

        k_out_start = batch_idx_q * stride_batch_out_kv + head_idx_kv * stride_head_out_kv + \
                                    cache_pos * HEAD_DIM
        k_out_offsets_top = k_out_start + offsets
        k_out_offsets_bottom = k_out_offsets_top + HEAD_DIM // 2
        tl.store(k_out_ptr + k_out_offsets_top, k_output_top)
        tl.store(k_out_ptr + k_out_offsets_bottom, k_output_bottom)

        tl.store(v_out_ptr + k_out_start + tl.arange(0, HEAD_DIM), v)

    # Store computed values using block pointer
    q_out_start = batch_idx_q * stride_batch_out_q + head_idx_q * HEAD_DIM
    q_out_offsets_top = q_out_start + offsets
    q_out_offsets_bottom = q_out_offsets_top + HEAD_DIM // 2
    tl.store(q_out_ptr + q_out_offsets_top, q_output_top)
    tl.store(q_out_ptr + q_out_offsets_bottom, q_output_bottom)

def fused_rope_cache_decode_out(
    qkv_proj: torch.Tensor, 
    cos: torch.Tensor, 
    sin: torch.Tensor,
    q_out: torch.Tensor, 
    k_cache: torch.Tensor, 
    v_cache: torch.Tensor, 
    cache_pos: int,
) -> None:
    n_batches, seqlen, qkv_concat_dim = qkv_proj.shape

    _, n_heads, _, head_dim = q_out.shape
    kv_heads = k_cache.shape[1]

    # seqlen = 1
    grid = (n_batches*n_heads,)

    fused_rope_cache_kernel[grid](
        qkv_proj, 
        cos,
        sin,
        q_out, 
        k_cache, # (batch_size, num_key_value_heads, max_seq_len, head_dim,)
        v_cache,
        qkv_proj.stride()[0],
        q_out.stride()[0],
        k_cache.stride()[0],
        k_cache.stride()[1], 
        cache_pos,
        n_heads,
        kv_heads,
        head_dim,
    )

