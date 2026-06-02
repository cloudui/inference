import math
import torch
import torch.nn.functional as F


def flash_decode_split_kv_reference(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, block_seq_kv: int = 64
) -> tuple[torch.Tensor, torch.Tensor]:
    q_heads = q.shape[0]
    k_heads, seq_len, head_dim = k.shape
    n_blocks = (seq_len + block_seq_kv - 1) // block_seq_kv
    gqa_ratio = q_heads // k_heads
    
    mid_o = torch.zeros((k_heads, n_blocks, gqa_ratio, head_dim), device=q.device, dtype=torch.float16)
    mid_lse = torch.zeros((k_heads, n_blocks, gqa_ratio), device=q.device, dtype=torch.float32)
    scale = 1 / math.sqrt(head_dim)

    for kv_head in range(k_heads):
        q_slice = q[gqa_ratio * kv_head : gqa_ratio * (kv_head + 1)].squeeze(1)
        for block_idx in range(n_blocks):
            k_slice = k[kv_head, block_idx * block_seq_kv : (block_idx + 1) * block_seq_kv]
            v_slice = v[kv_head, block_idx * block_seq_kv : (block_idx + 1) * block_seq_kv]
            
            scores = q_slice @ k_slice.T * scale
            max_scores = scores.amax(dim=-1, keepdim=True)
            sum_exp = (scores - max_scores).exp().sum(dim=-1)
            
            probs = F.softmax(scores, dim=-1)
            weighted_sum = probs @ v_slice
            
            mid_o[kv_head, block_idx] = weighted_sum
            mid_lse[kv_head, block_idx] = max_scores.view(-1) + torch.log(sum_exp)

    out = torch.zeros((q_heads, 1, head_dim), device=q.device, dtype=torch.float16)
    
    for q_head in range(q_heads):
        kv_head_idx = q_head // gqa_ratio
        gqa_idx = q_head % gqa_ratio
        lse_accum = torch.tensor(float("-inf"), device=q.device, dtype=torch.float32)
        for block_idx in range(n_blocks):
            block_acc = mid_o[kv_head_idx, block_idx, gqa_idx]
            block_lse = mid_lse[kv_head_idx, block_idx, gqa_idx]

            max_lse = max(lse_accum, block_lse)
            new_lse = max_lse + torch.log(torch.exp(lse_accum - max_lse) + torch.exp(block_lse - max_lse))
    
            scale_accum = torch.exp(lse_accum - new_lse)
            scale_block = torch.exp(block_lse - new_lse)
            out[q_head, 0] = scale_accum * out[q_head, 0] + scale_block * block_acc
    
            lse_accum = new_lse

    return mid_o, out


def pytorch_gqa_naive(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    q_heads = q.shape[0]
    k_heads = k.shape[0]
    head_dim = q.shape[-1]
    gqa_ratio = q_heads // k_heads
    
    k_expanded = k.repeat_interleave(gqa_ratio, dim=0)
    v_expanded = v.repeat_interleave(gqa_ratio, dim=0)
    
    scores = torch.matmul(q, k_expanded.transpose(-1, -2)) / (head_dim ** 0.5)
    probs = F.softmax(scores, dim=-1)
    output = torch.matmul(probs, v_expanded)
    
    return output


def test_reference_implementations():
    print("Running reference implementation comparison...")
    q_heads = 32
    k_heads = 8
    seq_len = 256
    head_dim = 128
    
    q = torch.randn(q_heads, 1, head_dim)
    k = torch.randn(k_heads, seq_len, head_dim)
    v = torch.randn(k_heads, seq_len, head_dim)
    
    _, out_split_kv = flash_decode_split_kv_reference(q, k, v)
    out_naive = pytorch_gqa_naive(q, k, v)
    
    assert torch.allclose(out_split_kv, out_naive, atol=1e-4), "Sanity check failed: outputs do not match!"
    print("Sanity check passed: reference outputs match perfectly!")


if __name__ == "__main__":
    test_reference_implementations()
