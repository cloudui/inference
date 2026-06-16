"""
LLaMA 3 8B Inference Engine

Minimal inference-only implementation targeting a single model architecture.
No training, no gradient tracking, no HuggingFace abstractions.
"""

import json
import torch
import torch.nn.functional as F
import math
from dataclasses import dataclass
from pathlib import Path
from safetensors.torch import load_file
from torch.profiler import record_function

from kernels import (
    rmsnorm, rmsnorm_out,
    swiglu, swiglu_out,
    flash_decode, flash_decode_out,
    apply_rope_decode, apply_rope_decode_out
)


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class LlamaConfig:
    hidden_size: int = 4096
    num_hidden_layers: int = 32
    num_attention_heads: int = 32       # Q heads
    num_key_value_heads: int = 8        # KV heads (GQA)
    intermediate_size: int = 14336
    vocab_size: int = 128256
    max_position_embeddings: int = 8192
    rms_norm_eps: float = 1e-6
    rope_theta: float = 500000.0
    head_dim: int = 128                 # hidden_size // num_attention_heads


# ── RoPE ──────────────────────────────────────────────────────────────────────

def precompute_rope_freqs(
    head_dim: int,
    max_seq_len: int,
    theta: float = 500000.0,
    device: torch.device = torch.device("cuda"),
) -> torch.Tensor:
    """Precomputes the complex RoPE frequency table.

    Returns:
        Complex tensor of shape (max_seq_len, head_dim // 2) containing
        cis(freq * position) values for rotary embedding.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    positions = torch.arange(max_seq_len, device=device).float()
    # (max_seq_len, head_dim // 2)
    freqs_table = torch.outer(positions, freqs)
    return torch.polar(torch.ones_like(freqs_table), freqs_table)

def rotate_half(x: torch.Tensor):
    """
    Setup QK the RoPe rotation vectorization
    """
    # x: (bh, seqlen, head_dim)
    x_tophalf = x[..., : x.shape[-1] // 2]
    x_bottomhalf = x[..., x.shape[-1] // 2 : ]
    return torch.concat((-x_bottomhalf, x_tophalf), dim=-1)

# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
def apply_rope(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Applies rotary positional embeddings to Q and K tensors.

    Args:
        q: (batch, n_heads, seq_len, head_dim)
        k: (batch, n_kv_heads, seq_len, head_dim)
        cos, sin: (batch, seqlen, head_dim)
    """
    # TODO: make sure batch dims match for rope ops
    # works for now
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed


# ── RMSNorm ───────────────────────────────────────────────────────────────────

class RMSNorm:
    """Holds the learned scale weight for RMSNorm. Forward dispatches to kernel."""

    def __init__(self, dim: int, eps: float = 1e-6):
        self.eps = eps
        self.weight = torch.ones(dim)

    def __call__(self, x: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
        if out is not None:
            rmsnorm_out(x, self.weight, out, self.eps)
            return out
        return rmsnorm(x, self.weight, self.eps)


# ── Attention ─────────────────────────────────────────────────────────────────

class Attention:
    """Multi-head attention with GQA support and KV cache for decode.

    Weight shapes (no bias):
        wq: (hidden_size, num_attention_heads * head_dim)
        wk: (hidden_size, num_key_value_heads * head_dim)
        wv: (hidden_size, num_key_value_heads * head_dim)
        wo: (num_attention_heads * head_dim, hidden_size)
    """

    def __init__(self, config: LlamaConfig):
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size

        # Projection weights — populated by weight loading
        # self.wq = torch.empty(config.hidden_size, self.num_heads * self.head_dim)
        # self.wk = torch.empty(config.hidden_size, self.num_kv_heads * self.head_dim)
        # self.wv = torch.empty(config.hidden_size, self.num_kv_heads * self.head_dim)
        qkv_concat_dim_size = self.num_heads * self.head_dim + 2 * self.num_kv_heads * self.head_dim
        self.wqkv = torch.empty(config.hidden_size, qkv_concat_dim_size)
        self.wo = torch.empty(self.num_heads * self.head_dim, config.hidden_size)

    def preallocate_buffers(self, batch_size: int, device: torch.device, dtype: torch.dtype, max_seq_len: int):
        qkv_concat_dim_size = self.num_heads * self.head_dim + 2 * self.num_kv_heads * self.head_dim
        self.qkv_proj_out = torch.empty(batch_size, 1, qkv_concat_dim_size, device=device, dtype=dtype)
        self.q_rope_out = torch.empty(batch_size, self.num_heads, 1, self.head_dim, device=device, dtype=dtype)
        self.k_rope_out = torch.empty(batch_size, self.num_kv_heads, 1, self.head_dim, device=device, dtype=dtype)
        min_block_seq_kv = 32
        n_blocks_max = (max_seq_len + min_block_seq_kv - 1) // min_block_seq_kv
        gqa_ratio = self.num_heads // self.num_kv_heads
        self.mid_o = torch.empty(batch_size, self.num_kv_heads, n_blocks_max, gqa_ratio, self.head_dim, device=device, dtype=dtype)
        self.mid_lse = torch.empty(batch_size, self.num_kv_heads, n_blocks_max, gqa_ratio, device=device, dtype=torch.float32)
        self.fd_out = torch.empty(batch_size, self.num_heads, 1, self.head_dim, device=device, dtype=dtype)
        self.wo_out = torch.empty(batch_size, 1, self.hidden_size, device=device, dtype=dtype)

    def __call__(
        self,
        x: torch.Tensor,
        rope_embeds: tuple[torch.Tensor, torch.Tensor],
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
        cache_position: int = 0,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        cos, sin = rope_embeds

        hidden_shape = (x.shape[0], x.shape[1], -1, self.head_dim,)

        with record_function("qkv_proj"):
            if not hasattr(self, "qkv_proj_out") or self.qkv_proj_out.shape[0] != x.shape[0] or self.qkv_proj_out.device != x.device or self.qkv_proj_out.dtype != x.dtype:
                self.preallocate_buffers(x.shape[0], x.device, x.dtype, max_seq_len=8192)

            qkv = torch.matmul(x, self.wqkv, out=self.qkv_proj_out)
            q_dim = self.num_heads * self.head_dim
            kv_dim = self.num_kv_heads * self.head_dim
            q, k, v = torch.split(qkv, [q_dim, kv_dim, kv_dim], dim=-1)

            q = q.view(hidden_shape).transpose(1, 2)
            k = k.view(hidden_shape).transpose(1, 2)
            v = v.view(hidden_shape).transpose(1, 2)

        with record_function("rope"):
            apply_rope_decode_out(q, cos, sin, self.q_rope_out)
            apply_rope_decode_out(k, cos, sin, self.k_rope_out)
            q = self.q_rope_out
            k = self.k_rope_out

        with record_function("kv_cache_write"):
            K, V = kv_cache
            K[:, :, cache_position:cache_position+1] = k
            V[:, :, cache_position:cache_position+1] = v

        with record_function("flash_decode"):
            flash_decode_out(q, K[:, :, : cache_position+1], V[:, :, : cache_position+1], self.mid_o, self.mid_lse, self.fd_out)
            
        with record_function("out_proj"):
            fd_out_reshaped = self.fd_out.transpose(1, 2).reshape(x.shape)
            out_tensor = out if out is not None else self.wo_out
            out = torch.matmul(fd_out_reshaped, self.wo, out=out_tensor)
        
        return out

# ── MLP (SwiGLU) ──────────────────────────────────────────────────────────────

class MLP:
    """LLaMA FFN: SwiGLU(x @ gate, x @ up) @ down.

    Weight shapes (no bias):
        w_gate_up: (hidden_size, 2 * intermediate_size)
        w_down:    (intermediate_size, hidden_size)
    """

    def __init__(self, config: LlamaConfig):
        self.w_gate_up = torch.empty(config.hidden_size, 2 * config.intermediate_size)
        self.w_down = torch.empty(config.intermediate_size, config.hidden_size)

    def preallocate_buffers(self, batch_size: int, device: torch.device, dtype: torch.dtype):
        intermediate_size = self.w_down.shape[0]
        self.gate_up_out = torch.empty(batch_size, 1, 2 * intermediate_size, device=device, dtype=dtype)
        self.act_out = torch.empty(batch_size, 1, intermediate_size, device=device, dtype=dtype)
        self.down_out = torch.empty(batch_size, 1, self.w_down.shape[1], device=device, dtype=dtype)

    def __call__(self, x: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, hidden_size)
        Returns:
            (batch, seq_len, hidden_size)
        """
        if not hasattr(self, "gate_up_out") or self.gate_up_out.shape[0] != x.shape[0] or self.gate_up_out.device != x.device or self.gate_up_out.dtype != x.dtype:
            self.preallocate_buffers(x.shape[0], x.device, x.dtype)

        # kernel dispatch
        with record_function("gate_up_proj"):
            gate_up = torch.matmul(x, self.w_gate_up, out=self.gate_up_out)
            intermediate_size = self.w_down.shape[0]
            gate, up = torch.split(gate_up, [intermediate_size, intermediate_size], dim=-1)
        with record_function("swiglu"):
            swiglu_out(up, gate, self.act_out)
            act = self.act_out
        with record_function("down_proj"):
            out_tensor = out if out is not None else self.down_out
            return torch.matmul(act, self.w_down, out=out_tensor)


# ── Decoder Layer ─────────────────────────────────────────────────────────────

class DecoderLayer:
    """Pre-norm transformer block: RMSNorm -> Attention + residual -> RMSNorm -> MLP + residual."""

    def __init__(self, config: LlamaConfig, layer_idx: int):
        self.layer_idx = layer_idx
        self.self_attn = Attention(config)
        self.mlp = MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def preallocate_buffers(self, batch_size: int, device: torch.device, dtype: torch.dtype, max_seq_len: int):
        self.input_layernorm_out = torch.empty(batch_size, 1, self.input_layernorm.weight.shape[0], device=device, dtype=dtype)
        self.post_attention_layernorm_out = torch.empty(batch_size, 1, self.post_attention_layernorm.weight.shape[0], device=device, dtype=dtype)
        self.attn_out = torch.empty(batch_size, 1, self.input_layernorm.weight.shape[0], device=device, dtype=dtype)
        self.mlp_out = torch.empty(batch_size, 1, self.input_layernorm.weight.shape[0], device=device, dtype=dtype)
        self.self_attn.preallocate_buffers(batch_size, device, dtype, max_seq_len)
        self.mlp.preallocate_buffers(batch_size, device, dtype)

    def __call__(
        self,
        hidden_states: torch.Tensor,
        rope_embeds: tuple[torch.Tensor, torch.Tensor],
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
        cache_position: int = 0,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch, seq_len, hidden_size)
        Returns:
            (batch, seq_len, hidden_size)
        """
        if not hasattr(self, "input_layernorm_out") or self.input_layernorm_out.shape[0] != hidden_states.shape[0] or self.input_layernorm_out.device != hidden_states.device or self.input_layernorm_out.dtype != hidden_states.dtype:
            self.preallocate_buffers(hidden_states.shape[0], hidden_states.device, hidden_states.dtype, max_seq_len=8192)

        residual = hidden_states
        with record_function("input_norm"):
            normed = self.input_layernorm(hidden_states, out=self.input_layernorm_out)
        with record_function("attn"):
            attn_out = self.self_attn(normed, rope_embeds, kv_cache, cache_position, out=self.attn_out)
        with record_function("attn_residual"):
            hidden_states = residual.add_(attn_out)

        residual = hidden_states
        with record_function("post_norm"):
            normed2 = self.post_attention_layernorm(hidden_states, out=self.post_attention_layernorm_out)
        with record_function("mlp"):
            mlp_out = self.mlp(normed2, out=self.mlp_out)
        with record_function("mlp_residual"):
            hidden_states = residual.add_(mlp_out)

        return hidden_states


# ── Full Model ────────────────────────────────────────────────────────────────

class Llama:
    """LLaMA 3 8B inference model.
    
    Architecture:
        token_embed -> [DecoderLayer x num_hidden_layers] -> RMSNorm -> lm_head
    """

    def __init__(self, config: LlamaConfig):
        self.config = config

        # Token embedding: (vocab_size, hidden_size)
        self.embed_tokens = torch.empty(config.vocab_size, config.hidden_size)

        # Transformer layers
        self.layers = [DecoderLayer(config, i) for i in range(config.num_hidden_layers)]

        # Final norm + output projection
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # lm_head: (hidden_size, vocab_size) — often tied with embed_tokens
        self.lm_head = torch.empty(config.hidden_size, config.vocab_size)

        # Precomputed RoPE frequencies
        freqs_cis = precompute_rope_freqs(
            config.head_dim, config.max_position_embeddings, config.rope_theta
        )

        self.cos = freqs_cis.real.contiguous()
        self.sin = freqs_cis.imag.contiguous()

    # ── KV Cache ──────────────────────────────────────────────────────────

    def allocate_kv_cache(
        self, batch_size: int = 1, max_seq_len: int | None = None, device: torch.device = torch.device("cuda")
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Pre-allocates KV cache for all layers.

        Returns:
            List of (k_cache, v_cache) tuples per layer, each of shape
            (batch, num_kv_heads, max_seq_len, head_dim).
        """
        max_seq_len = max_seq_len or self.config.max_position_embeddings
        caches = []
        for _ in self.layers:
            k_cache = torch.zeros(
                batch_size, self.config.num_key_value_heads, max_seq_len, self.config.head_dim,
                device=device, dtype=torch.float16,
            )
            v_cache = torch.zeros(
                batch_size, self.config.num_key_value_heads, max_seq_len, self.config.head_dim,
                device=device, dtype=torch.float16,
            )
            caches.append((k_cache, v_cache))
        return caches

    # ── Forward ───────────────────────────────────────────────────────────

    def preallocate_buffers(self, batch_size: int, device: torch.device = torch.device("cuda"), dtype: torch.dtype = torch.float16):
        self.norm_out = torch.empty(batch_size, 1, self.norm.weight.shape[0], device=device, dtype=dtype)
        self.lm_head_out = torch.empty(batch_size, 1, self.lm_head.shape[1], device=device, dtype=dtype)
        for layer in self.layers:
            layer.preallocate_buffers(batch_size, device, dtype, self.config.max_position_embeddings)

    @torch.inference_mode()
    def forward(
        self,
        token_ids: torch.Tensor,
        start_pos: int = 0,
        kv_caches: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> torch.Tensor:
        """Run a forward pass (prefill or single-token decode).

        Args:
            token_ids: (batch, seq_len) token indices
            start_pos: position offset for RoPE and KV cache writes
            kv_caches: per-layer KV caches from allocate_kv_cache()

        Returns:
            Logits tensor of shape (batch, seq_len, vocab_size)
        """
        if not hasattr(self, "norm_out") or self.norm_out.shape[0] != token_ids.shape[0] or self.norm_out.device != token_ids.device or self.norm_out.dtype != self.embed_tokens.dtype:
            self.preallocate_buffers(token_ids.shape[0], device=token_ids.device, dtype=self.embed_tokens.dtype)

        with record_function("embed_lookup"):
            hidden_states = self.embed_tokens[token_ids]

        cos = self.cos[start_pos : start_pos + token_ids.shape[1]]
        sin = self.sin[start_pos : start_pos + token_ids.shape[1]]

        for i, layer in enumerate(self.layers):
            with record_function(f"layer_{i}"):
                hidden_states = layer(
                    hidden_states, 
                    rope_embeds=(cos, sin),
                    kv_cache=kv_caches[i],
                    cache_position=start_pos
                )
        
        with record_function("final_norm"):
            x = self.norm(hidden_states, out=self.norm_out)
        with record_function("lm_head"):
            out = torch.matmul(x, self.lm_head, out=self.lm_head_out)

        return out

    # ── Weight Loading ────────────────────────────────────────────────────

    @staticmethod
    def from_pretrained(model_path: str, device: torch.device = torch.device("cuda")) -> "Llama":
        """Loads weights from a HuggingFace-format checkpoint directory.

        Expects safetensors files and a config.json. Maps HF weight names
        to our flat structure.

        Args:
            model_path: path to a directory containing safetensors + config.json
            device: target device for all tensors
        """
        model_dir = Path(model_path)

        # ── Load config ──────────────────────────────────────────────────
        with open(model_dir / "config.json") as f:
            raw = json.load(f)

        config = LlamaConfig(
            hidden_size=raw["hidden_size"],
            num_hidden_layers=raw["num_hidden_layers"],
            num_attention_heads=raw["num_attention_heads"],
            num_key_value_heads=raw["num_key_value_heads"],
            intermediate_size=raw["intermediate_size"],
            vocab_size=raw["vocab_size"],
            max_position_embeddings=raw["max_position_embeddings"],
            rms_norm_eps=raw.get("rms_norm_eps", 1e-6),
            rope_theta=raw.get("rope_theta", 500000.0),
            head_dim=raw["hidden_size"] // raw["num_attention_heads"],
        )

        model = Llama(config)

        # ── Load safetensors shards ──────────────────────────────────────
        state_dict = {}
        for f in sorted(model_dir.glob("*.safetensors")):
            state_dict.update(load_file(str(f), device="cpu"))

        # ── Map weights ──────────────────────────────────────────────────
        # Embedding and final norm (no transpose)
        model.embed_tokens = state_dict["model.embed_tokens.weight"]
        model.norm.weight = state_dict["model.norm.weight"]

        # lm_head: HF (vocab, hidden) → custom (hidden, vocab)
        if "lm_head.weight" in state_dict:
            model.lm_head = state_dict["lm_head.weight"].T
        else:
            # Weight-tied models share embed_tokens
            model.lm_head = model.embed_tokens.T

        # Per-layer weights
        for i in range(config.num_hidden_layers):
            p = f"model.layers.{i}."
            layer = model.layers[i]

            # Attention projections: HF (out, in) → custom (in, out)
            wq = state_dict[p + "self_attn.q_proj.weight"].T
            wk = state_dict[p + "self_attn.k_proj.weight"].T
            wv = state_dict[p + "self_attn.v_proj.weight"].T
            layer.self_attn.wqkv = torch.concat((wq, wk, wv), dim=-1)

            layer.self_attn.wo = state_dict[p + "self_attn.o_proj.weight"].T

            # Norms (1-D, no transpose)
            layer.input_layernorm.weight = state_dict[p + "input_layernorm.weight"]
            layer.post_attention_layernorm.weight = state_dict[p + "post_attention_layernorm.weight"]

            # MLP projections: HF (out, in) → custom (in, out)
            w_gate = state_dict[p + "mlp.gate_proj.weight"].T
            w_up = state_dict[p + "mlp.up_proj.weight"].T
            layer.mlp.w_gate_up = torch.concat((w_gate, w_up), dim=-1)
            layer.mlp.w_down = state_dict[p + "mlp.down_proj.weight"].T

        model._move_to_device(device)
        return model

    def _move_to_device(self, device: torch.device) -> None:
        """Moves all weight tensors to the specified device."""
        self.embed_tokens = self.embed_tokens.to(device)
        self.lm_head = self.lm_head.to(device)
        self.norm.weight = self.norm.weight.to(device)
        self.cos = self.cos.to(device)
        self.sin = self.sin.to(device)

        for layer in self.layers:
            layer.self_attn.wqkv = layer.self_attn.wqkv.to(device)
            layer.self_attn.wo = layer.self_attn.wo.to(device)
            layer.input_layernorm.weight = layer.input_layernorm.weight.to(device)
            layer.post_attention_layernorm.weight = layer.post_attention_layernorm.weight.to(device)
            layer.mlp.w_gate_up = layer.mlp.w_gate_up.to(device)
            layer.mlp.w_down = layer.mlp.w_down.to(device)