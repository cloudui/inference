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

from kernels import rmsnorm, swiglu, flash_decode, fused_rmsnorm_swiglu


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

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
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
        self.wq = torch.empty(config.hidden_size, self.num_heads * self.head_dim)
        self.wk = torch.empty(config.hidden_size, self.num_kv_heads * self.head_dim)
        self.wv = torch.empty(config.hidden_size, self.num_kv_heads * self.head_dim)
        self.wo = torch.empty(self.num_heads * self.head_dim, config.hidden_size)

    def __call__(
        self,
        x: torch.Tensor,
        rope_embeds: tuple[torch.Tensor, torch.Tensor],
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
        cache_position: int = 0,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, hidden_size)
            freqs_cis: (seq_len, head_dim // 2) complex
            kv_cache: optional (k_cache, v_cache) each (batch, n_kv_heads, max_seq_len, head_dim)
            cache_position: write position in the KV cache (for decode step)
        
        Returns:
            (batch, seq_len, hidden_size)
        """
        cos, sin = rope_embeds

        hidden_shape = (x.shape[0], x.shape[1], -1, self.head_dim,)

        # (b, seqlen, heads, head_dim)
        q = (x @ self.wq).view(hidden_shape).transpose(1, 2)
        k = (x @ self.wk).view(hidden_shape).transpose(1, 2)

        q, k = apply_rope(q, k, cos, sin)
        v = (x @ self.wv).view(hidden_shape).transpose(1, 2)

        K, V = kv_cache
        K[:, :, cache_position:cache_position+1] = k
        V[:, :, cache_position:cache_position+1] = v

        fd_out = flash_decode(q, K[:, :, : cache_position+1], V[:, :, : cache_position+1])
        out = fd_out.transpose(1, 2).reshape(x.shape) @ self.wo
        
        return out

# ── MLP (SwiGLU) ──────────────────────────────────────────────────────────────

class MLP:
    """LLaMA FFN: SwiGLU(x @ gate, x @ up) @ down.

    Weight shapes (no bias):
        w_gate: (hidden_size, intermediate_size)
        w_up:   (hidden_size, intermediate_size)
        w_down: (intermediate_size, hidden_size)
    """

    def __init__(self, config: LlamaConfig):
        self.w_gate = torch.empty(config.hidden_size, config.intermediate_size)
        self.w_up = torch.empty(config.hidden_size, config.intermediate_size)
        self.w_down = torch.empty(config.intermediate_size, config.hidden_size)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, hidden_size)
        Returns:
            (batch, seq_len, hidden_size)
        """
        # kernel dispatch
        x = swiglu(x @ self.w_up, x @ self.w_gate)
        return x @ self.w_down


# ── Decoder Layer ─────────────────────────────────────────────────────────────

class DecoderLayer:
    """Pre-norm transformer block: RMSNorm -> Attention + residual -> RMSNorm -> MLP + residual."""

    def __init__(self, config: LlamaConfig, layer_idx: int):
        self.layer_idx = layer_idx
        self.self_attn = Attention(config)
        self.mlp = MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

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
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, rope_embeds, kv_cache, cache_position)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

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

        cos = torch.cat([freqs_cis.real, freqs_cis.real], dim=-1)  # (max_seq, head_dim)
        sin = torch.cat([freqs_cis.imag, freqs_cis.imag], dim=-1)
        self.cos = cos.half()  # or .bfloat16() when you switch
        self.sin = sin.half()

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
        hidden_states = self.embed_tokens[token_ids]

        cos = self.cos[start_pos : start_pos + token_ids.shape[1]].unsqueeze(0).unsqueeze(0)
        sin = self.sin[start_pos : start_pos + token_ids.shape[1]].unsqueeze(0).unsqueeze(0)

        for i, layer in enumerate(self.layers):
            hidden_states = layer(
                hidden_states, 
                rope_embeds=(cos, sin),
                kv_cache=kv_caches[i],
                cache_position=start_pos
            )
        
        x = self.norm(hidden_states)
        out = x @ self.lm_head

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
            layer.self_attn.wq = state_dict[p + "self_attn.q_proj.weight"].T
            layer.self_attn.wk = state_dict[p + "self_attn.k_proj.weight"].T
            layer.self_attn.wv = state_dict[p + "self_attn.v_proj.weight"].T
            layer.self_attn.wo = state_dict[p + "self_attn.o_proj.weight"].T

            # Norms (1-D, no transpose)
            layer.input_layernorm.weight = state_dict[p + "input_layernorm.weight"]
            layer.post_attention_layernorm.weight = state_dict[p + "post_attention_layernorm.weight"]

            # MLP projections: HF (out, in) → custom (in, out)
            layer.mlp.w_gate = state_dict[p + "mlp.gate_proj.weight"].T
            layer.mlp.w_up = state_dict[p + "mlp.up_proj.weight"].T
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
            layer.self_attn.wq = layer.self_attn.wq.to(device)
            layer.self_attn.wk = layer.self_attn.wk.to(device)
            layer.self_attn.wv = layer.self_attn.wv.to(device)
            layer.self_attn.wo = layer.self_attn.wo.to(device)
            layer.input_layernorm.weight = layer.input_layernorm.weight.to(device)
            layer.post_attention_layernorm.weight = layer.post_attention_layernorm.weight.to(device)
            layer.mlp.w_gate = layer.mlp.w_gate.to(device)
            layer.mlp.w_up = layer.mlp.w_up.to(device)
            layer.mlp.w_down = layer.mlp.w_down.to(device)