# Llama-3 8B Inference Stack

Custom kernel inference stack for Llama3.1 8B model. One batch preallocated KV cache decode.
96% bandwidth saturation on RTX Pro 4500 Blackwell, 32GB @ 800GB/s 

README in progress.

Triton stack:
- Fused RoPE + KV cache update kernel
- Ping pong buffering
- FlashDecode, RMSNorm, SwiGLU kernels
- HuggingFace equivalent on FP16. 

# Stack
- Triton 
- CUDA
- FP16

## Tentative Additions
- H100/Hopper
- BF16/FP8