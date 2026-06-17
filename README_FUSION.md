# Cross-Layer Kernel Fusion Notes & TODOs

We have successfully fused the residual additions into the RMSNorm operations across the decoder layers. This improved decode throughput from ~36 tok/s to **47+ tok/s** at `batch=1` by eliminating CPU-bound kernel launch bubbles and reducing GPU global memory traffic.

However, this architecture introduces a few design trade-offs and issues that we need to address later.

---

## 1. API Encapsulation vs. Fusion Performance
### The Problem:
A standard PyTorch `DecoderLayer` is self-contained: it takes a single `hidden_states` tensor and returns a single `hidden_states` tensor.
To achieve cross-layer fusion (fusing layer $i-1$'s MLP residual add into layer $i$'s input layernorm), the layer must defer its final addition. This forces the layer to:
- Accept both `residual` and `mlp_out` as separate inputs.
- Return both `residual` and `mlp_out` as a tuple.

To prevent breaking standalone unit tests (which expect a single tensor input/output), we introduced a `fused: bool` parameter:
```python
def __call__(
    self,
    residual: torch.Tensor,
    mlp_out: torch.Tensor | None = None,
    ...,
    fused: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
```
While this works and passes all correctness tests, it is **API-wise ugly** because it compromises the clean signature of standard transformer block components.

### Backup File:
The clean, non-fused version of the `DecoderLayer` has been backed up to:
- [`DecoderLayer_clean.py.backup`](file:///workspace/inference/DecoderLayer_clean.py.backup)

---

## 2. CUDA Graph Interaction
### Observation:
At `batch=1` decode, the GPU executes cheap kernels (like vector addition) in $<1\ \mu\text{s}$, while the CPU launch overhead takes $\sim 10\ \mu\text{s}$. This causes the GPU to sit idle waiting for the CPU to launch successive kernels (a host-starvation bubble). Fusing the addition into the RMSNorm removes these bubbles.

### The Trade-off:
If we compile the entire model with **CUDA Graphs** (`torch.compile(mode="reduce-overhead")`):
1. **Launch Latency Disappears:** The entire model forward pass is captured into a single static CUDA graph and executed with a single CPU launch call. The launch overhead of 288+ individual kernels is reduced to 0, which means the CPU launch-time speedups from our fusion are nullified.
2. **Memory Bandwidth Wins Remain:** Fusing the addition and norm still saves **1 DRAM read per layer** by performing the addition in register space instead of HBM. Since single-token decode is entirely memory-bandwidth bound on the GPU, saving DRAM traffic still provides a permanent hardware-level speedup even under CUDA Graphs.
