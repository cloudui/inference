# Kernel & Performance Findings

**Note**: AI-logged on my behalf. Done via profiling and may be inaccurate. 

## 1. Weight Layout Investigation (2026-06-17)

### Question
HF dispatches `gemvx` (matrix-vector) kernels for linear projections during bs=1 decode.
Does storing weights as `(in, out)` and computing `x @ W` miss the gemV dispatch path?

### Finding: gemV dispatch is M-triggered, not transpose-triggered (on RTX Pro 4500 / cuBLAS)

**Both layouts dispatch the same `gemvx` kernel.** Profiling confirmed:

| Layout | Kernel | Self CUDA (258 calls) |
|---|---|---|
| `(in, out)` — `x @ W` (main) | `internal::gemvx` | 39.284ms |
| `(out, in)` — `x @ W.T` (refactor) | `internal::gemvx` | 37.190ms |

cuBLAS dispatches `gemvx` based on **M=1** (single-token decode), regardless of the `transB` flag.
The transpose flag distinction (which older cuBLAS versions used) does not apply here.

### Performance impact

| Branch | Throughput (tok/s) | Samples |
|---|---|---|
| `main` — `x @ W` | 38.8 – 41.0, median ~40 | 4 runs |
| `weight-refactor` — `x @ W.T` | 37.2 – 39.3, median ~38.5 | 5 runs |

**~4% regression from the `.T` view approach.** The overhead comes from:

1. **Per-call view construction** — `.T` creates a new tensor metadata object with swapped strides on every matmul call. Over 258 matmuls/step, this adds ~0.5–1ms of CPU overhead.
2. **Different cuBLAS tile configs** — `main` shows two specialized `gemvx` instantiations (46.5% + 19.9%) tuned to different matrix sizes. The `.T` path unified into one, suggesting cuBLAS selected slightly different tiling for the transposed stride pattern.

### Verdict

The `(out, in)` layout with `.T` is **architecturally cleaner** (simpler weight loading, matches `nn.Linear` convention, concat on contiguous tensors) but **measurably slower** due to per-call CPU overhead. At this stage (pre-CUDA-Graphs), the CPU dispatch cost is real. With CUDA Graphs, all CPU overhead would be eliminated and the difference would vanish.

**Current state:** `weight-refactor` branch has the cleaner layout. The ~4% regression is acceptable as a trade-off for code quality, and will be recovered by CUDA Graphs.

---

## 2. HF Performance Gap Analysis (2026-06-17)

### Benchmark: seq_len=512, bs=1, fp16, Llama-3 8B dims, random weights

| Implementation | tok/s | ms/step |
|---|---|---|
| HF (StaticCache, uncompiled) | ~38.4 | ~26.1 |
| Custom (main branch) | ~40 | ~25.0 |
| HF (DynamicCache, uncompiled) | ~43.3 | ~23.1 |
| HF (DynamicCache, compiled) | ~46 | ~21.9 |

**Surprise: HF with StaticCache (38.4 tok/s) is slower than both DynamicCache (43.3) and custom (40).**
StaticCache pre-allocates the full KV tensor, so SDPA attends over all allocated positions
(using an attention mask to ignore unfilled slots). This is more expensive than DynamicCache
where the KV tensor is exactly the right size — no masking needed, no wasted compute.

**The fair comparison is DynamicCache vs Custom: ~3 tok/s gap (~8%), all from dispatch overhead.**

### Where the time goes — kernel-level breakdown

#### Matmul (gemvx) — effectively tied

| | Custom | HF |
|---|---|---|
| Self CUDA time | 37.193ms | 37.520ms |
| Calls | 258 | 450 |
| Kernel | `gemvx` (1 variant) | `gemvx` (2 variants) |

Both spend virtually identical GPU time on matmuls. HF has more calls (450 vs 258) because it runs separate Q/K/V/O projections per layer instead of fused QKV — but per-call CUDA time is the same since the total weight FLOPS are identical.

#### Attention — HF uses flash_attention, custom uses split-KV Triton

| | Custom | HF |
|---|---|---|
| Per-call CUDA | 3.9μs (gen) + 4.4μs (reduce) = ~8.3μs | ~4.5μs (FA2 splitkv) |
| Per decode step (32 layers) | ~265μs | ~142μs |

Custom's 2-kernel split-KV is ~2x slower per-call vs FA2, but this is only **~123μs/step** —
a real but small contributor to the gap.

#### RoPE — custom is much leaner

| | Custom | HF |
|---|---|---|
| Total CUDA | 58μs (fused kernel) | ~1.3ms (mul + neg + cat + copy) |

Custom's fused Triton RoPE kernel is >20x faster than HF's decomposed PyTorch ops. **This saves ~1.2ms per step.**

#### RMSNorm — custom is leaner

| | Custom | HF |
|---|---|---|
| Total CUDA | 182μs (triton) | ~455μs (mean + mul + rsqrt, decomposed) |

Custom's fused Triton RMSNorm saves ~270μs per step.

#### HF overhead ops (DynamicCache)

| HF Op | CUDA time | Purpose |
|---|---|---|
| `aten::cat` | 586μs | KV cache concat (DynamicCache grows tensors) |
| `aten::copy_` | 372μs | Tensor copies for cache management |
| `aten::mul` | 726μs | RoPE + RMSNorm elementwise ops |
| `aten::neg` | 193μs | RoPE `rotate_half` |
| `aten::add` | 353μs | Residual connections (not in-place) |
| `aten::mean` | 227μs | RMSNorm variance computation |

Total HF overhead: ~2.5ms of decomposed elementwise ops that custom fuses into Triton kernels.

### Why custom loses on wall-clock despite winning on CUDA time

| Per decode step | Custom | HF (DynamicCache) |
|---|---|---|
| Self CUDA total | ~19.1ms | ~20.4ms |
| Self CPU total | ~39.5ms | ~45.6ms |
| Kernel launches | ~193 | ~450+ |
| **Wall-clock** | **~25ms** | **~23ms** |

**Custom does less GPU work AND less CPU work, but achieves worse wall-clock.** The reason
is CPU-GPU pipeline overlap:

HF launches ~450 small kernels through optimized C++ dispatch (`nn.Module._call_impl`).
While kernel N executes on GPU, kernels N+1 through N+5 are already queued — the GPU
never starves for work.

Custom launches ~193 larger, fused kernels separated by Python code (Triton dispatch,
`torch.split`, `record_function` context managers, conditional checks). Between each
fused kernel, the CPU does heavier per-call work before queueing the next launch, creating
GPU idle gaps ("pipeline bubbles"). The kernels individually are faster, but the gaps
between them eat the savings.

This is the classic **fusion paradox**: fewer, bigger kernels reduce total GPU work but can
worsen CPU-GPU overlap when the dispatch path is Python-heavy.

### Opportunities to close the gap

| Optimization | Expected Impact | Difficulty |
|---|---|---|
| **CUDA Graphs** — eliminate all CPU dispatch overhead | +3–5 tok/s | Medium |
| **Remove `mid_lse.fill_`** — initialize in the kernel or track block validity | +0.2ms/step | Easy |
| **Flash decode kernel tuning** — close the 123μs/step gap vs FA2 | +0.1ms/step | Hard |
| **INT8 weight quantization** — halve weight read traffic | ~2x throughput | Medium |

### Key insight

Custom's Triton kernels (RMSNorm, SwiGLU, fused RoPE) save ~3.7ms of GPU time vs HF's
decomposed ops, but this is offset by Python dispatch overhead creating pipeline bubbles.
**CUDA Graphs would recover the full gap and then some**, since custom has fewer total CUDA
microseconds — it's purely losing on dispatch/scheduling overhead, which graphs eliminate
entirely.
