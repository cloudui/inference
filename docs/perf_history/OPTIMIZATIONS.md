# From 42 to 51 tok/s: optimizing a Triton LLaMA-3 8B decode engine

**Setup.** Batch 1, fp16, 512-token context, RTX PRO 4500 Blackwell (~896 GB/s).
Each commit was re-benchmarked against one frozen benchmark: median of 5 runs × 128 decode steps, after 3 warmup runs.

**Ceiling.** Batch-1 decode streams 15.01 GB of fp16 weights per token (every layer plus `lm_head`; the embedding is just a row lookup). At the 896 GB/s spec that caps out at about **59.4 tok/s**. At the ~820 GB/s a well-tuned GEMV actually reaches, it's about **54.4 tok/s**.

| | tok/s | % of spec ceiling | % of practical ceiling |
|---|---:|---:|---:|
| First working engine | 42.3 | 71% | 78% |
| **Final** | **51.2** | **86%** | **94%** |
| HF eager (SDPA) | 44.8 | 75% | 82% |
| HF `torch.compile` + CUDA graphs | 48.2 | 81% | 89% |

![throughput history](throughput_history.png)

## The optimizations, ranked by measured impact

| # | Change | What I did | Impact |
|---|---|---|---:|
| 1 | **Don't autotune on sequence length** | Flash-decode's `@triton.autotune` key included `seq_len`, so every generated token triggered a full re-tune. Key it on `head_dim` only. | **2.2 → 44.9 tok/s (~20×)** in real generation |
| 2 | **Triton RoPE + fused QKV projection** | Replaced PyTorch RoPE (slice/neg/cat/mul/add, ~10 tiny kernels per layer) with one Triton kernel for q and one for k. Concatenated `wq`/`wk`/`wv` into a single `wqkv` GEMV. | **+3.8 tok/s (+9.1%)** |
| 3 | **Weight layout for GEMV** | Stored weights in (out, in) layout and multiplied by `W.T`, so each output row is read contiguously. The isolated GEMV for `wo` went from 590 to 752 GB/s. | **+2.9 tok/s (+6.2%)** |
| 4 | **Fused gate + up projection** | One 4096×28672 GEMV instead of two 4096×14336, then split the output (no copy). | **+1.1 tok/s (+2.5%)** |
| 5 | **Preallocated buffers, `out=` kernels** | No `torch.zeros`/`empty` per step. Every kernel writes into a buffer allocated once, which also removes three memsets per layer inside flash-decode. | **+0.7 tok/s (+1.6%)** |
| 6 | **Fused RoPE + KV-cache write** | One kernel reads the fused QKV output, rotates q and k, and writes k/v straight into the cache. This replaces split/transpose, 2 RoPE launches and 2 copy kernels. | **+0.5 tok/s (+1.1%)** |
| 7 | **Fused residual-add + RMSNorm** | Post-attention add+norm in one kernel, and the MLP residual add deferred into the next layer's input norm (cross-layer fusion). | **+0.4 tok/s** (two commits) |
| 8 | **Flash-decode kernel polish** | Raw pointers instead of block pointers in the reduce kernel, running max/denominator instead of per-block log-sum-exp, `exp2` with a log2(e)-prescaled scale, fixed 16 KV splits, reversed grid order. | **+0.2 tok/s** at 512 context (these target long context) |
| 9 | **CUDA graphs** (after this study) | Capture one decode step and replay it. The position moves into a GPU buffer the kernels read, the embedding writes into a static buffer, and warmup runs on a side stream so Triton autotuning finishes before capture. | **+1.0 tok/s (+2%)**, and throughput no longer depends on host CPU or profiler scopes (scopes on: 42.2 → 52.5) |

## Lessons worth a slide

- **Profiling hooks aren't free.** Wrapping every op in `torch.profiler.record_function` (483 scopes per token) costs CPU time even with no profiler attached. Without CUDA graphs the decode loop is CPU-bound, so this cut throughput by a third early on and still costs 12% at the end (51.2 → 45.2 tok/s). It also made results depend on how busy the cloud host's CPU was, the most likely reason the same code measured ~40 one day and 47–49 another. The fix: gate the scopes behind a flag (`INFERENCE_PROFILE=1`), off by default.
- **A CPU-bound benchmark can hide a GPU win.** Change #3 looked like a 4% regression with the hooks on, because the extra `.T` views cost CPU time. With CPU overhead out of the way, it's the second-biggest GPU improvement.
- **A fixed benchmark can hide the biggest bug.** Replaying the same positions warms Triton's autotune cache, so change #1 shows as "+0" in a commit-by-commit sweep while being a 20× fix in real generation. Benchmark with fresh positions too.
- **Know your ceiling.** At the end, the GEMVs alone take 18.7 of the 19.5 ms per token. What's left is GEMV bandwidth efficiency and lower-precision weights, not more fusion.

*Method:* each commit ran in its own `git worktree` against the same benchmark script, with profiler hooks stubbed out for the main numbers. Full per-commit data: `REPORT.md`, `performance_history.csv`.
