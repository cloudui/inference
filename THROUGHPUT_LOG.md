
# Me

## Me-8B:

### Base
============================================================
  Decode Throughput Benchmark
  seq_len=512  decode_steps=128  batch=1
============================================================

Warming up (10 steps)...

────────────────────────────────────────────────────────────
  Model:            Llama-3 8B
  Context length:   512 → 650
  Decode steps:     128
  Batch size:       1
────────────────────────────────────────────────────────────
  Total time:       4472.70 ms
  Per step:         34.943 ms/tok
  Throughput:       28.6 tok/s
────────────────────────────────────────────────────────────

### Preallocate kernel output tensors
============================================================
  Decode Throughput Benchmark
  seq_len=512  decode_steps=128  batch=1
============================================================

Warming up (10 steps)...

────────────────────────────────────────────────────────────
  Model:            Llama-3 8B
  Context length:   512 → 650
  Decode steps:     128
  Batch size:       1
────────────────────────────────────────────────────────────
  Total time:       3787.93 ms
  Per step:         29.593 ms/tok
  Throughput:       33.8 tok/s
────────────────────────────────────────────────────────────

# HF

## HF-8B not compiled

## HF-8B compiled
────────────────────────────────────────────────────────────
  Model:            HF Llama-3 8B
  Context length:   512 → 650
  Decode steps:     128
  Batch size:       1
  Dtype:            float16
  Compiled:         True
────────────────────────────────────────────────────────────
  Total time:       2798.40 ms
  Per step:         21.862 ms/tok
  Throughput:       45.7 tok/s
────────────────────────────────────────────────────────────