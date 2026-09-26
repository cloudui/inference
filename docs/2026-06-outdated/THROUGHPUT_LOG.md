
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

### Fuse rope + kv cache write
============================================================
  Decode Throughput Benchmark
  seq_len=512  decode_steps=128  batch=1
============================================================

Warming up (30 steps)...

────────────────────────────────────────────────────────────
  Model:            Llama-3 8B
  Context length:   512 → 670
  Decode steps:     128
  Batch size:       1
────────────────────────────────────────────────────────────
  Total time:       3232.07 ms
  Per step:         25.251 ms/tok
  Throughput:       39.6 tok/s
────────────────────────────────────────────────────────────

### Other optims
- remove kv cache slicing for FD
- remove initialization transpose on affine trans

This is actually like a 4% regression, probably due to transpose overhead. May be removed via CUDA graphs but we are yet to see. 
Maybe revert: 179f662a9d3c3102d658e81790fada083945f8ca 

============================================================
  Decode Throughput Benchmark
  seq_len=512  decode_steps=128  batch=1
============================================================

Warming up (30 steps)...

────────────────────────────────────────────────────────────
  Model:            Llama-3 8B
  Context length:   512 → 670
  Decode steps:     128
  Batch size:       1
────────────────────────────────────────────────────────────
  Total time:       3318.58 ms
  Per step:         25.926 ms/tok
  Throughput:       38.6 tok/s
────────────────────────────────────────────────────────────

### Add + RMSNorm fusion
============================================================
  Decode Throughput Benchmark
  seq_len=512  decode_steps=128  batch=1
============================================================

Warming up (30 steps)...

────────────────────────────────────────────────────────────
  Model:            Llama-3 8B
  Context length:   512 → 670
  Decode steps:     128
  Batch size:       1
────────────────────────────────────────────────────────────
  Total time:       3012.60 ms
  Per step:         23.536 ms/tok
  Throughput:       42.5 tok/s
────────────────────────────────────────────────────────────

# HF

Dynamic cache

## HF-8B not compiled

============================================================
  Hugging Face Decode Throughput Benchmark
  seq_len=512  decode_steps=128  batch=1
  dtype=float16  compiled=False  cache=dynamic
============================================================

Initializing HF LlamaModel on cuda (float16)...
Pre-filling DynamicCache to seq_len=512...
Warming up (30 steps)...
Running timed steps...

────────────────────────────────────────────────────────────
  Model:            HF Llama-3 8B
  Context length:   512 → 670
  Decode steps:     128
  Batch size:       1
  Dtype:            float16
  Compiled:         False
────────────────────────────────────────────────────────────
  Total time:       2970.31 ms
  Per step:         23.206 ms/tok
  Throughput:       43.1 tok/s
────────────────────────────────────────────────────────────

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