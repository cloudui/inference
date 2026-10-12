"""
bench_throughput_vllm.py — Decode tok/s benchmark for vLLM, same workload as bench_throughput.py.

Runs in its own virtualenv (vLLM pins its own torch):
    python -m venv /workspace/venvs/vllm && /workspace/venvs/vllm/bin/pip install vllm
    /workspace/venvs/vllm/bin/python benchmarks/bench_throughput_vllm.py [--seq-len 512] [--decode-steps 128]

Llama 3.1 8B config, random fp16 weights (load_format="dummy", no checkpoint needed).
vLLM can't be handed a pre-filled KV cache, so each run sends batch_size seq_len-token prompts at
once, lets vLLM prefill them, then generates warmup + decode_steps tokens for each. The engine is driven one step() at a
time and only decode steps are timed (host wall clock per step, which is what a user sees), so
prefill time and its variance don't leak into the number. With --batch-size N, a decode step is
one where all N requests produced a token, so steps that mix prefill and decode are skipped. Each run is appended to
benchmarks/results/throughput_runs.csv.
"""

import argparse
import json
import os
import statistics
import tempfile
import time
from pathlib import Path

LLAMA31_8B = {
    "architectures": ["LlamaForCausalLM"],
    "model_type": "llama",
    "hidden_size": 4096,
    "intermediate_size": 14336,
    "num_hidden_layers": 32,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "hidden_act": "silu",
    "max_position_embeddings": 131072,
    "rms_norm_eps": 1e-5,
    "rope_theta": 500000.0,
    "rope_scaling": {
        "factor": 8.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_max_position_embeddings": 8192,
        "rope_type": "llama3",
    },
    "vocab_size": 128256,
    "bos_token_id": 128000,
    "eos_token_id": 128001,
    "tie_word_embeddings": False,
    "attention_bias": False,
    "mlp_bias": False,
    "torch_dtype": "float16",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seq-len",      type=int, default=512,
                   help="Prompt tokens prefilled before decoding (the prior context)")
    p.add_argument("--decode-steps", type=int, default=128, help="Decode steps to measure")
    p.add_argument("--warmup",       type=int, default=30, help="Decode steps before timing starts")
    p.add_argument("--batch-size",   type=int, default=1,
                   help="Concurrent requests, all with the same prompt length (max_num_seqs)")
    p.add_argument("--runs",         type=int, default=3,
                   help="Requests to time (each re-prefills); reports the median run")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.99)
    p.add_argument("--max-num-batched-tokens", type=int, default=1024,
                   help="Prefill chunk size. Smaller chunks reserve less activation memory, leaving "
                        "room for a 112K KV cache; prefill isn't timed")
    p.add_argument("--no-log",       action="store_true")
    p.add_argument("--note",         type=str, default="")
    return p.parse_args()


def time_one_batch(engine, run_id, prompts, n_tokens, warmup):
    """Per-step wall time (s) for the decode steps after `warmup` generated tokens.

    A decode step is one where every request already had a token and gets exactly one more.
    """
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    params = SamplingParams(max_tokens=n_tokens, ignore_eos=True, temperature=0.0, detokenize=False)
    for i, prompt_ids in enumerate(prompts):
        engine.add_request(f"{run_id}-{i}", TokensPrompt(prompt_token_ids=prompt_ids), params)
    n_out = {f"{run_id}-{i}": 0 for i in range(len(prompts))}
    step_times = []
    while engine.has_unfinished_requests():
        t0 = time.perf_counter()
        outs = engine.step()
        dt = time.perf_counter() - t0
        before = dict(n_out)
        for o in outs:
            if o.outputs:
                n_out[o.request_id] = len(o.outputs[0].token_ids)
        if all(before[r] >= 1 and n_out[r] == before[r] + 1 for r in n_out):
            step_times.append(dt)
    assert all(n == n_tokens for n in n_out.values()), f"generated {n_out} of {n_tokens} tokens"
    return step_times[warmup:]


def main():
    args = parse_args()
    # FlashInfer's JIT sampling module finds no target arch on this sm_120 card and refuses to
    # build; greedy sampling doesn't need it, so use vLLM's PyTorch sampler
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    import torch
    import vllm
    from vllm import LLM

    model_dir = Path(tempfile.mkdtemp(prefix="llama31-8b-dummy-"))
    (model_dir / "config.json").write_text(json.dumps(LLAMA31_8B))

    max_len = args.seq_len + args.warmup + args.decode_steps + 16
    print(f"\n{'='*60}")
    print(f"  vLLM {vllm.__version__} Decode Throughput Benchmark")
    print(f"  seq_len={args.seq_len}  decode_steps={args.decode_steps}  batch={args.batch_size}  max_model_len={max_len}")
    print(f"{'='*60}\n")

    llm = LLM(
        model=str(model_dir),
        load_format="dummy",
        dtype="float16",
        skip_tokenizer_init=True,
        max_model_len=max_len,
        max_num_seqs=args.batch_size,
        # a chunk larger than max_model_len crashes vLLM 0.30 at startup (illegal address)
        max_num_batched_tokens=min(args.max_num_batched_tokens, max_len),
        enable_prefix_caching=False,   # every run must really attend over its own prefilled context
        gpu_memory_utilization=args.gpu_memory_utilization,
        seed=0,
    )
    engine = llm.llm_engine

    g = torch.Generator().manual_seed(0)
    n_tokens = 1 + args.warmup + args.decode_steps
    run_ms = []
    for r in range(args.runs + 1):  # run 0 warms up the engine and is discarded
        prompts = [torch.randint(0, LLAMA31_8B["vocab_size"], (args.seq_len,), generator=g).tolist()
                   for _ in range(args.batch_size)]
        steps = time_one_batch(engine, f"r{r}", prompts, n_tokens, args.warmup)
        if r > 0:
            run_ms.append(statistics.mean(steps) * 1e3)
            print(f"  run {r}: {run_ms[-1]:.3f} ms/tok over {len(steps)} steps "
                  f"(step median {statistics.median(steps) * 1e3:.3f} ms)")

    per_step_ms = statistics.median(run_ms)
    tok_per_sec = args.batch_size * 1e3 / per_step_ms
    print(f"\n{'─'*60}")
    print(f"  Model:            vLLM Llama-3.1 8B (dummy weights)")
    print(f"  Context length:   {args.seq_len} → {args.seq_len + n_tokens}")
    print(f"  Decode steps:     {args.decode_steps} x {args.runs} runs")
    print(f"  Batch size:       {args.batch_size}")
    print(f"{'─'*60}")
    print(f"  Per step:         {per_step_ms:.3f} ms/tok")
    print(f"  Throughput:       {tok_per_sec:.1f} tok/s")
    print(f"{'─'*60}")
    if not args.no_log:
        import run_log
        run_log.log_run({
            "impl": "vllm",
            "mode": "default",
            "model": "llama3.1-8b",
            "seq_len": args.seq_len,
            "kv_len": max_len,
            "decode_steps": args.decode_steps,
            "warmup": args.warmup,
            "batch_size": args.batch_size,
            "profiler_scopes": False,
            "cuda_graphs": True,
            "tok_s": round(tok_per_sec, 2),
            "ms_per_tok": round(per_step_ms, 3),
            "note": f"vllm {vllm.__version__}; runs {[round(x, 3) for x in run_ms]}; {args.note}".rstrip("; "),
        })
    print()


if __name__ == "__main__":
    main()
