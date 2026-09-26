"""
Fixed decode-throughput benchmark, derived from benchmarks/bench_throughput.py @ 40dc0b6.

Run against an arbitrary checkout:
    python bench_fixed.py --repo <worktree> [--hooks off|on] [--json-out f.json]

Fixed workload (identical for every commit):
    Llama-3 8B shape config, random fp16 weights (seed 0), batch=1,
    KV cache pre-filled with 512 random tokens, one "run" = 128 decode steps
    covering positions 512..639 (positions reset each run so every run is identical).
    3 warmup runs, then 5 timed runs; report median tok/s.

Adapters (benchmark-side only, no engine code changed):
  * Weight init: instead of assigning hard-coded attribute names (wqkv, w_gate_up, ...)
    which changed over history, walk the model and replace every weight tensor that
    __init__ created (torch.empty / torch.ones) with randn*0.02 fp16 of the SAME shape.
    Non-weight tensors (cos/sin/freqs tables) are just moved to CUDA unchanged.
  * --hooks off: torch.profiler.record_function is replaced by a no-op context manager
    BEFORE `import model`, so the ~400 record_function scopes per step added in 95a4e01
    cost nothing. Commits without record_function are unaffected.
  * The editable-install finder for /workspace/inference is removed from sys.meta_path so
    old checkouts can never silently import HEAD's modules.
"""
import argparse, contextlib, json, os, statistics, sys, time, traceback

p = argparse.ArgumentParser()
p.add_argument("--repo", required=True)
p.add_argument("--hooks", choices=["on", "off"], default="off")
p.add_argument("--seq-len", type=int, default=512)
p.add_argument("--decode-steps", type=int, default=128)
p.add_argument("--warmup-runs", type=int, default=3)
p.add_argument("--timed-runs", type=int, default=5)
p.add_argument("--batch-size", type=int, default=1)
p.add_argument("--json-out")
args = p.parse_args()
if args.json_out: args.json_out = os.path.abspath(args.json_out)

result = {"repo": args.repo, "hooks": args.hooks, "status": "error", "stage": "setup"}


def dump():
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(result, f, indent=1)
    print(json.dumps({k: v for k, v in result.items() if k not in ("runs", "traceback")}))


try:
    repo = os.path.abspath(args.repo)
    # isolate imports to the checkout
    sys.meta_path[:] = [f for f in sys.meta_path if "editable" not in repr(f).lower()]
    sys.path[:] = [x for x in sys.path if os.path.abspath(x or ".") not in ("/workspace/inference",)]
    sys.path.insert(0, repo)
    os.chdir(repo)

    import torch
    import torch.profiler, torch.autograd.profiler

    if args.hooks == "on":
        # model.py gates its scopes on INFERENCE_PROFILE since the profiler-toggle change;
        # older commits ignore this and always run their scopes
        os.environ["INFERENCE_PROFILE"] = "1"
    if args.hooks == "off":
        class _NoOp(contextlib.ContextDecorator):
            def __init__(self, *a, **k): pass
            def __enter__(self): return self
            def __exit__(self, *e): return False
        torch.profiler.record_function = _NoOp
        torch.autograd.profiler.record_function = _NoOp

    result["stage"] = "import"
    import model as M
    bad = [m.__file__ for n, m in list(sys.modules.items())
           if (n == "model" or n == "kernels" or n.startswith("kernels.")) and getattr(m, "__file__", None)
           and not os.path.abspath(m.__file__).startswith(repo)]
    assert not bad, f"imported modules from outside worktree: {bad}"
    result["n_record_function_in_model"] = open(M.__file__).read().count("record_function(")

    result["stage"] = "build"
    torch.manual_seed(0)
    dev = torch.device("cuda")
    cfg = M.LlamaConfig()
    model = M.Llama(cfg)

    WEIGHT_NAMES = {"embed_tokens", "lm_head", "weight"}
    randomized, moved = [], []

    def fill(obj, prefix):
        for name, val in list(vars(obj).items()):
            if isinstance(val, torch.Tensor):
                if name in WEIGHT_NAMES or name.startswith("w"):
                    setattr(obj, name, torch.randn(*val.shape, dtype=torch.float16, device=dev) * 0.02)
                    randomized.append(f"{prefix}{name}{tuple(val.shape)}")
                else:
                    setattr(obj, name, val.to(dev))
                    moved.append(f"{prefix}{name}{tuple(val.shape)}:{val.dtype}")

    fill(model, "")
    for sub in ("norm",):
        if hasattr(model, sub): fill(getattr(model, sub), sub + ".")
    for i, layer in enumerate(model.layers):
        for sub in ("self_attn", "mlp", "input_layernorm", "post_attention_layernorm"):
            if hasattr(layer, sub): fill(getattr(layer, sub), f"L{i}.{sub}.")
        fill(layer, f"L{i}.")
    result["randomized_example"] = [r for r in randomized if not r.startswith("L") or r.startswith("L0.")]
    result["moved"] = sorted(set(m.split(".", 1)[-1] if m.startswith("L") else m for m in moved))

    kv = model.allocate_kv_cache(batch_size=args.batch_size, max_seq_len=cfg.max_position_embeddings, device=dev)
    for K, V in kv:
        K[:, :, :args.seq_len] = torch.randn_like(K[:, :, :args.seq_len]) * 0.02
        V[:, :, :args.seq_len] = torch.randn_like(V[:, :, :args.seq_len]) * 0.02
    tok = torch.zeros(args.batch_size, 1, dtype=torch.long, device=dev)
    torch.cuda.synchronize()

    def one_run():
        torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        t0 = time.perf_counter(); e0.record()
        out = None
        for i in range(args.decode_steps):
            out = model.forward(tok, start_pos=args.seq_len + i, kv_caches=kv)
        t_enq = time.perf_counter()
        e1.record()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        return dict(wall_ms=(t1 - t0) * 1e3, cuda_ms=e0.elapsed_time(e1), enqueue_ms=(t_enq - t0) * 1e3,
                    finite=bool(torch.isfinite(out).all().item()) if out is not None else None)

    result["stage"] = "warmup"
    for _ in range(args.warmup_runs):
        one_run()
    result["stage"] = "timed"
    runs = [one_run() for _ in range(args.timed_runs)]
    toks = args.decode_steps * args.batch_size
    for r in runs:
        r["tok_s"] = toks / (r["wall_ms"] / 1e3)
    result["runs"] = runs
    result["median_tok_s"] = statistics.median(r["tok_s"] for r in runs)
    result["min_tok_s"] = min(r["tok_s"] for r in runs)
    result["max_tok_s"] = max(r["tok_s"] for r in runs)
    result["median_ms_per_tok"] = statistics.median(r["wall_ms"] for r in runs) / args.decode_steps
    result["median_enqueue_ms_per_tok"] = statistics.median(r["enqueue_ms"] for r in runs) / args.decode_steps
    result["output_finite"] = all(r["finite"] for r in runs)
    result["status"] = "ok"
    result["stage"] = "done"
    result["env"] = dict(torch=torch.__version__, gpu=torch.cuda.get_device_name())
except BaseException as e:
    result["error"] = f"{type(e).__name__}: {e}"[:500]
    result["traceback"] = traceback.format_exc()[-3000:]
dump()
