"""
bench_components.py — Lightweight component-level benchmark comparing Custom Llama and HF Llama.
Uses CUDA Events for zero-overhead, highly accurate GPU-side profiling.
"""

import argparse
import time
import torch
from transformers import LlamaConfig as HFLlamaConfig
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from transformers.cache_utils import DynamicCache

# Local imports
from model import Llama, LlamaConfig

DEVICE = torch.device("cuda")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seq-len", type=int, default=512, help="Context sequence length")
    p.add_argument("--steps", type=int, default=50, help="Number of timed decode steps")
    p.add_argument("--warmup", type=int, default=20, help="Number of warmup steps")
    p.add_argument("--small", action="store_true", help="Use a small 2-layer config for quick tests")
    return p.parse_args()

class HFComponentTimer:
    def __init__(self):
        self.starts = {}
        self.elapsed = {}
        self.global_elapsed = []

    def register_hooks(self, model):
        for name, module in model.named_modules():
            class_name = module.__class__.__name__
            targets = ["LlamaAttention", "LlamaMLP", "LlamaRMSNorm"]
            if class_name in targets:
                module.register_forward_pre_hook(self.make_pre_hook(name, class_name))
                module.register_forward_hook(self.make_post_hook(name, class_name))

    def make_pre_hook(self, name, class_name):
        def pre_hook(module, input):
            start = torch.cuda.Event(enable_timing=True)
            start.record()
            self.starts[id(module)] = start
        return pre_hook

    def make_post_hook(self, name, class_name):
        def post_hook(module, input, output):
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            start = self.starts.get(id(module))
            if start is not None:
                norm_name = class_name.replace("Llama", "")
                if norm_name not in self.elapsed:
                    self.elapsed[norm_name] = []
                self.elapsed[norm_name].append((start, end))
        return post_hook

    def get_results(self, num_steps):
        torch.cuda.synchronize()
        results = {}
        for name, events in self.elapsed.items():
            total_ms = 0.0
            for start, end in events:
                total_ms += start.elapsed_time(end)
            results[name] = total_ms / num_steps
        
        total_gpu_forward_ms = 0.0
        for start, end in self.global_elapsed:
            total_gpu_forward_ms += start.elapsed_time(end)
        results["Total_GPU_Forward"] = total_gpu_forward_ms / num_steps
        
        return results


class CustomComponentTimer:
    def __init__(self):
        self.elapsed = {"Attention": [], "MLP": [], "RMSNorm": []}
        self.orig_calls = {}
        self.global_elapsed = []

    def patch_and_register(self, model):
        from model import Attention, MLP, RMSNorm
        
        # Monkey patch class __call__ methods
        for name, cls in [("Attention", Attention), ("MLP", MLP), ("RMSNorm", RMSNorm)]:
            orig_call = cls.__call__
            self.orig_calls[cls] = orig_call
            
            def make_wrapped(c_name, o_call):
                def wrapped(self_obj, *args, **kwargs):
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    out = o_call(self_obj, *args, **kwargs)
                    end.record()
                    if hasattr(self_obj, "_timer_list") and self_obj._timer_list is not None:
                        self_obj._timer_list.append((start, end))
                    return out
                return wrapped
                
            cls.__call__ = make_wrapped(name, orig_call)

        # Attach target lists to model instances
        for layer in model.layers:
            layer.self_attn._timer_list = self.elapsed["Attention"]
            layer.mlp._timer_list = self.elapsed["MLP"]
            layer.input_layernorm._timer_list = self.elapsed["RMSNorm"]
            layer.post_attention_layernorm._timer_list = self.elapsed["RMSNorm"]
        model.norm._timer_list = self.elapsed["RMSNorm"]

    def restore(self):
        for cls, orig_call in self.orig_calls.items():
            cls.__call__ = orig_call

    def get_results(self, num_steps):
        torch.cuda.synchronize()
        results = {}
        for name, events in self.elapsed.items():
            total_ms = 0.0
            for start, end in events:
                total_ms += start.elapsed_time(end)
            results[name] = total_ms / num_steps
        
        total_gpu_forward_ms = 0.0
        for start, end in self.global_elapsed:
            total_gpu_forward_ms += start.elapsed_time(end)
        results["Total_GPU_Forward"] = total_gpu_forward_ms / num_steps
        
        return results


def setup_custom_model(args):
    if args.small:
        cfg = LlamaConfig(
            hidden_size=512,
            num_hidden_layers=2,
            num_attention_heads=8,
            num_key_value_heads=2,
            intermediate_size=1024,
            vocab_size=1024,
            max_position_embeddings=2048,
            head_dim=64,
        )
    else:
        cfg = LlamaConfig()

    model = Llama(cfg)
    
    def rand_fp16(*shape):
        return torch.randn(*shape, dtype=torch.float16, device=DEVICE) * 0.02

    model.embed_tokens = rand_fp16(cfg.vocab_size, cfg.hidden_size)
    model.lm_head      = rand_fp16(cfg.vocab_size, cfg.hidden_size)
    model.norm.weight   = rand_fp16(cfg.hidden_size)
    model.cos           = model.cos.to(DEVICE)
    model.sin           = model.sin.to(DEVICE)

    for layer in model.layers:
        qkv_concat_dim_size = cfg.num_attention_heads * cfg.head_dim + 2 * cfg.num_key_value_heads * cfg.head_dim
        layer.self_attn.wqkv = rand_fp16(qkv_concat_dim_size, cfg.hidden_size)
        layer.self_attn.wo = rand_fp16(cfg.hidden_size, cfg.num_attention_heads * cfg.head_dim)
        layer.input_layernorm.weight          = rand_fp16(cfg.hidden_size)
        layer.post_attention_layernorm.weight  = rand_fp16(cfg.hidden_size)
        layer.mlp.w_gate_up = rand_fp16(2 * cfg.intermediate_size, cfg.hidden_size)
        layer.mlp.w_down = rand_fp16(cfg.hidden_size, cfg.intermediate_size)

    kv_caches = model.allocate_kv_cache(batch_size=1, max_seq_len=cfg.max_position_embeddings, device=DEVICE)
    return model, kv_caches, cfg


def setup_hf_model(args):
    if args.small:
        hf_config = HFLlamaConfig(
            hidden_size=512,
            num_hidden_layers=2,
            num_attention_heads=8,
            num_key_value_heads=2,
            intermediate_size=1024,
            vocab_size=1024,
            max_position_embeddings=2048,
            rope_theta=10000.0,
            attn_implementation="sdpa",
        )
    else:
        hf_config = HFLlamaConfig(
            rope_theta=10000.0,
            attn_implementation="sdpa",
        )
        
    # Instantiate directly on GPU in target precision
    old_default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float16)
    with torch.device(DEVICE):
        model = LlamaForCausalLM(config=hf_config)
    torch.set_default_dtype(old_default_dtype)
    model.eval()
    
    hf_cache = DynamicCache()
    history_k = torch.randn(1, hf_config.num_key_value_heads, args.seq_len, hf_config.hidden_size // hf_config.num_attention_heads, device=DEVICE, dtype=torch.float16)
    history_v = torch.randn(1, hf_config.num_key_value_heads, args.seq_len, hf_config.hidden_size // hf_config.num_attention_heads, device=DEVICE, dtype=torch.float16)
    for i in range(hf_config.num_hidden_layers):
        hf_cache.update(history_k, history_v, layer_idx=i)
        
    return model, hf_cache, hf_config


def run_benchmark(model, kv_cache, is_hf, args):
    if is_hf:
        timer = HFComponentTimer()
        timer.register_hooks(model)
    else:
        timer = CustomComponentTimer()
        timer.patch_and_register(model)

    token_ids = torch.zeros(1, 1, dtype=torch.long, device=DEVICE)
    position_ids = torch.tensor([[args.seq_len]], device=DEVICE)

    # Warmup
    for i in range(args.warmup):
        pos = args.seq_len + i
        with torch.inference_mode():
            if is_hf:
                position_ids.fill_(pos)
                model(input_ids=token_ids, past_key_values=kv_cache, position_ids=position_ids, use_cache=True)
            else:
                model.forward(token_ids, start_pos=pos, kv_caches=kv_cache)
    torch.cuda.synchronize()

    # Reset timers without breaking instance references
    if not is_hf:
        for lst in timer.elapsed.values():
            lst.clear()
    else:
        timer.elapsed.clear()
    timer.global_elapsed.clear()

    # Timed run
    start_pos = args.seq_len + args.warmup
    cpu_latencies = []
    
    for i in range(args.steps):
        pos = start_pos + i
        
        g_start = torch.cuda.Event(enable_timing=True)
        g_end = torch.cuda.Event(enable_timing=True)
        
        t0 = time.perf_counter()
        
        g_start.record()
        with torch.inference_mode():
            if is_hf:
                position_ids.fill_(pos)
                model(input_ids=token_ids, past_key_values=kv_cache, position_ids=position_ids, use_cache=True)
            else:
                model.forward(token_ids, start_pos=pos, kv_caches=kv_cache)
        g_end.record()
        
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        
        cpu_latencies.append((t1 - t0) * 1000.0) # ms
        timer.global_elapsed.append((g_start, g_end))

    # Compile results
    results = timer.get_results(args.steps)
    results["Wall_Clock_Forward"] = sum(cpu_latencies) / args.steps
    
    if not is_hf:
        timer.restore()
        
    return results

def main():
    args = parse_args()
    
    print("==========================================================")
    print("  Zero-Overhead Component-Level CUDA Event Benchmark  ")
    print(f"  seq_len={args.seq_len}  steps={args.steps}  small={args.small}")
    print("==========================================================\n")

    print("Setting up Hugging Face Llama Model...")
    hf_model, hf_cache, hf_cfg = setup_hf_model(args)
    print("Running Hugging Face Llama Benchmark...")
    hf_results = run_benchmark(hf_model, hf_cache, is_hf=True, args=args)

    # Free memory
    del hf_model
    del hf_cache
    torch.cuda.empty_cache()

    print("Setting up Custom Llama Model...")
    custom_model, custom_caches, custom_cfg = setup_custom_model(args)
    print("Running Custom Llama Benchmark...")
    custom_results = run_benchmark(custom_model, custom_caches, is_hf=False, args=args)

    # Print Report
    print("\n" + "="*80)
    print(f"{'Component Profiling Results (Average ms per step)':^80}")
    print("="*80)
    print(f"{'Component':<25} | {'Hugging Face':<15} | {'Custom Engine':<15} | {'Speedup':<15}")
    print("-"*80)
    
    num_layers = hf_cfg.num_hidden_layers
    
    components = ["RMSNorm", "Attention", "MLP"]
    for c in components:
        hf_total = hf_results.get(c, 0.0)
        custom_total = custom_results.get(c, 0.0)
        speedup = f"{hf_total / custom_total:.2f}x" if custom_total > 0 else "N/A"
        print(f"{c + ' (Total ' + str(num_layers) + ' layers)':<25} | {hf_total:>12.3f} ms | {custom_total:>12.3f} ms | {speedup:>12}")

    print("-"*80)
    
    # Sum of active components
    hf_sum = sum(hf_results.get(c, 0.0) for c in components)
    custom_sum = sum(custom_results.get(c, 0.0) for c in components)
    sum_speedup = f"{hf_sum / custom_sum:.2f}x" if custom_sum > 0 else "N/A"
    print(f"{'Sum of Fused Components':<25} | {hf_sum:>12.3f} ms | {custom_sum:>12.3f} ms | {sum_speedup:>12}")
    
    # Overall timings
    hf_gpu = hf_results["Total_GPU_Forward"]
    custom_gpu = custom_results["Total_GPU_Forward"]
    gpu_speedup = f"{hf_gpu / custom_gpu:.2f}x" if custom_gpu > 0 else "N/A"
    print(f"{'Total GPU Forward Time':<25} | {hf_gpu:>12.3f} ms | {custom_gpu:>12.3f} ms | {gpu_speedup:>12}")
    
    # GPU Idle Bubbles / Gaps
    hf_bubbles = hf_gpu - hf_sum
    custom_bubbles = custom_gpu - custom_sum
    bubble_speedup = f"{hf_bubbles / custom_bubbles:.2f}x" if custom_bubbles > 0 else "N/A"
    print(f"{'GPU Idle Bubbles / Other':<25} | {hf_bubbles:>12.3f} ms | {custom_bubbles:>12.3f} ms | {'(Lower is better)'}")
    
    print("-"*80)
    
    hf_wall = hf_results["Wall_Clock_Forward"]
    custom_wall = custom_results["Wall_Clock_Forward"]
    wall_speedup = f"{hf_wall / custom_wall:.2f}x" if custom_wall > 0 else "N/A"
    print(f"{'Wall-Clock Forward Time':<25} | {hf_wall:>12.3f} ms | {custom_wall:>12.3f} ms | {wall_speedup:>12}")
    
    # Gaps & overhead
    hf_launch_overhead = hf_wall - hf_gpu
    custom_launch_overhead = custom_wall - custom_gpu
    print(f"{'CPU Launch Overhead':<25} | {hf_launch_overhead:>12.3f} ms | {custom_launch_overhead:>12.3f} ms | {'(Lower is better)'}")
    
    print("="*80)
    
    # Print key observations
    print("\nKey Takeaways:")
    if custom_gpu < hf_gpu:
        reduction = (hf_gpu - custom_gpu) / hf_gpu * 100
        print(f"  ✓ Custom model kernels are {reduction:.1f}% faster than HF on the GPU!")
    else:
        print(f"  ⚠️ Custom model kernels are slower on the GPU by {(custom_gpu - hf_gpu)/hf_gpu*100:.1f}%.")
        
    print(f"  • HF GPU Idle Bubbles:     {hf_bubbles:.3f} ms")
    print(f"  • Custom GPU Idle Bubbles: {custom_bubbles:.3f} ms")
    
    bubble_diff = custom_bubbles - hf_bubbles
    if bubble_diff > 0:
        print(f"  ⚠️ Custom model has {bubble_diff:.3f} ms more GPU idle bubble time per step than HF.")
        print("    This is the CPU launch dispatch gap! CUDA Graphs will eliminate this entirely.")
    else:
        print("  ✓ Custom model has excellent launch latency and minimal pipeline bubbles.")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
