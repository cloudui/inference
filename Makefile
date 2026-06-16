# Makefile for High-Performance Inference Engine
# Provides helper commands for running tests and benchmarks

.PHONY: help test test-attention test-decode test-flash-decode test-flash-decode-gpu test-forward benchmark benchmark-small benchmark-chrome benchmark-small-chrome benchmark-hf benchmark-hf-small benchmark-hf-chrome benchmark-hf-small-chrome bench-throughput bench-throughput-small bench-throughput-hf bench-throughput-hf-small

# Default target: show help message
help:
	@echo "======================================================================"
	@echo "                       Inference Engine Commands                      "
	@echo "======================================================================"
	@echo "Test Commands:"
	@echo "  make test                  - Run all tests (pytest & python scripts)"
	@echo "  make test-attention        - Run GPU correctness test for custom Attention"
	@echo "  make test-decode           - Run DecoderLayer correctness tests via pytest"
	@echo "  make test-flash-decode     - Run CPU reference comparison for flash-decode"
	@echo "  make test-flash-decode-gpu - Run Triton flash-decode GPU sanity check"
	@echo "  make test-forward          - Run Llama full model forward tests via pytest"
	@echo ""
	@echo "Benchmark Commands:"
	@echo "  make benchmark             - Run decode profiler (Llama-3 8B defaults)"
	@echo "  make benchmark-small       - Run decode profiler with a tiny 2-layer config"
	@echo "  make benchmark-chrome      - Run decode profiler and export Chrome trace"
	@echo "  make benchmark-hf          - Run HF decode profiler (Llama-3 8B defaults)"
	@echo "  make benchmark-hf-small    - Run HF decode profiler with a tiny 2-layer config"
	@echo "  make benchmark-hf-chrome   - Run HF decode profiler and export Chrome trace"
	@echo "  make benchmark-hf-small-chrome - Run HF decode profiler with tiny config and export trace"
	@echo "  make bench-throughput      - Run throughput benchmark (Llama-3 8B defaults)"
	@echo "  make bench-throughput-small - Run throughput benchmark with a tiny 2-layer config"
	@echo "  make bench-throughput-hf   - Run HF throughput benchmark (Llama-3 8B defaults)"
	@echo "  make bench-throughput-hf-small - Run HF throughput benchmark with a tiny 2-layer config"
	@echo "======================================================================"

# Run all tests
test: test-flash-decode test-flash-decode-gpu test-attention test-decode test-forward

# Individual Test Targets
test-attention:
	python tests/test_attention.py

test-decode:
	pytest tests/test_decode.py -v

test-flash-decode:
	python tests/test_flash_decode.py

test-flash-decode-gpu:
	python tests/test_flash_decode_gpu.py

test-forward:
	pytest tests/test_forward.py -v

# Benchmarking & Profiling Targets
benchmark:
	python benchmarks/profile_decode.py

benchmark-small:
	python benchmarks/profile_decode.py --small

benchmark-small-chrome:
	python benchmarks/profile_decode.py --small --export-chrome

benchmark-chrome:
	python benchmarks/profile_decode.py --export-chrome

# Throughput Benchmark Targets (Custom)
bench-throughput:
	python benchmarks/bench_throughput.py

bench-throughput-small:
	python benchmarks/bench_throughput.py --small

# HF Benchmark & Profiling Targets
benchmark-hf:
	python benchmarks/profile_decode_hf.py

benchmark-hf-small:
	python benchmarks/profile_decode_hf.py --small

benchmark-hf-chrome:
	python benchmarks/profile_decode_hf.py --export-chrome

benchmark-hf-small-chrome:
	python benchmarks/profile_decode_hf.py --small --export-chrome

bench-throughput-hf:
	python benchmarks/bench_throughput_hf.py

bench-throughput-hf-small:
	python benchmarks/bench_throughput_hf.py --small