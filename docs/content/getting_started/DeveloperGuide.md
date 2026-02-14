---
title: "Developer Guide"
date: 2024-05-09T15:26:15Z
lastmod: 2025-02-04T15:26:15Z
draft: false
---

# LLMIR Developer Guide

This guide provides an overview of how to develop with LLMIR.

## Quick Start

### Python (Recommended for most users)

```bash
pip install llmir
# Or with optional dependencies:
pip install llmir[dev]    # Development tools (pytest, black, mypy)
pip install llmir[full]   # Full stack with torch and transformers
```

```python
import llmir

config = llmir.KVCacheConfig(num_layers=32, num_heads=32, head_dim=128)
cache = llmir.PagedKVCache(config)

optimizer = llmir.LlamaOptimizer.for_llama3_8b()
kv_config = optimizer.get_optimized_kv_cache_config()
```

### C++ MLIR Dialect Build

For building the LLM dialect with MLIR 18:

```bash
git clone https://github.com/chenxingqiang/llmir.git
cd llmir

# Standalone LLM dialect build (see build_llm_dialect/)
cd build_llm_dialect
mkdir build && cd build
cmake -G Ninja ..
ninja
```

## Building LLMIR from Source

LLMIR is built on top of the MLIR ecosystem. Prerequisites:

1. C++ compiler (GCC or Clang) with C++17 support
2. CMake 3.13.4+
3. Python 3.8+ (for bindings)
4. Ninja or Make

### Clone and Build

```bash
git clone https://github.com/chenxingqiang/llmir.git
cd llmir
mkdir build && cd build
cmake -G Ninja ..
ninja
ninja check-llmir   # Run tests
```

## LLMIR Project Structure

```
include/mlir/Dialect/LLM/
  ├── IR/                     # Dialect ops: LLM.td, LLMTypes.td
  └── Runtime/                # PagedKVCache.h, QuantizedKVCache.h, etc.

lib/Dialect/LLM/
  ├── IR/                     # LLMDialect.cpp, LLMOps.cpp, LLMTypes.cpp
  ├── Transforms/             # KVCacheOptimization.cpp
  └── Runtime/                # AttentionOpt.cpp, PagedKVCache impl

build_llm_dialect/            # Standalone C++ dialect build (MLIR 18)
llm_dialect_build/            # C++ dialect test harness

python/mlir/dialects/llm/     # Python bindings
benchmark/LLM/                # Benchmarks
test/Dialect/LLM/             # MLIR lit tests
tests/                        # Python pytest (84 tests)
IEEE-conference/              # ICCD 2025 paper, figures, verification
examples/                     # demo_llmir_0.6b.py, etc.
```

## Running Benchmarks

```bash
# Real model benchmark (Qwen2.5-7B, vLLM comparison)
./run_real_benchmark.sh

# Comprehensive benchmark with vLLM and SGLang
./comprehensive_benchmark.sh

# Quick vLLM comparison
./vllm_comparison.sh

# Llama-3.1 benchmark (see benchmark/LLM/)
cd benchmark/LLM
./setup_llama31_benchmark.sh
./run_llama31_benchmark.sh
```

## Running Tests

```bash
# Python tests (84 tests)
python -m pytest tests/ -v

# C++ LLM dialect tests
cd build_llm_dialect/build
ninja
./tools/llmir-opt ../test/Dialect/LLM/kv_cache_ops.mlir -kv-cache-optimization
```

## Core Components (Implemented)

- **LLM MLIR Dialect**: `llm.append_kv`, `llm.lookup_kv`, `llm.paged_attention`; `!llm.paged_kv_cache`
- **KV Cache**: PagedKVCache, QuantizedKVCache (INT8/INT4), DistributedKVCache
- **Advanced**: SpeculativeKVCache, PrefixCache, ContinuousBatchingEngine
- **Model Optimizers**: LlamaOptimizer, MistralOptimizer, PhiOptimizer, ModelRegistry
- **Profiling**: Profiler, LatencyProfiler, ThroughputMonitor

## Contributing to LLMIR

Contributions are welcome. See [Contributing](/getting_started/Contributing/).

## Example: KV Cache in MLIR

Here's an example of how a paged KV cache might be represented in LLMIR (syntax may evolve as the project develops):

```mlir
// Create a paged KV cache type
!kv_cache_t = !llm.paged_kv_cache<f16, 12, 16, 64, 16, 4096>

// Append key-value pairs to the cache
%new_kv, %block_indices = llm.append_kv %kv_cache, %keys, %values, %seq_ids {
  block_size = 16 : i32,
  max_seq_len = 4096 : i32
} : (!kv_cache_t, tensor<2x1x16x64xf16>, tensor<2x1x16x64xf16>, tensor<2xi32>) 
    -> (!kv_cache_t, tensor<2x1xi32>)

// Perform paged attention with the KV cache
%output = llm.paged_attention %query, %new_kv, %block_indices, %seq_lens {
  num_heads = 16 : i32,
  head_dim = 64 : i32,
  scale = 0.125 : f32
} : (tensor<2x1x16x64xf16>, !kv_cache_t, tensor<2x128xi32>, tensor<2xi32>) 
    -> tensor<2x1x16x64xf16>
```

For more details on the project roadmap and architecture, please refer to our [GitHub repository](https://github.com/chenxingqiang/llmir.git).
