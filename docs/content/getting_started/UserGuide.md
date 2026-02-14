---
title: "User Guide"
date: 2025-02-04T15:26:15Z
draft: false
weight: 5
---

# LLMIR User Guide

This guide helps you get started using the llmir repository for LLM inference optimization.

## Installation

### Option 1: PyPI (Recommended)

```bash
pip install llmir
```

**Optional extras:**

| Extra | Purpose |
|-------|---------|
| `llmir[dev]` | pytest, black, mypy for development |
| `llmir[full]` | torch, transformers for full model workflows |

### Option 2: From Source

```bash
git clone https://github.com/chenxingqiang/llmir.git
cd llmir
pip install -e .
# Or with extras: pip install -e ".[full]"
```

### Option 3: C++ MLIR Dialect Only

If you only need the C++ dialect (no Python):

```bash
cd llmir/build_llm_dialect
mkdir build && cd build
cmake -G Ninja ..
ninja
```

---

## Use Cases

### 1. KV Cache for Custom Inference

```python
import llmir

# Configure for your model (e.g., Llama 7B: 32 layers, 32 heads, 128 head_dim)
config = llmir.KVCacheConfig(num_layers=32, num_heads=32, head_dim=128)
cache = llmir.PagedKVCache(config)

# In your inference loop:
# cache.append(keys, values, seq_ids)   # Append new KV pairs
# cache.lookup(block_indices, seq_lens) # Lookup for attention
```

### 2. Model-Specific Optimization

```python
# Llama 3.1 8B
optimizer = llmir.LlamaOptimizer.for_llama3_8b()
kv_config = optimizer.get_optimized_kv_cache_config()

# Mistral 7B (sliding window attention)
optimizer = llmir.MistralOptimizer.for_mistral_7b()
config = optimizer.get_optimized_kv_cache_config()

# Phi-3
optimizer = llmir.PhiOptimizer.for_phi3_mini()
config = optimizer.get_optimized_kv_cache_config()
```

### 3. Memory-Constrained: Quantized KV Cache

```python
quant_config = llmir.QuantizationConfig(quant_type=llmir.QuantizationType.INT8)
quant_cache = llmir.QuantizedKVCache(config, quant_config)
# ~4x memory reduction with INT8, ~8x with INT4
print(f"Compression: {quant_cache.get_compression_ratio()}x")
```

### 4. Performance Profiling

```python
profiler = llmir.Profiler()
profiler.start()

with profiler.trace("attention"):
    # Your attention / inference code
    run_attention_step()

profiler.stop()
report = profiler.get_report()
report.print_summary()  # Latency, throughput, memory
```

### 5. Benchmarking Against vLLM

From the repo root:

```bash
./run_real_benchmark.sh      # Qwen2.5-7B, PyTorch vs vLLM
./vllm_comparison.sh         # Quick vLLM comparison
./comprehensive_benchmark.sh # Full suite with SGLang
```

---

## Repository Layout (What to Use)

| Path | For Users |
|------|-----------|
| `src/` | Python package source (used by `pip install`) |
| `examples/` | `demo_llmir_0.6b.py` – runnable demos |
| `benchmark/LLM/` | Benchmark scripts, Llama-3.1 setup |
| `*.sh` (root) | `run_real_benchmark.sh`, `vllm_comparison.sh` |
| `tests/` | Test examples; run with `pytest tests/ -v` |
| `IEEE-conference/` | Paper, figures, verification reports |
| `build_llm_dialect/` | C++ dialect build (advanced) |

---

## Supported Models

| Family | Sizes | Optimizer |
|--------|-------|-----------|
| Llama 1/2/3/3.1 | 7B–405B | `LlamaOptimizer` |
| Mistral | 7B | `MistralOptimizer` |
| Mixtral | 8x7B, 8x22B | `MistralOptimizer` |
| Phi | Phi-2, Phi-3 | `PhiOptimizer` |

Use `ModelRegistry` for presets:

```python
registry = llmir.ModelRegistry.get_instance()
config = registry.get_config("llama3.1-8b")  # or "mixtral-8x7b", "phi-3-mini", etc.
```

---

## Hardware & Requirements

- **Python**: 3.8+
- **GPU**: NVIDIA (CUDA) for best performance; verified on A800 80GB
- **RAM**: Depends on model size; see `ModelMemoryEstimator` for planning
- **Optional**: vLLM, SGLang for baseline comparisons

---

## Getting Help

| Resource | Link |
|----------|------|
| **GitHub repo** | [github.com/chenxingqiang/llmir](https://github.com/chenxingqiang/llmir) |
| **Issues** | [github.com/chenxingqiang/llmir/issues](https://github.com/chenxingqiang/llmir/issues) |
| **Documentation** | [This site](/docs/architecture/) |
| **Forums** | [LLVM Discourse – LLMIR](https://llvm.discourse.group/c/llmir/31) |
| **Chat** | [LLVM Discord – LLMIR channel](https://discord.gg/xS7Z362) |
| **FAQ** | [FAQ](/getting_started/Faq/) |

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: llmir` | Run `pip install llmir` or install from source with `pip install -e .` |
| Benchmark script fails | Ensure vLLM is installed (`pip install vllm`) for vLLM comparison scripts |
| C++ dialect build fails | Check MLIR 18 is available; see `build_llm_dialect/README` |
| Out of memory | Use `QuantizedKVCache` (INT8/INT4) or reduce batch size; see `ModelMemoryEstimator` |

---

## Next Steps

- [Developer Guide](DeveloperGuide/) – Build from source, C++ dialect
- [Architecture](/docs/architecture/) – KV cache, quantization, distributed
- [Performance Evaluation](/docs/architecture/PerformanceEvaluation/) – Benchmark results
- [Publications](/pubs/) – Paper, performance highlights
