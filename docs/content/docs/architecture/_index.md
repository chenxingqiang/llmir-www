---
title: "Architecture"
date: 2024-05-09T15:26:15Z
lastmod: 2025-02-04T15:26:15Z
draft: false
weight: 2
---

# LLMIR Architecture

This section provides detailed information about the LLMIR architecture, its key components, and features.

## Key Features

LLMIR implements several key optimizations for LLM inference (Phases 1–6 complete):

* [KV Cache Optimization](KVCache): Block-based PagedKVCache, QuantizedKVCache (INT8/INT4), SpeculativeKVCache, PrefixCache
* [Quantization Support](Quantization): INT8 (4×) and INT4 (8×) compression for KV cache and weights
* [Distributed Deployment](DistributedDeployment): Multi-GPU sharding (layer-wise, head-wise, sequence-wise)
* [Performance Evaluation](PerformanceEvaluation): Verified benchmarks on A800 GPUs; vLLM/SGLang/TensorRT-LLM/MLC-LLM comparisons

## System Architecture

LLMIR (Large Language Model Intermediate Representation) is a compiler infrastructure for large language models based on MLIR, designed to optimize and accelerate LLM inference through specialized compilation techniques.

LLMIR follows a layered architecture:

```
                       ┌─────────────────┐
                       │   Application   │
                       │ vLLM / SGLang   │
                       └────────┬────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────┐
│                    LLMIR Compiler                │
│                                                  │
│  ┌──────────────┐    ┌───────────────────────┐   │
│  │ Front-end    │ → │  MLIR Optimization     │   │
│  │ Converters   │    │  Pipeline             │   │
│  └──────────────┘    └───────────┬───────────┘   │
│                                  │               │
│                      ┌───────────▼───────────┐   │
│                      │    Backend Generators │   │
│                      └───────────────────────┘   │
└──────────────────────────┬───────────────────────┘
                           │
                           ▼
            ┌─────────────────────────────┐
            │       Execution Layer       │
            │ CUDA / ROCm / LLVM / Accel  │
            └─────────────────────────────┘
```

### Front-end Converters

The front-end converts models and operations from existing frameworks into the LLMIR representation:

- **vLLM Integration**: vLLM-compatible API via `VLLMIntegration.h`; drop-in compatibility for vLLM-based applications
- **SGLang Support**: Maps SGLang computation graphs to LLMIR operations for cross-framework optimization

### MLIR Optimization Pipeline

The optimization pipeline (`KVCacheOptimization.cpp`) implements passes for LLM inference:

- **Block Size Optimization**: Automatic optimal block size selection (16/32/64/128 by sequence length)
- **Duplicate KV Fusion**: Fuse duplicate KV cache operations
- **Cross-Sequence Sharing**: Detect and enable prefix sharing opportunities
- **PagedAttention Optimization**: Scale factor and memory layout optimizations
- **Attention Variants**: Flash Attention, Fused Softmax, Sliding Window, Optimized Masked (1.28×–2.15× speedup)

### Backend Generators

Backend generators produce optimized code for different execution targets:

- **CUDA/HIP Code Generation**: For NVIDIA and AMD GPUs
- **LLVM IR Generation**: For CPUs and general platforms
- **Specialized Accelerator Code**: For ML accelerators like TPUs

### Runtime Library

LLMIR includes a runtime library (`include/mlir/Dialect/LLM/Runtime/`) with:

- **PagedKVCache**: Block-based allocation, `appendKV`, `lookupKV`; verified on A800 GPUs
- **QuantizedPagedKVCache**: INT8/INT4 with automatic quantization/dequantization
- **DistributedPagedKVCache**: Layer/head/sequence-wise sharding across multiple GPUs
- **SpeculativeKVCache**: Branch creation, rollback, draft token verification
- **PrefixCache**: Radix tree-based prefix reuse, system prompt caching
- **ContinuousBatchingEngine**: vLLM-style dynamic batch management
- **ModelOptimizations**: Llama, Mistral, Phi presets; `LlamaOptimizer`, `MistralOptimizer`, `PhiOptimizer`

## LLMIR Dialect

The core of LLMIR is a specialized MLIR dialect for LLM operations:

- **Types**: `!llm.paged_kv_cache`, `ShardedTensorType`, `QuantizedTensorType`
- **Operations**: `llm.append_kv`, `llm.lookup_kv`, `llm.paged_attention`
- **Build**: C++ dialect builds with MLIR 18; see [build_llm_dialect](https://github.com/chenxingqiang/llmir/tree/main/build_llm_dialect)

For detailed information about specific features, please visit the dedicated pages listed above.

## Development Status

LLMIR has completed Phases 1–6. Current status:

1. **Phase 1–2** ✅: Core infrastructure, MLIR dialect, KV cache, attention fusion
2. **Phase 3–4** ✅: Quantization, multi-GPU sharding, speculative decoding, prefix caching, adaptive block management
3. **Phase 5–6** ✅: Continuous batching, vLLM integration, benchmarks, Python bindings, model optimizers, profiling
4. **Phase 7** (Planned): HuggingFace integration, distributed training, Kubernetes support

Verification: 84/84 Python tests passed; C++ dialect verified with MLIR 18. See [Performance Evaluation](PerformanceEvaluation) and [Publications](/pubs/).

## References

For a comprehensive list of related work and publications that have influenced LLMIR, please see our [References](../References) page. 