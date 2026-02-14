---
title: "Documentation"
date: 2024-05-09T15:26:15Z
draft: false
weight: 1
---

# LLMIR Documentation

Welcome to the LLMIR technical documentation. This section provides detailed information about the architecture, components, and features of the LLMIR compiler infrastructure.

## Architecture Overview

LLMIR (Large Language Model Intermediate Representation) is a compiler infrastructure for large language models based on MLIR, designed to optimize and accelerate LLM inference through specialized compilation techniques.

[Read more about LLMIR's architecture →](/docs/architecture)

## Key Features

LLMIR includes several key features designed to optimize LLM inference:

### KV Cache Optimization

Efficient management of key-value caches for transformer-based LLMs, including block-based allocation, optimized memory access patterns, and specialized attention operations.

[Learn about KV Cache optimization →](/docs/architecture/KVCache)

### Quantization Support

Comprehensive quantization capabilities for reducing model size and improving inference performance, including various quantization strategies and hardware-specific optimizations.

[Explore quantization in LLMIR →](/docs/architecture/Quantization)

### Distributed Deployment

Support for executing large models across multiple devices and nodes through tensor parallelism, pipeline parallelism, and efficient memory management.

[Discover distributed deployment capabilities →](/docs/architecture/DistributedDeployment)

### Performance Evaluation

Benchmarking and evaluation methodologies for measuring LLMIR's impact on inference performance across different models and hardware platforms.

[View performance evaluation approaches →](/docs/architecture/PerformanceEvaluation)

## Development Status

LLMIR has completed Phases 1–6 of its development roadmap:

1. **Phase 1–2** ✅: Core infrastructure, MLIR dialect, KV cache management, attention fusion
2. **Phase 3** ✅: Quantization (INT8/INT4), tensor/pipeline parallelism, multi-GPU sharding
3. **Phase 4** ✅: Speculative decoding, prefix caching, adaptive block management
4. **Phase 5** ✅: Continuous batching, vLLM integration, comprehensive benchmarks
5. **Phase 6** ✅: Python bindings, model-specific optimizations (Llama, Mistral, Phi), profiling tools
6. **Phase 7** (Planned): HuggingFace integration, distributed training, Kubernetes support

**Verification**: C++ MLIR dialect builds with MLIR 18; 84/84 unit tests passed. See [Publications](/pubs/) for the ICCD 2025 paper.

For more information on contributing to LLMIR, please see the [Developer Guide](/getting_started/DeveloperGuide/). 