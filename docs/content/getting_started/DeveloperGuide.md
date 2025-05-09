---
title: "Developer Guide"
date: 2024-05-09T15:26:15Z
draft: false
---

# LLMIR Developer Guide

This guide provides an overview of how to develop with LLMIR.

## Building LLMIR

LLMIR is built on top of the MLIR ecosystem. To build LLMIR, you'll need:

1. A C++ compiler (GCC or Clang) with C++17 support
2. CMake (3.13.4 or higher)
3. Python (3.7 or higher)
4. Ninja or Make build system

### Clone the Repository

```bash
git clone https://github.com/chenxingqiang/llmir.git
cd llmir
```

### Configure the Build

```bash
mkdir build && cd build
cmake -G Ninja ..
```

### Build

```bash
ninja
```

## LLMIR Project Structure

The LLMIR project is structured as follows:

```
include/mlir/Dialect/LLM/       # MLIR dialect definitions
  ├── IR/                       # MLIR operations and types
  └── Runtime/                  # Runtime support headers

lib/Dialect/LLM/                # Implementation
  ├── IR/                       # MLIR operation implementations
  └── Runtime/                  # Runtime library implementations

test/Dialect/LLM/               # Tests
  ├── IR/                       # MLIR operation tests
  └── Runtime/                  # Runtime tests

examples/                       # Example applications
  └── kv_cache_example.cpp      # KV cache example
```

## Core Components (Under Development)

LLMIR is being developed in phases according to our [development plan](https://github.com/chenxingqiang/llmir.git). The core components are:

### Phase 1: Basic Infrastructure

- **LLM MLIR Dialect**: Specialized dialect defining operations and types for LLM inference
- **Custom Type System**: Types for representing KV caches, sharded tensors, etc.
- **Core Operations**: Attention, linear, layernorm, etc.

### Phase 2: Core Optimizations

- **KV Cache Management**: PagedAttention-style block-based KV cache handling
- **Attention Computation**: Fusion and optimization of attention operations
- **Memory Management**: Block allocation and recycling strategies

### Phase 3: Advanced Features

- **Quantization Support**: INT8/INT4 quantization transformations
- **Parallelism Strategies**: Tensor and pipeline parallelism
- **Backend Code Generation**: CUDA/CPU/accelerator support

## Contributing to LLMIR

LLMIR is in the early phases of development, and contributions are welcome. Here's how you can contribute:

1. Review the development plan in our repository
2. Choose an area to focus on (dialect design, optimization, etc.)
3. Follow standard MLIR development practices
4. Submit pull requests with well-tested changes

## Development Workflow

We recommend the following workflow for contributing to LLMIR:

1. Create a new branch for your feature
2. Implement the necessary changes with appropriate tests
3. Update documentation to reflect your changes
4. Submit a pull request for review

## Running Tests

Once tests are implemented, you can run them from the build directory:

```bash
ninja check-llmir
```

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
