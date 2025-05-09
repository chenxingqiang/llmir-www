---
title: "Developer Guide"
date: 2023-11-29T15:26:15Z
draft: false
---

# LLMIR Developer Guide

This guide provides an overview of how to develop with LLMIR.

## Building LLMIR

LLMIR is built on top of the MLIR ecosystem. To build LLMIR, you'll need:

1. A C++ compiler (GCC or Clang)
2. CMake (3.13.4 or higher)
3. Python (3.6 or higher)
4. Ninja or Make build system

### Clone the Repository

```bash
git clone https://github.com/chenxingqiang/llmir.git
cd llmir
```

### Configure the Build

```bash
mkdir build && cd build
cmake -G Ninja ../llvm -DLLVM_ENABLE_PROJECTS="llmir" -DLLVM_BUILD_EXAMPLES=ON -DLLVM_TARGETS_TO_BUILD="host"
```

### Build

```bash
ninja
```

## LLMIR Project Structure

The LLMIR project is organized into the following components:

### Core Components

- **IR**: Core representation for LLM operations
- **Dialect**: LLMIR-specific dialect definitions and operations
- **Transforms**: Optimization passes for LLM inference
- **Conversion**: Conversion utilities for importing from other frameworks

### Key Features

- KV Cache Optimization
- Quantization Support
- Distributed Deployment Interfaces
- Hardware-specific Operations

## Adding a New Operation

To add a new operation to LLMIR:

1. Define the operation in the appropriate tablegen file
2. Implement the operation semantics
3. Add canonicalization patterns if applicable
4. Add conversion patterns for importing/exporting
5. Update documentation

## Running Tests

From the build directory:

```bash
ninja check-llmir
```

## Getting Help

For help with LLMIR development, please use:

- LLMIR Discord channel for quick questions
- LLMIR section of LLVM Discourse for longer discussions
