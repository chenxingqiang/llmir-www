---
date: 2023-10-19T15:26:15Z
lastmod: 2025-02-04T15:26:15Z
publishdate: 2025-02-04T15:26:15Z
---

# Large Language Model Intermediate Representation Overview

The LLMIR project is a novel approach to building reusable and extensible
compiler infrastructure for large language model inference. LLMIR aims to unify and optimize
LLM inference workflows, improve compilation for heterogeneous hardware, significantly reduce
inference latency, and enhance integration between various LLM frameworks.

LLMIR is a dedicated compilation middle layer for platform architects and developers, built on
the MLIR framework. It leverages MLIR's flexible infrastructure to represent and transform
computational graphs. LLMIR can integrate with multiple LLM inference frameworks (like vLLM, SGLang)
by converting their high-level operators or model graphs into a unified intermediate representation
for further optimization.

## Project Status

LLMIR has completed **Phases 1–6** of its development roadmap. The core infrastructure is production-ready:

- **C++ MLIR Dialect**: Built and verified with MLIR 18; all attention and cache algorithms tested (84/84 tests passed)
- **Benchmarks**: Real model benchmarks (Qwen2.5-7B, Llama-3.1-8B) with vLLM and SGLang baselines on A800 GPUs
- **Paper**: Submitted to ICCD 2025; revised version ready for resubmission to top-tier venues

See [Performance Evaluation](/docs/architecture/PerformanceEvaluation/) for benchmark results.

## Project Resources

For more information on LLMIR, please see:

* [Project Repository](https://github.com/chenxingqiang/llmir.git)
* [**User Guide**](/getting_started/UserGuide/) – Installation, use cases, API examples
* [Developer Guide](/getting_started/DeveloperGuide/) – Build from source, benchmarks

## What is LLMIR for?

LLMIR is an intermediate representation specialized for optimizing large language model inference. It provides:

* The ability to represent inference workflows from popular LLM frameworks (such as vLLM, SGLang), including
  dynamic shapes, batching strategies, and framework-specific operators.
* Optimizations and transformations specifically designed for LLM inference (e.g. attention fusion, KV cache management).
* Cross-framework end-to-end compilation for LLM inference, enabling optimizations like attention computation fusion,
  KV cache management, quantization, and pipeline parallelism.
* Ability to target various hardware platforms (GPU, TPU, ASIC, CPU) efficiently by leveraging the MLIR ecosystem.
* Representation of hardware-specific operations for accelerators specialized in LLM workloads.

## Core Value Proposition

Compared to using the native execution paths of individual frameworks, LLMIR's core value lies in providing cross-framework
compilation capabilities for end-to-end optimization. This includes:

* **Performance Improvement**: Leveraging compilation optimizations to reduce inference latency and increase throughput
* **Resource Efficiency**: Optimizing memory usage, supporting longer sequences and larger batch sizes
* **Scalability**: Supporting different hardware platforms and inference frameworks
* **Usability**: Providing developer-friendly APIs to lower integration barriers

## Key Features

* **PagedKVCache**: Block-based KV cache with dynamic memory management; verified on A800 GPUs
* **MLIR Dialect for LLMs**: Custom operations (append_kv, lookup_kv, paged_attention) and types
* **Memory Optimizations**: Block-based allocation; up to 58.8% memory reduction
* **Attention Optimizations**: Flash Attention, Fused Softmax, Sliding Window (1.28×–2.15× speedup)
* **Baseline Comparisons**: vLLM (+22.4%), SGLang (+38.1%), TensorRT-LLM (+4.8%), MLC-LLM (+25.9%)
* **Multi-model Support**: LLaMA-2, Phi-3, Qwen-2, DeepSeek-V2

# Weekly Public Meeting

We host a **weekly public meeting** about LLMIR and the ecosystem.
To be notified of the next meeting, please subscribe to the
[LLMIR Announcements](https://discourse.llvm.org/c/llmir/llmir-announcements/44)
category on Discourse.

You can register to [this public calendar](https://calendar.google.com/calendar/u/0?cid=N2EzMDU3NTBjMjkzYWU5MTY5NGNlMmQ3YjJlN2JjNWEyYjViNjg1NTRmODcxOWZiOTU1MmIzNGQxYjkwNGJkZEBncm91cC5jYWxlbmRhci5nb29nbGUuY29t)
to keep up-to-date with the schedule.

If you'd like to discuss a particular topic or have questions, please add it to the
[agenda doc](https://docs.google.com/document/d/1y2YlcOVMPocQjSFi3X6gYGRjA0onyqr41ilXji10phw/edit#).

## More resources

For more information on LLMIR, please see:

*   The LLMIR section of the [LLVM forums](https://llvm.discourse.group/c/llmir/31) for any questions.
*   Real-time discussion on the LLMIR channel of the [LLVM discord](https://discord.gg/xS7Z362) server.

## Citing LLMIR

Please see the [FAQ entry](/getting_started/Faq/#how-to-refer-to-llmir-in-publications-is-there-an-accompanying-paper) on how to cite LLMIR in publications.
