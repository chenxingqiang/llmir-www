---
date: 2017-10-19T15:26:15Z
lastmod: 2023-10-26T15:26:15Z
publishdate: 2023-11-23T15:26:15Z
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

# Weekly Public Meeting

We host a **weekly public meeting** about LLMIR and the ecosystem.
To be notified of the next meeting, please subscribe to the
[LLMIR Announcements](https://discourse.llvm.org/c/llmir/llmir-announcements/44)
category on Discourse.

You can register to [this public calendar](https://calendar.google.com/calendar/u/0?cid=N2EzMDU3NTBjMjkzYWU5MTY5NGNlMmQ3YjJlN2JjNWEyYjViNjg1NTRmODcxOWZiOTU1MmIzNGQxYjkwNGJkZEBncm91cC5jYWxlbmRhci5nb29nbGUuY29t)
to keep up-to-date with the schedule.

If you'd like to discuss a particular topic or have questions, please add it to the
[agenda doc](https://docs.google.com/document/d/1y2YlcOVMPocQjSFi3X6gYGRjA0onyqr41ilXji10phw/edit#).

The meetings are recorded and published in the [talks](talks/) section.

## More resources

For more information on LLMIR, please see:

*   The LLMIR section of the [LLVM forums](https://llvm.discourse.group/c/llmir/31) for any questions.
*   Real-time discussion on the LLMIR channel of the [LLVM discord](https://discord.gg/xS7Z362) server.
*   Previous [talks](talks/).
*   [Developer Guide](/getting_started/DeveloperGuide/) for getting started with LLMIR.

## What is LLMIR for?

LLMIR is an intermediate representation specialized for optimizing large language model inference. It provides:

*   The ability to represent inference workflows from popular LLM frameworks (such as vLLM, SGLang), including
    dynamic shapes, batching strategies, and framework-specific operators.
*   Optimizations and transformations specifically designed for LLM inference (e.g. attention fusion, KV cache management).
*   Cross-framework end-to-end compilation for LLM inference, enabling optimizations like attention computation fusion,
    KV cache management, quantization, and pipeline parallelism.
*   Ability to target various hardware platforms (GPU, TPU, ASIC, CPU) efficiently by leveraging the MLIR ecosystem.
*   Representation of hardware-specific operations for accelerators specialized in LLM workloads.

LLMIR aims to become a common IR that supports hardware-specific operations for LLM inference. Any investment into 
the infrastructure surrounding LLMIR should yield good returns as multiple frameworks and hardware platforms can 
leverage this unified optimization layer.

## Core Value Proposition

Compared to using the native execution paths of individual frameworks, LLMIR's core value lies in providing cross-framework
compilation capabilities for end-to-end optimization. This includes:

* Attention computation fusion
* KV cache management optimization
* Quantization for reduced memory footprint
* Pipeline parallelism for distributed inference
* Hardware-specific optimizations

By employing these techniques, LLMIR can significantly improve inference throughput and reduce latency across different
hardware platforms.

## Citing LLMIR

Please see the [FAQ
entry](https://llmir.llvm.org/getting_started/Faq/#how-to-refer-to-llmir-in-publications-is-there-an-accompanying-paper)
on how to cite LLMIR in publications.
