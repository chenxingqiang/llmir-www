---
title: "LLMIR Related Publications"
date: 2024-05-09T15:26:15Z
lastmod: 2025-02-04T15:26:15Z
draft: false
weight: 1
---

# LLMIR Publications

## ICCD 2025 Submission

**LLMIR: A Compiler Infrastructure for Optimizing Large Language Model Inference**

* **Author**: Xingqiang Chen (Xiamen University & Turingai Inc.)
* **Venue**: IEEE/ACM International Conference on Computer-Aided Design (ICCD) 2025
* **Status**: Revised version ready for resubmission
* **Repository**: [github.com/chenxingqiang/llmir](https://github.com/chenxingqiang/llmir)

### Key Contributions

1. **LLM-Specific MLIR Dialect**: Custom types (PagedKVCache, ShardedTensor, QuantizedTensor) capturing LLM semantics
2. **Compiler-Level PagedAttention**: First IR-level representation enabling static analysis of dynamic memory patterns
3. **Multi-Stage Compilation Pipeline**: Model import → LLM dialect optimization → Code generation → Runtime integration
4. **Comprehensive Optimization Framework**: KV cache, multi-precision, parallelization, attention optimizations

### Performance Highlights (Revised Paper)

| Metric | Value |
|--------|-------|
| Average Throughput | 58,499 tokens/sec |
| Peak Throughput | 88,250 tokens/sec |
| vs vLLM | +22.4% |
| vs SGLang | +38.1% |
| vs TensorRT-LLM | +4.8% |
| vs MLC-LLM | +25.9% |
| Attention Speedup | 1.28× – 2.15× |
| 8-GPU Scaling | 94.5% efficiency |
| Memory Optimization | Up to 58.8% |

### Verification

- **C++ MLIR Dialect**: Built with MLIR 18; all 84 unit tests passed
- **A800 GPU Benchmarks**: Qwen2.5-7B, PyTorch vs vLLM comparison verified
- **Attention/Cache Algorithms**: All paper claims verified in [test report](https://github.com/chenxingqiang/llmir/blob/main/IEEE-conference/TEST_VERIFICATION_REPORT.md)

## Technical Documents

* [Development Plan](https://github.com/chenxingqiang/llmir) - 
  The comprehensive development roadmap for LLMIR, including architectural details and implementation strategies.

* [LLMIR Architecture Overview](/getting_started/DeveloperGuide/) - 
  Technical overview of the LLMIR system architecture and key components.

* [Submission Summary](https://github.com/chenxingqiang/llmir/blob/main/IEEE-conference/SUBMISSION_SUMMARY.md) - 
  ICCD 2025 submission details and revision notes.

## Contact for Research Collaboration

If you're interested in collaborating on LLMIR-related research or publications, please contact the project maintainers through our [GitHub repository](https://github.com/chenxingqiang/llmir).

