---
title: "KV Cache Optimization"
date: 2024-05-09T15:26:15Z
lastmod: 2025-02-04T15:26:15Z
draft: false
weight: 2
---

# KV Cache Optimization in LLMIR

KV cache management is one of the core optimizations in LLMIR, focusing on efficient handling of key-value pairs in attention mechanisms for transformer-based LLMs. This feature is **implemented and verified** (84/84 tests passed, A800 GPU benchmarks).

## The KV Cache Challenge

In large language model inference, the key-value cache stores computed key and value tensors from previous tokens to avoid redundant computation. As sequence lengths grow, efficiently managing this cache becomes critical for:

- Memory efficiency
- Computation speed
- Support for longer contexts
- Dynamic batch handling

## PagedAttention in LLMIR

LLMIR implements a paged attention mechanism inspired by vLLM's approach, which treats the KV cache as blocks of memory rather than a continuous buffer:

### Block-based KV Cache

```mlir
// Define a paged KV cache type
!kv_cache_t = !llm.paged_kv_cache<f16, 12, 16, 64, 16, 4096>
```

Key parameters:
- Element type: `f16`
- Number of layers: `12`
- Number of heads: `16`
- Head dimension: `64`
- Block size: `16` (tokens per block)
- Maximum sequence length: `4096`

### Key Cache Operations

LLMIR provides specialized operations for managing the KV cache:

```mlir
// Append new key-value pairs to the cache
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

## Runtime Implementation

The LLMIR runtime library includes an efficient implementation of the paged KV cache:

```cpp
// C++ Runtime API (Implemented)
class PagedKVCache {
public:
  PagedKVCache(int numLayers, int numHeads, int headDim,
               int blockSize, int maxSeqLen, ElementType type);

  // Append new KV pairs to the cache
  void appendKV(void* keyPtr, void* valuePtr, int batchSize,
                int seqLen, int* seqIds, int* blockIndices);

  // Lookup existing KV pairs
  void lookupKV(int* blockIndices, int* seqLens, int batchSize,
                void* outputKeys, void* outputValues);

private:
  // Block-based memory manager
  BlockAllocator allocator_;
  // Maps sequence IDs to block indices
  std::unordered_map<int, std::vector<int>> seqToBlocks_;
  // ... other implementation details
};
```

## Optimization Passes

LLMIR includes several optimization passes for the KV cache (see `KVCacheOptimization.cpp`):

1. **BlockifyKVCachePass**: Convert continuous KV caches to block-based representations
2. **PagedAttentionRewritePass**: Rewrite standard attention to use paged attention
3. **KVCacheAllocationOptimizationPass**: Optimize memory allocation for KV caches
4. **KVCachePruningPass**: Remove unused or stale entries in the KV cache
5. **KVCacheShardingPass**: Support sharded KV caches for large models

## Memory Management

The KV cache implementation uses block-based memory management to:

- Allocate memory in fixed-size blocks
- Efficiently handle varying sequence lengths
- Minimize memory fragmentation
- Enable fast memory reuse for finished sequences

## Implemented Enhancements

LLMIR's KV cache optimization includes:

- **Quantized KV cache**: INT8 (4×) and INT4 (8×) compression for reduced memory footprint
- **Distributed KV cache**: Multi-device sharding (layer-wise, head-wise, sequence-wise)
- **Speculative KV cache**: Branch creation and rollback for draft token verification
- **Prefix cache**: Radix tree-based prefix matching with LRU eviction
- **Block size optimization**: Automatic optimal block size selection (Algorithm 1 in ICCD 2025 paper)

This feature is implemented as part of Phases 2–4 of the LLMIR project. See [Performance Evaluation](/docs/architecture/PerformanceEvaluation/) for benchmark results.