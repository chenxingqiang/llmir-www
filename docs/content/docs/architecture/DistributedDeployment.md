---
title: "Distributed Deployment"
date: 2024-05-09T15:26:15Z
lastmod: 2025-02-04T15:26:15Z
draft: false
weight: 4
---

# Distributed Deployment in LLMIR

LLMIR supports executing large models across multiple GPUs and nodes through distributed KV cache sharding and parallel execution strategies.

## Distributed KV Cache

LLMIR implements **DistributedPagedKVCache** with configurable sharding strategies:

### Sharding Strategies

| Strategy | Description | Use Case |
|----------|-------------|----------|
| **Layer-wise** | Partition layers across GPUs | Large models (70B+) |
| **Head-wise** | Partition attention heads | Moderate models |
| **Sequence-wise** | Partition sequences in batch | High batch throughput |

### Configuration

```cpp
// Configure sharding across 4 GPUs
ShardingConfig config;
config.strategy = ShardingStrategy::LAYER_WISE;
config.numDevices = 4;
config.deviceIds = {0, 1, 2, 3};

DistributedPagedKVCache distCache(numLayers, numHeads, headDim, blockSize,
                                   maxSeqLen, elementType, config);

// Operations are automatically distributed
distCache.appendKV(keyPtr, valuePtr, batchSize, seqLen, seqIds, blockIndices);
```

### 8-GPU Scaling

The ICCD 2025 paper reports **94.5% efficiency** for 8-GPU layer-wise sharding. For Llama 70B, GPUs 0–3/4–7 form tensor-parallel groups for layers 0–39/40–79.

## Multi-Device Inference

- **Tensor Parallelism**: Split layers across devices
- **Pipeline Parallelism**: Stage layers for sequential execution
- **Memory Management**: Coordinated allocation across devices
- **Device Communication**: Efficient data exchange for distributed attention

## Python API

```python
from mlir.dialects.llm import DistributedKVCache, ShardingConfig, ShardingStrategy

config = ShardingConfig(
    strategy=ShardingStrategy.LAYER_WISE,
    num_devices=4,
    device_ids=[0, 1, 2, 3]
)
dist_cache = DistributedKVCache(kv_config, config)
```

## Verification

DistributedKVCache is part of the 84/84 Python test suite. See [Performance Evaluation](PerformanceEvaluation) for scaling results and [TEST_VERIFICATION_REPORT](https://github.com/chenxingqiang/llmir/blob/main/IEEE-conference/TEST_VERIFICATION_REPORT.md) for details.
