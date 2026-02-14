---
title: "Performance Evaluation"
date: 2024-05-09T15:26:15Z
lastmod: 2025-02-04T15:26:15Z
draft: false
weight: 5
---

# Performance Evaluation in LLMIR

LLMIR includes comprehensive benchmarking and evaluation methodologies to measure its impact on LLM inference performance across different models and hardware platforms.

## Verified Results (February 2025)

### Test Verification Summary

| Component | Status | Details |
|----------|--------|---------|
| Python Runtime | 84/84 PASSED | PagedKVCache, QuantizedKVCache, DistributedKVCache, SpeculativeKVCache, PrefixCache, Model optimizers |
| C++ MLIR Dialect | Verified | Built with MLIR 18; all attention and cache algorithms tested |
| A800 GPU Benchmarks | Verified | Qwen2.5-7B, PyTorch vs vLLM on NVIDIA A800 80GB |

### Real Model Benchmarks (A800 80GB)

**Qwen2.5-7B comparison:**

| Framework | Peak Throughput | Best Batch | vs PyTorch |
|-----------|-----------------|------------|------------|
| PyTorch (transformers) | 2,006.8 tok/s | 64 | Baseline |
| vLLM | 7,431.5 tok/s | 128 | **+270.3%** |

### Baseline Comparisons (from ICCD 2025 revised paper)

| Baseline | Throughput Improvement |
|----------|-------------------------|
| vLLM | +22.4% |
| SGLang | +38.1% |
| TensorRT-LLM | +4.8% |
| MLC-LLM | +25.9% |

### Attention Optimization Speedup

| Technique | Speedup Range | Memory Reduction |
|-----------|---------------|-------------------|
| Flash Attention | 1.28× – 1.69× | Minimal |
| Fused Softmax | 1.36× – 1.48× | 30–40% |
| Optimized Masked | 1.42× – 1.92× | Varies by mask |
| Sliding Window | 1.52× – 2.15× | 40–70% |

## Benchmark Framework

LLMIR provides a dedicated benchmarking framework to evaluate performance improvements:

```cpp
// LLMIR Benchmark API (Planned)
class LLMIRBenchmark {
public:
  // Configure benchmark parameters
  void setModel(const std::string& modelPath);
  void setHardware(const std::string& hardware);
  void setSequenceLength(int length);
  void setBatchSize(int batchSize);
  void setQuantizationMode(QuantMode quantMode);
  void setKVCacheStrategy(KVCacheMode kvMode);
  
  // Run benchmarks
  BenchmarkResult runThroughputTest(int iterations);
  BenchmarkResult runLatencyTest(int iterations);
  BenchmarkResult runMemoryTest();
  
  // Compare with baselines
  ComparisonResult compareWithBaseline(const std::string& baselineFramework);
};
```

## Key Performance Metrics

LLMIR will track and optimize for several key performance metrics:

### Throughput Metrics

- **Tokens per Second (TPS)**: Number of output tokens generated per second
- **Requests per Second (RPS)**: Number of inference requests processed per second
- **Effective TPS**: Combined throughput across multiple devices/nodes

### Latency Metrics

- **First Token Latency**: Time from request reception to first token generation
- **Inter-Token Latency**: Time between consecutive token generations
- **End-to-End Latency**: Total time from request to completion
- **Attention Computation Latency**: Time spent in attention operations
- **KV Cache Access Latency**: Time spent accessing the KV cache

### Memory Metrics

- **Peak Memory Usage**: Maximum memory consumed during inference
- **Memory Efficiency**: Ratio of active tensors to allocated memory
- **KV Cache Size**: Memory consumed by the key-value cache
- **Memory Bandwidth Utilization**: Efficiency of memory access patterns

### Scaling Metrics

- **Strong Scaling**: Speedup when increasing devices for fixed workload
- **Weak Scaling**: Performance with fixed workload per device while increasing devices
- **Device Utilization**: Percentage of device compute capacity used

## Benchmark Suite

LLMIR will include a comprehensive benchmark suite with:

### Model Selection

- **Size Variants**: Small (7B), Medium (13B), Large (70B+)
- **Architecture Types**: Decoder-only, Encoder-decoder
- **Model Families**: Llama, Mistral, Falcon, etc.

### Workload Patterns

- **Text Generation**: Standard autoregressive generation
- **Chat Completion**: Multi-turn dialogue generation
- **Long Context Processing**: Tests with very long input contexts
- **Mixed Batch Sizes**: Varying concurrent request volumes

### Hardware Targets

- **NVIDIA GPUs**: A100, H100, RTX series
- **AMD GPUs**: MI100, MI250, MI300
- **x86 CPUs**: Intel Xeon, AMD EPYC
- **ARM CPUs**: AWS Graviton, Apple Silicon

## Analysis Tools

LLMIR will provide tools for detailed performance analysis:

### Profiling

```cpp
// LLMIR Profiler API (Planned)
class LLMIRProfiler {
public:
  // Start/stop profiling
  void startProfiling(const std::string& name);
  void stopProfiling();
  
  // Event tracking
  void recordEvent(const std::string& name);
  void markOperationStart(const std::string& opName);
  void markOperationEnd(const std::string& opName);
  
  // Analysis
  ProfileData getOperationBreakdown();
  ProfileData getMemoryUsageTimeline();
  ProfileData getDeviceUtilization();
  
  // Export
  void exportChromeTraceFormat(const std::string& filename);
  void exportReport(const std::string& filename);
};
```

### Visualization

The benchmarking system will include visualizations to help understand performance:

- Operation timeline views
- Memory usage graphs
- Compute/memory utilization heatmaps
- Performance comparison charts

## Baseline Comparisons

LLMIR performance is compared against several baselines (verified in ICCD 2025 paper):

- **vLLM Native**: +22.4% throughput improvement
- **SGLang Native**: +38.1% throughput improvement
- **TensorRT-LLM**: +4.8% over NVIDIA's production compiler
- **MLC-LLM**: +25.9% over TVM-based deployment
- **HuggingFace Transformers**: PyTorch baseline for model verification

## Benchmark Scripts

Run benchmarks from the [llmir repository](https://github.com/chenxingqiang/llmir):

```bash
# Real model benchmark (Qwen2.5-7B, vLLM comparison)
./run_real_benchmark.sh

# Comprehensive benchmark with vLLM and SGLang baselines
./comprehensive_benchmark.sh

# Quick vLLM comparison
./vllm_comparison.sh
```

## Future Directions

As LLMIR matures, the performance evaluation framework will expand to include:

- **Automated Regression Testing**: Continuous performance monitoring
- **Bottleneck Identification**: Automatic detection of performance limitations
- **Optimization Recommendation**: Suggestions for performance improvements
- **Hardware-Specific Insights**: Targeted optimizations based on profiling
- **Performance Modeling**: Predictive modeling of optimization impacts

The performance evaluation system is operational. See the [GitHub repository](https://github.com/chenxingqiang/llmir) for benchmark scripts and [TEST_VERIFICATION_REPORT](https://github.com/chenxingqiang/llmir/blob/main/IEEE-conference/TEST_VERIFICATION_REPORT.md) for verification details. 