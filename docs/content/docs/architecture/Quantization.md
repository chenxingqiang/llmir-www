---
title: "Quantization Support"
date: 2024-05-09T15:26:15Z
lastmod: 2025-02-04T15:26:15Z
draft: false
weight: 3
---

# Quantization Support in LLMIR

Quantization reduces memory footprint and computation by using lower-precision representations. LLMIR provides quantization through specialized types and runtime implementations. **QuantizedKVCache is implemented and verified** (84/84 tests; INT8 4×, INT4 8× compression).

## Quantization in LLMIR

LLMIR supports various quantization strategies tailored for LLM inference:

### Custom Quantized Types

LLMIR defines specialized types for representing quantized tensors:

```mlir
// INT8 asymmetric quantized tensor
!llm.quantized_tensor<4x1024xi8, scale=f32, zp=i8, group_size=128>

// INT4 symmetric grouped quantized tensor
!llm.quantized_tensor<4x1024xi4, scale=f32, symmetric=true, group_size=128>

// Mixed precision quantized tensor
!llm.mixed_quantized_tensor<4x1024x!llm.mixed<i8,i4>, scale=f32>
```

### Quantization Operations

```mlir
// Quantize a tensor from FP16 to INT8
%quantized = llm.quantize(%input) {
  scale = dense<0.01> : tensor<256xf32>,
  zero_point = dense<-2> : tensor<256xi8>,
  bits = 8 : i32,
  symmetric = false
} : (tensor<1x256xf16>) -> tensor<1x256xi8>

// Dequantize from INT8 back to FP16
%dequantized = llm.dequantize(%quantized) {
  scale = dense<0.01> : tensor<256xf32>,
  zero_point = dense<-2> : tensor<256xi8>
} : (tensor<1x256xi8>) -> tensor<1x256xf16>

// Quantized matrix multiplication
%result = llm.quantized_matmul(%input, %weight, %scales, %zero_points) {
  bits = 8 : i32,
  group_size = 128 : i32
} : (tensor<?x?xf16>, tensor<?x?xi8>, tensor<?xf32>, tensor<?xi8>) -> tensor<?x?xf16>
```

## Quantization Methods

LLMIR supports multiple quantization strategies:

### Post-Training Quantization (PTQ)

- **Symmetric Quantization**: Uses a symmetric range around zero
- **Asymmetric Quantization**: Uses zero-point offsets for asymmetric ranges
- **Per-Channel/Per-Tensor**: Supports different scaling granularities
- **Group-wise Quantization**: Applies quantization parameters to groups of weights

### Quantization-Aware Inference (QAI)

- **Weight-only Quantization**: Keeps activations in higher precision
- **Activation Quantization**: Quantizes intermediate activations
- **Mixed-precision Inference**: Different precision for different parts of the model

## Optimization Passes

LLMIR includes quantization-related optimization passes:

1. **QuantizationCalibrationPass**: Analyze model to determine optimal quantization parameters
2. **WeightQuantizationPass**: Convert model weights to quantized formats
3. **ActivationQuantizationPass**: Add quantization/dequantization for activations
4. **QuantizedOperationFusionPass**: Fuse quantized operations for efficient execution
5. **HardwareSpecificQuantizationPass**: Customize quantization for specific hardware features

## Integration with Hardware Backends

LLMIR's quantization system is designed to integrate with various hardware backends:

- **NVIDIA Tensor Cores**: Support for INT8/INT4 computation
- **Intel AMX**: Optimizations for x86 Advanced Matrix Extensions
- **ARM Matrix multiplier**: Support for efficient ARM-based inference
- **Custom Accelerators**: Extensible for specialized ML hardware

## Runtime Support

The LLMIR runtime provides efficient implementations for quantized KV cache:

```cpp
// QuantizedPagedKVCache - Implemented (QuantizedKVCache.h)
QuantizationConfig config(QuantizationType::INT8, QuantizationStrategy::PER_TENSOR);
QuantizedPagedKVCache qCache(numLayers, numHeads, headDim, blockSize, maxSeqLen, config, elementType);
qCache.appendKV(...);  // Automatic quantization/dequantization
float ratio = qCache.getCompressionRatio();  // ~4x INT8, ~8x INT4

// Quantized Matrix Multiplication (API)
void quantizedMatMul(
    const void* input,           // Input activations (typically FP16)
    const int8_t* weights,       // Quantized weights (INT8/INT4)
    const float* scales,         // Quantization scales
    const int8_t* zeroPoints,    // Zero points (for asymmetric)
    void* output,                // Output buffer
    int M, int N, int K,         // Matrix dimensions
    int groupSize,               // Group size for scales
    int bits                     // Bit width (8/4)
);
```

## Implemented Features

- **QuantizedPagedKVCache**: INT8 (4×), INT4 (8×) compression; per-tensor and per-channel
- **QuantizationConfig**: `QuantizationType`, `QuantizationStrategy`, group size
- **Model-specific**: `LlamaOptimizer.getRecommendedQuantConfig()`, `PhiOptimizer` presets

## Future Directions

- Sparse-quantized representations
- Dynamic quantization
- Calibration tools
- Automated mixed precision

See [KV Cache Optimization](KVCache) and [Performance Evaluation](PerformanceEvaluation) for benchmark results. 