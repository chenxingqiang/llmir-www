---
title: "References"
date: 2024-05-09T15:26:15Z
draft: false
weight: 10
---

# References

This page lists relevant papers, projects, and resources that have influenced the development of LLMIR or are related to LLM optimization and compilation.

## Foundational Work

### MLIR Framework

* Chris Lattner, Mehdi Amini, Uday Bondhugula, Albert Cohen, Andy Davis, Jacques Pienaar, River Riddle, Tatiana Shpeisman, Nicolas Vasilache, and Oleksandr Zinenko. **"MLIR: Scaling compiler infrastructure for domain specific computation."** In 2021 IEEE/ACM International Symposium on Code Generation and Optimization (CGO), pp. 2-14. IEEE, 2021. [Link](https://ieeexplore.ieee.org/abstract/document/9370308)

### Large Language Model Inference

* Dao, Tri, et al. **"FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness."** Advances in Neural Information Processing Systems, 2022. [Link](https://proceedings.neurips.cc/paper_files/paper/2022/file/67d57c32e20fd0a7a302cb81d36e40d5-Paper.pdf)

* Sheng, Ying, et al. **"High-throughput Generative Inference of Large Language Models with a Single GPU."** In International Conference on Machine Learning, 2023. [Link](https://arxiv.org/abs/2303.06865)

## LLM Inference Frameworks

### vLLM

* Kwon, Woosuk, et al. **"Efficient Memory Management for Large Language Model Serving with PagedAttention."** In Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles, 2023. [Link](https://arxiv.org/abs/2309.06180)

### SGLang

* Xiao, Lianmin, et al. **"SGLang: Semi-Structured Gateway Language for Multi-Agent System."** arXiv preprint arXiv:2403.06071 (2024). [Link](https://arxiv.org/abs/2403.06071)

## Compiler Optimization for LLMs

* DeepSpeed Team. **"DeepSpeed Inference: Enabling Efficient Inference of Transformer Models at Unprecedented Scale."** arXiv preprint arXiv:2207.00032 (2022). [Link](https://arxiv.org/abs/2207.00032)

* Frantar, Elias, et al. **"GPTQ: Accurate Post-training Quantization for Generative Pre-trained Transformers."** In International Conference on Learning Representations, 2023. [Link](https://arxiv.org/abs/2210.17323)

* Korthikanti, Vijay Anand, et al. **"Reducing Activation Recomputation in Large Transformer Models."** In Proceedings of Machine Learning and Systems, 2023. [Link](https://proceedings.mlsys.org/paper_files/paper/2023/file/9fb11ca5c4611e9a545e15f04bea4afd-Paper-Conference.pdf)

## Hardware-Specific Optimizations

* Wang, Ze, et al. **"TensorRT-LLM: A Comprehensive and Efficient Large Language Model Inference Library."** arXiv preprint arXiv:2405.09386 (2024). [Link](https://arxiv.org/abs/2405.09386)

* Dao, Tri, et al. **"FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning."** arXiv preprint arXiv:2307.08691 (2023). [Link](https://arxiv.org/abs/2307.08691)

## Quantization Techniques

* Dettmers, Tim, et al. **"LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale."** In Advances in Neural Information Processing Systems, 2022. [Link](https://proceedings.neurips.cc/paper_files/paper/2022/file/721d1c440afcb96283fb84384663a667-Paper-Conference.pdf)

* Xiao, Guangxuan, et al. **"SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models."** In Proceedings of the International Conference on Machine Learning, 2023. [Link](https://arxiv.org/abs/2211.10438)

## Distributed LLM Inference

* Aminabadi, Reza Yazdani, et al. **"DeepSpeed-Inference: Enabling Efficient Inference of Transformer Models at Unprecedented Scale."** In Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis, 2022. [Link](https://arxiv.org/abs/2207.00032)

* Zheng, Lianmin, et al. **"Alpa: Automating Inter-and Intra-Operator Parallelism for Distributed Deep Learning."** In 16th USENIX Symposium on Operating Systems Design and Implementation, 2022. [Link](https://arxiv.org/abs/2201.12023)

## Related Projects

* [LLVM](https://llvm.org/): The LLVM Compiler Infrastructure Project
* [MLIR](https://mlir.llvm.org/): Multi-Level Intermediate Representation
* [vLLM](https://github.com/vllm-project/vllm): High-throughput and memory-efficient inference and serving engine for LLMs
* [SGLang](https://github.com/sgl-project/sglang): Semi-structured gateway language for large language models
* [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM): NVIDIA's LLM optimization library

---

This reference list will be updated as the LLMIR project progresses and new relevant research emerges in the field. 