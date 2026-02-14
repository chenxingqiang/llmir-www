---
title: "Getting Started"
date: 2023-11-29T15:26:15Z
lastmod: 2025-02-04T15:26:15Z
draft: false
---

# Getting Started with LLMIR

Welcome to LLMIR! This section helps you get started with the LLMIR compiler infrastructure for LLM inference optimization.

## Quick Start

```bash
pip install llmir
```

```python
import llmir
config = llmir.KVCacheConfig(num_layers=32, num_heads=32, head_dim=128)
cache = llmir.PagedKVCache(config)
```

See the [**User Guide**](/getting_started/UserGuide/) for installation options, use cases, and API examples. See [Developer Guide](/getting_started/DeveloperGuide/) for building from source and C++ dialect.

## Documentation

| Resource | Description |
|----------|-------------|
| [**User Guide**](/getting_started/UserGuide/) | Installation, use cases, API examples, repo layout |
| [Developer Guide](/getting_started/DeveloperGuide/) | Build from source, benchmarks, project structure |
| [Architecture](/docs/architecture/) | KV cache, quantization, distributed deployment |
| [Performance Evaluation](/docs/architecture/PerformanceEvaluation/) | Benchmark results, verification |
| [Publications](/pubs/) | ICCD 2025 paper, performance highlights |
| [FAQ](/getting_started/Faq/) | Common questions |

## Contributing

- [Contributing](/getting_started/Contributing/)
- [Testing Guide](/getting_started/TestingGuide/)
- [Debugging](/getting_started/Debugging/)
- [Reporting Issues](/getting_started/ReportingIssues/)