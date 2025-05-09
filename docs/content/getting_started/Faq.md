---
title: "FAQ"
date: "2024-05-09"
menu: "main"
weight: 10
---

# Frequently Asked Questions about LLMIR

## What is LLMIR?

LLMIR (Large Language Model Intermediate Representation) is a compiler infrastructure for large language models based on MLIR (Multi-Level Intermediate Representation). It's designed to optimize and accelerate LLM inference through specialized compilation techniques, providing a unified intermediate representation layer for different LLM frameworks.

## What problem does LLMIR solve?

LLMIR addresses several key challenges in LLM inference:

1. **Performance bottlenecks**: By applying compiler optimization techniques, LLMIR reduces inference latency and improves throughput.
2. **Memory efficiency**: Through optimizations like efficient KV cache management, LLMIR enables processing of longer sequences and larger batch sizes with less memory.
3. **Framework fragmentation**: LLMIR provides a unified compilation layer that works across different LLM frameworks (like vLLM and SGLang).
4. **Hardware diversity**: LLMIR enables efficient targeting of different hardware platforms (GPUs, CPUs, accelerators) from the same source representation.

## How does LLMIR relate to MLIR?

LLMIR is built on top of MLIR, which provides the foundational compiler infrastructure. LLMIR extends MLIR with:

1. **LLM-specific dialect**: Custom operations and types specifically designed for LLM inference
2. **Specialized optimizations**: Passes targeting key performance bottlenecks in LLM workloads
3. **Runtime components**: Libraries for efficient execution of compiled LLM models

## What is the current status of LLMIR?

LLMIR is currently in active development. We are following the development plan as outlined in our [GitHub repository](https://github.com/chenxingqiang/llmir.git). The project is in the early phases of building the core infrastructure and defining the MLIR dialect.

## What features does LLMIR support?

LLMIR is being developed in phases:

### Phase 1 (Current Focus)
- Basic infrastructure: MLIR dialect, type system, core operations

### Phase 2 (Planned)
- KV cache optimization
- Attention computation fusion
- Memory management optimizations

### Phase 3 (Future)
- Quantization support
- Parallelism strategies (tensor/pipeline)
- Advanced backend code generation

## How can I contribute to LLMIR?

Contributions to LLMIR are welcome! Here's how you can contribute:

1. Clone the [repository](https://github.com/chenxingqiang/llmir.git) and familiarize yourself with the code
2. Check the development plan to see which areas need attention
3. Follow standard MLIR development practices
4. Submit pull requests with well-tested changes

## Which LLM frameworks does LLMIR support?

LLMIR is being designed to initially support:

1. **vLLM**: For its efficient paged attention mechanism
2. **SGLang**: For its structured generation capabilities

Additional framework support may be added in the future.

## What hardware targets does LLMIR support?

LLMIR aims to support:

1. **NVIDIA GPUs** (via CUDA code generation)
2. **AMD GPUs** (via ROCm/HIP)
3. **x86/ARM CPUs** (via LLVM)
4. **Specialized accelerators** (future support)

## How does LLMIR compare to other LLM optimization frameworks?

Unlike framework-specific optimizers, LLMIR provides a cross-framework compilation layer. Compared to:

- **TensorRT-LLM**: LLMIR offers more flexibility in model representation and isn't tied to a specific vendor
- **Framework-native optimizers**: LLMIR enables cross-framework optimizations and consistent hardware targeting
- **Generic ML compilers**: LLMIR includes specialized optimizations for LLM inference patterns

## How can I use LLMIR in my project?

LLMIR is still in the early development phase. When ready for use, it will provide:

1. A C++ API for integrating into compilation workflows
2. Python bindings for easy integration with Python-based ML frameworks
3. Command-line tools for converting and optimizing models

## Where can I learn more about LLMIR?

The primary resources for learning about LLMIR are:

1. [GitHub Repository](https://github.com/chenxingqiang/llmir.git)
2. [Project Website](https://chenxingqiang.github.io/llmir-www/)
3. [Developer Guide](/getting_started/DeveloperGuide/)

## How to refer to MLIR in publications? Is there an accompanying paper?

MLIR has been presented in the 2021 IEEE/ACM International Symposium on Code
Generation and Optimization, the full text of the paper is [available from
IEEE](https://ieeexplore.ieee.org/abstract/document/9370308). A pre-publication
draft is available on [arXiv](https://arxiv.org/pdf/2002.11054) but may be
missing improvements and corrections. Please also note that MLIR keeps evolving
and IR snippets presented in the paper may no longer use modern syntax, refer to
the MLIR documentation for the new syntax.

To cite MLIR in academic or other publications, please use: _Chris Lattner,
Mehdi Amini, Uday Bondhugula, Albert Cohen, Andy Davis, Jacques Pienaar, River
Riddle, Tatiana Shpeisman, Nicolas Vasilache, and Oleksandr Zinenko. "MLIR:
Scaling compiler infrastructure for domain specific computation." In 2021
IEEE/ACM International Symposium on Code Generation and Optimization (CGO), pp.
2-14. IEEE, 2021._

The BibTeX entry is as follows.

```
@inproceedings{mlir,
  author={Lattner, Chris and Amini, Mehdi and Bondhugula, Uday and Cohen, Albert and Davis, Andy and Pienaar, Jacques and Riddle, River and Shpeisman, Tatiana and Vasilache, Nicolas and Zinenko, Oleksandr},
  booktitle={2021 {{IEEE/ACM}} International Symposium on Code Generation and Optimization (CGO)},
  title={{{MLIR}}: Scaling Compiler Infrastructure for Domain Specific Computation},
  year={2021},
  volume={},
  number={},
  pages={2-14},
  doi={10.1109/CGO51591.2021.9370308}
}
```

Please do **not** cite the arXiv preprint as it is not a formal peer-reviewed
publication.

## Why is \<small feature\> not available in MLIR?

On general basis, there is never a reason why a small feature is not available in MLIR other than nobody needed it enough to implement it. Consider submitting a patch. For larger features and dialects, follow the [request-for-comments](https://mlir.llvm.org/getting_started/DeveloperGuide/#guidelines-on-contributing-a-new-dialect-or-important-components) process.

## MLIR is too heavy framework, should I just reimplement my own compiler from scratch?

Maybe: it is hard to tell as it depends on your requirements, even C++ may already be too
large for some micro-controllers. In our experience most projects ends up growing beyond
what their original author intended, and reimplementing the features you would get from
MLIR would also have a footprint. MLIR footprint is representative of the features it
provides. More importantly we have a "you don't pay for what you don't use" approach:
MLIR is very modular and you can link a binary with a very minimal set of libraries.
If you use just the core IR, some pieces of the infrastructure, and a few dialects
you should expect a few MBs. We have
[three examples](https://github.com/llvm/llvm-project/tree/main/mlir/examples/minimal-opt)
in the repo showing some small possible configurations of MLIR, showing that the
core of MLIR can take around 1MB.

## What is the difference between the Tensor and Vector types?

1) Conceptual: vectors are meant to and occur in lower level dialects - often where you expect hardware to have registers of that size. Tensors model higher-level "closer to the source" abstract representation. This is reflected in the abstraction modeled by the operations from the [`vector` dialect](https://mlir.llvm.org/docs/Dialects/Vector/), while Tensors would be more naturally present in the operations of the [`linalg` dialect](https://mlir.llvm.org/docs/Dialects/Linalg/).
2) Tensors can be dynamically shaped, unranked, or have 0 dimensions ; but Vectors can't be.
3) You can have a memref (a buffer in memory) containing Vectors but you can't have a memref of a tensor type.
4) The set of allowed element types is different: the Tensor type isn't limited while Vector is limited to float and integer types.
5) Tensors accept an optional "encoding" attribute, vector don't at the moment.

## Registered, loaded, dependent: what's up with Dialects management?

Before creating an Operation, a Type, or an Attribute, the associated Dialect
must be already *loaded* in the `MLIRContext`. For example the Toy tutorial
explicitly loads the Toy Dialect before emitting the Toy IR from the AST.

The process of loading a Dialect in the context is not thread-safe, which forces
all involved Dialects to be loaded before the multi-threaded pass manager starts
the execution. To keep the system modular and layered, invoking a pass pipeline
should never require pre-loading dialects explicitly. This is achieved by
requiring every pass to declare a list of *dependent* Dialects: these are
Dialects for which an entity (Operation, Type, or Attribute) can be created by
the pass, other than for Dialects that would already be in the input.
For example, a `convertLinalgToLoops` pass would declare the `SCF` Dialect as
dependent, but does not need to declare `Linalg`. See also
[dependent dialects](https://mlir.llvm.org/docs/PassManagement/#dependent-dialects)
in the pass infrastructure documentation.

Finally, dialects can be *registered* with the context. The sole purpose of the
registration is to make these dialects available for the textual parser used by
tools like `mlir-opt` or `mlir-translate`. A compiler frontend emitting the IR
programmatically and invoking a pass pipeline would never need to register any
dialects.


## In dialect conversion, I want an operation to be removed after its users get converted, how do I do that?

This operation can be marked "illegal" and you can just do speculatively
`rewriter.eraseOp(op);`. The operation won't be actually removed right now,
instead when mark something as erased you are basically saying to the driver
"I expect all uses of this to go away by the time everything is over". The
conversion will fail if the operation you marked as erased doesn't actually get
erased at the end.

## Why is dialect X missing feature Y?

Most likely, nobody has had a need for it yet. Many MLIR components, dialects
even more than others, grew out of specific needs and are extended by volunteers
sending patches to add the missing bits. Everybody is welcome to contribute!

In some specfic cases, the dialect design might have explicitly decided against
implementing a feature or chose an alternative modeling that provides a similar
functionality. Such design decisions are usually noted in the dialect or
rationale documents.

## Many dialects define a `constant` operation, how do I get a constant value generically?

```c++
#include "mlir/IR/Matchers.h"

// Return the constant attribute, or null if the Operation isn't a constant.
Attribute getConstantAttr(Operation *constantOp) {
  Attribute constant;
  matchPattern(value.getDefiningOp(), m_Constant());
  return constant;
}
```

## What is the difference between traits and interfaces?

Both [traits](https://mlir.llvm.org/docs/Traits/) and
[interfaces](https://mlir.llvm.org/docs/Interfaces) can be used to inject common
behavior into operations, types and attributes without introducing duplication.
However, conceptually these are quite different.

Traits inject static behavior into operations/types/attributes whereas
interfaces dynamically dispatch behavior based on their runtime type.  For
instance, since
[`ModuleOp`](https://github.com/llvm/llvm-project/blob/f3e1f44340dc26e3810d601edf0e052813b7a11c/mlir/include/mlir/IR/BuiltinOps.td#L167)
implements the
[`SymbolTable`](https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/IR/SymbolTable.h#L338)
trait, `mlir::ModuleOp` exposes `lookupSymbol` as a member function.  However,
there is no type-erased way to access this functionality -- it is available only
via `mlir::ModuleOp`.  On the other hand, if an operation implements
[`CallOpInterface`](https://github.com/llvm/llvm-project/blob/902184e6cc263e4c66440c95a21665b6fdffe57c/mlir/include/mlir/Interfaces/CallInterfaces.td#L25),
its implementation of `getCallableForCallee` can be invoked in a type-erased
manner by `dyn_cast`ing the operation to a `CallOpInterface`.  The caller does
not need to know the concrete type of the operation to do this.

There is one similarity between interfaces and traits: both their presence can
be checked dynamically (i.e. without access to the concrete type).
Specifically, presence of traits can be checked using
[`Operation::hasTrait`](https://github.com/llvm/llvm-project/blob/902184e6cc263e4c66440c95a21665b6fdffe57c/mlir/include/mlir/IR/Operation.h#L470)
and presence of interfaces can be checked using `isa<>`.  However, this
similarity does not run deep, and was only added for practical ergonomic
reasons.

## How to convert a `memref` to a pointer?

It is impossible in the general case. Structured memory reference (`memref`) type **is not (only) a pointer**. This type supports multi-dimensional indexing and customizable data layouts to support advanced yet analyzable addressing modes. Implementing address computation requires understanding the layout and storing additional information such as sizes and layout parameters that would be impossible with a plain, single-typed pointer to a contiguous block of data. Even the single-dimensional `memref<?xi8>` with the default layout is *not a pointer* as it must store at the very least the size of the data (think C++ `std::string` vs. C `NULL`-terminated `const char *`).

It is, however, possible to define operations that create pointer-like types out of a `memref` as well as operations that, conversely, create `memref` out of pointers combined with additional information. Before implementing such operations, dialect authors are advised to carefully consider the implication of such operations on aliasing properties of the resulting IR.

Interoperability with C is often cited to motivate an opaque cast from `memref`s to pointers. The [LLVM IR target](https://mlir.llvm.org/docs/TargetLLVMIR/#ranked-memref-types) provides an interface compatible with C for a well-defined subset of `memrefs` with [strided layout](https://mlir.llvm.org/docs/Dialects/Builtin/#strided-memref). At the function boundary, it even provides a minimalist support for passing memrefs as [bare pointers](https://mlir.llvm.org/docs/TargetLLVMIR/#bare-pointer-calling-convention-for-ranked-memref) provided their sizes are known statically and their layout is trivially identity.

## What's with "op symbol declaration cannot have public visibility"?

A common mistake is to try to provide a function declaration (that is a function
without a body) but leaving it "public". Declaration must be private, only
definitions can be public in the MLIR symbol system. See the
[symbol visibility](https://mlir.llvm.org/docs/SymbolsAndSymbolTables/#symbol-visibility)
documentation.

## I'm confused about iterating on `getUsers()` vs `getUses()`: what's the difference?

The "users" of an SSA value are instances of `Operation`, while the "uses" refer to
the operands of these operations. For example considering `test.op(%0, %0) : ...`, when
iterating on the "uses" of `%0` you would see two instances of `OpOperand` (one for each
use in `test.op`), whereas iterating on the "users" of `%0` would yield directly two
`Operation *` corresponding to `test.op`. Note that you see `test.op` twice as it is
twice a user of `%0`, it's up to the call site to use a set to unique these if needed.
[The tutorial on use-def chains](https://mlir.llvm.org/docs/Tutorials/UnderstandingTheIRStructure/#traversing-the-def-use-chains) may help understand the details as well.

## How to programmatically obtain the "name" of the SSA value (`%foo`)?

The values names are _not part of the IR_ and are only there to make textual
representation of the IR easier for humans to read.  They are generated by the
IR printer on-the-fly and may differ depending on the printer configuration.
While it is technically possible to configure the printer to produce predictable
names, in particular names with specific prefixes via the
[`OpAsmOpInterface`](https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/IR/OpAsmInterface.td),
one is strongly discouraged from relying on the textual names. Therefore there
is intentionally no support for obtaining these names easily.

