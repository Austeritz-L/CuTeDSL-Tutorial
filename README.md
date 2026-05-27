# CuTeDSL-Tutorial

A long-term learning and research repository for exploring **CuTe DSL**, high-performance GPU kernels, and the algebraic ideas behind tensor layout mapping.

This project starts from **GEMM** and gradually moves toward **Attention**. The goal is not only to implement working kernels, but also to understand how mathematical expressions, tensor layouts, data movement, and GPU execution are connected.

---

## Motivation

High-performance GPU programming is not just about writing fast code. It is also about understanding how a mathematical computation is mapped onto hardware.

This repository is my attempt to study that mapping through CuTe DSL.

The main questions I want to explore are:

- What do `Shape`, `Stride`, and `Layout` mean in CuTe?
- How are logical tensor coordinates mapped to physical memory addresses?
- How is GEMM decomposed into tiled computations?
- How are tiles assigned to threads, warps, and CTAs?
- How does Tensor Core MMA interact with layout design?
- How can Attention be understood as a system built from GEMM, softmax, and reductions?
- How can a correct kernel be gradually optimized into a high-performance kernel?

---

## Main Topics

### 1. CuTe DSL Basics

The first part of this project focuses on the basic abstractions of CuTe DSL:

- Shape
- Stride
- Layout
- Tensor
- Tile
- Copy
- MMA

The goal is to understand these abstractions not only as APIs, but also as algebraic tools for describing data layout and computation mapping.

### 2. GEMM

GEMM is the starting point of this repository.

Planned topics include:

- Naive GEMM
- Tiled GEMM
- Shared memory tiling
- Tensor Core MMA
- Pipeline design
- Performance analysis

GEMM will be used as the main example to study how matrix multiplication is transformed from a mathematical expression into a GPU kernel.

### 3. Attention

After building a basic understanding of GEMM, this project will move toward Attention.

Planned topics include:

- `Q @ K^T`
- Softmax
- `P @ V`
- Online softmax
- Causal masking
- FlashAttention-style blocking

The goal is to understand Attention as a structured composition of GEMM, reduction, normalization, and memory-efficient data movement.

---

## Project Philosophy

The central idea of this repository is:

```text
Math
  -> Layout Algebra
  -> Data Movement
  -> GPU Mapping
  -> CuTe DSL Implementation
  -> Benchmark
```

Instead of treating performance optimization as a collection of tricks, I want to study it as a mapping problem:

1. What is the mathematical computation?
2. How is the data logically organized?
3. How is the data physically laid out in memory?
4. How is the work distributed across GPU threads?
5. How do hardware instructions participate in the computation?
6. Where are the actual performance bottlenecks?

---

## Current Status

This repository is just getting started.

The first step is to establish the learning direction and project motivation. Code, notes, experiments, and benchmarks will be added gradually.

---

## Long-Term Goal

The long-term goal of this repository is to build a systematic learning record around CuTe DSL and high-performance GPU kernels.

Ideally, this repository will eventually contain:

- CuTe DSL implementations of GEMM kernels
- CuTe DSL implementations of Attention kernels
- Notes on CuTe layout algebra
- Experiments on GPU mapping and data movement
- Benchmark results and performance analysis
- Blog-style explanations of the learning process

---

## Notes

This is a personal learning project. The content will evolve over time as I learn more about CuTe DSL, CUTLASS, GEMM, Attention, and GPU kernel optimization.

---

## License

To be decided.
