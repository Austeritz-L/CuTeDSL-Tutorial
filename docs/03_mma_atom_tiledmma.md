# MMA Atom 和 Tiled MMA

这一章解释 CuTe 如何描述矩阵乘法指令，尤其是 Ampere 上的
`mma.sync.aligned.m16n8k8`。

核心链路是：

```text
MMA op -> MMA atom trait -> tiled MMA -> per-lane fragments -> cute.gemm
```

在我们的教程里有两种 MMA：

- `MmaUniversalOp(Float32)`：教学用 scalar/universal atom，本质上是普通 FMA 逻辑
- `warp.MmaF16BF16Op(Float16, Float32, (16,8,8))`：Ampere Tensor Core warp-level MMA

## 源码入口

CuTe DSL 侧：

- `cutlass/python/CuTeDSL/cutlass/cute/atom.py`
  - `MmaAtom`
  - `TiledMma`
  - `ThrMma`
  - `make_mma_atom`
  - `make_tiled_mma`
- `cutlass/python/CuTeDSL/cutlass/cute/algorithm.py`
  - `gemm`
- `cutlass/python/CuTeDSL/cutlass/cute/nvgpu/common.py`
  - `MmaUniversalOp`
- `cutlass/python/CuTeDSL/cutlass/cute/nvgpu/warp/mma.py`
  - `MmaF16BF16Op`

CuTe C++ 侧：

- `cutlass/include/cute/atom/mma_atom.hpp`
  - `MMA_Atom`
  - `TiledMMA`
  - `ThrMMA`
  - `make_tiled_mma`
- `cutlass/include/cute/atom/mma_traits_sm80.hpp`
  - `MMA_Traits<SM80_16x8x8_...>`
- `cutlass/include/cute/arch/mma_sm80.hpp`
  - `SM80_16x8x8_F32F16F16F32_TN`

## MMA 的三层抽象

和 copy 类似，MMA 也可以分三层：

```text
MmaOp      = 具体数学/硬件操作
MmaAtom    = 这个操作的 trait：shape、线程数、A/B/C TV layout、数据类型
TiledMma   = 把一个 MMA atom 铺到更大的 MNK tile 上
ThrMma     = 当前线程在 TiledMma 里的切片
```

`MmaAtom` 关心的是一条 atom 级别的 MMA。对 Ampere m16n8k8 来说，atom 就是一条
warp-level Tensor Core 指令：

```text
mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32
```

`TiledMma` 关心的是如果一个 CTA/warp 里有多个 atom，应该怎么把 atom tile 排列到
更大的 `(M,N,K)` tile 上。我们当前教程故意使用最简单的：

```python
tiled_mma = cute.make_tiled_mma(mma_op, cute.make_layout((1, 1, 1)))
```

这表示：

```text
不在 atom 之上再做 tiling
一个 CTA/warp 正好计算一个 m16n8k8 MMA atom
```

## `MmaUniversalOp` 是什么

`navie_sgemm.py` 里使用：

```python
mma_atom = cute.make_mma_atom(cute.nvgpu.MmaUniversalOp(cutlass.Float32))
```

这个 atom 不是 Tensor Core 指令。它的作用是给 CuTe 的 `cute.gemm` 一个统一的
MMA 接口，让我们可以用同一套形式写 scalar GEMM：

```python
cute.gemm(mma_atom, tCrC, tCrA, tCrB, tCrC)
```

在我们的 scalar SGEMM 中，它可以近似理解为：

```python
for k in range(size(tCrA)):
    tCrC[0] += tCrA[k] * tCrB[k]
```

所以回答一个常见问题：

```text
MmaUniversalOp 会不会用 Tensor Core？
```

在这个上下文里不会。它是 universal/scalar 形式的 MMA atom，通常对应普通 CUDA core
上的标量 FMA 或等价 IR。要使用 Tensor Core，需要使用 warp-level MMA op，例如
`MmaF16BF16Op`。

## `MmaF16BF16Op`

Tensor Core 版本里使用：

```python
mma_op = cute.nvgpu.warp.MmaF16BF16Op(
    cutlass.Float16,
    cutlass.Float32,
    (16, 8, 8),
)
```

这三个参数分别是：

```text
A/B dtype      = Float16
accumulator    = Float32
instruction    = m16n8k8
```

DSL 源码 `warp/mma.py` 会检查：

- A/B dtype 必须是 `Float16` 或 `BFloat16`
- accumulator dtype 必须是 `Float16` 或 `Float32`
- shape 必须是 `(16,8,8)` 或 `(16,8,16)`

然后构造 MLIR 层的 SM80 MMA atom type：

```python
MmaAtomSM80Type.get(shape_mnk, ab_dtype, ab_dtype, acc_dtype)
```

C++ 最终对应到 `arch/mma_sm80.hpp` 的内联汇编。对我们这个配置来说，对应结构是：

```cpp
SM80_16x8x8_F32F16F16F32_TN
```

它的寄存器签名是：

```cpp
using DRegisters = float[4];
using ARegisters = uint32_t[2];
using BRegisters = uint32_t[1];
using CRegisters = float[4];
```

最终 PTX 形态是：

```text
mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32
```

这就是 Tensor Core 真正进入程序的位置。

## MMA Traits 保存了什么

C++ `MMA_Atom` 继承自 `MMA_Traits`。核心字段是：

```cpp
using ValTypeD = ...
using ValTypeA = ...
using ValTypeB = ...
using ValTypeC = ...

using Shape_MNK  = ...
using ThrID      = ...
using LayoutC_TV = ...
using LayoutA_TV = ...
using LayoutB_TV = ...
```

对 SM80 m16n8k8 fp16 输入、fp32 accumulate：

```cpp
using Shape_MNK = Shape<_16,_8,_8>;
using ThrID     = Layout<_32>;
using ALayout   = SM80_16x8_Row;
using BLayout   = SM80_8x8_Row;
using CLayout   = SM80_16x8_Row;
```

这些 layout 的意义是：

```text
LayoutA_TV: (thread, value) -> A(m,k)
LayoutB_TV: (thread, value) -> B(n,k)
LayoutC_TV: (thread, value) -> C(m,n)
```

也就是说，MMA traits 把 PTX 文档里的 fragment 分布编码成 CuTe layout。

这也是 CuTe 抽象最关键的地方：你不需要在 kernel 里手写 lane 0 该拿哪些 A/B/C
寄存器。MMA atom trait 已经保存了这件事。

## 为什么每个 lane 有 4 个 C accumulator

`m16n8k8` 的 C tile 是：

```text
16 x 8 = 128 个 C 元素
```

一个 warp 有 32 个 lane：

```text
128 / 32 = 4
```

所以对于 fp32 accumulate，每个 lane 拥有 4 个 fp32 accumulator register。

这和 C++ PTX wrapper 里：

```cpp
using DRegisters = float[4];
using CRegisters = float[4];
```

完全一致。

在我们的 debug 输出里：

```text
lane 0 owns 4 C accumulator registers
```

就是这个原因。

## `make_tiled_mma`

当前代码：

```python
tiled_mma = cute.make_tiled_mma(mma_op, cute.make_layout((1, 1, 1)))
```

`make_layout((1,1,1))` 表示 atom layout MNK：

```text
M 方向 1 个 atom
N 方向 1 个 atom
K 方向 1 个 atom
```

所以 tile shape 仍然是 atom shape：

```text
M = 16
N = 8
K = 8
```

C++ `make_tiled_mma` 会把这个 thread layout 扩展成 rank-3 的
`AtomLayoutMNK`，再构造：

```cpp
TiledMMA<MMA_Atom<MMA_Op>, AtomLayoutMNK, PermutationMNK>
```

内部关键字段叫：

```cpp
ThrLayoutVMNK
```

可以读成：

```text
V: atom 内部的 warp lanes
M/N/K: atom 在更大 tiled MMA 中沿 M/N/K 的排列
```

当前 `(1,1,1)` 的情况最简单：没有额外 M/N/K 方向的 atom replication。

## `get_slice(tidx)`

代码：

```python
thr_mma = tiled_mma.get_slice(tidx)
```

这一步把整个 TiledMma 变成当前线程视角的 `ThrMma`。

C++ 源码里：

```cpp
auto thr_vmnk = thr_layout_vmnk_.get_flat_coord(thr_idx);
return ThrMMA<TiledMMA, decltype(thr_vmnk)>{*this, thr_vmnk};
```

也就是说，`tidx` 会被映射成一个 `(V,M,N,K)` 坐标。之后
`partition_A/B/C` 都会用这个坐标取当前 lane 的 fragment。

这就是为什么 `partition_A/B/C` 依赖 `tidx`：不同 lane 持有不同的 A/B/C 寄存器。

## `partition_C`

代码：

```python
tCgC = thr_mma.partition_C(gC)
tCrC = tiled_mma.make_fragment_C(tCgC)
tCrC.fill(0.0)
```

`gC` 是当前 CTA 的 C tile view，形状大约是 `(16,8)`。

`partition_C(gC)` 做的是：

```text
C tile layout -> MMA C TV layout -> 当前 lane 的 C view
```

C++ 的 `ThrMMA::partition_C`：

```cpp
auto thr_tensor = make_tensor(ctensor.data(), this->thrfrg_C(ctensor.layout()));
auto thr_vmn = make_coord(V, make_coord(M, N));
return thr_tensor(thr_vmn, make_coord(_, ...));
```

其中 `thrfrg_C` 会经历几步：

1. 根据 permutation 重新组织 `(M,N)`
2. 用 atom shape `(16,8)` 做 `zipped_divide`
3. 用 `AtomLayoutC_TV` 把 atom 内坐标变成 `(thread,value)`
4. 再按 tiled MMA 的 thread layout 切线程

最后得到当前 lane 的 C view。

`make_fragment_C(tCgC)` 创建真正的 accumulator register tensor。对 fp32 m16n8k8，
每个 lane 是 4 个 float。

## `partition_A` 和 `partition_B`

代码：

```python
tCgA = thr_mma.partition_A(gA)
tCgB = thr_mma.partition_B(gB)

tCrA = tiled_mma.make_fragment_A(tCgA)
tCrB = tiled_mma.make_fragment_B(tCgB)
```

`gA` 是 `(M,K)` tile，`gB` 是 `(N,K)` tile。`partition_A/B` 的作用是：

```text
把逻辑 A/B tile 转成当前 lane 需要的 A/B fragment view
```

C++ 源码里：

```cpp
partition_A:
  thrfrg_A(atensor.layout())
  use coordinate (V, M, K)

partition_B:
  thrfrg_B(btensor.layout())
  use coordinate (V, N, K)
```

这也解释了为什么 A 是 `(M,K)`，B 是 `(N,K)`。对 `mma.sync.row.col` 来说，A
按 row-major 逻辑读，B 按 column-major 逻辑读；我们代码里把 B tensor 建成
`(N,K)`，这样访问形式仍然是行向量：

```python
torch.einsum("mk,nk->mn", a, b)
```

## `make_fragment_A/B/C`

`partition_A/B/C` 得到的是“某个 tensor 的 per-lane view”。这个 tensor 可能在
GMEM，也可能在 SMEM。

`make_fragment_A/B/C` 创建的是 MMA 指令需要的 register fragment。

C++ `MMA_Atom::make_fragment_A` 里有一个重要注释：它希望输入已经是
`partition_A` 之后的 tensor，因为它会检查 partitioned tensor 的 layout，
然后创建匹配的 fragment，以便后续 copy 能向量化或对齐。

这也是为什么我们的代码顺序是：

```python
tCgA = thr_mma.partition_A(gA)
tCrA = tiled_mma.make_fragment_A(tCgA)
```

而不是直接：

```python
tCrA = tiled_mma.make_fragment_A(gA)
```

## `cute.gemm`

Tensor Core 版本中：

```python
cute.gemm(tiled_mma, tCrC, tCrA, tCrB, tCrC)
```

`algorithm.py::gemm` 要求：

```text
All tensors must be partitioned according to the provided MMA Atom.
```

也就是说：

- `tCrA` 必须符合 A fragment layout
- `tCrB` 必须符合 B fragment layout
- `tCrC` 必须符合 C fragment layout

DSL `cute.gemm` 会把 `tiled_mma` unpack 成底层 trait，然后生成 `_cute_ir.gemm`。
在 SM80 m16n8k8 这个配置下，后端会降低到：

```text
mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32
```

如果使用 `MmaUniversalOp`，则是 scalar/universal 路径，不是 Tensor Core。

## `navie_tensorop.py` 的数据流

这个 kernel 故意不用 shared memory，方便观察寄存器 layout：

```text
GMEM A tile -> partition_A -> tCgA -> copy G2R -> tCrA
GMEM B tile -> partition_B -> tCgB -> copy G2R -> tCrB
GMEM C tile -> partition_C -> tCgC
tCrC = register accumulator
cute.gemm(tiled_mma, tCrC, tCrA, tCrB, tCrC)
tCrC -> copy R2G -> tCgC
```

真实高性能 GEMM 不会让 Tensor Core operand 直接从 GMEM 进 register fragment；
通常会走：

```text
GMEM -> SMEM -> ldmatrix -> RMEM -> mma.sync
```

但 `navie_tensorop.py` 对理解 MMA fragment 很有价值，因为它去掉了 shared memory
和 ldmatrix 的干扰。

## `ldsm_tensorop.py` 的数据流

这个 kernel 更接近真实 Tensor Core mainloop：

```text
GMEM A/B -> SMEM A/B
SMEM A/B -> ldmatrix -> RMEM A/B fragments
RMEM A/B/C -> mma.sync
RMEM C -> GMEM C
```

其中 MMA 相关代码仍然是：

```python
thr_mma = tiled_mma.get_slice(tidx)
tCgC = thr_mma.partition_C(gC)
tCrC = tiled_mma.make_fragment_C(tCgC)

tCsA = thr_mma.partition_A(sA)
tCsB = thr_mma.partition_B(sB)
tCrA = tiled_mma.make_fragment_A(tCsA)
tCrB = tiled_mma.make_fragment_B(tCsB)

cute.gemm(tiled_mma, tCrC, tCrA, tCrB, tCrC)
```

`ldmatrix` 只是负责把 shared memory 中的数据装进 `tCrA/tCrB`，MMA 指令消费的仍然
是这些 register fragments。

## 初学者应该抓住什么

看 MMA 代码时先问五个问题：

1. 用的是 universal atom 还是 Tensor Core atom？
2. MMA atom 的 shape 是多少？比如 `(16,8,8)`
3. 一个 CTA/warp 里有几个 atom？当前教程是 `(1,1,1)`
4. A/B/C tensor 是否经过了 `partition_A/B/C`？
5. A/B/C register fragment 是否通过 `make_fragment_A/B/C` 创建？

只要能回答这五个问题，就能读懂大部分 CuTe MMA kernel 的骨架。

## 和 TiledMMA 优化的关系

当前教程使用：

```text
1 CTA = 1 warp = 1 MMA atom
```

后续优化会变成：

```text
1 CTA = 多个 warp
1 warp = 多个 MMA atom
CTA tile = 多个 m16n8k8 atom 拼起来
```

那时 `make_tiled_mma` 的 atom layout 不再是 `(1,1,1)`。但底层逻辑不变：

```text
MMA traits 定义单条指令的 fragment layout
TiledMMA 把单条指令铺到更大的 tile
ThrMMA 根据 tidx 取当前线程的 A/B/C view
cute.gemm 消费已经 partition 好的 register fragments
```
