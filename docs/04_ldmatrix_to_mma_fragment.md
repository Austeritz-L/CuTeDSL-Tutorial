# 从 ldmatrix 到 MMA Fragment

这一章只讲一条路径：

```text
GMEM -> SMEM -> ldmatrix -> RMEM fragment -> mma.sync
```

重点是 `ldsm_tensorop.py` 中这段代码：

```python
s2R_copy_A = cute.make_copy_atom(
    cute.nvgpu.warp.LdMatrix8x8x16bOp(False, 4),
    mA.element_type,
)
s2R_copy_B = cute.make_copy_atom(
    cute.nvgpu.warp.LdMatrix8x8x16bOp(False, 4),
    mB.element_type,
)
tiled_s2r_A = cute.make_tiled_copy_A(s2R_copy_A, tiled_mma)
tiled_s2r_B = cute.make_tiled_copy_B(s2R_copy_B, tiled_mma)
thr_s2r_A = tiled_s2r_A.get_slice(tidx)
thr_s2r_B = tiled_s2r_B.get_slice(tidx)

tCsA_copy_view = thr_s2r_A.partition_S(sA)
tCrA_copy_view = thr_s2r_A.retile(tCrA)
tCsB_copy_view = thr_s2r_B.partition_S(sB)
tCrB_copy_view = thr_s2r_B.retile(tCrB)

cute.copy(tiled_s2r_A, tCsA_copy_view, tCrA_copy_view)
cute.copy(tiled_s2r_B, tCsB_copy_view, tCrB_copy_view)
```

## 源码入口

CuTe DSL 侧：

- `cutlass/python/CuTeDSL/cutlass/cute/nvgpu/warp/copy.py`
  - `LdMatrix8x8x16bOp`
- `cutlass/python/CuTeDSL/cutlass/cute/atom.py`
  - `make_tiled_copy_A`
  - `make_tiled_copy_B`
  - `ThrCopy.partition_S`
  - `TiledCopy.retile`

CuTe C++ 侧：

- `cutlass/include/cute/arch/copy_sm75.hpp`
  - `SM75_U32x1_LDSM_N`
  - `SM75_U32x2_LDSM_N`
  - `SM75_U32x4_LDSM_N`
- `cutlass/include/cute/atom/copy_traits_sm75.hpp`
  - `Copy_Traits<SM75_U32x*_LDSM_N>`
- `cutlass/include/cute/atom/copy_atom.hpp`
  - `TiledCopy::tidfrg_S`
  - `TiledCopy::retile`
  - `make_tiled_copy_A/B`

## `ldmatrix` 是什么

`ldmatrix` 是 PTX 的 warp-level matrix load 指令。它让一个 warp 协作从 shared
memory 读取矩阵 tile，并把数据放到每个 lane 的寄存器中。

对 Ampere 上常用的 fp16/bf16 MMA 来说，基础形式是：

```text
ldmatrix.sync.aligned.x1.m8n8.shared.b16
ldmatrix.sync.aligned.x2.m8n8.shared.b16
ldmatrix.sync.aligned.x4.m8n8.shared.b16
```

含义：

- `.sync`：warp 内线程同步参与
- `.aligned`：要求线程和地址满足对齐/一致性约束
- `.x1/.x2/.x4`：一次加载 1/2/4 个 8x8 matrix
- `.m8n8`：每个基础矩阵是 8x8
- `.shared`：source 在 shared memory
- `.b16`：每个元素 16 bit

CuTe DSL 的：

```python
LdMatrix8x8x16bOp(False, 4)
```

可以读成：

```text
ldmatrix.m8n8.shared.b16
不转置
一次加载 4 个 8x8 矩阵
```

也就是 PTX 的 `.x4` 形式。

## DSL 到 C++/PTX 的对应

DSL 源码 `warp/copy.py`：

```python
class LdMatrix8x8x16bOp(BaseOp):
    transpose: bool = False
    num_matrices: int = 1
```

它会检查：

```text
num_matrices 必须是 1、2、4
unpack_bits 不支持
```

然后构造：

```python
CopyAtomLdsmType.get(
    dtype,
    mode=(8,8),
    sz_pattern=u16,
    num_matrices,
    transpose_attr,
)
```

C++ 对应的底层 wrapper 在 `arch/copy_sm75.hpp`：

```cpp
SM75_U32x1_LDSM_N -> ldmatrix.sync.aligned.x1.m8n8.shared.b16
SM75_U32x2_LDSM_N -> ldmatrix.sync.aligned.x2.m8n8.shared.b16
SM75_U32x4_LDSM_N -> ldmatrix.sync.aligned.x4.m8n8.shared.b16
```

`_N` 表示 non-transpose。转置版本会走 `.trans` 形式。

## 为什么当前 A/B 都用 `.x4`

当前 `ldsm_tensorop.py` 的 MMA atom 是：

```text
mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
```

单条 m16n8k16 atom 的 logical operand tile 是：

```text
A: M x K = 16 x 16
B: N x K =  8 x 16
```

一个 `ldmatrix.m8n8` 的基础矩阵是：

```text
8 x 8
```

如果只看一条 atom，A 可以拆成 4 个 8x8 半精度矩阵块，B 可以拆成 2 个 8x8
矩阵块。当前代码使用 `TiledMMA` 把多个 atom group 铺成 `128x128x16` CTA tile，
再用 `make_tiled_copy_A/B` 将 ldmatrix 的 destination layout 对齐到 tiled MMA 的
A/B fragment layout。为了让每次 S2R 装载覆盖当前 tiled fragment 的向量化形状，
A 和 B 都采用 `.x4`：

```python
s2R_copy_A = cute.make_copy_atom(
    cute.nvgpu.warp.LdMatrix8x8x16bOp(False, 4),
    mA.element_type,
)
s2R_copy_B = cute.make_copy_atom(
    cute.nvgpu.warp.LdMatrix8x8x16bOp(False, 4),
    mB.element_type,
)
```

这里不要把 `.x4` 简单理解成“单条 atom 的 B 必须有 4 个 8x8”。真正需要匹配的是
`make_tiled_copy_B(s2R_copy_B, tiled_mma)` 生成的 copy TV layout 和 `tCrB` 的 tiled
register fragment layout。当前版本的静态打印会显示 `tCsB_copy_view` 和
`tCrB_copy_view` 的 shape，用它们检查 S2R copy 是否和 MMA fragment 对齐。

## 为什么不用手写 lane 到地址的映射

PTX 文档会描述每个 lane 提供哪些 shared memory 地址，以及每个 lane 得到哪些
register。直接手写这套映射很容易出错。

CuTe 的做法是把这个映射编码进 CopyAtom traits。C++ 的
`copy_traits_sm75.hpp` 里，例如 `.x4`：

```cpp
Copy_Traits<SM75_U32x4_LDSM_N>
  ThrID     = Layout<_32>
  SrcLayout = ...
  DstLayout = ...
  RefLayout = DstLayout
```

其中：

```text
SrcLayout: (src thread, src value) -> shared memory bit offset
DstLayout: (dst thread, dst value) -> register bit offset
```

也就是说，`ldmatrix` 的 PTX 语义被转换成了 layout algebra。

这就是 CuTe 的核心思想：底层硬件规定的 lane/value 排布，最终都可以变成
`(thread,value) -> coordinate/offset` 的 layout。

## `make_tiled_copy_A/B`

`ldmatrix` 读出来的数据最终要喂给 MMA 的 A/B register fragment。因此 copy 的
destination layout 必须和 MMA 的 A/B TV layout 一致。

代码：

```python
tiled_s2r_A = cute.make_tiled_copy_A(s2R_copy_A, tiled_mma)
tiled_s2r_B = cute.make_tiled_copy_B(s2R_copy_B, tiled_mma)
```

C++ 源码里：

```cpp
make_tiled_copy_A(copy_atom, mma) {
  return make_tiled_copy_impl(
      copy_atom,
      mma.get_layoutA_TV(),
      make_shape(tile_size<0>(mma), tile_size<2>(mma)));
}
```

```cpp
make_tiled_copy_B(copy_atom, mma) {
  return make_tiled_copy_impl(
      copy_atom,
      mma.get_layoutB_TV(),
      make_shape(tile_size<1>(mma), tile_size<2>(mma)));
}
```

所以：

```text
A copy tile shape = (M,K)
B copy tile shape = (N,K)
```

这一步是把 `ldmatrix` 的 copy layout 和 `mma.sync` 的 operand layout 绑定起来。

## `partition_S(sA/sB)`

代码：

```python
tCsA_copy_view = thr_s2r_A.partition_S(sA)
tCsB_copy_view = thr_s2r_B.partition_S(sB)
```

`partition_S` 根据 TiledCopy 的 source layout 切 shared memory tensor。它回答的
问题是：

```text
当前 lane 应该给 ldmatrix 提供哪些 shared memory 地址？
```

这里 `sA` 和 `sB` 是 shared memory tensor：

```python
sA_layout = cute.make_layout((self.bM, self.bK), stride=(self.bK, 1))
sB_layout = cute.make_layout((self.bN, self.bK), stride=(self.bK, 1))
sA = smem.allocate_tensor(mA.element_type, sA_layout, 16)
sB = smem.allocate_tensor(mB.element_type, sB_layout, 16)
```

当前教程先使用最简单的 row-major SMEM layout。这样比较容易理解，但不一定是性能
最好的 layout。

## `retile(tCrA/tCrB)`

代码：

```python
tCrA_copy_view = thr_s2r_A.retile(tCrA)
tCrB_copy_view = thr_s2r_B.retile(tCrB)
```

`tCrA` 和 `tCrB` 是 MMA register fragments：

```python
tCrA = tiled_mma.make_fragment_A(tCsA)
tCrB = tiled_mma.make_fragment_B(tCsB)
```

但是 `ldmatrix` copy atom 写寄存器时，有自己的 destination TV layout。
`retile` 的作用是把同一批寄存器换一种 view：

```text
MMA fragment view -> ldmatrix destination copy view
```

它不是重新分配寄存器，也不是搬数据。

可以把它理解成：

```text
tCrA 是“mma.sync 怎么看这批寄存器”
tCrA_copy_view 是“ldmatrix 怎么写这批寄存器”
```

两者指向同一批 RMEM storage。

## 完整 SMEM -> RMEM 路径

对 A 来说：

```text
sA
  -> thr_s2r_A.partition_S(sA)
  -> tCsA_copy_view
  -> cute.copy(tiled_s2r_A, ..., ...)
  -> tCrA_copy_view
  -> same storage as tCrA
  -> cute.gemm consumes tCrA
```

对 B 来说：

```text
sB
  -> thr_s2r_B.partition_S(sB)
  -> tCsB_copy_view
  -> cute.copy(tiled_s2r_B, ..., ...)
  -> tCrB_copy_view
  -> same storage as tCrB
  -> cute.gemm consumes tCrB
```

所以这几行代码的语义是：

```python
cute.copy(tiled_s2r_A, tCsA_copy_view, tCrA_copy_view)
cute.copy(tiled_s2r_B, tCsB_copy_view, tCrB_copy_view)
```

用 `ldmatrix` 从 shared memory 取 A/B tile，并按照 MMA atom 需要的 fragment
layout 写入寄存器。

## 为什么 `tCsA` 和 `tCsA_copy_view` 都存在

`ldsm_tensorop.py` 里有两种 view：

```python
tCsA = thr_mma.partition_A(sA)
tCsA_copy_view = thr_s2r_A.partition_S(sA)
```

它们都来自 `sA`，但用途不同。

`tCsA` 是 MMA 视角：

```text
如果 MMA 要从 sA 这个 tensor 得到 A fragment，它期望的 per-lane A view 是什么？
```

它被用来创建：

```python
tCrA = tiled_mma.make_fragment_A(tCsA)
```

`tCsA_copy_view` 是 ldmatrix copy 视角：

```text
如果 ldmatrix 要从 sA 读取，它期望当前 lane 提供哪些 SMEM 地址？
```

它被用来执行：

```python
cute.copy(tiled_s2r_A, tCsA_copy_view, tCrA_copy_view)
```

所以：

```text
tCsA           = 为了构造 MMA fragment
tCsA_copy_view = 为了执行 ldmatrix copy
```

B 同理。

## Shared Memory Layout 和 bank conflict

当前教程使用：

```python
sA_layout = cute.make_layout((128, 16), stride=(16, 1))
sB_layout = cute.make_layout((128, 16), stride=(16, 1))
```

这非常直观：

```text
row-major
offset(row, col) = row * K + col
```

优点是好理解。缺点是性能上可能有 shared memory bank conflict。

Shared memory 通常按 bank 组织。多个 lane 如果在同一条指令里访问落到同一个 bank
的不同地址，就可能发生 bank conflict。`ldmatrix` 是 warp-level load，它对地址
模式很敏感。

高性能 CUTLASS kernel 通常会对 shared memory layout 做 swizzle。swizzle 的目的
是：

```text
逻辑上仍然是 A(m,k)、B(n,k)
物理上改变 shared memory 地址分布
让同一个 warp 的 ldmatrix 访问更均匀地落到不同 bank
```

CuTe 里 swizzle 本质上仍然是 layout：

```text
logical coordinate -> swizzled physical offset
```

所以后续优化不是推翻当前代码，而是替换：

```python
sA_layout = ...
sB_layout = ...
```

并保证：

```text
GMEM->SMEM copy 写入的 layout
ldmatrix partition_S 读取的 layout
MMA fragment retile 的 layout
```

三者保持一致。

## 当前 kernel 的限制

`ldsm_tensorop.py` 是教学 kernel，不是最终高性能 kernel。它还没有：

- `cp.async`
- double buffering / multi-stage pipeline
- swizzled shared memory layout
- 更完整的 residue predicate

但它已经具备 Tensor Core 主循环的核心结构：

```text
vectorized G2S tiled copy 把 A/B tile load 到 SMEM
sync
ldmatrix to registers
mma.sync
store C
```

理解这条路径之后，再加 pipeline 和 swizzle 就是性能工程问题，而不是抽象理解问题。

## 调试建议

调试 ldmatrix 时优先打印静态 type：

```python
print(f"tCsA_copy_view = {tCsA_copy_view.type}")
print(f"tCrA_copy_view = {tCrA_copy_view.type}")
```

当前版本默认只保留这种静态打印。它发生在 JIT tracing/编译阶段，只展示
tensor/layout shape，不会在 GPU 上解引用数据，也不会因为打印 GMEM view 触发越界。

## 一句话总结

`ldmatrix` 在 CuTe 里就是一个特殊 CopyAtom：

```text
CopyAtom 描述 PTX ldmatrix 指令
TiledCopy 把 ldmatrix 对齐到 MMA A/B layout
partition_S 生成 SMEM source view
retile 生成 RMEM destination view
cute.copy 发出 ldmatrix
cute.gemm 发出 mma.sync
```
