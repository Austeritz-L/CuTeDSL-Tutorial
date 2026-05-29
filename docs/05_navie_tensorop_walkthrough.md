# Naive TensorOp GEMM Walkthrough

这一章把前三个 kernel 串起来读：

- `Gemm/navie_sgemm.py`
- `Gemm/navie_tensorop.py`
- `Gemm/ldsm_tensorop.py`

目标不是追求极致性能，而是建立一条清晰的理解路径：

```text
SIMT scalar GEMM
  -> Tensor Core register fragment
  -> shared memory + ldmatrix + Tensor Core
```

读完这章之后，应该能回答：

- `navie_tensorop.py` 中一个 CTA 如何映射到一个 m16n8k8 MMA atom？
- `ldsm_tensorop.py` 中一个 128x128x16 CTA 如何映射到 tiled m16n8k16 MMA？
- 每个 lane 拥有什么 A/B/C fragment？
- Tensor Core 到底从哪里进入程序？
- `shape:stride` 为什么足够解释 tensor view？
- 为什么需要 `partition_A/B/C`、`partition_S`、`retile`？

## 三个 kernel 的定位

### `navie_sgemm.py`

这是最简单的 SIMT baseline：

```text
一个 CTA 计算一个 C tile
一个线程计算一个 C 元素
A/B 用普通 copy 进入 register fragment
MmaUniversalOp 做 scalar FMA
```

核心目的是理解：

```text
local_tile
local_partition
make_fragment_like
cute.copy
cute.gemm universal atom
```

### `navie_tensorop.py`

这是最小 Tensor Core kernel：

```text
一个 CTA = 一个 warp = 一个 m16n8k8 MMA atom
A/B 从 GMEM 直接 copy 到 MMA register fragment
cute.gemm 发出 mma.sync
```

它不是高性能路径，但非常适合观察：

```text
MMA atom
TiledMma
ThrMma
partition_A/B/C
make_fragment_A/B/C
```

### `ldsm_tensorop.py`

这是更接近真实 Tensor Core GEMM 的版本：

```text
GMEM -> SMEM -> ldmatrix -> RMEM fragment -> tiled mma.sync
```

它仍然故意保持简单：

- GMEM->SMEM 使用 tiled CopyUniversalOp 做 128-bit vectorized copy
- 没有 `cp.async`
- 没有 pipeline
- SMEM layout 没有 swizzle

但主结构已经完整。

## 共同的 GEMM 问题定义

三个 kernel 都计算：

```text
C[M, N] = A[M, K] @ B[N, K]^T
```

代码里 B 被表示成 `(N, K)`，所以 reference 是：

```python
ref = torch.einsum("mk,nk->mn", a, b)
```

这和传统数学里的 `B[K,N]` 不同。这里的好处是每个 C(m,n) 的 dot product 可以直接
写成：

```text
A[m, k] * B[n, k]
```

也就是 A 和 B 都按 K 方向取一段连续向量。

## 第一阶段：CTA tile

三个 kernel 都用 `local_tile` 取 CTA tile。

对 C：

```python
gC = cute.local_tile(
    mC,
    tiler=self.cta_tiler,
    coord=(bidx, bidy, None),
    proj=(1, 1, None),
)
```

含义：

```text
当前 CTA 负责 C 的第 (bidx, bidy) 个 tile
```

对 A：

```python
gA = cute.local_tile(
    mA,
    tiler=self.cta_tiler,
    coord=(bidx, None, k_tile),
    proj=(1, None, 1),
)
```

含义：

```text
当前 CTA 在第 k_tile 次 K 循环中需要的 A tile
```

对 B：

```python
gB = cute.local_tile(
    mB,
    tiler=self.cta_tiler,
    coord=(None, bidy, k_tile),
    proj=(None, 1, 1),
)
```

含义：

```text
当前 CTA 在第 k_tile 次 K 循环中需要的 B tile
```

`proj` 的作用是把 3D GEMM tiler `(M,N,K)` 投影到每个 operand 自己的 2D tensor：

```text
C: (M,N)
A: (M,K)
B: (N,K)
```

## 第二阶段：SIMT scalar kernel

`navie_sgemm.py` 的 CTA tiler 默认可以是：

```text
(bM, bN, bK) = (16, 16, 16)
```

thread layout：

```python
thread_layout = cute.make_layout((self.bM, self.bN), stride=(self.bN, 1))
```

如果是 16x16：

```text
tid = m * 16 + n
```

每个线程负责一个 C 元素：

```python
tCgC = cute.local_partition(gC, thread_layout, tidx, proj=(1, 1))
tCrC = cute.make_fragment_like(tCgC, cutlass.Float32)
tCrC.fill(0.0)
```

对 A/B：

```python
tCgA = cute.local_partition(gA, thread_layout, tidx, proj=(1, None))
tCgB = cute.local_partition(gB, thread_layout, tidx, proj=(None, 1))
```

如果当前线程负责 `C[m,n]`，那么它拿到：

```text
tCgA = A[m, 0:bK]
tCgB = B[n, 0:bK]
```

加载到寄存器：

```python
tCrA = cute.make_fragment_like(tCgA, cutlass.Float32)
tCrB = cute.make_fragment_like(tCgB, cutlass.Float32)
cute.copy(copy_atom_A, tCgA, tCrA)
cute.copy(copy_atom_B, tCgB, tCrB)
```

计算：

```python
cute.gemm(mma_atom, tCrC, tCrA, tCrB, tCrC)
```

对这个 scalar universal atom，可以理解成：

```python
for k in range(bK):
    tCrC[0] += tCrA[k] * tCrB[k]
```

最后写回：

```python
cute.copy(copy_atom_C, tCrC, tCgC)
```

这就是最朴素的 tiled GEMM。

## 第三阶段：Tensor Core kernel

`navie_tensorop.py` 固定：

```text
CTA tile = (16, 8, 8)
threads  = 32
```

这正好匹配一条 Ampere warp-level MMA：

```text
mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32
```

构造 MMA op：

```python
mma_op = cute.nvgpu.warp.MmaF16BF16Op(
    cutlass.Float16,
    cutlass.Float32,
    (16, 8, 8),
)
```

构造 tiled MMA：

```python
tiled_mma = cute.make_tiled_mma(mma_op, cute.make_layout((1, 1, 1)))
```

这里 `(1,1,1)` 表示：

```text
不在 atom 之上再铺多个 MMA atom
当前 CTA/warp 就是一个 m16n8k8 atom
```

进入 kernel 后：

```python
thr_mma = tiled_mma.get_slice(tidx)
```

这一步把 `tidx` 映射到 MMA 的 lane/fragment 坐标。

## C accumulator fragment

代码：

```python
tCgC = thr_mma.partition_C(gC)
tCrC = tiled_mma.make_fragment_C(tCgC)
tCrC.fill(0.0)
```

这里的 `gC` 是逻辑 C tile：

```text
16 x 8 = 128 个 C 元素
```

一个 warp 32 个 lane，所以每个 lane 拥有：

```text
128 / 32 = 4 个 C accumulator
```

这和 `navie_tensorop.py` 里对单条 atom 的观察一致：

```text
lane 0 owns 4 C accumulator registers
```

也和 C++ PTX wrapper 一致：

```cpp
SM80_16x8x8_F32F16F16F32_TN:
  DRegisters = float[4]
  CRegisters = float[4]
```

`tCgC` 是当前 lane 的 global memory C view，`tCrC` 是当前 lane 的 register
accumulator。

## A/B register fragment

在每个 K tile：

```python
tCgA = thr_mma.partition_A(gA)
tCgB = thr_mma.partition_B(gB)

tCrA = tiled_mma.make_fragment_A(tCgA)
tCrB = tiled_mma.make_fragment_B(tCgB)

cute.copy(copy_A, tCgA, tCrA)
cute.copy(copy_B, tCgB, tCrB)
```

`partition_A/B` 做的是：

```text
逻辑 A/B CTA tile -> 当前 lane 的 A/B fragment view
```

`make_fragment_A/B` 创建寄存器 fragment。

这里的 copy 是 GMEM->RMEM：

```python
copy_A = cute.make_copy_atom(cute.nvgpu.CopyG2ROp(), ...)
copy_B = cute.make_copy_atom(cute.nvgpu.CopyG2ROp(), ...)
```

这条路径不够高效，但教学上有一个好处：它直接展示了 MMA atom 需要什么样的
register fragment。

## Tensor Core 计算在哪里发生

这一行：

```python
cute.gemm(tiled_mma, tCrC, tCrA, tCrB, tCrC)
```

如果 `tiled_mma` 来自：

```python
MmaF16BF16Op(Float16, Float32, (16,8,8))
```

那么它会走 Tensor Core MMA 路径，最终 lowering 到：

```text
mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32
```

如果 `mma_atom` 来自：

```python
MmaUniversalOp(Float32)
```

那就是 scalar/universal FMA 路径，不是 Tensor Core。

## 第四阶段：加入 shared memory 和 ldmatrix

当前 `ldsm_tensorop.py` 已经升级为 tiled MMA 版本：

```text
CTA tile    = (128, 128, 16)
MMA atom    = m16n8k16
atom layout = (2, 2, 1)
threads     = 128
```

区别是 A/B 不再直接从 GMEM copy 到 register fragment，而是：

```text
GMEM -> SMEM -> ldmatrix -> RMEM
```

先分配 shared memory：

```python
sA_layout = cute.make_layout((self.bM, self.bK), stride=(self.bK, 1))
sB_layout = cute.make_layout((self.bN, self.bK), stride=(self.bK, 1))

sA = smem.allocate_tensor(mA.element_type, sA_layout, 16)
sB = smem.allocate_tensor(mB.element_type, sB_layout, 16)
```

这里：

```text
sA: 128 x 16 row-major
sB: 128 x 16 row-major
```

## GMEM -> SMEM

当前 GMEM->SMEM copy 使用 Universal copy atom 加 tiled copy：

```python
g2S_copy_A = cute.make_copy_atom(
    cute.nvgpu.CopyUniversalOp(),
    mA.element_type,
    num_bits_per_copy=128,
)
tiled_g2s_A = self._make_gmem_tiled_copy_AB(g2S_copy_A, mA.element_type, 128)
```

`_make_gmem_tiled_copy_AB` 根据 `bK=16` 和 fp16 每次 128 bit copy 推出：

```python
copy_elems = 128 // 16  # 8 fp16
shape_dim_1 = bK // copy_elems  # 2
thread_layout = cute.make_layout((64, 2), stride=(2, 1))
value_layout = cute.make_layout((1, 8))
```

代码：

```python
tAgA = thr_g2s_A.partition_S(gA)
tAsA = thr_g2s_A.partition_D(sA)
tBgB = thr_g2s_B.partition_S(gB)
tBsB = thr_g2s_B.partition_D(sB)

cute.copy(tiled_g2s_A, tAgA, tAsA)
cute.copy(tiled_g2s_B, tBgB, tBsB)
cute.arch.sync_threads()
```

`sync_threads` 很重要。因为后面 `ldmatrix` 会从 `sA/sB` 读取，必须保证所有线程都已经
把 GMEM 数据写入 shared memory。

## SMEM -> RMEM：ldmatrix

构造 ldmatrix copy atom：

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

当前版本 A/B 都用 `.x4`。这里匹配的是 `make_tiled_copy_A/B(..., tiled_mma)`
生成的 S2R copy layout 与 tiled MMA A/B register fragment layout，而不是单独手算
一个最小 atom 需要几个 8x8。

绑定到 MMA layout：

```python
tiled_s2r_A = cute.make_tiled_copy_A(s2R_copy_A, tiled_mma)
tiled_s2r_B = cute.make_tiled_copy_B(s2R_copy_B, tiled_mma)
```

取当前线程 slice：

```python
thr_s2r_A = tiled_s2r_A.get_slice(tidx)
thr_s2r_B = tiled_s2r_B.get_slice(tidx)
```

构造 MMA fragment：

```python
tCsA = thr_mma.partition_A(sA)
tCsB = thr_mma.partition_B(sB)
tCrA = tiled_mma.make_fragment_A(tCsA)
tCrB = tiled_mma.make_fragment_B(tCsB)
```

构造 ldmatrix copy view：

```python
tCsA_copy_view = thr_s2r_A.partition_S(sA)
tCrA_copy_view = thr_s2r_A.retile(tCrA)
tCsB_copy_view = thr_s2r_B.partition_S(sB)
tCrB_copy_view = thr_s2r_B.retile(tCrB)
```

执行 ldmatrix：

```python
cute.copy(tiled_s2r_A, tCsA_copy_view, tCrA_copy_view)
cute.copy(tiled_s2r_B, tCsB_copy_view, tCrB_copy_view)
```

这两行之后，`tCrA` 和 `tCrB` 已经有了 MMA 指令需要的 A/B register fragments。

## MMA 和 epilogue

计算：

```python
cute.gemm(tiled_mma, tCrC, tCrA, tCrB, tCrC)
```

写回：

```python
cute.copy(r2G_copy_C, tCrC, tCgC)
```

这里 `tCgC` 是 `partition_C(gC)` 得到的 per-lane GMEM C view，所以每个 lane 把自己
的 4 个 accumulator 写回 C tile 中对应的位置。

## 静态 shape 输出怎么看

当前版本默认打印静态 type/layout，且没有运行时打印开关。典型写法：

```python
print(f"[DSL INFO]   tAgA = {tAgA.type}")
print(f"[DSL INFO]   tCrA = {tCrA.type}")
```

重点看三件事：

1. `gA/gB` 的 CTA tile shape 是否是当前 K tile 需要的 `(128,16)`
2. `tAgA/tAsA/tBgB/tBsB` 是否体现 tiled G2S 的 thread/value 分工
3. `tCsA_copy_view/tCrA_copy_view` 和 `tCsB_copy_view/tCrB_copy_view` 是否能对齐 ldmatrix 与 MMA fragment

这些 `print` 发生在 JIT tracing/编译阶段，只看 shape，不会在 GPU 上解引用数据。

## 三个 kernel 的抽象升级

### 从 `navie_sgemm.py` 到 `navie_tensorop.py`

变化：

```text
MmaUniversalOp -> MmaF16BF16Op
手写 thread_layout -> MMA atom 提供 partition_A/B/C
每线程一个 C 元素 -> 每 lane 四个 C accumulator
scalar FMA -> Tensor Core mma.sync
```

不变：

```text
local_tile 仍然负责 CTA tile
copy 仍然负责搬入 register fragment
cute.gemm 仍然是统一入口
```

### 从 `navie_tensorop.py` 到 `ldsm_tensorop.py`

变化：

```text
GMEM->RMEM 直接 copy
  -> GMEM->SMEM tiled Universal vector copy
  -> SMEM->RMEM ldmatrix
```

新增：

```text
shared memory layout
make_tiled_copy_tv
ThrCopy.partition_S/D
make_tiled_copy_A/B
partition_S
retile
sync_threads
```

不变：

```text
MMA atom 不变
MMA register fragment 不变
cute.gemm 不变
```

## 当前性能瓶颈

`ldsm_tensorop.py` 已经走了 Tensor Core，但还不是高性能 GEMM。主要瓶颈：

- 没有 `cp.async`，global load 和 compute 没有 overlap
- 没有 multi-stage pipeline
- SMEM layout 没有 swizzle，可能有 bank conflict
- residue predicate 还没有完整处理任意 M/N/K 尾部

下一步优化通常是：

```text
cp.async
SMEM swizzle
double buffering / multi-stage
尾部 predicate
```

但这些优化都建立在本章的骨架上。

## 最后用一句话串起来

`navie_sgemm.py` 教你：

```text
如何用 layout 把 CTA tile 分给线程
```

`navie_tensorop.py` 教你：

```text
如何用 MMA atom 把 CTA tile 分成 Tensor Core register fragments
```

`ldsm_tensorop.py` 教你：

```text
如何用 ldmatrix 把 shared memory tile 装进这些 register fragments
```

理解这三步之后，再读 CUTLASS 更复杂的 GEMM mainloop，就不会只看到一堆抽象名词，
而是能把它们还原成：

```text
tile selection
thread/lane partition
copy path
register fragment layout
mma instruction
store path
```
