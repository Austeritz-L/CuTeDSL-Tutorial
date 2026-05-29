# 128x128x16 LDSM TensorOp GEMM Walkthrough

这一章单独解释 `Gemm/ldsm_tensorop.py`。它当前实现的是一个教学版 Ampere
Tensor Core GEMM：

```text
CTA tile = 128 x 128 x 16
MMA atom = m16n8k16
threads  = 128
G2S      = CopyUniversalOp + make_tiled_copy_tv
S2R      = LdMatrix8x8x16bOp(False, 4) + make_tiled_copy_A/B
compute  = cute.gemm(tiled_mma, ...)
```

这不是最终高性能 GEMM：还没有 `cp.async`、pipeline、shared memory swizzle 和 完整 residue predicate。但它已经包含 Tensor Core GEMM mainloop 的核心抽象。

## 源码入口

教程 kernel：

- `Gemm/ldsm_tensorop.py`

CuTe DSL 侧：

- `cutlass/python/CuTeDSL/cutlass/cute/atom.py`
  - `make_tiled_mma`
  - `TiledMma.get_slice`
  - `ThrMma.partition_A/B/C`
  - `make_tiled_copy_tv`
  - `TiledCopy.get_slice`
  - `ThrCopy.partition_S/D`
  - `TiledCopy.retile`
- `cutlass/python/CuTeDSL/cutlass/cute/algorithm.py`
  - `copy`
  - `gemm`
- `cutlass/python/CuTeDSL/cutlass/cute/nvgpu/warp/copy.py`
  - `LdMatrix8x8x16bOp`

CuTe C++ 侧：

- `cutlass/include/cute/atom/mma_atom.hpp`
  - `TiledMMA`
  - `ThrMMA`
  - `partition_A/B/C`
- `cutlass/include/cute/atom/copy_atom.hpp`
  - `TiledCopy`
  - `ThrCopy`
  - `partition_S/D`

## Kernel 数据流

代码整体路径是：

```text
GMEM A/B
  -> tiled Universal copy
  -> SMEM A/B
  -> ldmatrix tiled copy
  -> RMEM A/B fragments
  -> tiled mma.sync
  -> RMEM C accumulators
  -> GMEM C
```

对应代码骨架：

```python
tiled_mma = cute.make_tiled_mma(...)
tiled_g2s_A = cute.make_tiled_copy_tv(...)
tiled_g2s_B = cute.make_tiled_copy_tv(...)

thr_mma = tiled_mma.get_slice(tidx)
thr_g2s_A = tiled_g2s_A.get_slice(tidx)
thr_s2r_A = tiled_s2r_A.get_slice(tidx)

tCgC = thr_mma.partition_C(gC)
tCrC = tiled_mma.make_fragment_C(tCgC)

tAgA = thr_g2s_A.partition_S(gA)
tAsA = thr_g2s_A.partition_D(sA)
cute.copy(tiled_g2s_A, tAgA, tAsA)

tCsA = thr_mma.partition_A(sA)
tCrA = tiled_mma.make_fragment_A(tCsA)
tCsA_copy_view = thr_s2r_A.partition_S(sA)
tCrA_copy_view = thr_s2r_A.retile(tCrA)
cute.copy(tiled_s2r_A, tCsA_copy_view, tCrA_copy_view)

cute.gemm(tiled_mma, tCrC, tCrA, tCrB, tCrC)
```

这几个名字的前缀可以这样读：

```text
tC... = thread view from MMA partition
tA... = thread view from G2S copy A
tB... = thread view from G2S copy B
g     = global memory
s     = shared memory
r     = register memory
```

例如 `tCgC` 表示当前线程 MMA 视角下的 C global-memory view；
`tCrC` 表示当前线程 MMA 视角下的 C register accumulator view。

## TiledMMA 设置

当前代码：

```python
self.cta_tiler = (128, 128, 16)
self.atom_layout_mnk = (2, 2, 1)
self.mma_inst_shape = (16, 8, 16)

permutation_mnk = (
    self.atom_layout_mnk[0] * self.mma_inst_shape[0],
    self.atom_layout_mnk[1] * self.mma_inst_shape[1] * 2,
    self.atom_layout_mnk[2] * self.mma_inst_shape[2],
)
```

所以：

```text
permutation_mnk = (32, 32, 16)
```

静态打印：

```text
Thr Layout VMNK: (32,2,2,1):(1,32,64,0)
Permutation MNK: (32:1,32:1,16:1)
Shape MNK:       (16,8,16)
```

`Thr Layout VMNK` 是线程布局：

```text
(V,M,N,K) -> thread_id
thread_id = V + 32*M + 64*N
```

含义：

```text
V = 32 lanes inside one warp-level MMA role
M = 2 atom groups along M
N = 2 atom groups along N
K = 1 atom group along K
```

所以 CTA 线程数是：

```text
32 * 2 * 2 * 1 = 128
```

`permutation_mnk=(32,32,16)` 不是整个 CTA tile。它是 TiledMMA 的基础 tiled
pattern。完整 CTA 是 `128x128x16`，因此同一套 pattern 会在 CTA 内重复：

```text
RestM = 128 / 32 = 4
RestN = 128 / 32 = 4
RestK = 16  / 16 = 1
```

这些 Rest modes 会体现在 `partition_A/B/C` 的结果 shape 中。

## `get_slice` 和 `partition` 的分工

这是 CuTe 里最容易混淆的地方。不要把 `get_slice` 理解成已经算出了地址。

`get_slice(tidx)` 只绑定线程身份：

```text
tidx -> 当前线程在 TiledMMA 或 TiledCopy 里的角色
```

对 MMA 来说，C++ 源码在 `mma_atom.hpp` 中是：

```cpp
auto thr_vmnk = thr_layout_vmnk_.get_flat_coord(thr_idx);
return ThrMMA<TiledMMA, decltype(thr_vmnk)>{*this, thr_vmnk};
```

也就是把 `tidx` 映射成 `(V,M,N,K)`。

`partition_*` 才把这个线程身份应用到具体 tensor layout 上：

```text
线程身份 + tensor layout -> 当前线程看到的 tensor view
```

同一个 `thr_mma` 可以作用到不同 tensor：

```python
tCgC = thr_mma.partition_C(gC)  # C 是 (M,N)
tCsA = thr_mma.partition_A(sA)  # A 是 (M,K)
tCsB = thr_mma.partition_B(sB)  # B 是 (N,K)
```

为什么要拆成两步？因为线程身份和地址不是同一件事。线程可以先知道自己在 MMA
规则里的 `(V,M,N,K)` 角色，但它要访问哪些地址，必须等看到具体 tensor 的 layout。
如果以后 `sA_layout` 换成 swizzle，`get_slice(tidx)` 不变，`partition_A(sA)` 的
结果会自动变成新的 shared-memory 地址 view。

Copy 也是同样逻辑：

```python
thr_g2s_A = tiled_g2s_A.get_slice(tidx)
tAgA = thr_g2s_A.partition_S(gA)
tAsA = thr_g2s_A.partition_D(sA)
```

`thr_g2s_A` 只知道当前线程在 G2S copy 里的 thread/value 角色；
`partition_S(gA)` 和 `partition_D(sA)` 才分别根据 GMEM/SMEM layout 算出读写 view。

## C accumulator layout

打印：

```text
gC   = "(128,128):(1024,1)"
tCgC = "((2,2),4,8):((1,8192),32768,16)"
tCrC = "((2,2),4,8):((1,2),4,16)"
```

`gC` 是当前 CTA 的 global-memory C tile：

```text
shape  = 128 x 128
stride = 1024, 1
```

`tCgC = thr_mma.partition_C(gC)` 是当前线程负责写回的 C 位置。shape：

```text
((2,2),4,8)
```

可以按三层理解：

```text
(2,2) = 单条 m16n8k16 atom 内每 lane 的 4 个 C accumulator
4     = CTA 内 M 方向的 Rest
8     = CTA 内 N 方向展开后的 Rest/value 组合
```

总元素数：

```text
2 * 2 * 4 * 8 = 128 fp32
```

这说明当前线程在整个 `128x128` CTA tile 中累计 128 个 C 元素。所有 128 个线程合计：

```text
128 threads * 128 values/thread = 16384 = 128 * 128
```

`tCgC` 和 `tCrC` 的 shape 相同，区别是 memory space 和 stride：

```text
tCgC: gmem, stride = ((1,8192),32768,16)
tCrC: rmem, stride = ((1,2),4,16)
```

也就是说：

```text
tCgC = 这些 accumulator 最后写回到 GMEM 的位置
tCrC = 这些 accumulator 在寄存器里的紧凑布局
```

## G2S tiled copy layout

当前 G2S copy 构造：

```python
g2s_copy_bits = 128
copy_elems = copy_bits // dtype.width  # 128 / 16 = 8 fp16
shape_dim_1 = self.bK // copy_elems    # 16 / 8 = 2

thread_layout = cute.make_layout((64, 2), stride=(2, 1))
value_layout = cute.make_layout((1, 8))
tiled_g2s_A = cute.make_tiled_copy_tv(copy_atom, thread_layout, value_layout)
```

打印：

```text
Tiler MN:        (64:1,16:1)
TV Layout tiled: ((2,64),8):((512,1),64)
```

`make_tiled_copy_tv` 的源码逻辑在 `atom.py` 中：

```python
tiler_mn, layout_tv = make_layout_tv(thr_layout, val_layout)
tiler_mn = product_each(tiler_mn)
return _make_tiled_copy(atom, layout_tv, tiler_mn)
```

所以它把：

```text
thread_layout: tile coordinate -> thread id
value_layout : tile coordinate -> value id
```

合成：

```text
layout_tv: (thread,value) -> tile coordinate
tiler_mn : 这个 tiled copy pattern 覆盖多大的 tile
```

这里的 `Tiler MN=(64,16)` 表示一个 G2S tiled copy pattern 覆盖：

```text
64 rows x 16 columns
```

但 A/B 的 CTA tile 是：

```text
128 x 16
```

所以 G2S partition 后会有一个 RestM：

```text
128 / 64 = 2
```

打印：

```text
gA   = "(128,16):(128,1)"
tAgA = "((8,1),2,1):((1,0),8192,0)"
tAsA = "((8,1),2,1):((1,0),1024,0)"
```

`tAgA` 是当前线程从 GMEM A 读取的 view。shape：

```text
((8,1),2,1)
```

含义：

```text
8 = 一次 128-bit copy 里的 8 个连续 fp16
2 = tiled copy pattern 在 128 行中重复两次
1 = K 方向没有额外 Rest
```

stride 解释：

```text
tAgA stride = ((1,0),8192,0)
tAsA stride = ((1,0),1024,0)
```

GMEM A 的 `gA` layout 是：

```text
(128,16):(128,1)
```

跨 64 行的 GMEM offset：

```text
64 * 128 = 8192
```

SMEM A 的 `sA` layout 是：

```text
(128,16):(16,1)
```

跨 64 行的 SMEM offset：

```text
64 * 16 = 1024
```

所以同一个 `thr_g2s_A`，应用到 `gA` 和 `sA` 后 shape 一样，但 stride 不同。这正是
`partition_S` 和 `partition_D` 分离的意义。

## S2R ldmatrix 和 MMA fragment layout

当前 S2R copy：

```python
s2R_copy_A = cute.make_copy_atom(
    cute.nvgpu.warp.LdMatrix8x8x16bOp(False, 4),
    mA.element_type,
)
tiled_s2r_A = cute.make_tiled_copy_A(s2R_copy_A, tiled_mma)
```

`make_tiled_copy_A/B` 会把 ldmatrix copy atom 的 destination layout 绑定到 `tiled_mma` 的 A/B register fragment layout。也就是说，S2R copy 的目标不是随便的 register tensor，而是 MMA 指令后面要消费的 `tCrA/tCrB`。

打印：

```text
tCsA = "((2,2,2),4,1):((1,128,8),512,0)"
tCrA = "((2,2,2),4,1):((1,2,4),8,0)"
```

`tCsA = thr_mma.partition_A(sA)` 是 MMA 视角下 A 应该从 SMEM 读哪些逻辑元素。
`tCrA = tiled_mma.make_fragment_A(tCsA)` 是 MMA 视角下的 A register fragment。

shape：

```text
((2,2,2),4,1)
```

含义：

```text
(2,2,2) = 单条 m16n8k16 atom 内每 lane 的 8 个 A operand values
4       = M 方向 Rest
1       = K 方向 Rest
```

再看 ldmatrix copy 视角：

```text
tCsA_copy_view = "((8,1),4,1):((1,0),512,0)"
tCrA_copy_view = "((8,1),4,1):((1,0),8,0)"
```

它们和 `tCsA/tCrA` 的元素总数一样：

```text
(2*2*2)*4*1 = 32
8*1*4*1     = 32
```

但是 layout 不同。原因是：

```text
tCrA           = MMA 怎么看这批寄存器
tCrA_copy_view = ldmatrix 怎么写这批寄存器
```

`retile` 的作用就是把同一批 register storage 换成 ldmatrix copy 需要的 view：

```python
tCrA_copy_view = thr_s2r_A.retile(tCrA)
```

它不搬数据，也不分配新寄存器，只是改变 view。

完整关系：

```text
sA
  -> tCsA_copy_view
  -> cute.copy(tiled_s2r_A, ..., tCrA_copy_view)
  -> same storage as tCrA
  -> cute.gemm consumes tCrA
```

B 的逻辑完全类似：

```text
tCsB = "((2,2),8,1):((1,8),256,0)"
tCrB = "((2,2),8,1):((1,2),4,0)"
tCsB_copy_view = "((8,1),4,1):((1,0),512,0)"
tCrB_copy_view = "((8,1),4,1):((1,0),8,0)"
```

## `cute.gemm` 消费什么

执行 MMA 的代码：

```python
cute.gemm(tiled_mma, tCrC, tCrA, tCrB, tCrC)
```

这里的输入必须已经符合 `tiled_mma` 的 fragment contract：

```text
tCrA: tiled_mma A register fragment
tCrB: tiled_mma B register fragment
tCrC: tiled_mma C accumulator fragment
```

`cute.gemm` 本身不会帮你从 GMEM/SMEM 找数据。它只消费已经准备好的 RMEM fragment。
当前 kernel 里，A/B fragment 的准备路径是：

```text
GMEM -> tiled G2S copy -> SMEM -> ldmatrix S2R copy -> RMEM fragment
```

对于 Ampere fp16/f32 m16n8k16，后端会 lowering 到类似：

```text
mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
```

## 一次 K tile 的完整时序

对每个 `k_tile`：

```text
1. local_tile 取 gA/gB
2. G2S partition_S/D 得到 tAgA/tAsA、tBgB/tBsB
3. cute.copy(tiled_g2s_*, ...) 把 GMEM 写入 SMEM
4. sync_threads，保证 SMEM 可见
5. ldmatrix copy 从 SMEM 装入 tCrA/tCrB
6. cute.gemm 做 tiled MMA，累加到 tCrC
7. sync_threads，下一轮 K tile 可以复用 SMEM
```

K-loop 完成后：

```python
cute.copy(r2G_copy_C, tCrC, tCgC)
```

把寄存器 accumulator 写回 `gC` 对应的位置。

## 读 shape 的方法

看到类似：

```text
!cute.memref<f16, smem, align<16>, "((8,1),4,1):((1,0),512,0)">
```

按这个顺序读：

```text
f16       = 元素类型
smem      = memory space
align<16> = 对齐
shape     = ((8,1),4,1)
stride    = ((1,0),512,0)
```

shape 说明当前 view 有多少逻辑元素以及它们的分层结构。
stride 说明这些逻辑元素在对应 memory space 中的实际 offset。

shape 里出现 `1` 和 stride 里出现 `0` 很常见。它通常表示这个 mode 是 broadcast
或 degenerate mode：逻辑上保留这一层，方便和其他 tensor/view 对齐，但实际地址不变。

## 总结

这个 kernel 最重要的抽象是：

```text
get_slice:
  把 tidx 转成当前线程在 TiledMMA/TiledCopy 里的角色

partition:
  把这个角色应用到具体 tensor layout，得到当前线程的 view

retile:
  不搬数据，只把同一批 register storage 换成另一种 copy/MMA 视角

copy/gemm:
  真正发出数据搬运或 MMA 指令
```

因此，CuTe 并不是简单隐藏底层逻辑。它把底层 PTX 对 lane/value/register 的规定写成
layout algebra，然后用 `get_slice + partition + retile` 把这些规定系统地作用到
具体 tensor 上。
