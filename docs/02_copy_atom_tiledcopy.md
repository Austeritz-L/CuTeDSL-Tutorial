# Copy Atom 和 Tiled Copy

这一章解释 CuTe 如何描述数据搬运。先看最小模型：

```text
source tensor view + destination tensor view + copy atom -> 生成具体 copy 指令
```

在我们的三个 GEMM kernel 里有两类 copy：

- 普通 copy：GMEM -> RMEM、RMEM -> GMEM、GMEM -> SMEM
- warp-level matrix copy：SMEM -> RMEM，也就是 `ldmatrix`

普通 copy 可以只用 `CopyAtom`。`ldmatrix` 这种 copy 必须和 MMA 的寄存器布局匹配，
所以需要 `TiledCopy`、`ThrCopy`、`partition_S` 和 `retile`。

## 源码入口

CuTe DSL 侧：

- `cutlass/python/CuTeDSL/cutlass/cute/atom.py`
  - `CopyAtom`
  - `TiledCopy`
  - `ThrCopy`
  - `make_copy_atom`
  - `make_tiled_copy_A`
  - `make_tiled_copy_B`
- `cutlass/python/CuTeDSL/cutlass/cute/algorithm.py`
  - `copy`
- `cutlass/python/CuTeDSL/cutlass/cute/nvgpu/common.py`
  - `CopyUniversalOp`
  - `CopyG2ROp`
  - `CopyR2GOp`
- `cutlass/python/CuTeDSL/cutlass/cute/nvgpu/warp/copy.py`
  - `LdMatrix8x8x16bOp`

CuTe C++ 侧：

- `cutlass/include/cute/atom/copy_atom.hpp`
  - `Copy_Atom`
  - `TiledCopy`
  - `ThrCopy`
  - `make_tiled_copy_A/B`
- `cutlass/include/cute/algorithm/copy.hpp`
  - `copy`
  - `copy_if`
- `cutlass/include/cute/arch/copy_sm75.hpp`
  - `SM75_U32x1_LDSM_N`
  - `SM75_U32x2_LDSM_N`
  - `SM75_U32x4_LDSM_N`
- `cutlass/include/cute/atom/copy_traits_sm75.hpp`
  - `Copy_Traits<SM75_U32x*_LDSM_N>`

## 三层抽象

copy 相关抽象可以分成三层：

```text
CopyOp      = 具体硬件/IR 操作，比如普通 load/store、ldmatrix
CopyAtom    = 这个操作的 trait：线程数、src/dst TV layout、value type
TiledCopy   = 把一个 CopyAtom 复制/铺开到更大的 tile
ThrCopy     = 某一个线程在 TiledCopy 里的切片
```

其中最关键的是 TV layout：

```text
T = thread
V = value

(thread, value) -> logical coordinate / bit offset
```

CopyAtom 不是只记录“我要 copy”。它还记录：

- 这个 copy 指令由多少线程共同执行
- 每个线程输入几个 value
- 每个线程输出几个 value
- source 侧 `(thread,value)` 怎么映射
- destination 侧 `(thread,value)` 怎么映射

这就是为什么 `ldmatrix` 可以被表达成 copy。它本质上也是：

```text
SMEM 中某些地址 -> 每个 lane 的寄存器
```

只是这个映射由 PTX 指令规定，不能随便写。

## `make_copy_atom`

普通 copy 的写法：

```python
copy_A = cute.make_copy_atom(
    cute.nvgpu.CopyG2ROp(),
    mA.element_type,
    num_bits_per_copy=mA.element_type.width,
)
```

或者：

```python
g2S_copy_A = cute.make_copy_atom(
    cute.nvgpu.CopyUniversalOp(),
    mA.element_type,
    num_bits_per_copy=mA.element_type.width,
)
```

`make_copy_atom(op, dtype, ...)` 的作用是把一个 copy operation 变成 copy atom。
在 DSL 源码里，`CopyAtom` 暴露了这些属性：

```python
copy_atom.thr_id
copy_atom.layout_src_tv
copy_atom.layout_dst_tv
copy_atom.value_type
```

这和 C++ 里的 `Copy_Atom` 一一对应。C++ 源码中：

```cpp
using ThrID        = typename Traits::ThrID;
using BitLayoutSrc = typename Traits::SrcLayout;
using BitLayoutDst = typename Traits::DstLayout;
using BitLayoutRef = typename Traits::RefLayout;
using ValType      = CopyInternalType;
```

也就是说 CopyAtom 的核心是 copy traits。不同 op 的 traits 不同。

## `CopyUniversalOp`、`CopyG2ROp`、`CopyR2GOp`

在我们的教程代码里：

```python
cute.nvgpu.CopyUniversalOp()
cute.nvgpu.CopyG2ROp()
cute.nvgpu.CopyR2GOp()
```

可以先这样理解：

- `CopyUniversalOp`：通用 copy，适合教学和普通路径，source/destination 由 tensor
  memory space 决定
- `CopyG2ROp`：更明确地表达 global memory 到 register memory
- `CopyR2GOp`：更明确地表达 register memory 到 global memory

`navie_tensorop.py` 里使用：

```python
cute.copy(copy_A, tCgA, tCrA)
cute.copy(copy_B, tCgB, tCrB)
cute.copy(copy_C, tCrC, tCgC)
```

这里：

```text
tCgA: per-lane 的 A 的 GMEM view
tCrA: per-lane 的 A 的 RMEM fragment
tCgB: per-lane 的 B 的 GMEM view
tCrB: per-lane 的 B 的 RMEM fragment
tCrC: per-lane 的 C accumulator fragment
tCgC: per-lane 的 C 的 GMEM view
```

`ldsm_tensorop.py` 的 GMEM->SMEM 路径使用 `CopyUniversalOp`：

```python
cute.copy(g2S_copy_A, tAgA, tAsA)
cute.copy(g2S_copy_B, tBgB, tBsB)
```

这里每个线程的 source/destination slice 是我们用 `local_partition` 手动构造的。

## `cute.copy` 的源码逻辑

DSL 侧 `algorithm.py::copy` 的文档说得很明确：source 和 destination 必须符合
copy atom 需要的 layout profile：

```text
(V, Rest...)
```

其中 `V` 是 copy atom 一次能消费的 value 维度，`Rest...` 是外层循环维度。

如果 `src` 和 `dst` 不是单次指令能直接处理的形状，copy 算法会沿着 `Rest...`
递归或展开，直到最内层形状匹配 copy atom 的 instruction granularity。

C++ 里的 `Copy_Atom::call` 也体现了同样的逻辑：

```cpp
if size(src) or size(dst) matches instruction:
    copy_unpack(traits, src, dst)
else:
    peel one rank and recurse
```

所以 `cute.copy` 不是一个固定的 `ld.global` 或 `st.global`。它是一个 copy algorithm：

```text
检查 tensor layout -> 匹配 atom -> 必要时展开 Rest 维 -> emit 具体 copy IR/PTX
```

## 为什么需要 TiledCopy

普通 copy 可以这样写：

```python
cute.copy(copy_atom, src, dst)
```

前提是 `src` 和 `dst` 已经是这个线程应该处理的 slice。

但是 `ldmatrix` 不一样。`ldmatrix` 是一个 warp-level 指令：

```text
32 个 lane 协作
从 shared memory 读取 8x8 / 多个 8x8 的矩阵
按照 PTX 固定规则写入每个 lane 的寄存器
```

也就是说 source 的 SMEM layout 和 destination 的 MMA register layout 必须对齐。
这个对齐关系不能只靠一个普通 CopyAtom 表达，需要把 copy atom 放到 MMA tile 的
坐标系统里。于是就有：

```python
tiled_s2r_A = cute.make_tiled_copy_A(s2R_copy_A, tiled_mma)
tiled_s2r_B = cute.make_tiled_copy_B(s2R_copy_B, tiled_mma)
thr_s2r_A = tiled_s2r_A.get_slice(tidx)
thr_s2r_B = tiled_s2r_B.get_slice(tidx)
```

`make_tiled_copy_A` 的意思是：

```text
构造一个 TiledCopy，它的 destination layout 匹配 tiled_mma 的 A operand TV layout
```

`make_tiled_copy_B` 同理，只是匹配 B operand。

C++ 源码里对应实现非常直接：

```cpp
make_tiled_copy_A(copy_atom, mma) {
  return make_tiled_copy_impl(
      copy_atom,
      mma.get_layoutA_TV(),
      make_shape(tile_size<0>(mma), tile_size<2>(mma)));
}

make_tiled_copy_B(copy_atom, mma) {
  return make_tiled_copy_impl(
      copy_atom,
      mma.get_layoutB_TV(),
      make_shape(tile_size<1>(mma), tile_size<2>(mma)));
}
```

所以 A 用 `(M,K)` tile，B 用 `(N,K)` tile。

## `ThrCopy.partition_S`

在 `ldsm_tensorop.py` 里：

```python
tCsA_copy_view = thr_s2r_A.partition_S(sA)
tCsB_copy_view = thr_s2r_B.partition_S(sB)
```

`partition_S` 的作用是：

```text
把 source tensor 按 TiledCopy 的 source TV layout 切成当前线程要提供给 copy atom 的 view
```

这里 source 是 shared memory tensor：

```text
sA: SMEM A tile, shape 16x8
sB: SMEM B tile, shape 8x8
```

C++ 的 `ThrCopy::partition_S` 做了两步：

```cpp
auto thr_tensor = make_tensor(stensor.data(), TiledCopy::tidfrg_S(stensor.layout()));
return thr_tensor(thr_idx_, _, repeat<rank_v<STensor>>(_));
```

也就是：

1. 用 `tidfrg_S` 把原 tensor layout 转成 `(thread, fragment, rest...)` 形状
2. 用当前 `thr_idx` 取出当前线程的那一份

这和 `local_partition` 很像，但布局不是用户手写的，而是由 TiledCopy 的 source
layout 推导出来。

## `retile`

在 `ldsm_tensorop.py` 里：

```python
tCrA = tiled_mma.make_fragment_A(tCsA)
tCrB = tiled_mma.make_fragment_B(tCsB)

tCrA_copy_view = thr_s2r_A.retile(tCrA)
tCrB_copy_view = thr_s2r_B.retile(tCrB)
```

`tCrA` 和 `tCrB` 是 MMA atom 需要的 register fragment。但是 `ldmatrix` copy atom
看到的 destination layout 是 copy 的 TV layout。两者描述的是同一批寄存器，只是
坐标系统不同。

`retile` 的作用就是：

```text
把已有 RMEM fragment 重新解释成 TiledCopy 期望的 destination view
```

它不应该理解为重新分配寄存器，也不是搬数据。它是 view/layout 变换：

```text
same register storage, different logical view
```

C++ 源码里的 `TiledCopy::retile` 做的事情比较复杂，但核心目的可以这样概括：

```text
根据 TiledLayout_TV 的 value 顺序，重建 fragment 的 V 维和 rest 维，
让 copy atom 写进去的位置正好等于 MMA atom 后续读取的位置。
```

所以在 `ldmatrix` 路径中：

```python
cute.copy(tiled_s2r_A, tCsA_copy_view, tCrA_copy_view)
```

不能直接写成：

```python
cute.copy(tiled_s2r_A, tCsA, tCrA)
```

因为 `tCsA` 是 MMA partition 视角的 SMEM view，`tCrA` 是 MMA fragment 视角的
RMEM view；`ldmatrix` 需要的是 copy atom 视角的 source/destination view。

## `partition_D` 和 `retile_D/S`

DSL 里常用：

```python
thr_copy.partition_S(src)
thr_copy.partition_D(dst)
thr_copy.retile(tensor)
```

C++ 里名字更细：

```cpp
partition_S
partition_D
retile_S
retile_D
```

含义是：

- `partition_S`：按 source layout 切 source tensor
- `partition_D`：按 destination layout 切 destination tensor
- `retile_S`：把已有 tensor 重新解释成 copy source view
- `retile_D`：把已有 tensor 重新解释成 copy destination view

DSL 的 `retile` 会根据 TiledCopy trait 调用底层 IR 做相同类型的重解释。

## `ldmatrix` 作为 CopyAtom

`LdMatrix8x8x16bOp(False, 2)` 出现在：

```python
s2R_copy_A = cute.make_copy_atom(
    cute.nvgpu.warp.LdMatrix8x8x16bOp(False, 2),
    mA.element_type,
)
```

参数含义：

```text
False -> 不转置，对应 PTX 没有 .trans
2     -> 一条 ldmatrix.x2，读取两个 8x8 矩阵
```

`LdMatrix8x8x16bOp(False, 1)` 对应 `ldmatrix.x1`。

DSL 源码 `warp/copy.py` 会把它构造成 `CopyAtomLdsmType`：

```python
CopyAtomLdsmType.get(
    dtype,
    mode=(8, 8),
    sz_pattern=u16,
    num_matrices,
    transpose_attr_or_none,
)
```

C++ 里最终对应到 `arch/copy_sm75.hpp`：

```cpp
ldmatrix.sync.aligned.x1.m8n8.shared.b16
ldmatrix.sync.aligned.x2.m8n8.shared.b16
ldmatrix.sync.aligned.x4.m8n8.shared.b16
```

也就是说，`ldmatrix` 在 CuTe 里不是魔法函数，而是一个带特殊 traits 的 copy atom。

## `ldsm_tensorop.py` 的两段 copy

### GMEM -> SMEM

第一段是教学用的简单 copy：

```text
gA/gB -> local_partition -> tAgA/tBgB
sA/sB -> local_partition -> tAsA/tBsB
copy universal atom
```

代码：

```python
tAgA = cute.local_partition(gA, g2s_thr_layout_A, tidx)
tAsA = cute.local_partition(sA, g2s_thr_layout_A, tidx)
tBgB = cute.local_partition(gB, g2s_thr_layout_B, tidx)
tBsB = cute.local_partition(sB, g2s_thr_layout_B, tidx)

cute.copy(g2S_copy_A, tAgA, tAsA)
cute.copy(g2S_copy_B, tBgB, tBsB)
```

这里我们自己决定 32 个线程怎么分工。

### SMEM -> RMEM

第二段必须满足 Tensor Core 的 fragment 需求：

```text
sA/sB -> partition_S -> ldmatrix source view
tCrA/tCrB -> retile -> ldmatrix destination view
ldmatrix copy atom
```

代码：

```python
tCsA_copy_view = thr_s2r_A.partition_S(sA)
tCrA_copy_view = thr_s2r_A.retile(tCrA)
tCsB_copy_view = thr_s2r_B.partition_S(sB)
tCrB_copy_view = thr_s2r_B.retile(tCrB)

cute.copy(tiled_s2r_A, tCsA_copy_view, tCrA_copy_view)
cute.copy(tiled_s2r_B, tCsB_copy_view, tCrB_copy_view)
```

这里我们不再手写每个 lane 读哪些 SMEM 地址。这个映射来自：

```text
ldmatrix CopyAtom trait + MMA A/B TV layout
```

## 初学者应该抓住什么

读 copy 代码时先问四个问题：

1. source tensor 在哪里？GMEM、SMEM 还是 RMEM？
2. destination tensor 在哪里？
3. 是普通 per-thread copy，还是 warp-level matrix copy？
4. source/destination view 是否已经符合 copy atom 的 `(V, Rest...)` 形状？

如果是普通 copy，通常 `local_partition + cute.copy` 就够了。

如果是 `ldmatrix`，必须额外检查：

```text
make_tiled_copy_A/B 是否绑定到了正确的 tiled_mma
partition_S 是否用在 SMEM source 上
retile 是否用在 MMA register fragment 上
```

## 和后续优化的关系

当前 `ldsm_tensorop.py` 还没有做：

- vectorized GMEM->SMEM copy
- `cp.async`
- 多 stage pipeline
- swizzle shared memory layout

但这些优化仍然建立在同一个抽象上：

```text
copy op/atom 描述指令
layout 描述线程和值怎么对应到 tensor 坐标
tiled copy 把单条指令铺到更大的 tile 上
```

后面做 cp.async 或 swizzle 时，真正变化的是 copy atom 和 SMEM layout，整体框架不会变。
