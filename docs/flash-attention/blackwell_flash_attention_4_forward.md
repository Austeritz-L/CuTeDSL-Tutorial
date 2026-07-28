# Blackwell FlashAttention-4 Forward 源码分析：Layout、TMEM 与异步流水

> 本文是第一版源码阅读笔记，重点分析 Blackwell / SM100 上的标准
> FlashAttention-4 forward 主路径。本文暂时不展开 Python 顶层 API、全部变体、
> backward 和 head-dim 256 专用 kernel，而是把注意力放在 device kernel 本身：
> **Layout 怎样服务于 tcgen05 MMA、数据怎样经过 GMEM/SMEM/TMEM/RMEM，以及各类
> warp 如何组成异步流水。**

## 0. 源码版本与阅读范围

本文基于以下源码版本：

```text
Dao-AILab/flash-attention
commit: 14c377950125c70b7a9dabf9c561fca53715ac7d

NVIDIA/cutlass
commit: 7ac18190ca2d876117abd1c06bc788e87f8c8734
```

主要阅读文件：

- [`flash_attn/cute/flash_fwd_sm100.py`](https://github.com/Dao-AILab/flash-attention/blob/14c377950125c70b7a9dabf9c561fca53715ac7d/flash_attn/cute/flash_fwd_sm100.py)
  - SM100 forward 的 tile、Layout、warp 分工与主流水。
- [`flash_attn/cute/softmax.py`](https://github.com/Dao-AILab/flash-attention/blob/14c377950125c70b7a9dabf9c561fca53715ac7d/flash_attn/cute/softmax.py)
  - Blackwell online softmax、条件 rescale 和 `exp2` 实现。
- [`flash_attn/cute/blackwell_helpers.py`](https://github.com/Dao-AILab/flash-attention/blob/14c377950125c70b7a9dabf9c561fca53715ac7d/flash_attn/cute/blackwell_helpers.py)
  - 手工拼装的 tcgen05 PTX、SMEM descriptor 和 instruction descriptor。
- [`flash_attn/cute/mma_sm100_desc.py`](https://github.com/Dao-AILab/flash-attention/blob/14c377950125c70b7a9dabf9c561fca53715ac7d/flash_attn/cute/mma_sm100_desc.py)
  - UMMA instruction descriptor 与 SMEM descriptor 编码。
- [`flash_attn/cute/pipeline.py`](https://github.com/Dao-AILab/flash-attention/blob/14c377950125c70b7a9dabf9c561fca53715ac7d/flash_attn/cute/pipeline.py)
  - 对 CUTLASS pipeline 的 index/phase 封装。
- [`cutlass/utils/blackwell_helpers.py`](https://github.com/NVIDIA/cutlass/blob/7ac18190ca2d876117abd1c06bc788e87f8c8734/python/CuTeDSL/cutlass/utils/blackwell_helpers.py)
  - `make_trivial_tiled_mma`、SMEM Layout atom 选择和 epilogue copy。
- [`cutlass/pipeline/sm100.py`](https://github.com/NVIDIA/cutlass/blob/7ac18190ca2d876117abd1c06bc788e87f8c8734/python/CuTeDSL/cutlass/pipeline/sm100.py)
  - `PipelineTmaUmma`、`PipelineUmmaAsync` 等 Blackwell pipeline。
- [`cutlass/cute/nvgpu/tcgen05`](https://github.com/NVIDIA/cutlass/tree/7ac18190ca2d876117abd1c06bc788e87f8c8734/python/CuTeDSL/cutlass/cute/nvgpu/tcgen05)
  - tcgen05 MMA、TMEM load/store 与 SMEM-to-TMEM copy 的 CuTe DSL atom。

FA4 源码仍在快速演进。本文所有形状、角色与同步关系都以以上 commit 为准。

## 1. 先建立一个具体 case

为避免只讨论抽象符号，本文使用下面这个主 case：

```text
GPU                = B200 / SM100
Q/K/V/O dtype      = BF16
head_dim           = 128
head_dim_v         = 128
m_block_size       = 128
n_block_size       = 128
q_stage            = 2
KV load            = TMA
O store            = TMA
attention          = dense non-causal
```

对满足条件的长序列 dense non-causal case，接口会自动启用 2-CTA 指令。
自动选择条件可以直接看
[`interface.py` 的 `use_2cta_instrs`](https://github.com/Dao-AILab/flash-attention/blob/14c377950125c70b7a9dabf9c561fca53715ac7d/flash_attn/cute/interface.py#L589-L607)：

```python
use_2cta_instrs = (
    arch // 10 in [10, 11]
    and not causal
    and not local
    and not is_split_kv
    ...
    and head_dim_padded in [128, 192]
    and head_dim_v_padded == 128
    and seqlen_q_packgqa > 2 * tile_m
)
```

为了把 Layout 讲清楚，本文会同时写出 1-CTA 与 2-CTA 的差别。它们的算法相同，
真正变化的是 MMA 的 M 维覆盖范围、cluster Layout 和跨 CTA 的 barrier 语义。

## 2. FA4 与 Hopper FA3 最本质的差别

Hopper forward 可以粗略理解为：

```text
TMA:
    GMEM -> SMEM(Q/K/V)

WGMMA:
    SMEM(Q/K) -> accumulator/register-like result

consumer warpgroup:
    softmax + P @ V
```

Blackwell FA4 把中间状态的中心搬到了 **TMEM**：

```text
TMA:
    GMEM -> SMEM(Q/K/V)

tcgen05.mma:
    SMEM(Q) x SMEM(K) -> TMEM(S)

softmax warps:
    TMEM(S) -> RMEM
    mask / row-max / exp2 / row-sum
    RMEM(P) -> TMEM(P)

tcgen05.mma:
    TMEM(P) x SMEM(V) -> TMEM(O)

correction warps:
    TMEM(O) -> RMEM
    conditional rescale
    RMEM -> TMEM(O)

epilogue:
    TMEM(O) -> RMEM -> SMEM(O) -> GMEM
```

这不是简单地把 `wgmma` 改成 `tcgen05.mma`。TMEM 让 MMA accumulator 不再长期占据
普通寄存器，FA4 因而可以把工作拆给多组高度专用的 warp：

- 一个 warp 只负责发射异步 MMA；
- 两个 warpgroup 分别处理两个 Q stage 的 softmax；
- 一个 warpgroup 专门修正旧的 O accumulator；
- load 和 epilogue 各有自己的 warp；
- 不同角色通过动态寄存器分配获取不同 register budget。

FA4 的流水线设计，本质是利用 Blackwell 的不对称资源：

```text
Tensor Core 很快
MUFU exp2、标量计算和数据搬运相对成为瓶颈

=> 不让一个 warpgroup 串行承担全部工作
=> 把 MMA、softmax、correction、load、store 拆开并重叠
```

## 3. Tile 的三层含义

源码在
[`FlashAttentionForwardSm100.__init__`](https://github.com/Dao-AILab/flash-attention/blob/14c377950125c70b7a9dabf9c561fca53715ac7d/flash_attn/cute/flash_fwd_sm100.py#L172-L181)
中同时定义了三种 tile：

```python
self.cta_group_size = 2 if self.use_2cta_instrs else 1

self.cta_tiler = (
    self.q_stage * m_block_size,
    n_block_size,
    self.head_dim_padded,
)

self.mma_tiler_qk = (
    self.cta_group_size * m_block_size,
    n_block_size,
    self.head_dim_padded,
)

self.mma_tiler_pv = (
    self.cta_group_size * m_block_size,
    self.head_dim_v_padded,
    n_block_size,
)
```

三者不能混为一谈。

### 3.1 `cta_tiler`

```text
cta_tiler = (q_stage * tile_m, tile_n, head_dim)
```

在主 case 中：

```text
cta_tiler = (256, 128, 128)
```

它描述一个 scheduler work tile 中包含两个 Q stage：

```text
stage 0: 128 rows
stage 1: 128 rows
```

`BlockInfo` 和 tile scheduler 使用的是这个粒度。

### 3.2 `mma_tiler_qk`

```text
QK:
    A = Q : (M, K)
    B = K : (N, K)
    C = S : (M, N)
```

1-CTA：

```text
mma_tiler_qk = (128, 128, 128)
```

2-CTA：

```text
mma_tiler_qk = (256, 128, 128)
```

2-CTA 指令跨两个 CTA 覆盖 M=256；每个 CTA 仍然拥有其中 128 行。

### 3.3 `mma_tiler_pv`

```text
PV:
    A = P : (M, K=N_block)
    B = V : (N=head_dim_v, K=N_block)
    C = O : (M, head_dim_v)
```

2-CTA 主 case：

```text
mma_tiler_pv = (256, 128, 128)
```

注意，PV 的 K 维是 attention 的 N tile。也就是说，QK 的逻辑 N 在第二次 GEMM
中变成 K：

```text
QK: (M,D) x (N,D) -> (M,N)
PV: (M,N) x (Dv,N) -> (M,Dv)
```

## 4. GMEM Layout：`select` 只是换坐标系

kernel 入口接收到的普通 dense tensor 是：

```text
Q: (batch, seqlen_q, head, dim)
K: (batch, seqlen_k, head_kv, dim)
V: (batch, seqlen_k, head_kv, dim_v)
O: (batch, seqlen_q, head, dim_v)
```

源码没有先执行一次真实 transpose，而是用 `cute.select` 重排 tensor Layout：

```python
Q_layout_transpose = [1, 3, 2, 0]
mQ = cute.make_tensor(
    mQ.iterator,
    cute.select(mQ.layout, mode=Q_layout_transpose),
)

KV_layout_transpose = [1, 3, 2, 0]
mK, mV = [
    cute.make_tensor(t.iterator, cute.select(t.layout, mode=KV_layout_transpose))
    for t in (mK, mV)
]

V_layout_transpose = [1, 0, 2, 3]
mV = cute.make_tensor(
    mV.iterator,
    cute.select(mV.layout, mode=V_layout_transpose),
)
```

对应源码：
[`flash_fwd_sm100.py#L425-L450`](https://github.com/Dao-AILab/flash-attention/blob/14c377950125c70b7a9dabf9c561fca53715ac7d/flash_attn/cute/flash_fwd_sm100.py#L425-L450)。

最终 kernel 使用的逻辑 view 是：

```text
Q: (seqlen_q, dim, head, batch)
K: (seqlen_k, dim, head_kv, batch)
V: (dim_v, seqlen_k, head_kv, batch)
O: (seqlen_q, dim_v, head, batch)
```

这里最重要的是：

```text
select(layout) != 搬数据
select(layout) == 用另一组 mode 顺序解释同一个 pointer
```

V 单独再交换前两个 mode，是为了让它符合 PV 的 B operand 逻辑：

```text
V as B = (N=head_dim_v, K=seqlen_k)
```

所以不能只看 PyTorch tensor 的表面 shape 来判断 MMA 是不是转置。真正决定
operand 坐标映射的是进入 `partition_A/B` 前的 CuTe tensor Layout。

## 5. TiledMMA Layout：从数学矩阵到 tcgen05 atom

MMA 在
[`flash_fwd_sm100.py#L485-L509`](https://github.com/Dao-AILab/flash-attention/blob/14c377950125c70b7a9dabf9c561fca53715ac7d/flash_attn/cute/flash_fwd_sm100.py#L485-L509)
中构造：

```python
cta_group = (
    tcgen05.CtaGroup.TWO
    if self.use_2cta_instrs
    else tcgen05.CtaGroup.ONE
)

q_major_mode = tcgen05.OperandMajorMode.K
k_major_mode = tcgen05.OperandMajorMode.K
v_major_mode = tcgen05.OperandMajorMode.MN

tiled_mma_qk = make_trivial_tiled_mma(
    q_dtype,
    q_major_mode,
    k_major_mode,
    Float32,
    cta_group,
    mma_tiler_qk[:2],
)

tiled_mma_pv = make_trivial_tiled_mma(
    v_dtype,
    OperandMajorMode.K,
    v_major_mode,
    Float32,
    cta_group,
    mma_tiler_pv[:2],
    OperandSource.TMEM,
)
```

### 5.1 QK：SS MMA

QK 的 A、B 都来自 shared memory：

```text
A source = SMEM
B source = SMEM
Q major  = K-major
K major  = K-major
C/S      = TMEM
```

可以把它称为：

```text
SS -> T
```

即 shared × shared，accumulator 写入 tensor memory。

### 5.2 PV：TS MMA

PV 的 P 来自 TMEM，V 来自 SMEM：

```text
A/P source = TMEM
B/V source = SMEM
P major    = K-major
V major    = MN-major
C/O        = TMEM
```

可以把它称为：

```text
TS -> T
```

这正是 Blackwell FA4 的关键数据路径：softmax 不需要把 P 写回 shared memory，
而是直接把 P 写入 TMEM，下一次 `tcgen05.mma` 从 TMEM 读取。

### 5.3 `make_trivial_tiled_mma` 实际做了什么

CUTLASS helper 对 BF16/FP16 构造：

```python
MmaF16BF16Op(
    dtype,
    acc_dtype,
    (*mma_tiler_mn, 16),
    cta_group,
    a_source,
    a_leading_mode,
    b_leading_mode,
)
```

也就是：

```text
instruction K = 16
instruction M/N = 当前 tiled MMA 的 M/N
```

约束来自
[`MmaF16BF16Op`](https://github.com/NVIDIA/cutlass/blob/7ac18190ca2d876117abd1c06bc788e87f8c8734/python/CuTeDSL/cutlass/cute/nvgpu/tcgen05/mma.py#L876-L1015)：

```text
CTA group 1:
    M in {64, 128}

CTA group 2:
    M in {128, 256}

K = 16
```

与 Ampere/Hopper 中“一个 warp/warpgroup 共同持有全部 accumulator fragment”的直觉不同，
tcgen05 MMA 的 C fragment 指向 TMEM。发射指令只需要一个 warp 中的 leader thread，
真正的矩阵计算在后台异步进行。

## 6. SMEM Layout：ComposedLayout、swizzle 与 stage

源码为 Q/K/V/O 构造的 Layout：

```python
sQ_layout = make_smem_layout_a(
    tiled_mma_qk, mma_tiler_qk, q_dtype, q_stage
)
sK_layout = make_smem_layout_b(
    tiled_mma_qk, mma_tiler_qk, k_dtype, kv_stage
)
sV_layout = make_smem_layout_b(
    tiled_mma_pv, mma_tiler_pv, v_dtype, kv_stage
)
sO_layout = make_smem_layout_epi(
    o_dtype, o_layout, epi_tile, q_stage
)
```

源码位置：
[`flash_fwd_sm100.py#L519-L533`](https://github.com/Dao-AILab/flash-attention/blob/14c377950125c70b7a9dabf9c561fca53715ac7d/flash_attn/cute/flash_fwd_sm100.py#L519-L533)。

### 6.1 为什么返回 `ComposedLayout`

典型的 SMEM Layout 可以拆成：

```text
ComposedLayout(
    inner = swizzle,
    offset = 0,
    outer = hierarchical base layout,
)
```

它表达的地址函数近似是：

```text
logical coordinate
    -> outer layout 得到基础 offset
    -> inner swizzle 对 offset 的部分 bit 做变换
    -> physical SMEM address
```

因此：

- `outer.shape` 告诉我们逻辑 tile 与 stage 层级；
- `outer.stride` 告诉我们未 swizzle 前的基础步长；
- `inner` 不是新的数据维度，而是 bank-conflict-aware 的地址变换。

kernel 最后这样建立 tensor：

```python
sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
```

对应
[`flash_fwd_sm100.py#L1031-L1042`](https://github.com/Dao-AILab/flash-attention/blob/14c377950125c70b7a9dabf9c561fca53715ac7d/flash_attn/cute/flash_fwd_sm100.py#L1031-L1042)。

### 6.2 Layout atom 如何选择

CUTLASS 的 `get_smem_layout_atom_ab` 先查看：

```text
operand 是 K-major 还是 MN-major
major mode 有多少元素
每个元素多少 bit
major mode 的总 bit 数能否被 1024/512/256 整除
```

然后选择：

```text
K_SW128 / K_SW64 / K_SW32 / K_INTER
MN_SW128 / MN_SW64 / MN_SW32 / MN_INTER
```

主 case 中，BF16 的 128 元素 major mode 为：

```text
128 * 16 bit = 2048 bit
```

可以使用 128B swizzle family。Q/K 是 K-major，V 是 MN-major，因此虽然逻辑 tile
都是 128×128，它们使用的 Layout atom 类型并不相同。

### 6.3 `partition_shape_A/B` 比手写 shape 更重要

helper 不是直接执行：

```python
make_layout((128, 128, stages))
```

而是先调用：

```python
tiled_mma.partition_shape_A(...)
tiled_mma.partition_shape_B(...)
```

得到与 MMA atom 相容的层级 shape，再把 stage append 到最后，最后用
`tile_to_mma_shape` 铺开 Layout atom。

所以 `sQ.shape` 的打印结果可能不是平坦的 `(128,128,2)`，而是类似：

```text
(MMA, MMA_M_or_N, MMA_K, PIPE)
```

这种层级结构不是多余复杂度。它保存了以下对应关系：

```text
MMA atom 内部坐标
    × atom 在 tile 中的重复坐标
    × K 方向重复
    × pipeline stage
```

后续：

```python
tiled_mma.make_fragment_A(sQ)
tiled_mma.make_fragment_B(sK)
```

才能直接得到 tcgen05 instruction descriptor 所需的 operand view。

### 6.4 stage 一定是 Layout 的一部分

Q 有 `q_stage=1/2`，K/V 的 `kv_stage` 根据 shared memory budget 动态计算。

主 case 中每个 K 或 V stage 大约占：

```text
128 * 128 * 2 bytes = 32 KiB
```

Q 的两个 stage 占：

```text
2 * 128 * 128 * 2 bytes = 64 KiB
```

源码以约 224 KiB 可用预算计算 KV stage 数：

```python
kv_stage = min(
    (224 * 1024 - smem_size_q_o) // smem_size_kv_per_stage,
    32,
)
```

见
[`flash_fwd_sm100.py#L341-L381`](https://github.com/Dao-AILab/flash-attention/blob/14c377950125c70b7a9dabf9c561fca53715ac7d/flash_attn/cute/flash_fwd_sm100.py#L341-L381)。

循环中的 `PipelineState(index, phase)` 与 Layout 最后一维一一对应：

```text
index = 当前使用哪个 stage
phase = 同一块 stage buffer 已经循环复用了几轮
```

### 6.5 K/V 为什么可以共用同一块 SMEM

SharedStorage 只真实分配了 `sK`，`sV` 是在同一个 pointer 上创建的新 view：

```python
sV = cute.make_tensor(
    cute.recast_ptr(sK.iterator, sV_layout.inner),
    sV_layout.outer,
)
```

这是安全的，因为主循环按：

```text
load K_i -> QK_i 消费 K_i -> 释放 stage
load V_i -> PV_i 消费 V_i -> 释放 stage
```

交替复用同一块 storage。Layout 使同一物理地址在 K 阶段解释为 K-major，
在 V 阶段解释为 MN-major。

这也是一个很好的 CuTe 例子：

```text
相同 pointer + 不同 Layout = 不同 tensor view
```

## 7. TMEM Layout：S、P、O 怎样塞进 512 columns

FA4 为 TMEM 申请最大 512 columns。主 case 的 offset 在初始化时静态计算：

```python
self.tmem_s_offset = [0, 128]
self.tmem_o_offset = [256, 384]
self.tmem_s_to_p_offset = 64
self.tmem_p_offset = [64, 192]
```

来源：
[`flash_fwd_sm100.py#L303-L316`](https://github.com/Dao-AILab/flash-attention/blob/14c377950125c70b7a9dabf9c561fca53715ac7d/flash_attn/cute/flash_fwd_sm100.py#L303-L316)。

可以画成：

```text
TMEM column

0                 128               256               384              512
|-----------------|-----------------|-----------------|-----------------|
| S0              | S1              | O0              | O1              |
| FP32 128 cols   | FP32 128 cols   | FP32 128 cols   | FP32 128 cols   |
|        P0 starts|        P1 starts|                 |                 |
|        at +64   |        at +64   |                 |                 |
```

### 7.1 `tStS`：假的 iterator，真的 Layout

源码先构造 C fragment：

```python
qk_acc_shape = thr_mma_qk.partition_shape_C(mma_tiler_qk[:2])
tStS = thr_mma_qk.make_fragment_C(
    cute.append(qk_acc_shape, s_stage)
)
```

注释明确说这是一个 “fake tensor”：此时尚未从 allocator 取出真实 TMEM pointer，
但代码已知会申请完整 512 columns，因此可以先用从 0 开始的 Layout 描述 S。

这里体现出：

```text
Tensor = iterator + Layout

iterator 可以稍后由 TMEM allocator 约定
Layout 可以先把逻辑坐标、stage 与 column offset 全部确定
```

### 7.2 P 为什么从 S 的 `+64 columns` 开始

S accumulator 是 FP32，P 在 BF16 主 case 中是 16-bit。相同数量的 P 元素只需要
S 一半的 TMEM column 宽度。

softmax warps 已经把当前 S fragment load 到寄存器后，可以把转换后的 P 写回 S
buffer 的后半段：

```python
tStP = cute.make_tensor(
    tSAcc.iterator + self.tmem_s_to_p_offset,
    tStP_layout,
)
```

这是一次 in-place lifetime reuse：

```text
S 的生命周期:
    QK 完成 -> softmax load S 到 RMEM -> S 的相应区域可复用

P 的生命周期:
    softmax 在 RMEM 算出 P -> 写入复用后的 TMEM -> PV MMA 读取
```

它减少了 TMEM column 占用，使两个 S stage 和两个 O stage 能同时放入 512 columns。

### 7.3 TMEM copy 也由 TiledCopy 描述

softmax 使用：

```python
tmem_load_op = tcgen05.copy.Ld32x32bOp(Repetition(32))
tmem_load_atom = cute.make_copy_atom(tmem_load_op, Float32)
thr_tmem_load = tcgen05.make_tmem_copy(
    tmem_load_atom, tSAcc
).get_slice(tidx)
tStS_t2r = thr_tmem_load.partition_S(tSAcc)
```

写 P 使用：

```python
St32x32bOp(Repetition(16))
```

对应源码：
[`flash_fwd_sm100.py#L1923-L1959`](https://github.com/Dao-AILab/flash-attention/blob/14c377950125c70b7a9dabf9c561fca53715ac7d/flash_attn/cute/flash_fwd_sm100.py#L1923-L1959)。

和 `ldmatrix` 类似，`make_tmem_copy` 的核心仍然是 TV Layout：

```text
(thread, value) -> TMEM logical coordinate
```

区别在于这里的 instruction atom 是 `tcgen05.ld/st`，而不是 SMEM `ldmatrix`。

## 8. CTA cluster 与 2-CTA MMA

2-CTA 模式设置：

```python
self.cluster_shape_mn = (2, 1)
self.cluster_shape_mnk = (2, 1, 1)
```

随后：

```python
cta_layout_vmnk = cute.tiled_divide(
    cute.make_layout(cluster_shape_mnk),
    (tiled_mma_qk.thr_id.shape,),
)
```

这里多出来的 V mode 可以理解为 tcgen05 MMA 内部的 CTA ownership mode。

kernel 通过：

```python
mma_tile_coord_v = bidx % size(tiled_mma_qk.thr_id.shape)
is_leader_cta = mma_tile_coord_v == 0
```

区分 CTA pair 中的 leader。只有 leader CTA 的 MMA warp 真正进入 `mma()` 主发射逻辑：

```python
if process_tile and is_leader_cta:
    ...
    gemm_Si(...)
    gemm_Pi(...)
```

但两个 CTA 都有 softmax 与 correction warps，分别处理属于自己的 128 行。

这意味着 2-CTA 不是：

```text
两个 CTA 各自做一遍 1-CTA kernel
```

而是：

```text
一个 tcgen05.cta_group::2 MMA 横跨两个 CTA
leader 发射
两个 CTA 各自消费属于自己的 TMEM datapath rows
barrier arrival count 和 mask 覆盖整个 CTA pair
```

`PipelineTmaUmma`、`PipelineUmmaAsync` 会根据 `cta_layout_vmnk` 自动选择：

```text
CtaGroup.ONE / CtaGroup.TWO
leader CTA
peer CTA rank
TMEM completion mask
跨 CTA consumer arrival count
```

因此 2-CTA 能否正确工作，不只是 MMA shape 的问题；所有 UMMA-bridging pipeline
都必须使用 cluster 级参与者数量。源码对此有明确注释：

```python
softmax_correction_threads_cluster = ThreadCooperativeGroup(
    32 * len(softmax_warps + correction_warps) * cta_group_size
)
```

见
[`flash_fwd_sm100.py#L902-L927`](https://github.com/Dao-AILab/flash-attention/blob/14c377950125c70b7a9dabf9c561fca53715ac7d/flash_attn/cute/flash_fwd_sm100.py#L902-L927)。

## 9. 16 个 warp 的角色分工

标准 `q_stage=2` 配置有 16 个 warp，也就是 512 threads：

| Warp | 角色 | 主要工作 |
|---:|---|---|
| 0–3 | Softmax stage 0 | S0: TMEM→RMEM，mask、max、exp2、sum，P0: RMEM→TMEM |
| 4–7 | Softmax stage 1 | 对 S1/P1 做同样工作 |
| 8–11 | Correction | 读取 TMEM O，根据新的 row max 缩放旧 O，最后完成 normalize/epilogue |
| 12 | MMA | 分配 TMEM，发射 QK 与 PV tcgen05 MMA |
| 13 | Epilogue | SMEM O → GMEM，通常用 TMA |
| 14 | Load | TMA load Q/K/V |
| 15 | Empty / CLC | 空闲或承担动态 CLC scheduler |

定义在：
[`flash_fwd_sm100.py#L263-L301`](https://github.com/Dao-AILab/flash-attention/blob/14c377950125c70b7a9dabf9c561fca53715ac7d/flash_attn/cute/flash_fwd_sm100.py#L263-L301)。

当 `q_stage=1`、non-TMA paged KV 或 non-TMA O 等条件改变时，部分 warp 会被重新分配，
所以不要把上表当成所有 specialization 的固定 ABI。

### 9.1 动态寄存器重分配

各角色进入主函数前调用：

```python
cute.arch.setmaxregister_decrease(num_regs_other)
cute.arch.setmaxregister_increase(num_regs_softmax)
cute.arch.setmaxregister_decrease(num_regs_correction)
```

softmax warps通常获得最多寄存器，因为 S fragment、P conversion 和 reduction 都在 RMEM
中进行；MMA/load warp 只需少量控制状态。主 case 的寄存器数还会根据 1/2-CTA、
causal、head dim、SM100/SM103 和 FP8 单独调优。

这与 warp specialization 配套：

```text
角色拆分只解决并行性
setmaxnreg 进一步把 CTA 的 register file 预算从轻角色转给重角色
```

## 10. 六条关键 pipeline

kernel 并不是只维护一条 Q/K/V double-buffer pipeline，而是维护多条不同方向的
handshake：

| Pipeline | Producer | Consumer | 表达的资源状态 |
|---|---|---|---|
| `pipeline_q` | TMA load warp | MMA warp | Q stage 已写入 SMEM / 已被 UMMA 消费 |
| `pipeline_kv` | TMA load warp | MMA warp | 当前 K 或 V stage 已写入 / 已消费 |
| `pipeline_s_p_o` | MMA warp | softmax + correction | S 已完成；反向信号表示 P 已写入且旧 O 已完成 rescale |
| `pipeline_p_lastsplit` | softmax warps | MMA warp | P 的最后一段已完成，可继续 PV |
| `pipeline_o_acc` | MMA warp | correction warps | 最后一轮 PV 的 O accumulator 已完成 |
| `pipeline_sm_stats` | softmax warps | correction warps | row scale / row max / row sum 可读取 |
| `pipeline_o_epi` | correction warps | epilogue warp | normalize 后的 sO 可写回 |

创建代码：
[`flash_fwd_sm100.py#L928-L1026`](https://github.com/Dao-AILab/flash-attention/blob/14c377950125c70b7a9dabf9c561fca53715ac7d/flash_attn/cute/flash_fwd_sm100.py#L928-L1026)。

### 10.1 `PipelineTmaUmma`

`pipeline_q` 和 `pipeline_kv` 的典型类型是：

```text
producer op = TMA load
consumer op = TCGen05 MMA
```

full barrier 追踪 TMA transaction bytes：

```text
producer acquire:
    等待 empty
    arrive_and_expect_tx(copy_bytes)

TMA instruction:
    数据传输完成时递减 tx count

MMA consumer:
    等 full phase
    发射 tcgen05
    通过 UMMA completion 语义 release empty
```

所以 `producer_commit` 是 no-op：真正完成 full barrier 的不是 load warp 再执行一条
commit，而是 TMA engine 完成相应字节数。

### 10.2 `PipelineUmmaAsync`

`pipeline_s_p_o` 的 producer 是 UMMA，consumer 是普通线程。

MMA warp 调用：

```python
pipeline_s_p_o.producer_commit_w_index(stage)
```

底层把 `tcgen05.commit`/TMEM completion 与 mbarrier 关联。softmax warp 等待该 barrier
后，才能安全读取 TMEM S。

反方向的 empty signal 被重新解释为：

```text
softmax: P 已写好
correction: 旧 O 已按新 max 缩放好

二者都 arrive 后:
    MMA warp 才能用 P @ V 更新同一 stage 的 O
```

这条 pipeline 同时保护了两个有写后读冲突的 TMEM 区域。

## 11. Load 流水：为什么是 K、Q、Q、V、K、V

load warp 的主要顺序可以从
[`load()`](https://github.com/Dao-AILab/flash-attention/blob/14c377950125c70b7a9dabf9c561fca53715ac7d/flash_attn/cute/flash_fwd_sm100.py#L1333-L1555)
直接读出。

对第一个 N block：

```text
load K0
load Q stage 0
load Q stage 1
load V0
```

后续：

```text
load K1
load V1
load K2
load V2
...
```

Q 只在一个 work tile 开头加载一次，之后被所有 K blocks 重用。K/V 则按 reverse N-block
顺序流过共享的 circular buffer。

源码使用：

```python
kv_producer_state = make_pipeline_state(Producer, kv_stage)
...
load_K(... producer_state=kv_producer_state)
kv_producer_state.advance()
load_V(... producer_state=kv_producer_state)
kv_producer_state.advance()
```

注意 K 和 V 共用一条 state 序列以及同一片 SMEM，因此一个 stage 的时间轴可能是：

```text
phase p:     K_i
phase p+1:   V_i
phase p+2:   K_(i+1)
```

不能把 `kv_stage` 理解成“有 kv_stage 份 K 和另外 kv_stage 份 V”。

## 12. MMA 主循环：PV 与下一轮 QK 交错

MMA warp 先为两个 Q stage 计算第一块 score：

```text
Q0 × K0 -> S0
Q1 × K0 -> S1
```

接下来每个 K/V block 的核心顺序是：

```text
等待 V_i

stage 0:
    等 P0_i 写好、O0 旧值 rescale 完成
    P0_i × V_i -> O0
    Q0 × K_(i+1) -> S0_(i+1)

stage 1:
    等 P1_i 写好、O1 旧值 rescale 完成
    P1_i × V_i -> O1
    Q1 × K_(i+1) -> S1_(i+1)
```

对应源码：
[`flash_fwd_sm100.py#L1698-L1800`](https://github.com/Dao-AILab/flash-attention/blob/14c377950125c70b7a9dabf9c561fca53715ac7d/flash_attn/cute/flash_fwd_sm100.py#L1698-L1800)。

这里存在三个层次的 overlap：

1. **Tensor Core 内部异步**：发射 tcgen05 后，MMA warp 不保存 accumulator。
2. **两个 Q stage**：S0 与 S1 分给不同 softmax warpgroup。
3. **跨 N block**：当前 `P_i @ V_i` 与下一块 `Q @ K_(i+1)` 交错发射。

可以画成一个简化时间图：

```text
time --------------------------------------------------------------->

Load:
    K0 Q0 Q1 V0 | K1 V1 | K2 V2 | K3 V3 ...

MMA:
    Q0K0 Q1K0   | P0V0 Q0K1 P1V0 Q1K1 | P0V1 Q0K2 ...

Softmax-0:
          S0_0 -> P0_0      | S0_1 -> P0_1      | S0_2 ...

Softmax-1:
               S1_0 -> P1_0 |      S1_1 -> P1_1 |      S1_2 ...

Correction:
                       rescale O0/O1 | rescale O0/O1 | ...
```

## 13. Softmax：S 从 TMEM 到 RMEM，再把 P 写回 TMEM

每个 softmax warpgroup只负责一个 `stage`。`softmax_step` 的顺序是：

```text
1. 等待 QK 的 S 完成
2. TMEM(S) -> RMEM
3. 应用 score_mod / mask
4. 计算或更新 row max
5. 生成旧 O 的 rescale factor
6. score * scale - row_max * scale
7. exp2，并转成 BF16/FP16/FP8 P
8. RMEM(P) -> TMEM(P)
9. 更新 row sum
```

对应：
[`flash_fwd_sm100.py#L2255-L2404`](https://github.com/Dao-AILab/flash-attention/blob/14c377950125c70b7a9dabf9c561fca53715ac7d/flash_attn/cute/flash_fwd_sm100.py#L2255-L2404)。

### 13.1 每个线程为什么只维护一个 row max / row sum

`SoftmaxSm100.create()` 中：

```python
num_rows = 1
row_max = make_rmem_tensor(1, Float32)
row_sum = make_rmem_tensor(1, Float32)
```

这说明 TMEM copy 的 TV Layout 已经把一个 score row 的不同列分给一个 4-thread reduction
group。每个 lane/线程逻辑上跟踪一个 row state，再用 width=4 的 reduction 合并该行。

所以 Layout 在这里决定了 reduction topology：

```text
TMEM Copy TV Layout
    -> 每个线程拿到哪些 S 元素
    -> 哪 4 个线程共同覆盖一行
    -> warp reduction width 为什么是 4
```

仅看 `row_max.shape == 1` 会误以为一个线程只处理一个 score；实际上它持有该行的一组
列 fragment，并为这一行保存一个在线状态。

### 13.2 条件 rescale

普通 online softmax 在发现新最大值 `m_new` 时，需要：

```text
alpha = exp2((m_old - m_new) * scale_log2)
O_old *= alpha
l_old *= alpha
```

FA4 的 `SoftmaxSm100.update_row_max` 增加了阈值：

```python
if acc_scale_log2 >= -rescale_threshold:
    row_max_new = row_max_old
    row_max_safe = row_max_old
    acc_scale = 1.0
```

BF16/FP16 forward 的 threshold 当前为 8.0。含义是：

```text
如果新 max 只比旧 max 大一点:
    暂时不更新全局 max
    不 rescale O
    允许 P 的指数值在安全范围内大于 1

只有差距足够大:
    才更新 max
    correction warpgroup rescale O
```

这减少了 TMEM O 的 load → multiply → store 次数。

### 13.3 correction 为什么单独使用一个 warpgroup

softmax warp把 `acc_scale` 写入 `sScale` 后立即继续计算 `exp2` 和 P；correction warpgroup
同时读取 scale，并在必要时执行：

```text
TMEM(O) -> RMEM
O *= acc_scale
RMEM -> TMEM(O)
```

源码：
[`correction_rescale`](https://github.com/Dao-AILab/flash-attention/blob/14c377950125c70b7a9dabf9c561fca53715ac7d/flash_attn/cute/flash_fwd_sm100.py#L2724-L2773)。

于是同一个 N-block 内可以出现：

```text
softmax warps:    计算当前 P 的 exp2
correction warps: 缩放之前的 O
MMA warp:         准备下一次异步 PV/QK
```

这正是 FA4 针对 Blackwell Tensor Core 与标量计算吞吐不对称做的 pipeline co-design。

## 14. 软件 `exp2` 与 SM103 `ld.red`

### 14.1 为什么混用硬件 `exp2` 与软件 emulation

SM100 上 `exp2` 依赖 MUFU。如果所有 score 元素都挤到 MUFU，Tensor Core 再快也会被
softmax 卡住。

`apply_exp2_convert` 把 fragment 每 32 个元素分组，并根据 tuning frequency 混合：

```python
cute.math.exp2(..., fastmath=True)
utils.ex2_emulation_2(x0, x1)
```

源码：
[`softmax.py#L359-L401`](https://github.com/Dao-AILab/flash-attention/blob/14c377950125c70b7a9dabf9c561fca53715ac7d/flash_attn/cute/softmax.py#L359-L401)。

它不是为了让单次指数运算更便宜，而是把一部分工作从拥塞的 MUFU 转移到普通算术
pipeline，提高整体吞吐。

调优 key 同时包含：

```text
1CTA / 2CTA
causal / non-causal
head dim
SM100 / SM103
FP8 / BF16
```

说明 `exp2` 策略不是算法常量，而是硬件资源平衡参数。

### 14.2 SM103 的 `tcgen05.ld.red`

在 SM103/B300 上，未使用 score/mask modifier 且 head dim 合适时，源码选择：

```python
tcgen05.copy.LdRed32x32bOp(...)
```

TMEM controller 在 load S 的同时返回每个 32-wide fragment 的 max，softmax 代码只需合并
这些预计算结果：

```python
row_max, acc_scale =
    softmax.update_row_max_precomputed(hw_row_max, is_first)
```

对应：
[`flash_fwd_sm100.py#L2313-L2348`](https://github.com/Dao-AILab/flash-attention/blob/14c377950125c70b7a9dabf9c561fca53715ac7d/flash_attn/cute/flash_fwd_sm100.py#L2313-L2348)。

这再次体现 Blackwell Ultra 的新思路：数据从 TMEM 出来时顺便做 reduction，减少软件
fmax tree。

## 15. P 的 split-arrive：只写完一部分就启动 PV

初始化时：

```python
self.split_P_arrive = n_block_size // 4 * 3
```

主 case 中为 96 columns。

softmax warp分段把 P 写入 TMEM。当完成前 96 columns 后：

```python
fence_view_async_tmem_store()
pipeline_s_p_o.consumer_release_w_index(stage)
```

MMA warp可以开始 PV 的前半部分；softmax warp继续写剩余 P。全部 P 写完后，再通过
`pipeline_p_lastsplit` 发出第二个信号。

对应：
[`flash_fwd_sm100.py#L2384-L2400`](https://github.com/Dao-AILab/flash-attention/blob/14c377950125c70b7a9dabf9c561fca53715ac7d/flash_attn/cute/flash_fwd_sm100.py#L2384-L2400)。

因此 P 的 producer-consumer 不是：

```text
写完整个 128 columns
    -> PV 开始
```

而是：

```text
写完 96 columns
    -> PV 开始消费前一段
同时写最后 32 columns
    -> 第二次 barrier 放行剩余 PV
```

这是比普通 double buffering 更细粒度的流水。

## 16. Epilogue：TMEM O 如何回到 GMEM

最终 correction warp 对 O 乘以：

```text
1 / row_sum
```

然后按 epilogue subtile 执行：

```text
TMEM(O)
  -> tcgen05.ld
RMEM FP32
  -> packed multiply / dtype convert
SMEM(O)
  -> TMA store 或普通 vector store
GMEM(O)
```

`correction_epilogue` 中再次使用 Layout composition：

```python
tOtO_i = cute.logical_divide(
    tOtO, make_layout((m_block_size, corr_tile_size))
)
tOsO_i = cute.logical_divide(
    tOsO, make_layout((m_block_size, corr_tile_size))
)
```

然后：

```python
tiled_tmem_load = tcgen05.make_tmem_copy(...)
tiled_smem_store = cute.make_tiled_copy_D(
    smem_copy_atom, tiled_tmem_load
)
```

`make_tiled_copy_D` 的意义是让 SMEM store 的 destination Layout 与 TMEM load 得到的
寄存器 fragment 相匹配，避免手工推导 per-thread O mapping。

源码：
[`flash_fwd_sm100.py#L2813-L2851`](https://github.com/Dao-AILab/flash-attention/blob/14c377950125c70b7a9dabf9c561fca53715ac7d/flash_attn/cute/flash_fwd_sm100.py#L2813-L2851)。

## 17. 一次 N-block 迭代的完整依赖图

把前面的 Layout 与 pipeline 合在一起，可以得到：

```text
GMEM K_i
  |
  | TMA + pipeline_kv.full
  v
SMEM K_i (K-major swizzled Layout)
  |
  | tcgen05 SS MMA
  v
TMEM S_i (FP32 C-fragment Layout)
  |
  | pipeline_s_p_o.full
  v
RMEM S_i owned by softmax TV Layout
  |
  +-- mask / score mod
  +-- row max
  +-- conditional acc_scale ----------+
  +-- exp2 + dtype convert             |
  |                                    v
  |                            correction warpgroup
  |                            TMEM O_old -> RMEM
  |                            O_old *= acc_scale
  |                            RMEM -> TMEM O
  v
TMEM P_i (reuses tail of S buffer)
  |                                    |
  +------ pipeline_s_p_o.empty <-------+
  |
  | P first split ready
  v
tcgen05 TS MMA: TMEM P_i x SMEM V_i
  |
  | pipeline_p_lastsplit releases remaining split
  v
TMEM O_new
```

这里最容易忽略的是 `pipeline_s_p_o` 的 empty phase必须同时等：

```text
P ready
O rescale ready
```

因为 PV 同时读取 P，并读写 O accumulator。只等待其中一个都会产生 TMEM hazard。

## 18. Blackwell 新特性在源码中的对应位置

### 18.1 `tcgen05.mma`

作用：

- 异步矩阵乘；
- accumulator 位于 TMEM；
- 支持 SMEM×SMEM 和 TMEM×SMEM；
- 支持 1-CTA / 2-CTA instruction。

源码入口：

```text
make_trivial_tiled_mma
gemm_ptx_precomputed_varname
gemm_ptx_partial
```

### 18.2 Tensor Memory

作用：

- 存放 S、P、O；
- 释放普通 register file 压力；
- 允许 MMA warp与 softmax/correction warp通过显式 TMEM load/store 交接数据；
- 允许 S/P 生命周期重叠复用。

源码入口：

```text
TmemAllocator
make_tmem_copy
Ld32x32bOp / St32x32bOp
tmem_s_offset / tmem_p_offset / tmem_o_offset
```

### 18.3 2-CTA MMA

作用：

- 一条 MMA 覆盖 CTA pair；
- 扩大 M tile；
- 两个 CTA 分担 softmax/correction；
- cluster barrier 和 TMEM mask保证跨 CTA 完成语义。

源码入口：

```text
CtaGroup.TWO
cluster_shape_mn = (2,1)
is_leader_cta
softmax_correction_threads_cluster
```

### 18.4 TMA 与 transaction mbarrier

作用：

- GMEM↔SMEM 大块搬运；
- barrier 用 tx byte count 追踪真正的数据完成，而非仅追踪发射线程到达。

源码入口：

```text
PipelineTmaUmma
make_tiled_tma_atom_A/B
CopyBulkTensorTileG2SOp
```

### 18.5 TMEM load-reduce

作用：

- SM103 上 `TMEM -> RMEM` 的同时做 max reduction；
- 减少 softmax 软件 reduction。

源码入口：

```text
LdRed32x32bOp
update_row_max_precomputed
```

### 18.6 CLC scheduler

作用：

- persistent kernel 中动态领取下一个 work tile；
- 使用专门 warp、CLC response buffer 和异步 fetch pipeline；
- 改善不规则 attention workload 的分配。

源码入口：

```text
ClcDynamicPersistentTileScheduler
PipelineClcFetchAsync
clc_scheduler_warp
```

CLC 不改变单个 attention tile 的数学计算，但会改变 CTA cluster如何持续获得后续 tile。

## 19. 读 Layout 时最常见的几个误区

### 误区一：看到 tensor shape 就以为知道物理布局

错误：

```text
sQ.shape 是 128×128，所以它就是普通 row-major 128×128
```

实际：

```text
shape 只描述坐标域
stride + hierarchical mode + swizzle 才决定地址函数
```

### 误区二：`partition_A/B/C` 在搬数据

它们通常只构造 view：

```text
原 tensor Layout
    -> 按 TiledMMA 的 TV Layout 分区
    -> 当前 MMA slice 的 tensor view
```

真正执行数据运动的是：

```text
TMA
cute.copy
tcgen05.ld/st
tcgen05.mma
```

### 误区三：TMEM fragment 等同于普通 RMEM fragment

`make_fragment_C` 在 SM100 路径中描述的是 TMEM accumulator Layout。它的 iterator
可以是 TMEM column address，而不是每个线程持有的一组普通寄存器。

### 误区四：2-CTA 是两个独立 tile

2-CTA MMA 有一个跨 CTA 的 instruction tile。每个 CTA 拥有自己的 datapath rows，
但发射、TMEM completion 和 barrier 是 cluster-coupled 的。

### 误区五：pipeline 数量等于 buffer 数量

一个物理 TMEM stage可能同时被多条逻辑 pipeline保护：

```text
S ready
P ready
O rescale ready
O final ready
```

pipeline表达的是资源状态与 ownership 转移，不只是“有几个缓冲区”。

## 20. 建议的源码阅读顺序

第一遍只建立结构：

```text
FlashAttentionForwardSm100.__init__
    -> tile / warp role / TMEM offsets

__call__
    -> GMEM view / TiledMMA / SMEM Layout / SharedStorage

kernel
    -> pipeline 创建 / 角色分流
```

第二遍跟一块 dense tile：

```text
load
    -> K0, Q0, Q1, V0

mma
    -> QK0
    -> PV_i 与 QK_(i+1) 交错

softmax_loop / softmax_step
    -> S load / mask / max / exp2 / P store

correction_loop
    -> O rescale / normalize / LSE

correction_epilogue
    -> TMEM -> SMEM -> GMEM
```

第三遍再深入 CuTe/CUTLASS：

```text
make_trivial_tiled_mma
make_smem_layout_a/b
make_tmem_copy
PipelineTmaUmma
PipelineUmmaAsync
mma_sm100_desc
blackwell_helpers 中的内联 PTX
```

建议每看到一个 tensor 名字，都记录五件事：

| 问题 | 示例 |
|---|---|
| memory space 是什么 | GMEM / SMEM / TMEM / RMEM |
| 逻辑坐标是什么 | `(M,K)`、`(N,K)`、`(M,N)` |
| Layout 从哪里来 | 原 tensor、TiledMMA partition、composition |
| 谁写它 | TMA、MMA、softmax、correction |
| 谁读它，靠什么同步 | MMA/warpgroup + 哪一条 pipeline |

## 21. 第一版结论

FA4 forward 的核心不是某一条更快的 MMA 指令，而是围绕 Blackwell TMEM 与异步
tcgen05 重新设计了整个数据生命周期：

```text
Layout:
    把 Q/K/V 映射成 UMMA 合法的 swizzled SMEM operand
    把 S/P/O 映射到有限的 TMEM columns
    把 TMEM fragment映射给 softmax/correction 的线程

Pipeline:
    TMA 与 MMA 重叠
    两个 Q stage 的 softmax 重叠
    PV_i 与 QK_(i+1) 重叠
    exp2(P_i) 与 rescale(O_old) 重叠
    P 的部分写入与 PV 的前半段重叠

Blackwell:
    tcgen05 异步 MMA
    TMEM accumulator
    2-CTA MMA
    TMEM load/store/reduce
    cluster-aware mbarrier
    动态寄存器再分配
```

从 CuTe DSL 的角度看，这份源码最值得学习的不是某个孤立 API，而是：

```text
先用 Layout 精确定义 ownership 和地址映射
再用 Copy/MMA atom 定义硬件操作
最后用 pipeline 把每块存储的生命周期拼成时序
```

下一版可以在本文基础上继续补充：

1. 打印主 case 的实际 `sQ/sK/sV/tStS/tOrP/tOtO` Layout；
2. 逐项展开 `tcgen05` instruction descriptor 的 bit field；
3. 对比 1-CTA 与 2-CTA 生成的 PTX；
4. 单独分析 backward 的 2-CTA 与 dQ/dK/dV pipeline；
5. 分析 head-dim 256 专用 forward 与标准 forward 的结构差异。
