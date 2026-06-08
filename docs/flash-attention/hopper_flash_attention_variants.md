# Hopper FlashAttention: Dense Attention 与变体逻辑梳理

> 目标：以 dense full attention 为基准，梳理 Hopper / SM90 forward 路径下 causal、local / sliding-window、varlen 等 attention 变体在计算逻辑、tile 范围、调度方式和源码实现上的差异。

## 目录

- [0. 阅读入口与源码定位](#0-阅读入口与源码定位)
- [1. Dense Full Attention 作为基准](#1-dense-full-attention-作为基准)
- [2. Causal Attention](#2-causal-attention)
- [3. Local / Sliding-Window Attention](#3-local--sliding-window-attention)
- [4. Varlen Attention](#4-varlen-attention)
- [5. Packed GQA / MQA Attention](#5-packed-gqa--mqa-attention)
- [6. Paged KV Attention](#6-paged-kv-attention)
- [7. 后续可继续补充的章节](#7-后续可继续补充的章节)

## 0. 阅读入口与源码定位

### 0.1 主要源码文件

- `flash_attn/cute/interface.py`
  - 上层接口归一化参数。
  - 选择 tile size。
  - 判断 dense / varlen / causal / local / packed GQA / paged KV 等场景。
  - 构造 `FlashAttentionForwardSm90(...)`。

- `flash_attn/cute/flash_fwd_sm90.py`
  - SM90 forward 主路径。
  - 做 Q/K/V/O/LSE layout select。
  - 创建 TMA atom。
  - 选择 scheduler。
  - producer 加载 Q/K/V。
  - consumer 执行 QK、mask、softmax、PV、epilogue。

- `flash_attn/cute/tile_scheduler.py`
  - `SingleTileScheduler`
  - `SingleTileLPTScheduler`
  - `SingleTileVarlenScheduler`

- `flash_attn/cute/block_info.py`
  - 根据 causal / local / window / seqlen 计算每个 M tile 对应的 K/V N block 范围。

- `flash_attn/cute/seqlen_info.py`
  - varlen 下读取真实 `offset_q`、`offset_k`、`seqlen_q`、`seqlen_k`。

### 0.2 基准Attention计算模型

Dense full attention 的逻辑是：

```text
对每个 batch b、Q head hq、query 位置 i：

scores[j] = dot(Q[b, i, hq], K[b, j, hkv]) * scale
P = softmax(scores over all j = 0..seqlen_k-1)
O[b, i, hq] = sum_j P[j] * V[b, j, hkv]
```

Dense full 的关键特征：

- 每个 query row 都看完整 K 序列。
- K/V 是规则 dense tensor。
- 每个 batch 的 `seqlen_q` 和 `seqlen_k` 是固定的。
- 每个 M tile 的 K/V N block 范围基本一样。
- CTA 工作量比较均匀。

## 1. Dense Full Attention 作为基准

### 1.1 Tensor 形态

```text
Q: [batch, seqlen_q, nheads_q,  head_dim]
K: [batch, seqlen_k, nheads_kv, head_dim]
V: [batch, seqlen_k, nheads_kv, head_dim_v]
O: [batch, seqlen_q, nheads_q,  head_dim_v]
```

### 1.2 SM90 内部 layout select

SM90 `__call__` 中会把 dense tensor 变成更适合 kernel 的布局：

```text
Q/O:  [batch, seqlen, head, dim] -> [seqlen, dim, head, batch]
K/V:  [batch, seqlen, head, dim] -> [seqlen, dim, head, batch]
LSE:  [batch, head, seqlen]      -> [seqlen, head, batch]
```

### 1.3 Dense full 的 tile 工作

一个 CTA 通常对应：

```text
work tile = (m_block, head_idx, batch_idx)
```

其中：

```text
m_block 表示 Q 方向上的 tile
head_idx 表示 attention head
batch_idx 表示 batch
```

对于 dense full：

```text
n_block_min = 0
n_block_max = ceil(seqlen_k / tile_n)
```

也就是每个 Q tile 都遍历完整 K/V tiles。

### 1.4 Dense full 的 scheduler

Dense full 通常使用：

```text
SingleTileScheduler
```

它的映射比较直接：

```text
blockIdx.x -> m_block
blockIdx.y -> head_idx
blockIdx.z -> batch_idx
```

因为每个 CTA 工作量接近，所以不需要 LPT 负载均衡。

## 2. Causal Attention

### 2.1 与 dense full 的数学差异

Dense full 中每个 query 可以看所有 key：

```text
j in [0, seqlen_k)
```

Causal attention 中 query 只能看自己及之前的 key：

```text
j <= i + offset
```

其中 `offset` 用于处理 `seqlen_q != seqlen_k` 的情况。

矩阵形态从完整矩形变为下三角：

```text
dense:
Q\K  0 1 2 3 4
 0   x x x x x
 1   x x x x x
 2   x x x x x

causal:
Q\K  0 1 2 3 4
 0   x . . . .
 1   x x . . .
 2   x x x . .
```

### 2.2 每个 M tile 的 K/V 范围如何变化

Causal 下，不同 `m_block` 能看的 K/V `n_block` 数不同：

```text
m_block 0    -> 能看的 K blocks 少
m_block 1    -> 能看的 K blocks 更多
...
m_block last -> 能看的 K blocks 最多
```

源码中由 `BlockInfo.get_n_block_min_max(...)` 计算：

```text
n_block_max = ceil(seqlen_k / tile_n)

如果 causal:
    m_idx_max = (m_block + 1) * tile_m
    n_idx = m_idx_max + seqlen_k - seqlen_q
    n_block_max = min(n_block_max, ceil(n_idx / tile_n))
```

这里的：

```text
seqlen_k - seqlen_q
```

是为了支持 Q/K 长度不一致，例如 decode 或 prefill 相关形态。

### 2.3 CTA 内部如何处理 mask

在 consumer mainloop 中，N blocks 会被分成几类：

```text
1. 最后一个 N block
   - K 长度可能不是 tile_n 的整数倍。
   - 需要 seqlen mask。

2. causal 边界附近的 N block
   - tile 内一部分可见，一部分不可见。
   - 需要 causal mask。

3. 完全可见的 N block
   - 不需要复杂 causal mask。
   - 可以走更简单的 QK / softmax / PV。
```

因此 causal 不是对所有 score 元素都做同样复杂的判断，而是：

```text
先在 tile 级别裁剪掉完全不可见的 N blocks，
再只对边界 blocks 做 mask。
```

### 2.4 Causal 的工作量倾斜

假设有 8 个 Q blocks，causal 下工作量大致类似：

```text
m_block:  0   1   2   3   4   5   6   7
work:     1   2   3   4   5   6   7   8
```

后面的 `m_block` 更重，因为它能看更多 K/V blocks。

如果按照自然顺序发射：

```text
blockIdx.x 0 -> m_block 0, work 1
blockIdx.x 1 -> m_block 1, work 2
...
blockIdx.x 7 -> m_block 7, work 8
```

重活会集中在 launch 后半段，容易造成尾部阶段只有少量 SM 在跑重 CTA，其他 SM 空闲。

### 2.5 LPT Scheduler 如何做负载均衡

Causal dense 使用：

```text
SingleTileLPTScheduler
```

LPT 表示：

```text
Longest Processing Time first
```

核心映射是把 block 顺序反过来：

```text
block = num_block - 1 - block
```

也就是：

```text
blockIdx.x 0 -> m_block last
blockIdx.x 1 -> m_block last-1
...
```

这样重 CTA 会优先铺到多个 SM 上，后面剩下较轻 CTA 用来填空。

示例：4 个 SM，8 个 CTA，工作量 `[1,2,3,4,5,6,7,8]`。

自然顺序近似：

```text
SM0: 1 + 5 = 6
SM1: 2 + 6 = 8
SM2: 3 + 7 = 10
SM3: 4 + 8 = 12

makespan = 12
```

LPT 反向顺序：

```text
SM0: 8
SM1: 7 + 1 = 8
SM2: 6 + 2 = 8
SM3: 5 + 3 = 8

makespan = 8
```

现实不会这么理想，但直觉是：

```text
重活先发，短活收尾，减少 tail。
```

### 2.6 LPT Scheduler 中的 L2 swizzle

`SingleTileLPTScheduler` 除了反转 block，还会做 L2 locality 相关的 swizzle。

一个 CTA 的坐标是：

```text
(m_block, head_idx, batch_idx)
```

对于同一个 `(batch_idx, head_idx)`，不同 `m_block` 会访问同一条 K/V 序列：

```text
(batch0, head0, m_block0) -> K/V batch0 head0
(batch0, head0, m_block1) -> K/V batch0 head0
(batch0, head0, m_block2) -> K/V batch0 head0
```

如果这些 CTA 时间上靠得近，K/V tile 更可能还留在 L2 中。

源码中会估算一个 `(batch, head)` 的 K/V working set 大小：

```text
size_one_kv_head =
    seqlen_k * (head_dim + head_dim_v) * element_size
```

然后估算 L2 能放下多少个这样的 `(batch, head)`：

```text
swizzle = power_of_two(size_l2 / size_one_kv_head)
```

这个 `swizzle` 可以理解为：

```text
一个 L2 section 中包含多少个 batch-head pair。
```

### 2.7 L2 section 调度示例

假设：

```text
num_batch = 1
num_head = 50
num_block = M
swizzle = 16
lpt = True
```

那么 flattened 的 batch-head pair 是：

```text
hb = batch_idx * num_head + head_idx
```

section 划分为：

```text
section 0: heads 0..15
section 1: heads 16..31
section 2: heads 32..47
section 3: heads 48..49
```

在 `section 0` 内，顺序近似为：

```text
m_block = M-1:
    head 0, head 1, ..., head 15

m_block = M-2:
    head 0, head 1, ..., head 15

m_block = M-3:
    head 0, head 1, ..., head 15

...

m_block = 0:
    head 0, head 1, ..., head 15
```

然后再进入：

```text
section 1: heads 16..31
section 2: heads 32..47
section 3: heads 48..49
```

所以它不是：

```text
head 0: m0, m1, ..., mlast
head 1: m0, m1, ..., mlast
```

而是：

```text
m last:   head group
m last-1: head group
m last-2: head group
...
```

目的：

```text
1. LPT: 重的 causal m_block 先发。
2. L2 swizzle: 相邻 CTA 访问同一小组 batch-head 的 K/V。
```

### 2.8 Causal 小结

Causal 相比 dense full 的核心变化：

```text
1. 每个 Q tile 的 K/V 范围不同。
2. 越靠后的 Q tile 工作量越重。
3. BlockInfo 负责计算 [n_block_min, n_block_max)。
4. 边界 N blocks 做 causal mask，完全可见 blocks 走简化路径。
5. Dense causal 使用 SingleTileLPTScheduler。
6. LPT 通过反向发射重 blocks 来减少尾部负载倾斜。
7. L2 swizzle 通过 batch-head section 提高 K/V cache reuse 概率。
```

## 3. Local / Sliding-Window Attention

### 3.1 与 dense full 的数学差异

Local attention 中每个 query 只看窗口内 key：

```text
j in [i - window_left, i + window_right]
```

矩阵形态从完整矩形变为带状区域：

```text
Q\K  0 1 2 3 4 5 6
 0   x x . . . . .
 1   x x x . . . .
 2   . x x x . . .
 3   . . x x x . .
```

### 3.2 Local 与 causal 的关系

Causal 可以看成一种特殊的单边 local：

```text
window_right = 0
window_left = infinity
```

接口层 `_resolve_causal_local_window(...)` 会把某些 window 参数归一化成 causal 或 local。

例如：

```text
window_size_left is None and window_size_right == 0
```

会被视为 causal。

### 3.3 Local 的 N block 范围

Local 同样通过 `BlockInfo.get_n_block_min_max(...)` 计算每个 `m_block` 的 K/V 范围。

右边界：

```text
m_idx_max = (m_block + 1) * tile_m
n_idx = m_idx_max + seqlen_k - seqlen_q
n_idx_right = n_idx + window_right
n_block_max = ceil(n_idx_right / tile_n)
```

左边界：

```text
m_idx_min = m_block * tile_m
n_idx = m_idx_min + seqlen_k - seqlen_q
n_idx_left = n_idx - window_left
n_block_min = max(floor(n_idx_left / tile_n), 0)
```

### 3.4 Local CTA 内部的 mask 分区

Local 下 mainloop 会区分：

```text
1. K/V 末尾 seqlen mask。
2. window right 边界 mask。
3. 完全在窗口内的 no-mask blocks。
4. window left 边界 mask。
```

这样可以避免所有 N blocks 都走复杂 mask。

### 3.5 Local 为什么通常不用 dense causal 的 LPT scheduler

Local window 下，每个 Q tile 可见的 K/V block 数接近固定窗口大小。

因此不同 `m_block` 的工作量通常更接近：

```text
m_block 0    -> window 范围被边界截断，略少
m_block 中间 -> 固定窗口大小
m_block last -> window 范围被边界截断，略少
```

它不像 causal 那样：

```text
work: 1, 2, 3, ..., N
```

强烈单调增长。

所以 local 的主要问题不是 CTA 工作量严重倾斜，而是：

```text
1. tile 内无效列更多。
2. 边界 mask 更多。
3. 大 tile_n 会浪费更多 K/V load 和 score 计算。
```

### 3.6 Local 的 tile_n 为什么更保守

Local window 内，一个 N tile 中可能有不少列其实在窗口外。

如果 `tile_n` 太大：

```text
1. producer 仍然加载整块 K/V。
2. consumer 仍然处理更大的 score tile。
3. 但其中一部分列最后被 mask 掉。
```

因此 SM90 tile size 选择中，local 在较大 head_dim 下会用更小的 `tile_n`。

例如：

```text
128 < head_dim <= 192:
    local -> tile_n = 96

192 < head_dim <= 256:
    local -> tile_n = 64
```

### 3.7 Local 小结

Local 相比 dense full 的核心变化：

```text
1. 每个 query row 只看窗口内 K。
2. BlockInfo 同时计算左边界和右边界。
3. mainloop 把边界 mask 和 no-mask blocks 分开处理。
4. 工作量通常不像 causal 那样强烈倾斜。
5. 因此 dense local 通常仍使用 SingleTileScheduler。
6. 性能重点更多在 tile_n 选择和减少无效列浪费。
```

## 4. Varlen Attention

### 4.1 Varlen 与 dense full 的数据组织差异

Dense batch 通常是：

```text
Q: [batch, max_seqlen_q, heads, dim]
K: [batch, max_seqlen_k, heads_kv, dim]
V: [batch, max_seqlen_k, heads_kv, dim_v]
```

Varlen 是把所有 batch 的 token 拼成一条：

```text
Q: [total_q, heads, dim]
K: [total_k, heads_kv, dim]
V: [total_k, heads_kv, dim_v]
```

并用前缀和记录每个 batch 的起点：

```text
lengths_q = [100, 300, 10]
cu_seqlens_q = [0, 100, 400, 410]
total_q = 410
```

含义：

```text
batch0 Q 范围: Q[0:100]
batch1 Q 范围: Q[100:400]
batch2 Q 范围: Q[400:410]
```

K/V 同理使用 `cu_seqlens_k`。

### 4.2 Varlen 的核心问题

Dense 中每个 batch 的 M block 数相同：

```text
num_m_blocks = ceil(max_seqlen_q / tile_m)
```

所以可以直接：

```text
grid_x = num_m_blocks
grid_y = num_head
grid_z = batch
```

Varlen 中每个 batch 的 Q 长度不同：

```text
batch0: len_q = 100 -> 1 个 m_block
batch1: len_q = 300 -> 3 个 m_block
batch2: len_q = 10  -> 1 个 m_block
```

真实 work tiles 是 ragged 的：

```text
batch0: block0
batch1: block0, block1, block2
batch2: block0
```

因此不能用 dense 的规则三维 grid 直接表达。

### 4.3 total_blocks_max 的含义

源码中 varlen scheduler 的 grid 上界：

```text
total_blocks_max =
    (total_q + batch * (cluster_m * tile_m - 1)) // tile_m

grid_x = total_blocks_max * num_head
```

先忽略 `cluster_m`，令 `cluster_m = 1`：

```text
total_blocks_max =
    (total_q + batch * (tile_m - 1)) // tile_m
```

它是：

```text
sum_i ceil(len_q_i / tile_m)
```

的一个安全上界。

原因是：

```text
ceil(len_i / tile_m)
```

最多比：

```text
len_i / tile_m
```

多不到 1 个 block。

所以把所有 batch 加起来，可以用：

```text
total_q + batch * (tile_m - 1)
```

估算一个上界。

### 4.4 total_blocks_max 示例

假设：

```text
tile_m = 128
num_head = 2
lengths_q = [100, 300, 10]
total_q = 410
batch = 3
```

真实 M blocks：

```text
batch0: ceil(100 / 128) = 1
batch1: ceil(300 / 128) = 3
batch2: ceil(10  / 128) = 1

real_total_m_blocks = 5
```

上界：

```text
total_blocks_max =
    (410 + 3 * 127) // 128
  = 791 // 128
  = 6
```

所以 launch：

```text
grid_x = 6 * num_head = 12
```

但真实只需要：

```text
5 * num_head = 10
```

因此：

```text
tile 0..9  -> valid work
tile 10..11 -> invalid padding work
```

也就是说：

```text
host 先按安全上界发射 12 个 CTA，
kernel 内每个 CTA 根据 tile_idx 和 cu_seqlens 判断自己是否有真实工作。
```

### 4.5 Varlen 中 tile_idx 如何映射到 batch/head/block

Varlen scheduler 会把 `blockIdx.x` 当成一个 flat tile id：

```text
tile_idx = blockIdx.x
```

然后在 kernel 内反解：

```text
tile_idx -> batch_idx, head_idx, m_block
```

仍以上面的例子：

```text
tile_m = 128
num_head = 2
lengths_q = [100, 300, 10]
num_m_blocks = [1, 3, 1]
```

每个 batch 的 tile 数是：

```text
batch0: 1 block * 2 heads = 2 tiles
batch1: 3 blocks * 2 heads = 6 tiles
batch2: 1 block * 2 heads = 2 tiles
```

真实 tiles：

```text
tile 0: batch0 head0 block0
tile 1: batch0 head1 block0

tile 2: batch1 head0 block0
tile 3: batch1 head0 block1
tile 4: batch1 head0 block2
tile 5: batch1 head1 block0
tile 6: batch1 head1 block1
tile 7: batch1 head1 block2

tile 8: batch2 head0 block0
tile 9: batch2 head1 block0

tile 10: invalid
tile 11: invalid
```

### 4.6 为什么 tile 5 是 batch1 head1 block0

先算 batch 边界：

```text
batch0 end tile = 1 * 2 = 2
batch1 end tile = (1 + 3) * 2 = 8
batch2 end tile = (1 + 3 + 1) * 2 = 10
```

`tile_idx = 5` 满足：

```text
2 <= 5 < 8
```

所以属于 `batch1`。

batch1 的起始 tile 是 2：

```text
mh_block = 5 - 2 = 3
```

普通 varlen non-causal 分支源码中采用：

```text
head_idx = mh_block // num_m_blocks
block = mh_block - head_idx * num_m_blocks
```

对于 batch1：

```text
num_m_blocks = 3
mh_block = 3

head_idx = 3 // 3 = 1
block = 3 - 1 * 3 = 0
```

所以：

```text
tile 5 = batch1, head1, block0
```

这说明普通 varlen non-causal 的 batch 内排列是：

```text
head-major, block-fastest
```

即：

```text
head0 block0
head0 block1
head0 block2
head1 block0
head1 block1
head1 block2
```

而不是：

```text
block0 head0
block0 head1
block1 head0
block1 head1
```

### 4.7 Varlen 如何在 kernel 内找 batch

源码不是提前构建 `tile_idx -> batch_idx` 表，而是在 kernel 内用 warp prefix sum 反解。

大致流程：

```text
1. 一个 warp 的 lanes 读取一组 batch 的 seqlen。
2. 每个 lane 计算对应 batch 的 num_m_blocks。
3. 做 warp prefix sum。
4. 得到每个 batch 的结束 tile 边界。
5. 判断当前 tile_idx 落在哪个 batch 范围内。
```

示例：

```text
num_m_blocks = [1, 3, 1]
prefix = [1, 4, 5]
乘 num_head = 2 后：
tile end = [2, 8, 10]
```

因此：

```text
tile_idx 5 -> batch1
tile_idx 9 -> batch2
tile_idx 10 -> invalid
```

### 4.8 Varlen 如何找真实 Q/K/V 地址

一旦 scheduler 得到：

```text
batch_idx, head_idx, m_block
```

`SeqlenInfoQK` 会读取：

```text
offset_q = cu_seqlens_q[batch_idx]
offset_k = cu_seqlens_k[batch_idx]

seqlen_q = cu_seqlens_q[batch_idx + 1] - offset_q
seqlen_k = cu_seqlens_k[batch_idx + 1] - offset_k
```

例如：

```text
cu_seqlens_q = [0, 100, 400, 410]

batch1:
    offset_q = 100
    seqlen_q = 300
```

如果：

```text
m_block = 1
tile_m = 128
```

那么这个 CTA 处理的 Q token 范围是：

```text
local q positions:
    128..255

global positions in Q[total_q]:
    offset_q + 128 .. offset_q + 255
  = 228 .. 355
```

K/V 同理：

```text
K base = K + offset_k
V base = V + offset_k
```

### 4.9 Varlen non-causal 与 varlen causal/local 的调度差异

普通 varlen non-causal 使用简单映射：

```text
head-major, block-fastest
```

原因：

```text
1. non-causal 中每个 m_block 工作量接近。
2. 连续跑同一个 head 的多个 blocks，有利于复用同一 batch/head 的 K/V。
```

但是 varlen causal/local 会进入 `SingleTileVarlenScheduler` 的 LPT / head-swizzle 分支。

其原因是：

```text
causal/local 下，不同 m_block 的工作量或边界行为不再完全均匀。
causal 尤其有明显的后部 block 更重的问题。
```

在该分支中，会对当前 batch 的 `num_m_blocks` 做类似 dense causal 的反转：

```text
block = num_m_blocks - 1 - block
```

并根据当前 batch 的 K/V working set 估算 L2 中适合放多少 heads。

### 4.10 Varlen 与 causal/local 叠加时的流程

Varlen 只解决：

```text
这个 tile 属于哪个真实 batch？
真实 Q/K/V offset 是多少？
真实 seqlen_q/seqlen_k 是多少？
```

Causal/local 再基于这个 batch 的真实长度计算：

```text
n_block_min
n_block_max
mask 边界
```

所以叠加流程是：

```text
1. tile_idx -> batch_idx/head_idx/m_block
2. cu_seqlens -> offset_q/offset_k/seqlen_q/seqlen_k
3. BlockInfo -> 当前 m_block 的 K/V N block 范围
4. mainloop -> 边界 mask + no-mask blocks
```

### 4.11 Varlen 小结

Varlen 相比 dense full 的核心变化：

```text
1. Q/K/V 从 dense batch tensor 变成 total_q / total_k 拼接 tensor。
2. cu_seqlens_q/k 记录每个 batch 的真实起点。
3. host 按 total_blocks_max 发射一个安全上界数量的 CTA。
4. kernel 内通过 tile_idx 和 cu_seqlens 反解 batch/head/block。
5. 多出来的 CTA 被标记 invalid，不做实际 Q/K/V load 和 MMA。
6. SeqlenInfoQK 负责读取真实 offset 和真实 seqlen。
7. 后续 causal/local/block range 都基于真实 seqlen 计算。
8. varlen non-causal 使用 head-major/block-fastest 简单映射。
9. varlen causal/local 会进入 LPT/head-swizzle 分支。
```

## 5. Packed GQA / MQA Attention

### 5.1 普通 MHA / GQA / MQA 的 head 映射

Dense MHA 中：

```text
nheads_q == nheads_kv
Q head h -> KV head h
```

GQA / MQA 中：

```text
nheads_q > nheads_kv
qhead_per_kvhead = nheads_q / nheads_kv
```

普通非 packed GQA 的映射是：

```text
head_idx_kv = head_idx_q // qhead_per_kvhead
```

例如：

```text
nheads_q = 8
nheads_kv = 2
qhead_per_kvhead = 4

Q head 0,1,2,3 -> KV head 0
Q head 4,5,6,7 -> KV head 1
```

MQA 是 GQA 的极端形式：

```text
nheads_kv = 1
qhead_per_kvhead = nheads_q
```

也就是所有 Q heads 共享同一个 KV head。

### 5.2 非 packed GQA 的 CTA 视角

如果不 packed，scheduler 仍然按 Q head 维度切：

```text
CTA = (m_block, q_head, batch)
```

例如：

```text
CTA0: q_head0 -> kv_head0
CTA1: q_head1 -> kv_head0
CTA2: q_head2 -> kv_head0
CTA3: q_head3 -> kv_head0
```

这几个 CTA 的 Q head 不同，但 K/V head 相同。

因此它们会重复围绕同一份 K/V tile 做 load / cache reuse：

```text
Q 不同
K/V 相同
```

即使 L2 cache 能提供一些复用，K/V 复用仍然发生在多个独立 CTA 之间，而不是一个 CTA 内部。

### 5.3 Packed GQA 的核心思想

Packed GQA 不改变 attention 数学，只改变 layout 和调度视角。

原始 SM90 内部 Q layout 类似：

```text
Q: [seqlen_q, dim, nheads_q, batch]
```

packed 后逻辑上变成：

```text
Q: [(qhead_per_kvhead, seqlen_q), dim, nheads_kv, batch]
```

也就是说：

```text
head 维从 nheads_q 变成 nheads_kv
多出来的 qhead_per_kvhead 被 fold 到 M 维
```

### 5.4 packed M 维的含义

假设：

```text
nheads_q = 8
nheads_kv = 2
qhead_per_kvhead = 4
tile_m = 128
```

非 packed 时，一个 CTA 处理：

```text
128 tokens * 1 Q head
```

packed 后，一个 CTA 处理：

```text
32 tokens * 4 Q heads = 128 packed rows
```

所以从 KV head 视角看：

```text
tile_m = 128
qhead_per_kvhead = 4

真实 token 个数 = tile_m / qhead_per_kvhead = 32
```

因为每个 token 位置要放进同一个 KV head group 下的 4 个 Q heads。

### 5.5 packed row 到真实 token/head 的映射

packed row 的反解关系是：

```text
packed_row = m_block * tile_m + row

token_idx = packed_row // qhead_per_kvhead
q_head_in_group = packed_row % qhead_per_kvhead
q_head = kv_head * qhead_per_kvhead + q_head_in_group
```

例如：

```text
tile_m = 128
qhead_per_kvhead = 4
m_block = 0
kv_head = 0
```

则：

```text
row 0 -> token0, Q head0
row 1 -> token0, Q head1
row 2 -> token0, Q head2
row 3 -> token0, Q head3

row 4 -> token1, Q head0
row 5 -> token1, Q head1
row 6 -> token1, Q head2
row 7 -> token1, Q head3

...

row 124 -> token31, Q head0
row 125 -> token31, Q head1
row 126 -> token31, Q head2
row 127 -> token31, Q head3
```

### 5.6 packed 后 scheduler 的 head 维

非 packed：

```text
CTA = (m_block, q_head, batch)
head_idx 表示 Q head
head_idx_kv = head_idx // qhead_per_kvhead
```

packed：

```text
CTA = (packed_m_block, kv_head, batch)
head_idx 表示 KV head
```

因此一个 CTA 内部：

```text
1. sQ 的 tile_m 行来自多个 Q heads。
2. sK/sV 来自一个 KV head。
3. QK 对每一行独立计算。
4. softmax state 对每一行独立维护。
5. PV 结果按 packed row 反解写回对应 Q head 和 token。
```

关键点：

```text
packed GQA 不会把多个 Q heads 的 score 混在一起。
M 维上的每一行仍然有独立的 row_max、row_sum 和 O accumulator。
```

### 5.7 packed GQA 与 causal/local 的关系

Causal/local 判断可见 K 时，需要使用真实 token index，而不是 packed row index。

因此当 `qhead_per_kvhead > 1` 时，`BlockInfo` 中会把 M index 转回 token index：

```text
m_idx_max = ceil((m_block + 1) * tile_m / qhead_per_kvhead)
m_idx_min = floor(m_block * tile_m / qhead_per_kvhead)
```

原因：

```text
packed M 维长度 = seqlen_q * qhead_per_kvhead
causal/local 的位置语义仍然基于真实 token index
```

### 5.8 packed GQA 的性能收益

Packed GQA 的核心收益：

```text
把多个共享同一 KV head 的 Q heads 放进同一个 M tile，
让一次 K/V 加载服务多个 Q heads。
```

非 packed：

```text
CTA0: q_head0 -> kv_head0
CTA1: q_head1 -> kv_head0
CTA2: q_head2 -> kv_head0
CTA3: q_head3 -> kv_head0
```

这些 CTA 都围绕同一份 K/V。

packed：

```text
CTA0: kv_head0
      M rows = 32 tokens * 4 Q heads
```

K/V tile 只需为这个 CTA 加载一次，然后服务多个 Q heads 的 rows。

收益包括：

```text
1. K/V load 复用更直接。
2. head 维 CTA 数从 nheads_q 降到 nheads_kv。
3. 减少重复 K/V tile 生产成本。
4. 降低对 L2 cache 事后命中的依赖。
5. 对 MQA 和 decode / KV cache 场景尤其有价值。
```

注意：

```text
packed GQA 不减少 QK/PV 的数学计算总量。
每个 Q head 仍然要算自己的 attention。
它减少的是重复 K/V 读取、producer 工作和 head 维调度碎片。
```

### 5.9 packed GQA 的代价和限制

Packed GQA 的代价：

```text
1. 一个 tile_m 覆盖的真实 token 数减少为 tile_m / qhead_per_kvhead。
2. Q/O/LSE 地址映射更复杂。
3. TMA Q/O 需要 packgqa wrapper。
4. tile_m 最好能被 qhead_per_kvhead 整除。
5. 某些 splitKV、block sparse、特殊 kernel 路径可能关闭 pack。
```

源码中如果：

```text
tile_m % qhead_per_kvhead != 0
```

则 packed Q 的 TMA 路径可能不能直接使用。

### 5.10 Packed GQA 小结

Packed GQA 相比 dense / 普通 GQA 的核心变化：

```text
1. 普通 GQA: 多个 Q heads 共享一个 KV head。
2. 非 packed: scheduler 仍按 Q head 切，多个 CTA 读同一 KV head。
3. packed: scheduler 按 KV head 切，把 Q heads fold 到 M 维。
4. 一个 tile_m 行表示 token * q_head_in_group。
5. CTA 内共享同一份 sK/sV。
6. softmax 仍然按 row 独立，不混合 Q heads。
7. 性能收益主要来自 K/V 复用和调度维度重排。
```

## 6. Paged KV Attention

### 6.1 Dense KV 的地址模型

普通 dense / varlen K/V 中，一个 batch 内部的 K/V 是逻辑连续的。

Dense 形态：

```text
K: [batch, seqlen_k, nheads_kv, head_dim]
V: [batch, seqlen_k, nheads_kv, head_dim_v]
```

给定：

```text
batch_idx
head_idx_kv
n_block
tile_n = 128
```

如果：

```text
n_block = 3
```

则读取：

```text
K[batch_idx, 384:512, head_idx_kv, :]
V[batch_idx, 384:512, head_idx_kv, :]
```

这是规则连续的 2D tile，适合 TMA。

### 6.2 Paged KV 的 tensor 形态

Paged KV 中，K/V 不再按 batch 直接连续存储，而是放在全局 page pool 中：

```text
K: [num_pages, page_size, nheads_kv, head_dim]
V: [num_pages, page_size, nheads_kv, head_dim_v]

page_table: [batch, max_num_pages_per_seq]
```

含义：

```text
num_pages:
    全局 K/V cache 池中的物理 page 数。

page_size:
    每个 page 能存多少个 token 的 K/V。

nheads_kv:
    KV head 数。

head_dim / head_dim_v:
    K/V 向量维度。
```

访问：

```text
K[p, t, h, d]
```

表示：

```text
物理 page p 中，第 t 个 token slot，第 h 个 KV head，第 d 维 K 值。
```

### 6.3 page_table 的含义

`page_table` 是逻辑序列到物理 page 的映射表：

```text
page_table[b, logical_page_idx] = physical_page_idx
```

也就是：

```text
batch 中第 b 条 sequence/request 的第 logical_page_idx 个逻辑 page，
实际放在全局 K/V page pool 的 physical_page_idx 号物理 page。
```

注意：

```text
一个 batch 不是一个 request。
一个 batch 包含多个 request / sequence。
batch 维上的每个元素通常对应一个 request。
```

如果：

```text
batch = 4
```

则：

```text
request0 -> page_table[0, :]
request1 -> page_table[1, :]
request2 -> page_table[2, :]
request3 -> page_table[3, :]
```

### 6.4 page_table 示例

假设：

```text
page_size = 4
batch = 3
max_num_pages_per_seq = 5

page_table =
batch0: [10, 11, 12, -1, -1]
batch1: [ 7, 30,  2, 19, -1]
batch2: [ 5, -1, -1, -1, -1]
```

表示：

```text
batch0 / request0:
    logical token 0..3   -> physical page 10
    logical token 4..7   -> physical page 11
    logical token 8..11  -> physical page 12

batch1 / request1:
    logical token 0..3   -> physical page 7
    logical token 4..7   -> physical page 30
    logical token 8..11  -> physical page 2
    logical token 12..15 -> physical page 19

batch2 / request2:
    logical token 0..3   -> physical page 5
```

K/V 的物理 page pool 是全局共享的，但每条 request 通过自己的 `page_table[b, :]` 看到一条逻辑连续序列。

### 6.5 逻辑 token 到物理 K/V 的地址翻译

给定：

```text
batch_idx
logical token j
head_idx_kv
```

地址翻译为：

```text
logical_page_idx = j // page_size
offset_in_page = j % page_size
physical_page = page_table[batch_idx, logical_page_idx]

K_logical[batch_idx, j, head_idx_kv, :] =
    K[physical_page, offset_in_page, head_idx_kv, :]

V_logical[batch_idx, j, head_idx_kv, :] =
    V[physical_page, offset_in_page, head_idx_kv, :]
```

所以 Paged KV 中没有直接的：

```text
K[batch, seqlen, head, dim]
```

而是：

```text
logical K/V sequence
    -> page_table
    -> physical K/V page pool
```

### 6.6 为什么需要 Paged KV

Paged KV 主要用于 KV cache / serving 场景。

生成式推理中：

```text
1. 每个 request 长度不同。
2. 每个 request 会持续增长。
3. 如果按最大长度给每个 request 分配连续 cache，显存浪费很大。
4. 动态增删 request 会产生碎片。
```

Paged KV 类似虚拟内存分页：

```text
request A 的逻辑 page0 -> physical page 10
request A 的逻辑 page1 -> physical page 37
request A 的逻辑 page2 -> physical page 12

request B 的逻辑 page0 -> physical page 4
request B 的逻辑 page1 -> physical page 9
```

优点：

```text
1. K/V cache 可以按 page 动态分配。
2. 长短序列混合时显存碎片更少。
3. decode 时可以持续追加新 pages。
4. batch 内 request 可以有不同 cache 长度。
```

### 6.7 page_size == tile_n: TMA paged path

如果：

```text
page_size == tile_n
```

那么一个 N tile 正好对应一个 logical page。

例如：

```text
page_size = 128
tile_n = 128
page_table[0] = [5, 9, 3, 20]
```

则：

```text
n_block 0 -> logical tokens 0..127   -> physical page 5
n_block 1 -> logical tokens 128..255 -> physical page 9
n_block 2 -> logical tokens 256..383 -> physical page 3
n_block 3 -> logical tokens 384..511 -> physical page 20
```

此时 producer 只需要：

```text
page_idx = page_table[batch_idx, n_block]
```

然后 TMA 加载：

```text
K[page_idx, 0:tile_n, head_idx_kv, :]
V[page_idx, 0:tile_n, head_idx_kv, :]
```

这与 dense TMA 很像：

```text
dense:
    src_idx = n_block

paged TMA:
    src_idx = page_table[batch_idx, n_block]
```

所以当前 SM90 实现中：

```text
page_size == tile_n
    -> paged_kv_non_tma = False
    -> use_tma_KV = True
```

### 6.8 page_size != tile_n: cp.async fallback

源码中 SM90 forward 构造参数：

```text
paged_kv_non_tma = page_size not in [None, tile_n]
```

也就是：

```text
page_size is None:
    非 paged K/V，TMA

page_size == tile_n:
    paged K/V，TMA

page_size != tile_n:
    paged K/V，non-TMA
    PagedKVManager + cp.async
```

### 6.9 情况 A: page_size < tile_n

例如：

```text
page_size = 64
tile_n = 128
```

一个 N tile 会跨多个 pages：

```text
n_block 0 wants logical tokens 0..127

= page_table[b, 0] offset 0..63
+ page_table[b, 1] offset 0..63
```

shared memory 中要拼成：

```text
sK[0:64, :]    <- K[page0, 0:64, :]
sK[64:128, :]  <- K[page1, 0:64, :]
```

理论上可以设计成多次 partial TMA，但当前 SM90 CuTe 路径没有实现这种：

```text
一个 n_block 内多次 page_table lookup
多次 TMA
每次写 shared memory 的不同 row offset
```

因此走 `PagedKVManager + cp.async` 手工拼 tile。

### 6.10 情况 B: page_size > tile_n

例如：

```text
page_size = 256
tile_n = 128
```

一个 page 包含多个 N tiles：

```text
n_block 0 -> page0 offset 0..127
n_block 1 -> page0 offset 128..255
```

每个块内部确实仍然连续。

但当前 paged TMA path 假设：

```text
n_block == logical_page_idx
offset_in_page == 0
```

也就是：

```text
page_idx = page_table[batch_idx, n_block]
```

当 `page_size > tile_n` 时，正确关系应该是：

```text
logical_page_idx = (n_block * tile_n) // page_size
offset_in_page = (n_block * tile_n) % page_size
physical_page = page_table[batch_idx, logical_page_idx]
```

这需要 TMA source coordinate 表达 page 内 offset。

当前实现没有为 paged path 构造：

```text
physical_page + offset_in_page
```

这种 TMA 坐标逻辑，所以同样归入 non-TMA paged path。

### 6.11 page_size != tile_n 并非理论上不能 TMA

需要特别注意：

```text
page_size != tile_n 并不是硬件理论上绝对不能 TMA。
```

更准确地说：

```text
当前 SM90 CuTe forward 的 TMA abstraction 和 producer pipeline
围绕 one logical N block <-> one physical page <-> one full TMA tile 设计。
```

当 page 和 tile 不一一对应时：

```text
page_size < tile_n:
    一个 N tile 跨多个 pages，需要 partial / gather TMA。

page_size > tile_n:
    一个 page 包含多个 N tiles，需要 page 内 offset TMA。
```

当前实现选择统一使用更通用的：

```text
PagedKVManager + cp.async
```

来覆盖这些情况。

### 6.12 cp.async fallback 的语义

non-TMA paged KV 可以理解为手工 gather：

```text
目标:
    sK[0:tile_n, :]
    sV[0:tile_n, :]

对 tile 内 logical row:
    logical_token = n_block * tile_n + row
    logical_page = logical_token // page_size
    offset = logical_token % page_size
    physical_page = page_table[batch_idx, logical_page]

    copy K[physical_page, offset, head_idx_kv, :] -> sK[row, :]
    copy V[physical_page, offset, head_idx_kv, :] -> sV[row, :]
```

真实实现会多线程分工、向量化并使用 `cp.async`，但语义就是把逻辑连续 K/V tile 拼到 shared memory。

### 6.13 Paged KV 与 causal/local 的叠加

Paged KV 只影响：

```text
K/V 从哪里读。
```

Causal/local 决定：

```text
哪些 K/V n_blocks 可见。
```

因此执行顺序是：

```text
1. scheduler 给出 m_block, head_idx, batch_idx
2. BlockInfo 根据 causal/local/seqlen 计算 n_block_min/n_block_max
3. mainloop 遍历这些 n_blocks
4. 对每个 n_block：
       dense KV:
           直接按 n_block 读连续 K/V tile
       paged KV:
           通过 page_table 找 physical page / offset
5. 执行 QK -> mask -> softmax -> PV
```

### 6.14 Paged KV 与 varlen 的区别

Varlen 解决的是：

```text
不同 batch 的序列长度不同。
每个 batch 在 total_k 中有不同 offset。
但一个 batch 内部 K/V 仍然连续。
```

访问形式：

```text
K[offset_k + j]
```

Paged KV 解决的是：

```text
一个 batch / request 内部的逻辑 K/V 也被拆成 pages。
逻辑连续，物理不连续。
```

访问形式：

```text
logical_page = j // page_size
offset = j % page_size
physical_page = page_table[batch, logical_page]
K[physical_page, offset]
```

总结：

```text
varlen:
    batch -> offset -> 连续区域

paged:
    batch + logical page -> physical page
```

### 6.15 Paged KV 小结

Paged KV 相比 dense KV 的核心变化：

```text
1. K/V 不再按 batch 连续存储，而是存入全局 page pool。
2. page_table 的每一行对应 batch 中一条 request 的逻辑页表。
3. logical token 需要先转换成 logical page 和 offset。
4. logical page 再通过 page_table 映射到 physical page。
5. page_size == tile_n 时，一个 N tile 正好一个 page，可以走当前 TMA path。
6. page_size != tile_n 时，当前 SM90 实现走 PagedKVManager + cp.async。
7. Paged KV 主要改变 K/V 地址计算，不改变 attention 数学。
```

## 7. 后续可继续补充的章节

### 7.1 KV Cache / Decode

待补充问题：

- prefill 与 decode 的 shape 差异。
- `seqlen_q` 很小、`seqlen_k` 很长时，瓶颈如何变化。
- causal decode 如何通过 `seqlen_k - seqlen_q` 对齐位置。
- SM90 不支持 SplitKV 对长 K 的影响。

### 7.2 Softcap / score_mod / mask_mod / learnable sink

待补充问题：

- 哪些变体改变 score。
- 哪些变体改变可见性。
- 哪些变体只改变 softmax finalize。
- custom mask 为什么会关闭内置 causal/local 混合逻辑。
