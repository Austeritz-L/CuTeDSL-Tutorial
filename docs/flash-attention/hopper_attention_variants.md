# Hopper FlashAttention: Dense Attention 与变体逻辑梳理

> 目标：以 dense full attention 为基准，梳理 Hopper / SM90 forward 路径下 causal、local / sliding-window、varlen 等 attention 变体在计算逻辑、tile 范围、调度方式和源码实现上的差异。

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

### 0.2 基准Attention模型

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

## 5. 当前已讨论内容的对比表

| 场景 | 改变了什么 | Scheduler | 关键源码逻辑 |
| --- | --- | --- | --- |
| Dense full | 每个 Q row 看完整 K | `SingleTileScheduler` | 规则 grid，所有 M tile 看完整 N blocks |
| Causal | 每个 Q row 只看过去 | `SingleTileLPTScheduler` | `BlockInfo.get_n_block_min_max` + LPT 反转 + L2 swizzle |
| Local | 每个 Q row 只看窗口 | 通常 `SingleTileScheduler` | 左右 window 裁剪 `n_block_min/max`，边界 mask |
| Varlen non-causal | batch 内 seqlen 不规则 | `SingleTileVarlenScheduler` 普通分支 | `tile_idx -> batch/head/block`，`cu_seqlens -> offset` |
| Varlen causal/local | varlen + 可见范围约束 | `SingleTileVarlenScheduler` LPT/head-swizzle 分支 | 先解 varlen 坐标，再按真实 seqlen 做 block range |

## 6. 后续可继续补充的章节

### 6.1 Packed GQA / MQA

待补充问题：

- 普通 GQA/MQA 如何从 Q head 映射到 KV head。
- packed GQA 如何把 `qhead_per_kvhead` fold 到 M 维。
- packed 后 scheduler 的 head 维为什么变成 KV head。
- 一个 CTA 内如何处理来自多个 Q heads 的 rows。
- packed GQA 下 Q/O/LSE load/store 如何做地址反解。

### 6.2 Paged KV

待补充问题：

- 普通 K/V 连续地址与 paged K/V page table 地址的区别。
- `page_size == tile_n` 为什么可以走 TMA。
- `page_size != tile_n` 为什么需要 `PagedKVManager + cp.async`。
- decode / serving 中 paged KV 如何和 causal 叠加。

### 6.3 KV Cache / Decode

待补充问题：

- prefill 与 decode 的 shape 差异。
- `seqlen_q` 很小、`seqlen_k` 很长时，瓶颈如何变化。
- causal decode 如何通过 `seqlen_k - seqlen_q` 对齐位置。
- SM90 不支持 SplitKV 对长 K 的影响。

### 6.4 Softcap / score_mod / mask_mod / learnable sink

待补充问题：

- 哪些变体改变 score。
- 哪些变体改变可见性。
- 哪些变体只改变 softmax finalize。
- custom mask 为什么会关闭内置 causal/local 混合逻辑。

