# Hopper FlashAttention Kernel 实现细节

本文聚焦 `FlashAttentionForwardSm90` 的 kernel 具体实现，尽量从硬件执行视角解释 Hopper forward FlashAttention：分块大小、shared memory layout、TMA pipeline、WGMMA 指令组织、online softmax、producer-consumer 调度、epilogue 写回。

源码位置：

- SM90 kernel：`flash-attention/flash_attn/cute/flash_fwd_sm90.py`
- forward 基类和 epilogue：`flash-attention/flash_attn/cute/flash_fwd.py`
- named barrier：`flash-attention/flash_attn/cute/named_barrier.py`
- pipeline 工具：`flash-attention/flash_attn/cute/pipeline.py`

## 1. Kernel 的数学目标

Forward FlashAttention 对每个 batch/head/Q block 做：

```text
S = Q @ K^T * softmax_scale
S = apply_mask(S)
P = softmax(S)
O = P @ V
LSE = logsumexp(S)
```

Hopper kernel 不会 materialize 全局 `S` 或 `P`。它按 K/V 的 N block 流式扫描，使用 online softmax 累积：

```text
for n_block in K/V blocks:
    S_tile = Q_tile @ K_tile^T
    P_tile, row_scale = online_softmax_update(S_tile)
    O = O * row_scale + P_tile @ V_tile
```

这样全局内存只读 Q/K/V，写 O/LSE，中间 `S/P` 只在寄存器或 shared memory 中短暂存在。

## 2. 典型 Hopper tile 配置

SM90 tile 由接口层 `_tile_size_fwd_sm90` 选择。

常见配置：

| head_dim | tile_m | tile_n | tile_hdim | MMA WGs | threads | K/V stages |
| --- | --- | --- | --- | --- | --- | --- |
| 64 | 192 | 128 | 64 | 3 | 512 | 2 |
| 96 causal/local | 192 | 128 | 96 | 3 | 512 | 2 |
| 96 non-causal | 192 | 144 | 96 | 3 | 512 | 2 |
| 128 | 128 | 128 | 128 | 2 | 384 | 2 |
| 192 | 128 | 96/112/128 | 192 | 2 | 384 | 2 |
| 256 | 128 | 64/80 | 256 | 2 | 384 | 2 |

其中：

- `tile_m`: Q rows per CTA。
- `tile_n`: K/V rows per pipeline iteration。
- `tile_hdim`: `head_dim` pad 到 16 倍数。
- `tile_hdimv`: `head_dim_v` pad 到 16 倍数。
- `num_stages=2`: K/V shared memory pipeline 双缓冲。
- `MMA WGs = tile_m // 64`。

为什么 `tile_m` 以 64 为单位：

- SM90 WGMMA 的常见输出 tile 以 64 行为一个 warpgroup 粒度。
- `atom_layout_mnk=(tile_m // 64, 1, 1)` 表示沿 M 方向复制多个 WGMMA atom。
- `tile_m=128` 时有 2 个 consumer warpgroup；`tile_m=192` 时有 3 个。

## 3. CTA 线程组织

kernel block 里线程分成两类：

```text
warp_idx 0..3     producer warpgroup, 128 threads
warp_idx 4..7     consumer WG0, 128 threads
warp_idx 8..11    consumer WG1, 128 threads
warp_idx 12..15   consumer WG2, 128 threads, 仅 tile_m=192 时存在
```

总线程数：

```text
num_threads = 128 * (num_mma_warpgroups + 1)
```

源码分支：

```python
if warp_idx < 4:
    setmaxregister_decrease(num_producer_regs)
    self.load(...)
else:
    setmaxregister_increase(num_mma_regs)
    tidx = tidx - 128
    self.mma(...)
```

寄存器预算：

| MMA WG 数 | producer regs | consumer regs |
| --- | --- | --- |
| 1 | 56 | 256 |
| 2 | 24 或 40 | 240 或 224 |
| 3 | 32 | 160 |

这不是简单优化项，而是 Hopper warpgroup specialization 的重要组成部分：producer 负责搬运，consumer 持有大量 accumulator 和 softmax fragment。

## 4. Shared memory 分配

`FlashAttentionForwardSm90._get_shared_storage_cls()` 定义 shared storage。

主要对象：

```text
sQ: tile_m x tile_hdim
sK: tile_n x tile_hdim x num_stages
sV: tile_n x tile_hdimv x num_stages
sP: tile_m x tile_n, 仅 mma_pv_is_rs=False 时存在
sO: tile_m x tile_hdimv, 复用 sQ 的 storage 做 epilogue staging
mbar_ptr_Q: 1 * 2
mbar_ptr_K: num_stages * 2
mbar_ptr_V: num_stages * 2
```

shared memory usage 在基类 `can_implement` 中估算：

```text
Q bytes = tile_m * head_dim * 2
K bytes = tile_n * head_dim * num_stages * 2
V bytes = tile_n * head_dim_v * num_stages * 2
total ~= Q + K + V
```

如果 `Q_in_regs=True`，Q 和 V 的 storage 有复用逻辑；当前 Hopper forward 构造时通常是 `Q_in_regs=False`。

`sP` 是否存在取决于 `mma_pv_is_rs`：

- `mma_pv_is_rs=True`: softmax 后的 P 留在寄存器，作为 PV WGMMA 的 A operand。
- `mma_pv_is_rs=False`: P 先写到 shared memory，再从 shared memory 作为 PV 的 A operand。

## 5. Shared memory layout 和 swizzle

SM90 使用 CuTe 的 swizzled shared memory layout：

```python
sQ_layout_atom = warpgroup.make_smem_layout_atom(
    sm90_utils_basic.get_smem_layout_atom(ROW_MAJOR, dtype, tile_hdim),
    dtype,
)
```

Q/K/V/O 都通过类似函数生成 layout atom，再 tile 到完整 shape：

```text
sQ_layout: tile_m x tile_hdim
sK_layout: tile_n x tile_hdim x num_stages
sV_layout: tile_n x tile_hdimv x num_stages
sO_layout: tile_m x tile_hdimv
```

这些 layout 的目的：

- 满足 WGMMA 对 shared memory operand 的对齐和 swizzle 要求。
- 降低 bank conflict。
- 让 TMA 写入的 shared memory tile 可以直接被 WGMMA 消费。

## 6. TMA 和 cp.async

Hopper 主路径优先使用 TMA：

```text
Q/K/V: CopyBulkTensorTileG2SOp
O:     CopyBulkTensorTileS2GOp
```

TMA 的特点：

- 由少数线程发起，通常 producer 的 warp 0。
- 使用 memory barrier 跟 consumer 同步。
- 适合规则的 2D tile，从 global bulk copy 到 shared。

非 TMA 路径：

- paged KV 且 `page_size != tile_n` 时，K/V 走 `PagedKVManager + cp.async`。
- Q 在某些 pack GQA 对不齐场景也可能走 cp.async。

cp.async copy atom 在基类中定义：

```python
cpasync.CopyG2SOp(cache_mode=GLOBAL)
num_bits_per_copy = 128
```

也就是说 cp.async 路径按 128-bit vectorized copy 组织。

## 7. Pipeline 结构

SM90 forward 有三条 pipeline：

```text
pipeline_q: 1 stage
pipeline_k: num_stages, 通常 2
pipeline_v: num_stages, 通常 2
```

Q 只需要一个 stage，因为一个 CTA 的 Q tile 在整个 K/V streaming 过程中重复使用。

K/V 需要 double buffer，因为 mainloop 一边消费当前 `n_block`，一边预取下一个 `n_block`。

TMA pipeline 的参与者：

```text
producer group: 1 thread, 用于 TMA 发起
consumer group: MMA warps
```

cp.async pipeline 的参与者：

```text
producer group: 128 load threads
consumer group: MMA warps
```

kernel 开始时会执行：

```text
pipeline_init_arrive
pipeline_init_wait
```

用 cluster/barrier 语义保证 pipeline barrier 初始化完成后再进入 producer/consumer 正式逻辑。

## 8. Named barrier

SM90 forward 使用 `NamedBarrierFwd`：

```python
Epilogue = 1
WarpSchedulerWG1 = 2
WarpSchedulerWG2 = 3
WarpSchedulerWG3 = 4
PFull = 5
PEmpty = 6
```

主要用途：

- `Epilogue`: O 从寄存器写 shared、shared 再 TMA/copy 写 global 时同步。
- `WarpSchedulerWG*`: intra-warpgroup overlap 中协调不同 consumer warpgroup 的 QK/PV 节奏。
- `PFull/PEmpty`: P 走 shared memory 时用于生产/消费同步。

## 9. Load producer 逻辑

producer 入口是 `FlashAttentionForwardSm90.load`。

核心变量：

```python
kv_producer_state = make_pipeline_state(Producer, num_stages)
q_producer_phase = 1
tile_scheduler = TileSchedulerCls()
work_tile = tile_scheduler.initial_work_tile_info()
```

每个 work tile 对应：

```text
m_block, head_idx, batch_idx, _
```

对每个 work tile：

1. 构造当前 batch 的 seqlen 信息。
2. 找到 Q head 对应的 KV head：
   - 普通 GQA: `head_idx_kv = head_idx // qhead_per_kvhead`
   - pack GQA: `head_idx_kv = head_idx`
3. 构造 Q/K/V 的 global tile view。
4. 构造 TMA copy closure 或 cp.async manager。
5. 根据 causal/local/block sparse 决定 N block 访问范围。

### 9.1 N block 方向

非 block-sparse 主路径：

```python
n_block_min, n_block_max = block_info.get_n_block_min_max(seqlen, m_block)
n_block = n_block_max - 1
```

也就是从右往左扫描 K/V block：

```text
n_block_max - 1, n_block_max - 2, ..., n_block_min
```

原因：

- causal/local mask 通常右侧边界更关键。
- 第一块常常需要处理 seqlen/casual 边界 mask。
- 右到左扫描方便和现有 online softmax/mask 区域划分配合。

### 9.2 第一轮 load

第一轮会先发起：

```text
K[n_block_max - 1] -> sK[stage]
Q[m_block]         -> sQ
```

然后根据模式加载 V。

TMA Q：

```python
pipeline_q.producer_acquire_w_index_phase(0, q_producer_phase)
load_Q(tma_bar_ptr=pipeline_q.sync_object_full.get_barrier(0))
q_producer_phase ^= 1
```

cp.async Q：

```python
pack_gqa.load_Q(...)
cp_async_commit_group()
pipeline_q.producer_commit_w_index(0)
q_producer_phase ^= 1
```

### 9.3 K/V double buffer

无 intra-wg overlap 或非 TMA KV：

```text
load K[current stage]
load V[current stage]
advance stage
load K[next stage]
load V[next stage]
advance stage
...
```

开启 `intra_wg_overlap` 且 KV 使用 TMA 时：

```text
先 load K[last]
之后循环：
    load K[next]
    load V[previous]
最后 load V[first]
```

这种错位加载让 consumer 可以在某些阶段把下一轮 QK 和当前轮 PV 交叠起来。

## 10. `load_KV`

`load_KV` 封装了 K/V 的具体搬运。

TMA 路径：

```text
tma_load_fn(src_idx=block 或 page_idx, producer_state=state)
pipeline_kv.producer_commit(state)
```

cp.async paged KV 路径：

```text
paged_kv_manager.load_KV(...)
cp_async_commit_group()
pipeline_kv.producer_commit(state)
```

这里的 `producer_state.index` 对应 shared memory stage，也就是 `sK[:, :, stage]` 或 `sV[:, :, stage]`。

## 11. MMA consumer 入口

consumer 入口是 `FlashAttentionForwardSm90.mma`。

每个 consumer warpgroup 通过 `warp_group_idx` 切出自己的 tiled MMA slice：

```python
warp_group_idx = tidx // 128
wg_mma_qk = tiled_mma_qk.get_slice(warpgroup_layout(warp_group_idx))
wg_mma_pv = tiled_mma_pv.get_slice(warpgroup_layout(warp_group_idx))
```

对于 `tile_m=128`：

```text
WG0 负责 Q rows 0..63
WG1 负责 Q rows 64..127
```

对于 `tile_m=192`：

```text
WG0 负责 Q rows 0..63
WG1 负责 Q rows 64..127
WG2 负责 Q rows 128..191
```

这就是 `atom_layout_mnk=(tile_m // 64, 1, 1)` 在这里的实际效果。

## 12. QK WGMMA

QK 的逻辑：

```text
acc_S = Q_tile @ K_tile^T
```

MMA 配置：

```text
A operand: Q, major K
B operand: K, major K
C/D: fp32 accumulator
tiler_mn: (64, tile_n)
```

`partition_fragment_ABC` 会把 `sQ`、`sK` partition 成当前 warpgroup 的 WGMMA operand fragment：

```python
tSrQ, tSrK, acc_S = partition_fragment_ABC(
    wg_mma_qk,
    (tile_m, tile_n, tile_hdim),
    sQ,
    sK,
)
```

执行时用：

```python
acc_S = mma_qk_fn(B_idx=state.index, wg_wait=-1)
```

其中：

- `B_idx=state.index` 选择当前 K pipeline stage。
- `wg_wait=-1` 表示发起 WGMMA 后不立即等待所有 outstanding group，后面显式 `warpgroup.wait_group(...)`。

底层对应 Hopper WGMMA 指令族，语义上是：

```text
wgmma.mma_async.sync.aligned.m64n{tile_n_part}k16...
```

CuTe 会根据 dtype、layout、tile_n、K 维循环生成具体 WGMMA 指令序列。K 维按 16 的粒度参与 Tensor Core MMA，所以 head dim 会 pad 到 16 的倍数。

## 13. Score modifier 和 mask

QK 得到 `acc_S` 后，会依次处理：

1. 可选 `score_mod`。
2. causal/local/seqlen mask。
3. online softmax。

mask 对象：

```python
AttentionMask(
    tile_m,
    tile_n,
    window_size_left,
    window_size_right,
    qhead_per_kvhead_packgqa,
)
```

典型 mask 类型：

- seqlen 越界 mask。
- causal mask。
- local window mask。
- pack GQA 下的 head/tile 对齐 mask。

被 mask 的 score 会写成 `-inf` 或等价极小值，使 softmax 后概率为 0。

## 14. Online softmax

每个 Q row 维护 softmax 状态：

```text
m_i = 当前扫描过 score 的 row max
l_i = 当前扫描过 exp(score - m_i) 的 sum
O_i = 当前已累计的输出
```

每处理一个 `S_tile`：

```text
m_new = max(m_old, max(S_tile))
alpha = exp2(m_old - m_new)
p = exp2(S_tile - m_new)
l_new = l_old * alpha + sum(p)
O = O * alpha + p @ V
```

源码中：

- `softmax.online_softmax(...)` 更新 row max / row sum，并返回 `row_scale`。
- `softmax.rescale_O(acc_O, row_scale)` 对旧的 O accumulator 做缩放。
- 最后 `softmax.finalize(...)` 生成 LSE，并得到最终归一化需要的 scale。

注意这里使用 `softmax_scale_log2`，因为底层常用 `exp2` 近似/指令路径：

```text
score * softmax_scale * log2(e)
```

## 15. P 的两种路径：RS 和 SMEM

### 15.1 `mma_pv_is_rs=True`

P 留在寄存器：

```text
acc_S fp32
  -> softmax
  -> convert to fp16/bf16 P fragment in register
  -> PV WGMMA A operand comes from register
```

优点：

- 避免 P 写 shared 再读 shared。
- 对 head_dim 64/128 等配置很高效。

代价：

- register pressure 更高。
- P fragment、O accumulator、softmax state 同时存在。

### 15.2 `mma_pv_is_rs=False`

P 写入 shared memory：

```text
acc_S fp32
  -> softmax
  -> convert to fp16/bf16
  -> store sP
  -> PV WGMMA 从 sP 读 A operand
```

通常用于某些 `head_dim <= 96` 且 tile 形状更适合 shared P 的路径。

需要额外同步：

```text
PFull / PEmpty named barrier
fence / sync before PV WGMMA consumes sP
```

## 16. PV WGMMA

PV 的逻辑：

```text
acc_O += P_tile @ V_tile
```

MMA 配置：

```text
A operand: P, register 或 shared
B operand: V^T shared view
C/D: fp32 accumulator
tiler_mn: (64, tile_hdimv)
```

V 在 shared 中原始 layout 是：

```text
sV: tile_n x tile_hdimv x num_stages
```

consumer 创建转置 view：

```python
sVt = transpose_view(sV)
```

PV WGMMA 看到的是：

```text
V^T: tile_hdimv x tile_n
```

执行时：

```python
mma_pv_fn(B_idx=state.index, wg_wait=0 或 -1)
```

同样通过 `B_idx` 选择 V 的 pipeline stage。

## 17. Mainloop 的三种消费函数

SM90 consumer 里有几个关键函数。

### 17.1 `mma_one_n_block`

普通一块 N block 的顺序：

```text
wait K
QK WGMMA
wait QK done
release K
score_mod / mask
online softmax
S -> P
rescale O
wait V
PV WGMMA
release V
advance pipeline state
```

这是最容易理解的标准路径。

### 17.2 `first_half_block_overlap` / `last_half_block_overlap`

用于 intra-wg overlap 的边界处理。

因为交叠需要前后两个 N block 的 K/V 都有合适的 pipeline 状态，第一块和最后一块要拆开处理：

```text
first_half:
    wait K
    QK
    softmax
    准备 P

last_half:
    wait V
    PV
```

### 17.3 `mma_one_n_block_intrawg_overlap`

核心交叠路径：

```text
current V state = 当前 block
next K state    = 下一 block

wait K[next]
launch QK[next]

wait V[current]
launch PV[current]

wait QK partly/all
release K[next]
score/mask/softmax for next

wait PV
release V[current]
prepare P[next]
```

也就是把：

```text
QK(next block)
PV(current block)
```

在同一 consumer warpgroup 内尽量重叠。

这要求 producer 端 K/V load 也错位：

```text
load K[next] before load V[current]
```

## 18. Mainloop 区域划分

非 block-sparse 路径按 mask 情况把 N block 分成几个区域。

典型顺序：

1. 最右侧 first block：经常需要 seqlen mask。
2. causal/local mask 区域：需要每个元素判断是否可见。
3. 中间 no-mask 区域：可以省掉 mask 分支，性能最好。
4. local 左边界区域：local attention 下还需要左窗口 mask。

这种划分的目标是：

- 只在需要的时候应用复杂 mask。
- 对完全可见的 N block 走更轻的路径。
- 保持 online softmax 的扫描顺序一致。

## 19. Epilogue：O 和 LSE 写回

Epilogue 在基类 `FlashAttentionForwardBase.epilogue`。

步骤：

1. `acc_O fp32 -> rO fp16/bf16`
2. consumer threads 到达 `NamedBarrierFwd.Epilogue`
3. `rO -> sO`
4. 写 LSE 到 global
5. `sO -> O global`

### 19.1 O 写回为什么经过 shared

即使 O 已经在寄存器里，kernel 仍然先把 O 写到 `sO`：

```text
register accumulator
  -> shared memory sO
  -> global memory O
```

原因：

- TMA store 要从 shared memory 到 global memory。
- 即使非 TMA store，也可以通过 shared staging 做更规整的 vectorized global copy。

### 19.2 TMA store O

TMA O store 使用：

```text
CopyBulkTensorTileS2GOp
```

大致顺序：

```text
fence_view_async_shared
barrier_arrive(Epilogue)
warp_idx == 4 发起 store_O()
cp_async_bulk_commit_group()
cp_async_bulk_wait_group(0, read=True)
```

这里 `warp_idx == 4` 是第一个 consumer warpgroup 的第一个 warp，用它来发起 TMA store。

### 19.3 LSE 写回

LSE 是每个 Q row 一个 fp32 值。写回时只让对应 column 0 的 thread 写，避免重复写。

pack GQA 情况下通过 `PackGQA.store_LSE` 处理 M 维里打包的多个 Q head。

## 20. 一次 CTA 的完整时间线

```text
kernel entry
  |
  | warp 0 prefetch TMA descriptors
  | allocate shared storage
  | create pipeline_q/k/v
  | initialize barriers
  v
producer WG                         consumer WGs
-----------                         ------------
load Q tile
load K last block
load V last block      --->         wait Q/K
                                    QK WGMMA
                                    mask + online softmax
                                    P prepare
                                    wait V
                                    PV WGMMA

prefetch next K/V      --->         consume next N block
...
producer tail                       finalize softmax
                                    rescale O
                                    epilogue store O/LSE
```

开启 intra-wg overlap 后，中间部分更接近：

```text
producer:
    load K[i-1]
    load V[i]

consumer:
    QK[i-1] overlaps PV[i]
```

## 21. 分块和资源的具体例子

### 21.1 head_dim=128

典型配置：

```text
tile_m = 128
tile_n = 128
tile_hdim = 128
tile_hdimv = 128
num_stages = 2
MMA WGs = 2
threads = 384
```

shared memory 近似：

```text
sQ = 128 * 128 * 2 bytes = 32 KB
sK = 128 * 128 * 2 stages * 2 bytes = 64 KB
sV = 128 * 128 * 2 stages * 2 bytes = 64 KB
total ~= 160 KB
```

每个 consumer warpgroup 处理 64 行 Q：

```text
QK: 64 x 128 output score tile
PV: 64 x 128 output O tile
```

如果 `mma_pv_is_rs=True`，P 不落 shared，寄存器压力较大但少一次 shared memory 往返。

### 21.2 head_dim=64

典型配置：

```text
tile_m = 192
tile_n = 128
tile_hdim = 64
num_stages = 2
MMA WGs = 3
threads = 512
```

shared memory 近似：

```text
sQ = 192 * 64 * 2 = 24 KB
sK = 128 * 64 * 2 * 2 = 32 KB
sV = 128 * 64 * 2 * 2 = 32 KB
total ~= 88 KB
```

因为 headdim 小，shared memory 和寄存器压力都较低，所以可以把 `tile_m` 做到 192，提高每个 CTA 的 Q rows 工作量。

### 21.3 head_dim=256

典型配置：

```text
tile_m = 128
tile_n = 64 或 80
tile_hdim = 256
num_stages = 2
MMA WGs = 2
threads = 384
```

如果 `tile_n=64`：

```text
sQ = 128 * 256 * 2 = 64 KB
sK = 64 * 256 * 2 * 2 = 64 KB
sV = 64 * 256 * 2 * 2 = 64 KB
total ~= 192 KB
```

这已经接近 shared memory 压力较大的区域，所以 N block 需要缩小。

## 22. Hopper 指令层要点

### 22.1 TMA bulk copy

语义：

```text
global tensor tile -> shared tensor tile
shared tensor tile -> global tensor tile
```

对应 Hopper 的 bulk tensor copy 能力，CuTe 用：

```text
cpasync.CopyBulkTensorTileG2SOp
cpasync.CopyBulkTensorTileS2GOp
```

并通过 memory barrier 和 pipeline state 交接。

### 22.2 WGMMA

语义：

```text
wgmma.mma_async
```

QK：

```text
fp16/bf16 Q shared
fp16/bf16 K shared
fp32 accumulator S
```

PV：

```text
fp16/bf16 P register/shared
fp16/bf16 V shared
fp32 accumulator O
```

Hopper WGMMA 是 warpgroup 级别，即 128 threads 协作执行。kernel 的 `warpgroup.wait_group(0/1)` 用来等待 outstanding WGMMA group 完成。

### 22.3 cp.async fallback

非 TMA load 使用：

```text
cp.async global.shared
cp_async_commit_group
pipeline producer commit
```

copy 粒度是 128-bit。

## 23. 和 Ampere 写法的关键区别

本文不展开 Ampere，但为了理解 Hopper，这里只列必要差异：

- Hopper 使用 WGMMA，计算单位是 128-thread warpgroup；Ampere 常见是 warp-level MMA。
- Hopper 主路径使用 TMA bulk tensor copy；Ampere 主要依赖 cp.async tiled copy。
- Hopper 使用 producer-consumer warpgroup specialization；Ampere 更常见同一批 warp 做 load+compute 的 software pipeline。
- Hopper 用 memory barrier/pipeline 同步 TMA 和 WGMMA consumer。

## 24. 阅读源码的定位表

| 想看什么 | 入口 |
| --- | --- |
| tile_m/tile_n 怎么来 | `interface.py::_tile_size_fwd_sm90` |
| tiled MMA 怎么创建 | `flash_fwd_sm90.py::_get_tiled_mma` |
| shared storage 有哪些 tensor | `flash_fwd_sm90.py::_get_shared_storage_cls` |
| TMA/cp.async copy atom | `flash_fwd.py::_setup_attributes` 和 `flash_fwd_sm90.py::__call__` |
| kernel producer/consumer 分流 | `flash_fwd_sm90.py::kernel` |
| Q/K/V load 顺序 | `flash_fwd_sm90.py::load` |
| 单个 K/V block 怎么算 | `flash_fwd_sm90.py::mma_one_n_block` |
| intra-wg overlap | `flash_fwd_sm90.py::mma_one_n_block_intrawg_overlap` |
| O/LSE 写回 | `flash_fwd.py::epilogue` |

## 25. 最短心智模型

Hopper FlashAttention forward 可以浓缩成：

```text
一个 CTA 处理一个 Q block x 一个 head x 一个 batch。

producer warpgroup:
    用 TMA/cp.async 把 Q/K/V tile 搬到 shared memory。
    Q 单 stage，K/V 双 stage。

consumer warpgroups:
    每个 warpgroup 负责 64 行 Q。
    QK 用 WGMMA 得到 score。
    score 在寄存器中 mask + online softmax。
    P 留寄存器或写 shared。
    PV 用 WGMMA 累加 O。

epilogue:
    O accumulator 转 dtype，先写 shared，再 TMA/copy 写 global。
    LSE 按 Q row 写 global。
```

如果要继续向更底层看，下一步应该追 `sm90_utils_basic.make_trivial_tiled_mma`、`sm90_utils.gemm_zero_init/gemm_w_idx` 和 CuTe 生成的 MLIR/PTX，那里可以看到具体 WGMMA shape 和指令序列。
