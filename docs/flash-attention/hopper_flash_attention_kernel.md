# Hopper FlashAttention Kernel 实现链路

本文承接上一份“上层接口链路”文档，继续深入 Hopper / SM90 FlashAttention forward kernel 的 device-side 实现。

上一份文档解释的是：

```text
Python API
  -> _flash_attn_fwd
  -> FlashAttentionForwardSm90.__call__
  -> kernel.launch 之前的配置
```

本文解释的是：

```text
FlashAttentionForwardSm90.kernel
  -> producer load Q/K/V
  -> consumer QK / mask / online softmax / PV
  -> epilogue store O/LSE
```

源码路径基于本地仓库：

```text
flash-attention-src/
  flash_attn/cute/flash_fwd_sm90.py
  flash_attn/cute/flash_fwd.py
  flash_attn/cute/softmax.py
  flash_attn/cute/block_info.py
  flash_attn/cute/seqlen_info.py
  flash_attn/cute/mask.py
  flash_attn/cute/paged_kv.py
  flash_attn/cute/tile_scheduler.py
```

本文仍然沿用一个真实推理场景风格的 case：

```text
模型: Qwen2.5-7B 风格
GPU: H100 / SM90
dtype: bf16
阶段: chunked prefill with paged KV cache

cached prefix = 2048 tokens
current chunk = 1024 tokens
visible KV = 3072 tokens
page_size = 128

q: [1, 1024, 28, 128]
k_cache_paged: [num_physical_pages, 128, 4, 128]
v_cache_paged: [num_physical_pages, 128, 4, 128]
page_table: [1, max_pages_per_seq]
seqused_k: [1] = [3072]
causal = True

head_dim = 128
head_dim_v = 128
qhead_per_kvhead = 7
tile_m = 128
tile_n = 128
num_stages = 2
mma_pv_is_rs = True
intra_wg_overlap = True
```

## 目录

1. [Kernel 总体结构](#1-kernel-总体结构)
2. [Kernel 开头: descriptor、shared storage、pipeline](#2-kernel-开头-descriptorshared-storagepipeline)
3. [BlockInfo、SeqlenInfo、AttentionMask、Scheduler](#3-blockinfoseqleninfoattentionmaskscheduler)
4. [Producer: `load`](#4-producer-load)
5. [Consumer: `mma`](#5-consumer-mma)
6. [Mainloop: 从右往左扫 K/V blocks](#6-mainloop-从右往左扫-kv-blocks)
7. [QK、mask、online softmax、PV](#7-qkmaskonline-softmaxpv)
8. [Online Softmax 实现](#8-online-softmax-实现)
9. [Mask 实现和 chunked prefill causal offset](#9-mask-实现和-chunked-prefill-causal-offset)
10. [Epilogue: 写 O 和 LSE](#10-epilogue-写-o-和-lse)
11. [Qwen Case 代入: 一个 work tile 发生了什么](#11-qwen-case-代入-一个-work-tile-发生了什么)
12. [Kernel 的关键设计收益](#12-kernel-的关键设计收益)
13. [源码阅读顺序](#13-源码阅读顺序)
14. [总结](#14-总结)

## 1. Kernel 总体结构

SM90 forward kernel 的入口在：

```python
# flash_fwd_sm90.py
@cute.kernel
def kernel(
    self,
    mQ,
    mK,
    mV,
    mO,
    mLSE,
    mCuSeqlensQ,
    mCuSeqlensK,
    mSeqUsedQ,
    mSeqUsedK,
    mPageTable,
    tma_atom_Q,
    tma_atom_K,
    tma_atom_V,
    tma_atom_O,
    softmax_scale_log2,
    softmax_scale,
    window_size_left,
    window_size_right,
    learnable_sink,
    blocksparse_tensors,
    sQ_layout,
    sK_layout,
    sV_layout,
    sO_layout,
    sP_layout,
    gmem_tiled_copy_Q,
    gmem_tiled_copy_K,
    gmem_tiled_copy_V,
    gmem_tiled_copy_O,
    tiled_mma_qk,
    tiled_mma_pv,
    tile_sched_params,
    TileScheduler,
    SharedStorage,
    aux_tensors,
    fastdiv_mods=None,
):
```

这些参数几乎全部来自 `FlashAttentionForwardSm90.__call__`。kernel 本身不再重新决定 tile、MMA、TMA descriptor、shared layout、scheduler 类型，而是直接消费这些已经配置好的对象。

kernel 内部可以粗略分成五段：

```text
1. TMA descriptor prefetch
2. shared memory / mbarrier / pipeline 初始化
3. 获取 shared memory tensor view
4. 构造 BlockInfo / SeqlenInfo / AttentionMask / TileScheduler 闭包
5. 按 warpgroup 分成 producer 和 consumer:
      producer: load Q/K/V
      consumer: QK, mask, online softmax, PV, epilogue
```

源码上最关键的分流是：

```python
if warp_idx < 4:  # Producer
    cute.arch.setmaxregister_decrease(self.num_producer_regs)
    self.load(...)
else:             # Consumer
    cute.arch.setmaxregister_increase(self.num_mma_regs)
    self.mma(...)
```

SM90 的一个 warpgroup 是 128 threads，也就是 4 warps。对本 case：

```text
block = 384 threads

warp_idx 0..3:
    producer warpgroup

warp_idx 4..7:
    consumer warpgroup 0

warp_idx 8..11:
    consumer warpgroup 1
```

也就是说：

```text
1 个 producer WG + 2 个 consumer WG
```

这是 Hopper FlashAttention kernel 的基本执行模型。

## 2. Kernel 开头: descriptor、shared storage、pipeline

### 2.1 TMA descriptor prefetch

源码：

```python
warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

if warp_idx == 0:
    for tma_atom in (tma_atom_Q, tma_atom_K, tma_atom_V, tma_atom_O):
        if const_expr(tma_atom is not None):
            cpasync.prefetch_descriptor(tma_atom)
```

`tma_atom_Q/K/V/O` 是 `__call__` 阶段创建好的 TMA copy atom。descriptor prefetch 的目的，是让 TMA descriptor 更早进入合适的 cache / descriptor path，减少后面第一次 TMA bulk copy 的延迟。

本 case 中：

```text
use_tma_Q = True
use_tma_KV = True
use_tma_O = True
```

所以 Q/K/V/O 都有 TMA atom，会被 warp 0 prefetch。

### 2.2 SharedStorage 分配

源码：

```python
smem = cutlass.utils.SmemAllocator()
storage = smem.allocate(SharedStorage)
```

`SharedStorage` 是 `__call__` 里通过 `_get_shared_storage_cls()` 生成的类型。本 case 的 shared storage 包含：

```text
mbar_ptr_Q: Q pipeline mbarriers
mbar_ptr_K: K pipeline mbarriers
mbar_ptr_V: V pipeline mbarriers
sV
sQ
sK
sP
```

本 case `mma_pv_is_rs=True`，所以 `sP` 的 cosize 为 0，实际不需要 P shared memory。

### 2.3 Pipeline 初始化

源码：

```python
ThreadCooperativeGroup = partial(pipeline.CooperativeGroup, pipeline.Agent.Thread)
tma_warp = ThreadCooperativeGroup(1)
load_threads = ThreadCooperativeGroup(self.num_threads_per_warp_group)
mma_warps = ThreadCooperativeGroup(self.num_mma_threads // cute.arch.WARP_SIZE)
```

含义：

```text
tma_warp:
    TMA path 下由 1 个 warp 发起 bulk copy。

load_threads:
    cp.async fallback 时，整个 producer warpgroup 的 128 threads 可以参与普通 async copy。

mma_warps:
    consumer 侧参与 WGMMA/softmax/PV 的 warps。
```

Q pipeline：

```python
if self.use_tma_Q:
    pipeline_q = PipelineTmaAsync.create(
        barrier_storage=mbar_ptr_Q,
        num_stages=1,
        producer_group=tma_warp,
        consumer_group=mma_warps,
        tx_count=self.tma_copy_bytes["Q"],
        defer_sync=True,
    )
else:
    pipeline_q = PipelineCpAsync.create(...)
```

K/V pipeline：

```python
if self.use_tma_KV:
    pipeline_k = PipelineTmaAsync.create(
        barrier_storage=storage.mbar_ptr_K.data_ptr(),
        num_stages=self.num_stages,
        producer_group=tma_warp,
        consumer_group=mma_warps,
        tx_count=self.tma_copy_bytes["K"],
        defer_sync=True,
    )
    pipeline_v = PipelineTmaAsync.create(
        barrier_storage=storage.mbar_ptr_V.data_ptr(),
        num_stages=self.num_stages,
        producer_group=tma_warp,
        consumer_group=mma_warps,
        tx_count=self.tma_copy_bytes["V"],
        defer_sync=True,
    )
else:
    pipeline_k = PipelineCpAsync.create(...)
    pipeline_v = PipelineCpAsync.create(...)
```

本 case：

```text
Q pipeline:
    TMA, 1 stage

K pipeline:
    TMA, 2 stages

V pipeline:
    TMA, 2 stages
```

`tx_count` 来自 `__call__` 中的：

```text
Q: 128 * 128 * 2 bytes = 32 KB
K: 128 * 128 * 2 bytes = 32 KB
V: 128 * 128 * 2 bytes = 32 KB
```

TMA mbarrier 需要知道每次 transaction 期望到达多少 bytes，consumer 才能正确 wait。

### 2.4 Shared memory tensor view

源码：

```python
sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)

if const_expr(not self.Q_in_regs):
    sV = storage.sV.get_tensor(sV_layout.outer, swizzle=sV_layout.inner)
else:
    sV = storage.sQ.get_tensor(...)

sVt = layout_utils.transpose_view(sV)

sP = None
if const_expr(sP_layout is not None):
    sP = storage.sP.get_tensor(sP_layout.outer, swizzle=sP_layout.inner)

sO = storage.sQ.get_tensor(sO_layout.outer, swizzle=sO_layout.inner, dtype=self.dtype)
```

本 case：

```text
sQ: [128, 128]
sK: [128, 128, 2]
sV: [128, 128, 2]
sVt: transpose view of sV
sP: None
sO: reuse sQ storage for output epilogue
```

为什么 `sVt` 要 transpose view：

```text
PV WGMMA 需要把 V 当成 [head_dim_v, tile_n] 的 B operand view。
原始 sV 是 [tile_n, head_dim_v, stage]。
transpose_view 不拷贝数据，只给 WGMMA 一个适合的 layout view。
```

为什么 `sO` 可以复用 `sQ`：

```text
计算结束后 Q tile 已经不会再被 WGMMA 使用。
epilogue 需要一个 shared memory staging buffer 把 acc_O 写回 gmem。
所以 sO 复用 sQ storage，减少 shared memory footprint。
```

## 3. BlockInfo、SeqlenInfo、AttentionMask、Scheduler

kernel 中会构造几个“类闭包”，producer 和 consumer 都用同一套逻辑。

### 3.1 BlockInfo

源码：

```python
block_info = BlockInfo(
    self.tile_m,
    self.tile_n,
    self.is_causal,
    self.is_local,
    False,  # is_split_kv
    window_size_left,
    window_size_right,
    qhead_per_kvhead_packgqa=self.qhead_per_kvhead if self.pack_gqa else 1,
)
```

`BlockInfo` 主要负责：

```text
给定一个 Q block，计算需要遍历哪些 K/V n_blocks。
```

核心函数：

```python
def get_n_block_min_max(self, seqlen_info, m_block, split_idx=0, num_splits=1):
    n_block_max = ceil_div(seqlen_info.seqlen_k, self.tile_n)
    if self.is_causal or (self.is_local and self.window_size_right is not None):
        m_idx_max = (m_block + 1) * self.tile_m
        if self.qhead_per_kvhead_packgqa > 1:
            m_idx_max = ceil_div(m_idx_max, self.qhead_per_kvhead_packgqa)
        n_idx = m_idx_max + seqlen_info.seqlen_k - seqlen_info.seqlen_q
        n_idx_right = n_idx if self.is_causal else n_idx + self.window_size_right
        n_block_max = min(n_block_max, ceil_div(n_idx_right, self.tile_n))
    ...
    return n_block_min, n_block_max
```

对 chunked prefill，`seqlen_k - seqlen_q` 就是 prefix offset：

```text
seqlen_q = 1024
seqlen_k = 3072
seqlen_k - seqlen_q = 2048
```

这表示当前 chunk 的第 0 个 query token 对应全局 position 2048。

### 3.2 SeqlenInfoCls

源码：

```python
SeqlenInfoCls = partial(
    SeqlenInfoQK.create,
    seqlen_q_static=mQ.shape[0] if not self.pack_gqa else mQ.shape[0][1],
    seqlen_k_static=mK.shape[0] if mPageTable is None else mK.shape[0] * mPageTable.shape[1],
    mCuSeqlensQ=mCuSeqlensQ,
    mCuSeqlensK=mCuSeqlensK,
    mSeqUsedQ=mSeqUsedQ,
    mSeqUsedK=mSeqUsedK,
    ...
)
```

paged KV 情况下：

```text
mK.shape[0] = page_size
mPageTable.shape[1] = max pages per seq
seqlen_k_static = page_size * max_pages_per_seq
```

但真实长度由 `mSeqUsedK` 决定：

```python
if mSeqUsedK is not None:
    seqlen_k = mSeqUsedK[batch_idx]
else:
    seqlen_k = seqlen_k_static
```

本 case：

```text
mSeqUsedK[0] = 3072
```

所以 kernel 内 `seqlen.seqlen_k = 3072`。

### 3.3 AttentionMaskCls

源码：

```python
AttentionMaskCls = partial(
    AttentionMask,
    self.tile_m,
    self.tile_n,
    window_size_left=window_size_left,
    window_size_right=window_size_right,
    qhead_per_kvhead_packgqa=self.qhead_per_kvhead if self.pack_gqa else 1,
)
```

consumer 每个 work tile 里会：

```python
mask = AttentionMaskCls(seqlen)
mask_fn = partial(
    mask.apply_mask,
    batch_idx=batch_idx,
    head_idx=head_idx,
    m_block=m_block,
    thr_mma=thr_mma_qk,
    mask_causal=self.is_causal,
    mask_local=self.is_local,
    aux_tensors=aux_tensors,
    fastdiv_mods=fastdiv_mods,
)
```

`mask.apply_mask` 会基于：

```text
m_block
n_block
seqlen_q
seqlen_k
causal/local/window
pack_gqa head mapping
```

把 `acc_S` 中不可见的位置设成 `-inf`。

### 3.4 TileSchedulerCls

源码：

```python
TileSchedulerCls = partial(TileScheduler.create, tile_sched_params)
```

producer 和 consumer 都创建自己的 scheduler：

```python
tile_scheduler = TileSchedulerCls()
work_tile = tile_scheduler.initial_work_tile_info()
```

它们拿到同样的 `(m_block, head_idx, batch_idx, split_idx)` 序列，保证 producer 加载的 tile 和 consumer 计算的 tile 对齐。

本 case 因为 dense causal：

```text
TileScheduler = SingleTileLPTScheduler
```

它会反转 block 顺序，优先处理后面的 Q block，因为后面的 Q block 可见更多 K/V，工作量更重。

## 4. Producer: `load`

producer 路径源码入口：

```python
if warp_idx < 4:
    setmaxregister_decrease(self.num_producer_regs)
    self.load(...)
```

`load` 的核心任务：

```text
对每个 work tile:
    1. 根据 scheduler 得到 m_block/head/batch
    2. load Q tile 到 sQ
    3. 根据 causal/local 得到 n_block range
    4. 按从右到左的顺序 load K/V tiles 到 sK/sV pipeline stages
    5. 如果 paged TMA，查 page_table 得到 physical page_idx
    6. 如果 non-TMA paged，使用 PagedKVManager 逐行查 page table 并 cp.async
```

### 4.1 哪些 warp 负责 load

源码：

```python
warp_idx_in_wg = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4
tidx, _, _ = cute.arch.thread_idx()

is_load_warp = warp_idx_in_wg == 0 or const_expr(not self.use_tma_KV or not self.use_tma_Q)
is_kv_load_warp = warp_idx_in_wg == 0 or const_expr(not self.use_tma_KV)
```

含义：

```text
TMA Q/KV path:
    producer WG 的 warp 0 发 TMA。

cp.async fallback:
    需要更多 producer threads 参与普通 async copy。
```

本 case Q/K/V 都走 TMA：

```text
is_load_warp = warp_idx_in_wg == 0
is_kv_load_warp = warp_idx_in_wg == 0
```

### 4.2 producer pipeline state

源码：

```python
q_producer_phase = Int32(1)
kv_producer_state = pipeline.make_pipeline_state(
    pipeline.PipelineUserType.Producer,
    self.num_stages,
)

tile_scheduler = TileSchedulerCls()
work_tile = tile_scheduler.initial_work_tile_info()
```

K/V 使用 `num_stages=2` 的 pipeline state：

```text
stage 0
stage 1
stage 0
stage 1
...
```

Q 只有 1 stage，但使用 phase bit 来区分 full/empty barrier 的交替。

### 4.3 当前 work tile 的 Q/KV head

源码：

```python
m_block, head_idx, batch_idx, _ = work_tile.tile_idx
seqlen = SeqlenInfoCls(batch_idx)

mQ_cur = seqlen.offset_batch_Q(mQ, batch_idx, dim=3)[None, None, head_idx]

head_idx_kv = (
    head_idx // self.qhead_per_kvhead
    if const_expr(not self.pack_gqa)
    else head_idx
)
```

本 case `pack_gqa=True`，所以：

```text
head_idx_kv = head_idx
```

因为 pack GQA 后 scheduler 的 head 维已经是 KV head 视角。

### 4.4 TMA load Q

源码：

```python
if self.use_tma_Q:
    gQ = cute.local_tile(mQ_cur, (self.tile_m, self.tile_hdim), (m_block, 0))
    load_Q, _, _ = copy_utils.tma_get_copy_fn(
        tma_atom_Q,
        0,
        cute.make_layout(1),
        gQ,
        sQ,
        single_stage=True,
    )
```

`gQ` 是当前 CTA 对应的 global Q tile：

```text
gQ: [tile_m, tile_hdim] = [128, 128]
```

真正发起 TMA：

```python
pipeline_q.producer_acquire_w_index_phase(0, q_producer_phase)
load_Q(tma_bar_ptr=pipeline_q.sync_object_full.get_barrier(0))
q_producer_phase ^= 1
```

注意这里把 pipeline Q 的 full barrier 指针传给 TMA load。TMA copy 完成后会 arrive 到这个 barrier，consumer 等待它。

### 4.5 TMA load K/V: non-paged 和 paged

TMA path：

```python
if self.use_tma_KV:
    if mPageTable is not None:
        mK_cur = mK[None, None, head_idx_kv, None]
        mV_cur = mV[None, None, head_idx_kv, None]
        gK = cute.local_tile(mK_cur, (self.tile_n, self.tile_hdim), (0, 0, None))
        gV = cute.local_tile(mV_cur, (self.tile_n, self.tile_hdimv), (0, 0, None))
    else:
        mK_cur = seqlen.offset_batch_K(mK, batch_idx, dim=3)[None, None, head_idx_kv]
        mV_cur = seqlen.offset_batch_K(mV, batch_idx, dim=3)[None, None, head_idx_kv]
        gK = cute.local_tile(mK_cur, (self.tile_n, self.tile_hdim), (None, 0))
        gV = cute.local_tile(mV_cur, (self.tile_n, self.tile_hdimv), (None, 0))
```

paged KV 下，`mK` 已经是：

```text
[page_size, dim, head_kv, num_pages]
```

所以：

```text
mK_cur = mK[:, :, head_idx_kv, :]
gK = local_tile(..., (tile_n, tile_hdim), (0, 0, None))
```

`None` 这一维保留 page 维，后面通过 `src_idx=page_idx` 选择 physical page。

创建 TMA load closure：

```python
tma_load_K_fn, _, _ = copy_utils.tma_get_copy_fn(
    tma_atom_K,
    0,
    cute.make_layout(1),
    gK,
    sK,
)
tma_load_K_fn = copy_utils.tma_producer_copy_fn(tma_load_K_fn, pipeline_k)

tma_load_V_fn, _, _ = copy_utils.tma_get_copy_fn(
    tma_atom_V,
    0,
    cute.make_layout(1),
    gV,
    sV,
)
tma_load_V_fn = copy_utils.tma_producer_copy_fn(tma_load_V_fn, pipeline_v)
```

### 4.6 paged TMA 的 page table 映射

在非 block sparse 路径里：

```python
n_block_min, n_block_max = block_info.get_n_block_min_max(seqlen, m_block)
n_block = n_block_max - 1 if self.use_tma_KV else max(n_block_max - 1, 0)

page_idx = (
    mPageTable[batch_idx, n_block]
    if const_expr(mPageTable is not None and self.use_tma_KV)
    else None
)
```

然后：

```python
pipeline_k.producer_acquire(kv_producer_state)
load_K(block=n_block, producer_state=kv_producer_state, page_idx=page_idx)
```

`load_KV` 里：

```python
if self.use_tma_KV:
    src_idx = block if const_expr(page_idx is None) else page_idx
    tma_load_fn(src_idx=src_idx, producer_state=producer_state)
else:
    paged_kv_manager.load_KV(...)
    cp_async_commit_group()
pipeline_kv.producer_commit(producer_state)
```

所以 paged TMA 的完整含义是：

```text
logical n_block
  -> page_idx = page_table[batch, n_block]
  -> tma_load_fn(src_idx=page_idx)
  -> 从 physical page 读一个 [tile_n, head_dim] tile
  -> 写入 sK/sV 的当前 pipeline stage
```

本 case `page_size=tile_n=128`，所以一个 logical n_block 正好对应一个 page。

### 4.7 producer 的 K/V overlap load 顺序

本 case：

```text
intra_wg_overlap=True
use_tma_KV=True
```

producer 会进入 overlap 分支：

```python
for i in range(n_block_max - 1 - n_block_min):
    n_block_prev = n_block_max - i - 1
    n_block = n_block_prev - 1

    page_idx = mPageTable[batch_idx, n_block]
    page_idx_prev = mPageTable[batch_idx, n_block_prev]

    kv_producer_state_prev = kv_producer_state.clone()
    kv_producer_state.advance()

    pipeline_k.producer_acquire(kv_producer_state)
    load_K(block=n_block, producer_state=kv_producer_state, page_idx=page_idx)

    pipeline_v.producer_acquire(kv_producer_state_prev)
    load_V(block=n_block_prev, producer_state=kv_producer_state_prev, page_idx=page_idx_prev)
```

这段看起来有点绕，但目标是让 producer 提前形成：

```text
K(next) 和 V(current) 错位加载
```

对应 consumer 侧：

```text
QK(next) overlap PV(current)
```

稳定态可以理解成：

```text
producer:
    load K block j-1
    load V block j

consumer:
    QK block j-1
    PV block j
```

因为 causal mainloop 从右往左扫 K blocks，所以这里的 next/current 在源码上表现为 `n_block_prev` 和 `n_block`。

## 5. Consumer: `mma`

consumer 路径源码入口：

```python
else:  # Consumer
    setmaxregister_increase(self.num_mma_regs)
    tidx = thread_idx - 128
    self.mma(...)
```

注意 `tidx = tidx - 128`，因为 consumer 的 thread indexing 从 producer 之后开始重新编号。

### 5.1 MMA thread slice 和 fragment partition

源码：

```python
warp_group_idx = cute.arch.make_warp_uniform(tidx // self.num_threads_per_warp_group)
warp_group_thread_layout = cute.make_layout(
    self.num_wg_mma,
    stride=self.num_threads_per_warp_group,
)

thr_mma_qk = tiled_mma_qk.get_slice(tidx)
wg_mma_qk = tiled_mma_qk.get_slice(warp_group_thread_layout(warp_group_idx))
wg_mma_pv = tiled_mma_pv.get_slice(warp_group_thread_layout(warp_group_idx))
```

本 case：

```text
num_wg_mma = 2
warp_group_idx = 0 or 1
```

每个 consumer WG 拿到自己的 Q rows：

```text
WG 0: Q tile 的前 64 rows
WG 1: Q tile 的后 64 rows
```

QK fragment partition：

```python
_, tSrQ, tSrK = sm90_utils.partition_fragment_ABC(
    wg_mma_qk,
    (self.tile_m, self.tile_n, self.tile_hdim),
    sQ,
    sK,
)

mma_qk_fn = partial(
    sm90_utils.gemm_zero_init,
    tiled_mma_qk,
    (self.tile_m, self.tile_n),
    tSrQ,
    tSrK,
)
```

PV fragment partition：

```python
acc_O, tOrP, tOrVt = sm90_utils.partition_fragment_ABC(
    wg_mma_pv,
    (self.tile_m, self.tile_hdimv, self.tile_n),
    sP,
    sVt,
)

mma_pv_fn = partial(sm90_utils.gemm_w_idx, tiled_mma_pv, acc_O, tOrP, tOrVt)
```

本 case `sP=None` 且 `mma_pv_is_rs=True`，所以 `tOrP` 对应 register source fragment，P 不落 shared。

### 5.2 Softmax 状态

源码：

```python
softmax = Softmax.create(
    softmax_scale_log2,
    num_rows=acc_O.shape[0][0] * acc_O.shape[1],
    softmax_scale=softmax_scale,
)
```

`Softmax.create`：

```python
row_max = cute.make_rmem_tensor(num_rows, Float32)
row_sum = cute.make_rmem_tensor(num_rows, Float32)
```

每个 consumer thread 持有自己负责的若干 rows 的：

```text
row_max
row_sum
```

这些状态贯穿多个 K/V n_blocks，用于 online softmax。

### 5.3 Work tile 循环

consumer 主循环：

```python
tile_scheduler = TileSchedulerCls()
work_tile = tile_scheduler.initial_work_tile_info()

while work_tile.is_valid_tile:
    m_block, head_idx, batch_idx, _ = work_tile.tile_idx
    seqlen = SeqlenInfoCls(batch_idx)
    mask = AttentionMaskCls(seqlen)
    ...
    n_block_min, n_block_max = block_info.get_n_block_min_max(seqlen, m_block)
    pipeline_q.consumer_wait_w_index_phase(0, q_consumer_phase)
    ...
    mainloop over n_blocks
    ...
    pipeline_q.consumer_release_w_index(0)
    ...
    finalize softmax
    epilogue
    tile_scheduler.advance_to_next_work()
```

consumer 先等 Q：

```python
pipeline_q.consumer_wait_w_index_phase(0, q_consumer_phase)
```

这保证 producer 的 Q TMA 已经完成，`sQ` 可以被 QK WGMMA 读取。

## 6. Mainloop: 从右往左扫 K/V blocks

### 6.1 为什么从右往左

non-block-sparse 主路径：

```python
n_block_min, n_block_max = block_info.get_n_block_min_max(seqlen, m_block)

if self.intra_wg_overlap:
    process_first_half_block(n_block=n_block_max - 1, ...)
...
n_block_max -= 1

for n_tile in range(...):
    n_block = n_block_max - 1 - n_tile
    mma_one_n_block(...)
```

也就是从 `n_block_max - 1` 往 `n_block_min` 扫。

原因包括：

```text
1. causal 下右侧边界块通常需要 mask。
2. 先处理最右侧块，可以把 seqlen/causal mask 逻辑单独拿出来。
3. intra_wg_overlap 需要 first half / last half 拆分，
   从右往左可以配合 producer 的 K(next)/V(current) 错位加载。
```

### 6.2 三类 n_block

mainloop 把 n_blocks 分成几类：

```text
1. first iteration:
   最右侧 n_block，通常需要 seqlen mask，也可能需要 causal mask。

2. causal/local mask iterations:
   causal 或 local 边界区域，需要 mask。

3. no-mask iterations:
   完全可见的 K/V blocks，不需要 causal/local/seqlen mask。

4. local-left mask iterations:
   local attention 左边界，本文 case 不涉及。
```

源码结构：

```python
# first iteration
process_first_half_block(n_block=n_block_max - 1, mask_seqlen=True)
n_block_max -= 1

# causal/local masked iterations
if self.is_causal or self.is_local:
    n_block_min_causal_local_mask = block_info.get_n_block_min_causal_local_mask(...)
    for n_tile in range(n_block_max - n_block_min_causal_local_mask):
        mma_one_n_block(..., mask_seqlen=False)
    n_block_max = min(n_block_max, n_block_min_causal_local_mask)

# no-mask iterations
n_block_min_before_local_mask = block_info.get_n_block_min_before_local_mask(...)
for n_tile in range(n_block_max - n_block_min_before_local_mask):
    mma_one_n_block(..., mask_seqlen=False)
```

这种拆分的收益：

```text
尽可能让中间大量 full blocks 走无 mask 快路径；
只在边界 blocks 做 mask。
```

## 7. QK、mask、online softmax、PV

### 7.1 非 overlap 版本: `mma_one_n_block`

源码主干：

```python
pipeline_k.consumer_wait(smem_pipe_read, pipeline_k.consumer_try_wait(smem_pipe_read))

# S = Q @ K.T
acc_S = mma_qk_fn(B_idx=smem_pipe_read.index, wg_wait=-1)
self.warp_scheduler_barrier_arrive()
warpgroup.wait_group(0)
pipeline_k.consumer_release(smem_pipe_read)

if score_mod_fn is not None:
    score_mod_fn(acc_S, n_block=n_block, seqlen=seqlen)
if mask_fn is not None:
    mask_fn(acc_S=acc_S, n_block=n_block)

row_scale = softmax.online_softmax(acc_S, is_first=is_first_n_block, check_inf=check_inf)

tOrP_acc = layout_utils.reshape_acc_to_frgA(acc_S)
tOrP_cur = tOrP if self.mma_pv_is_rs else make_rmem_tensor_like(...)
utils.cvt_f16(tOrP_acc, tOrP_cur)

if not self.mma_pv_is_rs:
    copy P to sP

softmax.rescale_O(acc_O, row_scale)

pipeline_v.consumer_wait(smem_pipe_read, pipeline_v.consumer_try_wait(smem_pipe_read))
self.warp_scheduler_barrier_sync()

# O += P @ V
mma_pv_fn(B_idx=smem_pipe_read.index, wg_wait=0)
pipeline_v.consumer_release(smem_pipe_read)
smem_pipe_read.advance()
```

语义：

```text
1. 等 K stage 可读。
2. QK WGMMA 得到 acc_S。
3. 对 acc_S 做 score_mod 和 mask。
4. online softmax，把 acc_S 原地变成 P。
5. 根据 row_scale rescale 历史 acc_O。
6. 等 V stage 可读。
7. PV WGMMA，把 P @ V 累加到 acc_O。
```

### 7.2 overlap 版本: `first_half_block_overlap`

开启 `intra_wg_overlap=True` 时，mainloop 被拆成 first half、middle overlap、last half。

first half：

```python
pipeline_k.consumer_wait(kv_consumer_state, pipeline_k.consumer_try_wait(kv_consumer_state))
acc_S = mma_qk_fn(B_idx=kv_consumer_state.index, wg_wait=0)
pipeline_k.consumer_release(kv_consumer_state)

if score_mod_fn is not None:
    score_mod_fn(acc_S, n_block=n_block, seqlen=seqlen)

mask_fn(acc_S, n_block=n_block, mask_seqlen=True)

row_scale = softmax.online_softmax(acc_S, is_first=is_first_block)

tOrP_acc = layout_utils.reshape_acc_to_frgA(acc_S)
tOrP_cur = tOrP if self.mma_pv_is_rs else make_rmem_tensor_like(...)
tOrP_cur.store(tOrP_acc.load().to(self.dtype))
```

它只完成：

```text
QK(first)
mask
online softmax
准备 P
```

不做 PV。PV 会在下一步和后续 QK 交叠。

### 7.3 overlap 稳定态: `mma_one_n_block_intrawg_overlap`

源码主干：

```python
smem_pipe_read_v = smem_pipe_read.clone()
smem_pipe_read.advance()

pipeline_k.consumer_wait(smem_pipe_read, pipeline_k.consumer_try_wait(smem_pipe_read))
self.warp_scheduler_barrier_sync()

# S(next) = Q @ K(next).T
acc_S = mma_qk_fn(B_idx=smem_pipe_read.index, wg_wait=-1)

if self.rescale_O_before_gemm:
    softmax.rescale_O(acc_O, scores_scale)

pipeline_v.consumer_wait(smem_pipe_read_v, pipeline_v.consumer_try_wait(smem_pipe_read_v))

# O += P(current) @ V(current)
mma_pv_fn(B_idx=smem_pipe_read_v.index, wg_wait=-1)

self.warp_scheduler_barrier_arrive()
warpgroup.wait_group(1)
pipeline_k.consumer_release(smem_pipe_read)

if score_mod_fn is not None:
    score_mod_fn(acc_S, n_block=n_block, seqlen=seqlen)
if mask_fn is not None:
    mask_fn(acc_S=acc_S, n_block=n_block)

row_scale = softmax.online_softmax(acc_S, check_inf=check_inf)

warpgroup.wait_group(0)
pipeline_v.consumer_release(smem_pipe_read_v)

tOrP_acc = layout_utils.reshape_acc_to_frgA(acc_S)
utils.cvt_f16(tOrP_acc, tOrP_cur)

if not self.rescale_O_before_gemm:
    softmax.rescale_O(acc_O, row_scale)
else:
    scores_scale.store(row_scale.load())
```

这段的核心是：

```text
QK(next) 和 PV(current) 同时在飞。
```

时间线可以理解为：

```text
已有:
    P(current) 已经由上一次 QK + softmax 准备好。

本次:
    1. 发起 QK(next)
    2. 等 V(current)
    3. 发起 PV(current)
    4. 等 QK(next) 完成
    5. 对 S(next) 做 mask/softmax，准备 P(next)
    6. 等 PV(current) 完成
```

所以一轮结束后：

```text
acc_O 已经累加了 current block 的贡献；
P(next) 已经准备好，留给下一轮 PV。
```

### 7.4 overlap last half

最后一个 half block：

```python
if self.rescale_O_before_gemm:
    softmax.rescale_O(acc_O, scores_scale)

pipeline_v.consumer_wait(kv_consumer_state, pipeline_v.consumer_try_wait(kv_consumer_state))
mma_pv_fn(B_idx=kv_consumer_state.index, zero_init=zero_init, wg_wait=0)
pipeline_v.consumer_release(kv_consumer_state)
kv_consumer_state.advance()
```

作用：

```text
把 first/middle 准备好的最后一个 P block 对应的 PV 做完。
```

这解释了为什么 overlap path 需要：

```text
first_half: QK + softmax only
middle: QK(next) overlap PV(current)
last_half: PV(last)
```

## 8. Online Softmax 实现

FlashAttention 的核心不是先 materialize 全部 S，再 softmax，而是对每个 K/V tile 做 online softmax。

### 8.1 状态

`Softmax` 类：

```python
@dataclass
class Softmax:
    scale_log2: Float32
    num_rows: Constexpr[int]
    row_max: cute.Tensor
    row_sum: cute.Tensor
    softmax_scale: Float32 | None = None
```

创建：

```python
row_max = cute.make_rmem_tensor(num_rows, Float32)
row_sum = cute.make_rmem_tensor(num_rows, Float32)
```

对每个 Q row，online softmax 维护：

```text
m_i = 当前已处理 blocks 的最大 score
l_i = 当前已处理 blocks 的 exp sum
acc_O_i = 当前已处理 blocks 的未归一化 O 累积
```

### 8.2 `online_softmax`

源码核心：

```python
acc_S_mn = layout_utils.reshape_acc_to_mn(acc_S)
row_scale = make_fragment_like(self.row_max, Float32)

for r in range(size(row_max)):
    acc_S_row = acc_S_mn[r, None].load()

    row_max_cur = fmax_reduce(
        acc_S_row,
        init_val=row_max[r] if not is_first else None,
    )
    row_max_cur = warp_reduction_max(row_max_cur, threads_in_group=4)

    row_max_prev = row_max[r]
    row_max[r] = row_max_cur

    if check_inf:
        row_max_cur = 0.0 if row_max_cur == -inf else row_max_cur

    acc_S_row_exp = exp2(acc_S_row * scale_log2 - row_max_cur * scale_log2)

    if is_first:
        acc_S_row_sum = fadd_reduce(acc_S_row_exp)
        row_scale[r] = 1.0
    else:
        row_scale[r] = exp2((row_max_prev - row_max_cur) * scale_log2)
        acc_S_row_sum = fadd_reduce(
            acc_S_row_exp,
            init_val=row_sum[r] * row_scale[r],
        )

    row_sum[r] = acc_S_row_sum
    acc_S_mn[r, None].store(acc_S_row_exp)

return row_scale
```

数学对应：

```text
m_new = max(m_old, max(S_block))
p_block = exp(S_block - m_new)
scale_old = exp(m_old - m_new)
l_new = l_old * scale_old + sum(p_block)
O_old *= scale_old
O_new = O_old + p_block @ V_block
```

源码里 `acc_S` 被原地改成 `exp(score - row_max)`，也就是 P block。

`row_scale` 用于 rescale 旧的 `acc_O`：

```python
softmax.rescale_O(acc_O, row_scale)
```

### 8.3 为什么用 `scale_log2`

`compute_softmax_scale_log2` 在 kernel 前把 scale 转成 log2 形式。kernel 内：

```python
exp2(acc_S_row * scale_log2 - row_max_cur_scaled)
```

这样可以用 `exp2` fast path。

数学上：

```text
exp(x * softmax_scale)
= exp2(x * softmax_scale * log2(e))
```

所以 `scale_log2 = softmax_scale * log2(e)`。

### 8.4 finalize

所有 N blocks 处理完后：

```python
row_scale = softmax.finalize(sink_val=sink_val)
softmax.rescale_O(acc_O, row_scale)
```

`finalize`：

```python
row_sum.store(warp_reduce(row_sum.load(), operator.add, width=4))

for r in range(size(row_sum)):
    row_scale[r] = rcp_approx(row_sum[r]) * final_scale
    row_sum[r] = (
        (row_max[r] * scale_log2 + log2(row_sum_cur)) * ln(2)
        if valid
        else -inf
    )
return row_scale
```

最终：

```text
acc_O *= 1 / row_sum
LSE = row_max * scale_log2 * ln(2) + log(row_sum)
```

注意 `row_sum` 在 finalize 后被改写成 LSE。

## 9. Mask 实现和 chunked prefill causal offset

### 9.1 causal offset

chunked prefill：

```text
cached prefix = 2048
current chunk = 1024
visible KV = 3072
```

对当前 chunk 内局部 query index `q_local`：

```text
global_q_pos = q_local + (seqlen_k - seqlen_q)
             = q_local + 2048
```

causal 条件：

```text
k_idx <= global_q_pos
```

源码中这个 offset 反复以：

```python
seqlen_k - seqlen_q
```

的形式出现。

`BlockInfo.get_n_block_min_max` 里：

```python
n_idx = m_idx_max + seqlen_info.seqlen_k - seqlen_info.seqlen_q
n_block_max = min(n_block_max, ceil_div(n_idx, tile_n))
```

`AttentionMask.apply_mask` 里有类似：

```python
causal_row_offset = (
    1 + self.seqlen_k - n_block * self.tile_n - self.seqlen_q - thr_col_offset
)
```

这类表达式都是在把当前 block 内局部 row/col 坐标映射到全局 causal 可见性。

### 9.2 为什么要分 block range 和 element mask

FlashAttention 先用 `BlockInfo` 粗粒度裁剪 n_blocks：

```text
完全在未来的 K/V blocks 不进入 mainloop。
```

然后对边界 block 用 `AttentionMask` 细粒度 mask：

```text
同一个 128-column tile 内，有些列可见，有些列不可见。
不可见的 score 写成 -inf。
```

两层结合：

```text
block-level skipping:
    少加载、少计算完全无效 K/V blocks。

element-level masking:
    正确处理 causal/local/seqlen 边界。
```

## 10. Epilogue: 写 O 和 LSE

SM90 forward 使用基类 `FlashAttentionForwardBase.epilogue`。

### 10.1 acc_O 到 shared memory

源码：

```python
rO = cute.make_fragment_like(acc_O, self.dtype)
rO.store(acc_O.load().to(self.dtype))

cute.arch.barrier(
    barrier_id=int(NamedBarrierFwd.Epilogue),
    number_of_threads=self.num_epilogue_threads,
)

smem_copy_atom_O = utils.get_smem_store_atom(...)
smem_thr_copy_O = cute.make_tiled_copy_C(smem_copy_atom_O, tiled_mma).get_slice(tidx)
taccOrO = smem_thr_copy_O.retile(rO)
taccOsO = smem_thr_copy_O.partition_D(sO)
cute.copy(smem_copy_atom_O, taccOrO, taccOsO)
```

这一步把 consumer register accumulator 中的 `acc_O` 转成 bf16，并写入 `sO`。

为什么不直接每个 thread 写 global：

```text
acc_O 的 fragment layout 是 WGMMA accumulator layout。
写 global 需要更规整的 layout 和更好的 vectorization。
所以先通过 smem copy atom 写入 sO，再由 TMA 或 gmem copy 写回 O。
```

### 10.2 写 LSE

源码：

```python
if mLSE is not None:
    mLSE_cur = seqlen.offset_batch_Q(mLSE, batch_idx, dim=2)[None, head_idx]
    if not self.pack_gqa:
        gLSE = cute.local_tile(mLSE_cur, (self.tile_m,), (m_block,))
        ...
        if taccOcO[0][1] == 0:
            for m in range(...):
                if valid row:
                    taccOgLSE[m, 0] = lse[m]
    else:
        pack_gqa.store_LSE(...)
```

只有对应 column 0 的 thread 写 LSE，因为每行只需要一个 LSE 值。

本 case如果 `return_lse=False` 且不需要 backward，`mLSE=None`，这一段跳过。

### 10.3 写 O: TMA path

本 case `use_tma_O=True`：

```python
if self.use_tma_O:
    cute.arch.fence_view_async_shared()
    cute.arch.barrier_arrive(
        barrier_id=int(NamedBarrierFwd.Epilogue),
        number_of_threads=self.num_epilogue_threads + WARP_SIZE,
    )
    gO = cute.local_tile(mO_cur, (self.tile_m, self.tile_hdimv), (m_block, 0))
    store_O, _, _ = copy_utils.tma_get_copy_fn(
        tma_atom_O,
        0,
        cute.make_layout(1),
        sO,
        gO,
        single_stage=True,
    )
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    if warp_idx == 4:
        cute.arch.barrier(...)
        store_O()
        cp_async_bulk_commit_group()
        cp_async_bulk_wait_group(0, read=True)
```

这里由 consumer 区域中的一个 warp 发起 TMA store。`sO` 已经由所有 consumer threads 写好，所以先 fence + barrier，再发 TMA S2G。

如果不能走 TMA O，则 fallback 到普通 tiled copy：

```python
gmem_thr_copy_O = gmem_tiled_copy_O.get_slice(tidx)
...
cute.copy(gmem_tiled_copy_O, tOrO, tOgO, pred=...)
```

## 11. Qwen Case 代入: 一个 work tile 发生了什么

现在把上面的源码执行流代入本 case。

### 11.1 静态配置

进入 kernel 前已经确定：

```text
tile_m = 128
tile_n = 128
tile_hdim = 128
tile_hdimv = 128
num_stages = 2
num_threads = 384
num_wg_mma = 2
use_tma_Q = True
use_tma_KV = True
use_tma_O = True
mma_pv_is_rs = True
intra_wg_overlap = True
pack_gqa = True
qhead_per_kvhead = 7
TileScheduler = SingleTileLPTScheduler
```

shared memory：

```text
sQ: 128 x 128
sK: 128 x 128 x 2
sV: 128 x 128 x 2
sO: 128 x 128, reusing sQ storage
sP: none
```

MMA：

```text
QK:
    each consumer WG: [64,128] x [128,128]^T -> [64,128]

PV:
    each consumer WG: [64,128] x [128,128] -> [64,128]
```

### 11.2 scheduler 选中一个 tile

假设 scheduler 选到：

```text
m_block = 7
head_idx = 0
batch_idx = 0
```

因为 LPT 会优先处理后面的 block，`m_block=7` 是当前 chunk 的最后 128 个 Q rows。

局部 Q rows：

```text
q_local = 896..1023
```

全局位置：

```text
q_global = 2048 + q_local
         = 2944..3071
```

所以这个 Q block 可以看几乎所有 K：

```text
K visible up to 3071
```

`seqlen.seqlen_k=3072`，`tile_n=128`：

```text
total K blocks = 24
```

`BlockInfo.get_n_block_min_max` 大致得到：

```text
n_block_min = 0
n_block_max = 24
```

mainloop 从：

```text
n_block = 23
```

开始往左扫到 0。

### 11.3 producer 对这个 tile 做什么

1. load Q：

```text
gQ = Q[m_block=7, head_idx=0, batch=0]
shape = [128,128]
TMA -> sQ
```

2. load 最右侧 K：

```text
n_block = 23
page_idx = page_table[0, 23]
TMA K page_idx -> sK[..., stage 0]
```

3. overlap preload：

```text
load K block 22 -> stage 1
load V block 23 -> stage 0
load K block 21 -> stage 0
load V block 22 -> stage 1
...
```

producer 的核心节奏是：

```text
K(next) / V(current) 错位进入 shared pipeline
```

### 11.4 consumer 对这个 tile 做什么

consumer 先等 Q：

```text
wait pipeline_q full
```

然后 first half：

```text
wait K block 23
QK block 23
mask block 23
online softmax, is_first=True
生成 P block 23, 留在 register
```

对 `m_block=7`，`n_block=23` 是 causal 边界 block。虽然这个 Q block 是最后一个 block，大部分可见，但 block 内仍然要对未来列做精细 mask。

中间 overlap 稳定态：

```text
QK block 22 overlaps PV block 23
softmax block 22 prepares P block 22

QK block 21 overlaps PV block 22
softmax block 21 prepares P block 21

...
```

每处理一个新 block，online softmax 更新：

```text
row_max
row_sum
row_scale
```

并用 `row_scale` rescale 旧的 `acc_O`，保持数值正确。

最后 last half：

```text
PV block 0
```

然后 finalize：

```text
acc_O *= 1 / row_sum
row_sum -> LSE
```

最后 epilogue：

```text
acc_O register -> sO
sO TMA store -> O[m_block=7, head_idx, batch]
```

### 11.5 对较早的 Q block 会怎样

假设：

```text
m_block = 0
q_local = 0..127
q_global = 2048..2175
```

这个 block 的最大可见 K：

```text
2175
```

对应 K block：

```text
ceil((2175 + 1) / 128) = 17
```

所以它不会扫满 24 个 blocks，而是只扫到大约：

```text
n_block_max = 17
```

这就是 causal LPT scheduler 要优先调度后面 block 的原因：后面 Q blocks 的 K/V mainloop 更长，工作量更大。

## 12. Kernel 的关键设计收益

### 12.1 不 materialize attention matrix

传统 attention：

```text
S = Q @ K^T
P = softmax(S)
O = P @ V
```

如果完整保存 S/P，内存开销是：

```text
seqlen_q x seqlen_k
```

FlashAttention kernel 只在寄存器里处理一个：

```text
[tile_m, tile_n]
```

score tile，并在线更新 `acc_O`。

### 12.2 TMA + shared pipeline 隐藏 memory latency

producer 发 TMA：

```text
global K/V page tile -> shared stage
```

consumer 等 stage ready 后做 WGMMA。

K/V double buffering：

```text
stage 0 / stage 1 交替
```

让 load 和 compute 更容易重叠。

### 12.3 QK/PV intra-warpgroup overlap

开启 `intra_wg_overlap=True` 后：

```text
QK(next) overlaps PV(current)
```

这减少 Tensor Core pipeline 空泡。

### 12.4 Register source P

本 case `mma_pv_is_rs=True`：

```text
S -> online_softmax -> P in register -> PV WGMMA
```

不需要：

```text
P register -> shared
shared -> PV WGMMA
```

减少 shared memory traffic，也减少一次同步。但对某些 tile/head_dim 组合，register pressure 可能过高，源码会选择 noRS。

### 12.5 causal block skipping + boundary mask

`BlockInfo` 先跳过完全不可见的 K blocks：

```text
block-level skip
```

`AttentionMask` 再处理边界 block 内的不可见元素：

```text
element-level mask
```

这样既正确，又避免对未来 blocks 做无效 load/compute。

### 12.6 LPT scheduler 降低尾部等待

causal 下后面的 Q blocks 工作量更大。LPT 反转 block 顺序：

```text
先处理重 blocks，再处理轻 blocks
```

减少最后少数重 block 拖尾。

## 13. 源码阅读顺序

建议按下面顺序读 kernel：

```text
1. flash_fwd_sm90.py::kernel
   看 kernel 参数、pipeline 初始化、producer/consumer 分流。

2. flash_fwd_sm90.py::load
   看 producer 如何 load Q/K/V，尤其 paged TMA 如何用 page_table。

3. flash_fwd_sm90.py::load_KV
   看 TMA path 和 cp.async fallback 的分界。

4. flash_fwd_sm90.py::mma
   看 consumer 如何 partition MMA fragments、创建 softmax 状态、进入 mainloop。

5. flash_fwd_sm90.py::first_half_block_overlap
   看 overlap path 的启动：QK + softmax + prepare P。

6. flash_fwd_sm90.py::mma_one_n_block_intrawg_overlap
   看 QK(next) overlap PV(current) 的稳定态。

7. flash_fwd_sm90.py::last_half_block_overlap
   看最后一个 PV 如何收尾。

8. softmax.py::Softmax.online_softmax / finalize / rescale_O
   看 online softmax 的数值逻辑。

9. block_info.py::BlockInfo.get_n_block_min_max
   看 causal/local 怎么裁剪 K/V block range。

10. mask.py::AttentionMask.apply_mask
   看 tile 内元素级 mask。

11. flash_fwd.py::FlashAttentionForwardBase.epilogue
   看 acc_O/LSE 如何写回。

12. paged_kv.py::PagedKVManager
   看 page_size != tile_n 时的 cp.async paged fallback。
```

## 14. 总结

Hopper FlashAttention forward kernel 可以用一句话概括：

```text
一个 CTA 处理一个 Q tile / head / batch work tile，
producer warpgroup 负责把 Q/K/V tile 搬到 shared，
consumer warpgroups 用 WGMMA 做 QK 和 PV，
中间用 online softmax 跨 K/V blocks 维护 row_max、row_sum 和 acc_O，
最后 epilogue 写回 O/LSE。
```

对 Qwen2.5 风格 case：

```text
tile_m=128, tile_n=128
page_size=128
head_dim=128
GQA ratio=7
causal chunked prefill
```

kernel 的执行形态是：

```text
1 producer WG:
    TMA load Q
    paged TMA load K/V
    K(next) / V(current) 错位加载

2 consumer WGs:
    每个 WG 负责 64 Q rows
    QK WGMMA
    causal/seqlen mask
    online softmax
    P in register
    PV WGMMA
    QK(next) overlap PV(current)

epilogue:
    acc_O normalize
    optional LSE
    acc_O -> sO
    TMA store O
```

这就是 FlashAttention 的 kernel 实现核心：不是单纯把 attention 分块，而是把分块、TMA、WGMMA、mbarrier pipeline、online softmax、寄存器分配、scheduler 和 paged KV 映射组织成一条稳定的 Hopper 执行流水线。
