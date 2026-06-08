# Hopper FlashAttention Pipeline 与 mbarrier 技术解析

本文围绕 Hopper / SM90 FlashAttention forward kernel 中的 pipeline 同步机制展开，重点解释 PTX `mbarrier`、CUTLASS/CuTeDSL pipeline 组件，以及 FlashAttention 源码如何将二者组合成 Q/K/V producer-consumer pipeline。

文档结构分为三章：

1. [mbarrier 机制详解](#1-mbarrier-机制详解)
2. [CUTLASS pipeline 组件拆解](#2-cutlass-pipeline-组件拆解)
3. [Hopper FlashAttention 的 pipeline 应用](#3-hopper-flashattention-的-pipeline-应用)

参考对象：

```text
NVIDIA PTX ISA:
    Parallel Synchronization and Communication Instructions
    mbarrier
    barrier / fence / elect.sync

FlashAttention CuTe 源码:
    flash-attention-src/flash_attn/cute/flash_fwd_sm90.py
    flash-attention-src/flash_attn/cute/pipeline.py
    flash-attention-src/flash_attn/cute/paged_kv.py

CUTLASS/CuTeDSL pipeline:
    cutlass.pipeline.PipelineTmaAsync
    cutlass.pipeline.PipelineCpAsync
    cutlass.pipeline.PipelineAsync
    cutlass.pipeline.PipelineState
    cutlass.pipeline.CooperativeGroup
```

本文讨论的 FlashAttention case 与前文保持一致：

```text
GPU: SM90 / Hopper
dtype: bf16
head_dim = 128
head_dim_v = 128
tile_m = 128
tile_n = 128
num_stages = 2
Q/K/V load path = TMA
Q pipeline stages = 1
K/V pipeline stages = 2
consumer warpgroup count = 2
producer warpgroup count = 1
```

## 1. mbarrier 机制详解

### 1.1 mbarrier 的定义和使用范围

PTX `mbarrier` 是位于 shared memory 中的 barrier object，用于表达线程到达、phase 完成检测，以及异步内存事务完成追踪。

PTX 文档对 `mbarrier` 的用途可以归纳为：

```text
1. 同一 CTA 内任意线程子集的同步。
2. CTA cluster 内的单向同步。
3. 跟踪 asynchronous memory operation 的完成，并在完成后使结果对等待线程可见。
```

`mbarrier` 对象本身是 shared memory 中的 opaque object，基本约束是：

```text
type: .b64
alignment: 8 bytes
state space: .shared
```

典型声明形式：

```ptx
.shared .b64 bar;
```

`mbarrier` 初始化后，不能被普通 load/store 当作普通 shared memory 数据使用。除 `mbarrier.init` 外，对未初始化的 mbarrier object 执行其他 mbarrier 操作属于 undefined behavior；对已初始化的 mbarrier object 执行非 mbarrier 操作同样属于 undefined behavior。

### 1.2 生命周期

`mbarrier` 的生命周期为：

```text
mbarrier.init
    -> zero or more synchronization phases
    -> mbarrier.inval
```

初始化语义：

```ptx
mbarrier.init.shared.b64 [bar], totalCount;
```

其中 `totalCount` 表示 initial expected arrival count。初始化后，mbarrier 进入初始 phase，其内部状态包含：

```text
current phase
pending arrival count
expected arrival count for next phase
tx-count
```

失效语义：

```ptx
mbarrier.inval.shared.b64 [bar];
```

`mbarrier.inval` 后，该 shared memory 位置可以重新用于其他目的。

### 1.3 phase 模型

`mbarrier` 以 phase 为单位工作。每个 phase 都有独立的完成条件：

```text
phase complete iff:
    pending arrival count == 0
    and tx-count == 0
```

当 phase 完成后，mbarrier 进入下一 phase，并将 pending arrival count 重新初始化为 expected arrival count。

phase 模型是 pipeline 中循环复用 shared memory stage 的基础。例如 K/V double-buffer：

```text
iteration 0 -> stage 0, phase 0
iteration 1 -> stage 1, phase 0
iteration 2 -> stage 0, phase 1
iteration 3 -> stage 1, phase 1
```

stage index 只能表示使用哪块 shared memory buffer；phase 用于区分同一 stage 在不同迭代中的同步状态。如果没有 phase，consumer 可能错误地将上一轮的 full signal 解释为当前轮数据可读，producer 也可能错误地将上一轮的 empty signal 解释为当前轮 buffer 可写。

PTX wait 指令既支持基于 `mbarrier.arrive` 返回的 `state` 检测，也支持 `.parity` 形式基于 phase parity 检测。phase parity 是 phase 的奇偶：

```text
even phase -> parity 0
odd phase  -> parity 1
```

### 1.4 arrive 操作

`mbarrier.arrive` 表示一个参与方到达当前 phase。

基本形式：

```ptx
mbarrier.arrive.shared.b64 state, [bar];
```

语义：

```text
pending arrival count -= 1
返回当前 phase 的 state
```

`arrive` 不阻塞执行线程。线程执行 arrive 后可以继续执行后续指令。后续线程可以用返回的 `state` 或 phase parity 检查当前 phase 是否完成。

带 count 的形式可以一次减少多个 arrival：

```ptx
mbarrier.arrive.shared.b64 state, [bar], count;
```

### 1.5 wait 操作

PTX 提供两类等待/检测：

```ptx
mbarrier.test_wait.shared.b64 p, [bar], state;
mbarrier.try_wait.shared.b64  p, [bar], state;
```

语义区别：

```text
test_wait:
    非阻塞检测 phase 是否完成。
    若完成，predicate 返回 true；否则返回 false。

try_wait:
    可能阻塞或挂起执行线程。
    phase 完成或达到系统相关时间限制后返回。
```

常见轮询形式：

```ptx
wait_loop:
    mbarrier.test_wait.shared.b64 p, [bar], state;
    @!p bra wait_loop;
```

在 acquire/release 语义下，mbarrier wait 还承担内存可见性约束。对于 producer-consumer 场景，consumer 在 acquire wait 完成后，可以观察到 producer 在 release arrive 之前对 shared memory 的写入，以及被 mbarrier 跟踪的异步内存事务结果。

### 1.6 arrive_drop 操作

`mbarrier.arrive_drop` 同时完成当前 phase arrival，并从后续 phase 的 expected arrival count 中移除当前参与方。

形式：

```ptx
mbarrier.arrive_drop.shared.b64 state, [bar];
```

语义：

```text
current phase:
    pending arrival count -= count

subsequent phases:
    expected arrival count -= count
```

该操作适用于某些参与线程或线程组后续不再参与该 mbarrier 的场景。

### 1.7 tx-count 与异步内存事务

`tx-count` 是 `mbarrier` 相比普通 barrier 的核心扩展。它用于跟踪当前 phase 关联的异步内存事务。

相关 PTX 操作：

```ptx
mbarrier.expect_tx.shared.b64   [bar], txCount;
mbarrier.complete_tx.shared.b64 [bar], completeCount;
```

语义：

```text
expect_tx:
    tx-count += txCount

complete_tx:
    tx-count -= completeCount
```

因此 phase completion 条件是：

```text
pending arrival count == 0
and tx-count == 0
```

对于 TMA/cp.async 这类 asynchronous memory operation，仅等待 producer 线程执行到某个位置是不充分的。producer 发起异步 copy 后，执行线程可能继续向前执行，但数据仍在后台传输。consumer 必须等待异步 copy 完成后才能安全读取 shared memory。`mbarrier` 的 `tx-count` 正是用于表达这一约束。

PTX 还提供组合形式：

```ptx
mbarrier.arrive.expect_tx.shared.b64 state, [bar], txCount;
```

其语义可视为：

```text
expect_tx(txCount)
arrive()
```

在 TMA pipeline 中，`txCount` 常对应一次 TMA transaction 覆盖的字节数。例如 FlashAttention 中一个 bf16 K tile：

```text
tile_n = 128
head_dim = 128
element_size = 2 bytes

K tile bytes = 128 * 128 * 2 = 32768 bytes
```

这一数值会成为 CUTLASS `PipelineTmaAsync` 的 `tx_count` 参数。

### 1.8 mbarrier 与传统 barrier 的技术区别

| 项目 | `bar.sync` / CTA barrier | `mbarrier` |
| --- | --- | --- |
| 存储位置 | CTA barrier resource | shared memory object |
| 数量限制 | 固定 CTA barrier resource 数量 | 受 shared memory 容量限制 |
| arrive/wait | 通常绑定 | 分离 |
| phase | 隐式或简单 | 显式 phase/parity 模型 |
| 异步事务追踪 | 不直接追踪 | 通过 tx-count 追踪 |
| 典型用途 | CTA 全体或子集同步 | producer-consumer pipeline、TMA/cp.async completion tracking |

对于 Hopper FlashAttention，关键需求不是单次 CTA 同步，而是多 stage shared memory buffer 的循环复用，以及 TMA 数据到达 shared memory 的精确同步。因此 `mbarrier` 是更合适的同步原语。

## 2. CUTLASS pipeline 组件拆解

### 2.1 CUTLASS pipeline 对 mbarrier 的抽象目标

CUTLASS/CuTeDSL pipeline 的目标是将底层 mbarrier 协议封装为统一的 producer-consumer API：

```text
producer_acquire
producer_commit
consumer_wait
consumer_release
```

这些 API 隐含了两组同步对象：

```text
full barrier:
    producer 写入或发起异步 copy 后标记 full；
    consumer 等待 full 后读取 shared memory。

empty barrier:
    consumer 读取完成后标记 empty；
    producer 等待 empty 后复用 shared memory stage。
```

因此 CUTLASS pipeline 与 mbarrier 的关系可以表示为：

```text
PTX mbarrier:
    phase
    arrive
    wait
    tx-count

CUTLASS pipeline:
    full/empty barrier
    PipelineState(index, phase)
    producer/consumer API
    PipelineTmaAsync / PipelineCpAsync
```

### 2.2 full / empty barrier 协议

对每个 pipeline stage，存在两个逻辑状态：

```text
empty:
    shared memory stage 可由 producer 写入。

full:
    shared memory stage 可由 consumer 读取。
```

producer 侧协议：

```text
producer_acquire(state):
    等待 empty barrier 对应 phase 完成。
    确认该 stage 可写。

producer_commit(state):
    标记 full barrier。
    通知 consumer 可等待该 stage。
```

consumer 侧协议：

```text
consumer_wait(state):
    等待 full barrier 对应 phase 完成。
    确认该 stage 可读。

consumer_release(state):
    标记 empty barrier。
    通知 producer 该 stage 可复用。
```

对于 TMA pipeline，`producer_commit` 或更准确地说 full barrier 的完成，不仅依赖 producer arrival，也依赖 TMA transaction 的 tx-count 清零。

### 2.3 PipelineState

CUTLASS `PipelineState` 用于表示当前 pipeline 操作的 stage 和 phase。

概念字段：

```text
stages:
    pipeline stage 数量。

index:
    当前 stage index。

phase:
    当前 stage 使用轮次。
```

对于 `num_stages=2`：

```text
iteration: 0  1  2  3  4
index:     0  1  0  1  0
phase:     0  0  1  1  2
```

stage index 决定访问哪块 shared memory buffer，phase 决定等待哪一轮的 full/empty barrier。

### 2.4 CooperativeGroup

CUTLASS pipeline 创建时需要指定 producer 和 consumer 的参与线程组。

FlashAttention SM90 中使用：

```python
ThreadCooperativeGroup = partial(pipeline.CooperativeGroup, pipeline.Agent.Thread)

tma_warp = ThreadCooperativeGroup(1)
load_threads = ThreadCooperativeGroup(self.num_threads_per_warp_group)
mma_warps = ThreadCooperativeGroup(self.num_mma_threads // cute.arch.WARP_SIZE)
```

含义：

```text
tma_warp:
    TMA path 下的 producer group。
    通常由一个 warp 发起 TMA 操作。

load_threads:
    cp.async fallback path 下的 producer group。
    通常由一个 producer warpgroup 的 128 threads 参与拷贝。

mma_warps:
    consumer group。
    对应执行 QK/PV WGMMA 的 consumer warps。
```

这些 group 信息会影响 mbarrier expected arrival count、arrive mask、以及 full/empty barrier 的参与者协议。

### 2.5 PipelineTmaAsync

`PipelineTmaAsync` 用于 TMA-based producer-consumer pipeline。它与 mbarrier 的关系最直接，因为 TMA completion 需要通过 mbarrier `tx-count` 表达。

FlashAttention 中创建 TMA pipeline：

```python
pipeline_k = pipeline_custom.PipelineTmaAsync.create(
    barrier_storage=storage.mbar_ptr_K.data_ptr(),
    num_stages=self.num_stages,
    producer_group=tma_warp,
    consumer_group=mma_warps,
    tx_count=self.tma_copy_bytes["K"],
    defer_sync=True,
)
```

参数含义：

```text
barrier_storage:
    shared memory 中 mbarrier 对象数组的地址。

num_stages:
    pipeline stage 数。K/V 为 2，Q 为 1。

producer_group:
    发起 TMA 的线程组。

consumer_group:
    等待并消费 shared memory tile 的线程组。

tx_count:
    每个 TMA transaction 期望完成的数据量。
```

从 mbarrier 角度看，`PipelineTmaAsync` 的 full barrier 需要跟踪：

```text
producer arrival
TMA transaction completion
phase parity
```

### 2.6 PipelineCpAsync

`PipelineCpAsync` 用于普通 `cp.async` path。

FlashAttention 中当 K/V 不能用 TMA 时，例如 paged KV 的 `page_size != tile_n`，会走 `PagedKVManager + cp.async`：

```python
pipeline_k = pipeline_custom.PipelineCpAsync.create(
    barrier_storage=storage.mbar_ptr_K.data_ptr(),
    num_stages=self.num_stages,
    producer_group=load_threads,
    consumer_group=mma_warps,
    defer_sync=True,
    elect_one_release=True,
    syncwarp_before_release=False,
)
```

该路径中，producer 执行：

```python
paged_kv_manager.load_KV(...)
cute.arch.cp_async_commit_group()
pipeline_kv.producer_commit(producer_state)
```

相比 TMA path，cp.async path 中每个 thread/warp 参与普通异步拷贝，pipeline 的 full/empty barrier 仍然承担 stage 可读/可写协议，但数据完成机制与 TMA transaction barrier 不完全相同。

### 2.7 FlashAttention 的 `pipeline_custom`

FlashAttention 在：

```text
flash_attn/cute/pipeline.py
```

对 CUTLASS pipeline 做了轻量扩展。扩展目标不是重新实现 mbarrier，而是适配 FlashAttention kernel 的调用方式和性能需求。

#### 2.7.1 PipelineStateSimple

源码：

```python
class PipelineStateSimple:
    def __init__(self, stages: int, phase_index: Int32):
        self._stages = stages
        self._phase_index = phase_index

    @property
    def index(self) -> Int32:
        if const_expr(self._stages == 1):
            return Int32(0)
        else:
            return self._phase_index % self._stages

    @property
    def phase(self) -> Int32:
        if const_expr(self._stages == 1):
            return self._phase_index
        else:
            return self._phase_index // self._stages

    def advance(self):
        if const_expr(self._stages == 1):
            self._phase_index ^= 1
        else:
            self._phase_index += 1
```

该类将 stage index 和 phase 信息压缩到一个 `phase_index` 中。对 power-of-two stages，index/phase 推导可以转化为简单整数运算。

#### 2.7.2 make_pipeline_state

源码：

```python
def make_pipeline_state(type: PipelineUserType, stages: int):
    if type is PipelineUserType.Producer:
        return PipelineStateSimple(stages, Int32(stages))
    elif type is PipelineUserType.Consumer:
        return PipelineStateSimple(stages, Int32(0))
```

producer 和 consumer 使用不同初始 phase，是为了匹配 full/empty barrier 的初始状态：

```text
初始 shared memory stage 是 empty。
producer 首次 acquire 应该能通过 empty barrier。
consumer 首次 wait 应该等待 full barrier。
```

因此 producer/consumer 的 phase parity 不能简单设成相同值。

#### 2.7.3 `_w_index_phase` 方法

FlashAttention 增加了基于 index/phase 的便捷方法：

```python
producer_acquire_w_index_phase(index, phase)
producer_commit_w_index(index)
consumer_wait_w_index_phase(index, phase)
consumer_release_w_index(index)
```

源码模式：

```python
state = _make_state(index, phase)
self.producer_acquire(state, ...)
```

该封装主要用于 Q pipeline。Q pipeline 是 single-stage，但仍然需要 phase bit 区分不同 work tile：

```python
q_producer_phase = Int32(1)
pipeline_q.producer_acquire_w_index_phase(0, q_producer_phase)
...
q_producer_phase ^= 1
```

#### 2.7.4 elect_one 封装

FlashAttention 对部分 `producer_commit` / `consumer_release` 操作增加 `elect_one` 包装：

```python
with cute.arch.elect_one():
    parent_method(self, state)
```

PTX `elect.sync` 从 membermask 指定的线程集合中确定性选出一个 leader thread。该机制用于避免多个线程重复对同一 barrier 执行 arrive。

在 cp.async fallback path 中，FlashAttention 创建 pipeline 时设置：

```python
elect_one_release=True
syncwarp_before_release=False
```

其目的在于让每个 warp 或线程组中只有被选中的线程执行 release arrive，从而与 barrier expected arrival count 保持一致并减少同步开销。

#### 2.7.5 PipelineTmaAsync.producer_acquire 扩展

FlashAttention 改写了 `PipelineTmaAsync.producer_acquire`：

```python
def producer_acquire(
    self,
    state,
    try_acquire_token=None,
    extra_tx_count=0,
):
    if try_acquire_token is None or try_acquire_token == 0:
        self.sync_object_empty.wait(state.index, state.phase)

    if extra_tx_count == 0:
        self.sync_object_full.arrive(state.index, self.producer_mask)
    else:
        tx_count = self.sync_object_full.tx_count + extra_tx_count
        self.sync_object_full.arrive_and_expect_tx(state.index, tx_count)
```

该方法直接体现了 CUTLASS pipeline 与 mbarrier 机制之间的联系：

```text
sync_object_empty.wait:
    对应 producer 等待 empty barrier。
    保证该 stage 已被 consumer release。

sync_object_full.arrive / arrive_and_expect_tx:
    对应 producer 对 full barrier 执行 arrive。
    若存在 tx-count，则使用 mbarrier arrive.expect_tx 语义。
```

对于 TMA path，`arrive_and_expect_tx` 是关键操作，因为 consumer 等待 full barrier 时必须等待 TMA 数据到达 shared memory。

## 3. Hopper FlashAttention 的 pipeline 应用

### 3.1 kernel 中的 pipeline 初始化

FlashAttention SM90 forward kernel 中，pipeline 创建发生在：

```text
flash_fwd_sm90.py::kernel
```

关键源码：

```python
ThreadCooperativeGroup = partial(pipeline.CooperativeGroup, pipeline.Agent.Thread)
tma_warp = ThreadCooperativeGroup(1)
load_threads = ThreadCooperativeGroup(self.num_threads_per_warp_group)
mma_warps = ThreadCooperativeGroup(self.num_mma_threads // cute.arch.WARP_SIZE)
```

Q pipeline：

```python
if const_expr(self.use_tma_Q):
    pipeline_q = pipeline_custom.PipelineTmaAsync.create(
        barrier_storage=mbar_ptr_Q,
        num_stages=1,
        producer_group=tma_warp,
        consumer_group=mma_warps,
        tx_count=self.tma_copy_bytes["Q"],
        defer_sync=True,
    )
else:
    pipeline_q = pipeline_custom.PipelineCpAsync.create(...)
```

K/V pipeline：

```python
if const_expr(self.use_tma_KV):
    pipeline_k = pipeline_custom.PipelineTmaAsync.create(
        barrier_storage=storage.mbar_ptr_K.data_ptr(),
        num_stages=self.num_stages,
        producer_group=tma_warp,
        consumer_group=mma_warps,
        tx_count=self.tma_copy_bytes["K"],
        defer_sync=True,
    )
    pipeline_v = pipeline_custom.PipelineTmaAsync.create(
        barrier_storage=storage.mbar_ptr_V.data_ptr(),
        num_stages=self.num_stages,
        producer_group=tma_warp,
        consumer_group=mma_warps,
        tx_count=self.tma_copy_bytes["V"],
        defer_sync=True,
    )
else:
    pipeline_k = pipeline_custom.PipelineCpAsync.create(...)
    pipeline_v = pipeline_custom.PipelineCpAsync.create(...)
```

该代码把第一章的 mbarrier 机制具体化为三条 pipeline：

```text
pipeline_q:
    single-stage pipeline
    full barrier tracks Q TMA completion

pipeline_k:
    two-stage pipeline
    full barrier tracks K TMA completion

pipeline_v:
    two-stage pipeline
    full barrier tracks V TMA completion
```

### 3.2 TMA transaction bytes 与 mbarrier tx-count

在 `__call__` 阶段，FlashAttention 计算：

```python
self.tma_copy_bytes = {
    name: cute.size_in_bytes(mX.element_type, cute.select(layout, mode=[0, 1]))
    for name, mX, layout in [
        ("Q", mQ, self.sQ_layout),
        ("K", mK, self.sK_layout),
        ("V", mV, self.sV_layout),
    ]
}
```

对 Qwen-style case：

```text
Q tile: 128 x 128 x bf16 = 32768 bytes
K tile: 128 x 128 x bf16 = 32768 bytes
V tile: 128 x 128 x bf16 = 32768 bytes
```

这些值传入：

```python
PipelineTmaAsync.create(..., tx_count=self.tma_copy_bytes["K"])
```

这正是 PTX `mbarrier.expect_tx` / `arrive.expect_tx` 中 tx-count 的上层来源。

因此源码链路是：

```text
shared memory layout
    -> tile bytes
    -> PipelineTmaAsync.tx_count
    -> mbarrier full barrier expect_tx
    -> consumer_wait 等待 TMA completion
```

### 3.3 producer 使用 Q pipeline

producer 侧初始化：

```python
q_producer_phase = Int32(1)
```

TMA Q path：

```python
pipeline_q.producer_acquire_w_index_phase(0, q_producer_phase)
load_Q(tma_bar_ptr=pipeline_q.sync_object_full.get_barrier(0))
q_producer_phase ^= 1
```

对应 mbarrier 语义：

```text
producer_acquire_w_index_phase:
    等 Q stage 的 empty barrier。
    对 Q full barrier 执行 arrive 或 arrive.expect_tx。

load_Q:
    发起 TMA，并将 TMA completion 关联到 Q full barrier。

q_producer_phase ^= 1:
    single-stage pipeline 的 phase parity 翻转。
```

Q pipeline 是 single-stage，但仍需 phase，因为不同 work tile 会复用同一个 `sQ` buffer。

### 3.4 producer 使用 K/V pipeline

K/V producer state：

```python
kv_producer_state = pipeline.make_pipeline_state(
    pipeline.PipelineUserType.Producer,
    self.num_stages,
)
```

第一次 K load：

```python
pipeline_k.producer_acquire(kv_producer_state)
load_K(block=n_block, producer_state=kv_producer_state, page_idx=page_idx)
```

`load_KV`：

```python
if const_expr(self.use_tma_KV):
    src_idx = block if const_expr(page_idx is None) else page_idx
    tma_load_fn(src_idx=src_idx, producer_state=producer_state)
else:
    paged_kv_manager.load_KV(block, sX[None, None, producer_state.index], K_or_V)
    cute.arch.cp_async_commit_group()
pipeline_kv.producer_commit(producer_state)
```

对于 TMA path：

```text
producer_acquire:
    等 empty。
    对 full barrier 建立 tx-count。

tma_load_fn:
    发起 TMA。
    src_idx 指定 global tile 或 paged KV physical page。

producer_commit:
    完成 pipeline 协议中的 producer commit 部分。
    数据真正可读仍由 full barrier phase completion 决定。
```

对于 paged KV，`page_idx` 来源于：

```python
page_idx = mPageTable[batch_idx, n_block]
```

因此：

```text
page_table:
    决定 TMA 从哪个 physical page 读取。

mbarrier:
    决定对应 shared stage 的数据何时可读。
```

二者职责不同，但在 TMA load 操作中汇合。

### 3.5 producer 的 K(next) / V(current) 调度

FlashAttention 开启 `intra_wg_overlap=True` 时，producer 采用错位加载：

```python
kv_producer_state_prev = kv_producer_state.clone()
kv_producer_state.advance()

pipeline_k.producer_acquire(kv_producer_state)
load_K(block=n_block, producer_state=kv_producer_state, page_idx=page_idx)

pipeline_v.producer_acquire(kv_producer_state_prev)
load_V(block=n_block_prev, producer_state=kv_producer_state_prev, page_idx=page_idx_prev)
```

该调度依赖两个条件：

```text
1. PipelineState 能区分 current stage 和 next stage。
2. full/empty mbarrier 能保证 stage 复用安全。
```

源码中的 `clone` 和 `advance` 对应：

```text
kv_producer_state_prev:
    current V stage

kv_producer_state after advance:
    next K stage
```

由此 producer 形成：

```text
load K(next)
load V(current)
```

### 3.6 consumer 使用 Q pipeline

consumer 侧初始化：

```python
q_consumer_phase = Int32(0)
```

等待 Q：

```python
pipeline_q.consumer_wait_w_index_phase(0, q_consumer_phase)
```

该 wait 对应：

```text
等待 Q full barrier 当前 phase 完成。
若 Q 走 TMA，则意味着 Q TMA transaction 已完成。
```

使用完 Q 后：

```python
pipeline_q.consumer_release_w_index(0)
q_consumer_phase ^= 1
```

对应：

```text
consumer 对 Q empty barrier arrive。
producer 后续可以复用 sQ。
```

### 3.7 consumer 使用 K/V pipeline

first half block：

```python
pipeline_k.consumer_wait(kv_consumer_state, pipeline_k.consumer_try_wait(kv_consumer_state))
acc_S = mma_qk_fn(B_idx=kv_consumer_state.index, wg_wait=0)
pipeline_k.consumer_release(kv_consumer_state)
```

语义：

```text
consumer_wait:
    等 K full barrier。
    确保 K tile 已经在 sK[stage]。

QK WGMMA:
    读取 sQ 和 sK[stage]。

consumer_release:
    标记 K stage empty。
```

middle overlap block：

```python
smem_pipe_read_v = smem_pipe_read.clone()
smem_pipe_read.advance()

pipeline_k.consumer_wait(smem_pipe_read, ...)
acc_S = mma_qk_fn(B_idx=smem_pipe_read.index, wg_wait=-1)

pipeline_v.consumer_wait(smem_pipe_read_v, ...)
mma_pv_fn(B_idx=smem_pipe_read_v.index, wg_wait=-1)

pipeline_k.consumer_release(smem_pipe_read)
...
pipeline_v.consumer_release(smem_pipe_read_v)
```

该代码对应：

```text
K(next):
    使用 advance 后的 state。

V(current):
    使用 clone 保存的 state。

QK(next) and PV(current):
    可重叠执行。
```

这段逻辑说明 pipeline 不只是“同步工具”，还直接塑造了 kernel 的执行调度。

### 3.8 TMA path 与 cp.async path 在 FlashAttention 中的分界

分界在 `__call__` / constructor 阶段确定：

```python
paged_kv_non_tma = page_size not in [None, tile_n]
self.use_tma_KV = not paged_kv_non_tma
```

当：

```text
page_size == tile_n
```

K/V paged cache 中一个 logical n_block 对应一个 page，TMA descriptor 可以描述规则 tile，走 `PipelineTmaAsync`。

当：

```text
page_size != tile_n
```

一个 K/V tile 可能跨多个 page，TMA descriptor 无法直接表达规则连续 tile，走 `PagedKVManager + PipelineCpAsync`。

cp.async path 中，`PagedKVManager` 负责：

```text
row_idx -> page_idx/page_offset
page_table lookup -> physical page
construct pointer
cp.async row fragments
```

pipeline 仍负责：

```text
stage full/empty 同步
producer/consumer 协议
phase 匹配
```

### 3.9 FlashAttention pipeline 的完整对应关系

| FlashAttention 源码对象 | CUTLASS pipeline 概念 | PTX / mbarrier 概念 |
| --- | --- | --- |
| `mbar_ptr_Q/K/V` | barrier storage | shared memory mbarrier object |
| `PipelineTmaAsync.create(..., tx_count=...)` | TMA pipeline full/empty sync object | mbarrier with tx-count |
| `PipelineCpAsync.create(...)` | cp.async pipeline sync object | barrier/mbarrier-backed full/empty protocol |
| `producer_acquire` | wait empty + prepare full | mbarrier wait + arrive/expect_tx |
| `tma_load_fn(..., producer_state)` | issue async copy for stage | TMA transaction associated with mbarrier |
| `producer_commit` | producer full commit | full barrier arrival protocol |
| `consumer_wait` | wait full | mbarrier test/try wait |
| `consumer_release` | mark empty | empty barrier arrive |
| `PipelineState.index` | shared stage index | mbarrier array index |
| `PipelineState.phase` | stage phase | phase parity / state |

### 3.10 对 Qwen-style case 的执行摘要

静态配置：

```text
Q pipeline:
    TMA
    stages = 1
    tx_count = 32768 bytes

K pipeline:
    TMA
    stages = 2
    tx_count = 32768 bytes

V pipeline:
    TMA
    stages = 2
    tx_count = 32768 bytes
```

producer：

```text
Q:
    acquire Q empty
    TMA Q -> sQ

K/V:
    acquire K stage
    TMA K(next) -> sK[next_stage]
    acquire V stage
    TMA V(current) -> sV[current_stage]
```

consumer：

```text
Q:
    wait Q full

K:
    wait K full
    QK WGMMA
    release K empty

V:
    wait V full
    PV WGMMA
    release V empty
```

关键同步保证：

```text
consumer_wait(K) 返回时:
    K stage phase 匹配
    producer arrival 完成
    K TMA tx-count 清零

consumer_release(K) 执行后:
    producer 后续可以复用该 K stage
```

### 3.11 总结

`mbarrier`、CUTLASS pipeline、FlashAttention pipeline 不是三个独立概念，而是同一套同步机制在不同抽象层级上的表达。

```text
PTX mbarrier 层:
    定义 phase、arrival、wait、tx-count、异步事务完成条件。

CUTLASS pipeline 层:
    将 mbarrier 组织成 full/empty barrier 协议。
    提供 producer_acquire / producer_commit / consumer_wait / consumer_release。
    使用 PipelineState 管理 stage index 与 phase。

FlashAttention kernel 层:
    用 PipelineTmaAsync 管理 Q/K/V TMA load。
    用 PipelineCpAsync 处理非 TMA fallback。
    通过 K(next) / V(current) 错位调度实现 QK(next) 与 PV(current) overlap。
```

Hopper FlashAttention 的性能依赖这套机制：

```text
1. TMA 异步搬运不会阻塞 producer。
2. mbarrier tx-count 确保 consumer 只在数据真正到达 shared memory 后读取。
3. full/empty barrier 保证 shared memory stage 安全复用。
4. PipelineState phase 防止循环 buffer 中旧信号被误用。
5. CUTLASS pipeline 把这些底层细节封装成可组合的 kernel 编程接口。
```

因此，在阅读 FlashAttention SM90 kernel 时，看到：

```python
pipeline_k.producer_acquire(...)
tma_load_fn(...)
pipeline_k.consumer_wait(...)
pipeline_k.consumer_release(...)
```

应当将其理解为一套完整的 mbarrier-backed stage protocol，而不是简单的函数调用序列。每一次 acquire/wait/release 都对应 shared memory stage 的所有权转移、phase 匹配，以及在 TMA path 下的异步事务完成确认。
