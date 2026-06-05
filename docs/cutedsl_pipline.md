# CuTeDSL CUTLASS Pipeline 组件详解

> 文件名按请求使用 `pipline`，正文统一写作 `pipeline`。

本文解释 CUTLASS/CuTeDSL 中 `cutlass.pipeline` 提供的组件、源码位置、核心语义和使用方式。重点是 Hopper/Blackwell kernel 中常见的 `mbarrier`、stage、phase、producer/consumer pipeline，以及它们如何映射到 TMA、cp.async、WGMMA/UMMA 等异步硬件路径。

参考源码：

```text
/root/lizhiyuan/cutlass/python/CuTeDSL/cutlass/pipeline/__init__.py
/root/lizhiyuan/cutlass/python/CuTeDSL/cutlass/pipeline/helpers.py
/root/lizhiyuan/cutlass/python/CuTeDSL/cutlass/pipeline/sm90.py
/root/lizhiyuan/cutlass/python/CuTeDSL/cutlass/pipeline/sm100.py
```

FlashAttention 中使用这些组件的位置：

```text
/root/lizhiyuan/flash-attention/flash_attn/cute/flash_fwd_sm90.py
/root/lizhiyuan/flash-attention/flash_attn/cute/pipeline.py
```

## 目录

1. [Pipeline 要解决什么问题](#1-pipeline-要解决什么问题)
2. [最小心智模型：full/empty + stage/phase](#2-最小心智模型-fullempty--stagephase)
3. [`cutlass.pipeline` 暴露了什么](#3-cutlasspipeline-暴露了什么)
4. [`Agent`：谁参与同步](#4-agent谁参与同步)
5. [`CooperativeGroup`：参与者数量和 arrival count](#5-cooperativegroup参与者数量和-arrival-count)
6. [`PipelineOp`：同步对象代表哪类硬件操作](#6-pipelineop同步对象代表哪类硬件操作)
7. [`SyncObject`：同步对象的统一接口](#7-syncobject同步对象的统一接口)
8. [`MbarrierArray`：pipeline 的核心同步对象](#8-mbarrierarraypipeline-的核心同步对象)
9. [`NamedBarrier`：硬件 named barrier](#9-namedbarrier硬件-named-barrier)
10. [`TmaStoreFence`：TMA store fence pipeline](#10-tmastorefencetma-store-fence-pipeline)
11. [`PipelineState` 和 `PipelineUserType`](#11-pipelinestate-和-pipelineusertype)
12. [`PipelineAsync`：通用 full/empty pipeline](#12-pipelineasync通用-fullempty-pipeline)
13. [`PipelineCpAsync`：cp.async producer pipeline](#13-pipelinecpasynccpasync-producer-pipeline)
14. [`PipelineTmaAsync`：Hopper TMA load pipeline](#14-pipelinetmaasynchopper-tma-load-pipeline)
15. [`PipelineTmaStore`：TMA store pipeline](#15-pipelinetmastoretma-store-pipeline)
16. [`PipelineOrder`：多组顺序控制](#16-pipelineorder多组顺序控制)
17. [SM100 pipeline：TMA/UMMA/TMEM](#17-sm100-pipelinetmaummatmem)
18. [组件选择表](#18-组件选择表)
19. [在 FlashAttention SM90 中的映射](#19-在-flashattention-sm90-中的映射)
20. [常见错误和调试心法](#20-常见错误和调试心法)

## 1. Pipeline 要解决什么问题

GPU kernel 中的高性能 GEMM/Attention 通常不是：

```text
load tile
compute tile
load next tile
compute next tile
```

而是希望：

```text
producer 正在 load 下一块 tile
consumer 正在 compute 当前 tile
```

Hopper 上典型路径是：

```text
producer: elected thread issues TMA GMEM -> SMEM
consumer: warpgroup waits data ready, then WGMMA reads SMEM
```

问题是：

1. producer 什么时候可以覆盖某个 shared memory buffer？
2. consumer 什么时候可以读取某个 shared memory buffer？
3. 同一个 buffer 被循环复用时，如何区分第 1 轮和第 2 轮？
4. TMA 是硬件后台搬运，不是普通线程 load/store，如何等待“写完 N bytes”？

`cutlass.pipeline` 的回答是：

```text
用 mbarrier array 管 full/empty 状态，
用 PipelineState 管 circular buffer 的 stage 和 phase，
用不同 PipelineOp 适配 cp.async、TMA、UMMA 等硬件路径。
```

### 小 case：2-stage K tile pipeline

假设 shared memory 有：

```text
sK[0]
sK[1]
```

producer 用 TMA 依次写 K0/K1/K2/K3：

```text
K0 -> sK[0]
K1 -> sK[1]
K2 -> sK[0]
K3 -> sK[1]
```

consumer 用 WGMMA 依次读：

```text
WGMMA reads sK[0] for K0
WGMMA reads sK[1] for K1
WGMMA reads sK[0] for K2
WGMMA reads sK[1] for K3
```

必须保证：

```text
consumer 不读未写完的 sK[stage]
producer 不覆盖未读完的 sK[stage]
```

这就是 pipeline 的核心。

## 2. 最小心智模型：full/empty + stage/phase

每个 stage 有两个 barrier：

```text
full[stage]  : producer 写完后 signal，consumer wait
empty[stage] : consumer 读完后 signal，producer wait
```

`stage` 是 circular buffer 的槽位：

```text
num_stages = 2
stage: 0 -> 1 -> 0 -> 1 -> ...
```

`phase` 是同一个 stage 第几轮复用的标签：

```text
tile   stage   phase
K0     0       0
K1     1       0
K2     0       1
K3     1       1
K4     0       0
K5     1       0
```

没有 phase 时，consumer 等 `full[0]` 无法判断等到的是 K0 的完成还是 K2 的完成。phase 让同一个 mbarrier 的多轮复用变得可区分。

通用公式：

```text
stage = count % num_stages
phase = (count / num_stages) % 2
```

producer 伪代码：

```python
state = producer_initial_state(num_stages=2)

for k_tile in tiles:
    stage = state.index
    phase = state.phase

    empty[stage].wait(phase)
    full[stage].arrive_and_expect_tx(bytes_per_tile)
    tma_copy(gmem[k_tile], smem[stage], barrier=full[stage])

    state.advance()
```

consumer 伪代码：

```python
state = consumer_initial_state(num_stages=2)

for k_tile in tiles:
    stage = state.index
    phase = state.phase

    full[stage].wait(phase)
    wgmma(smem[stage])
    empty[stage].arrive()

    state.advance()
```

对应 `cutlass.pipeline` API：

```python
pipeline.producer_acquire(state)
# issue TMA / cp.async / normal writes
pipeline.producer_commit(state)
state.advance()

pipeline.consumer_wait(state)
# compute / read smem
pipeline.consumer_release(state)
state.advance()
```

## 3. `cutlass.pipeline` 暴露了什么

`__init__.py` 暴露的主要组件：

```python
Agent
CooperativeGroup
PipelineOp
SyncObject
MbarrierArray
NamedBarrier
TmaStoreFence
PipelineUserType
PipelineState
make_pipeline_state
pipeline_init_arrive
pipeline_init_wait
agent_sync
PipelineAsync
PipelineCpAsync
PipelineTmaAsync
PipelineTmaStore
PipelineOrder
PipelineTmaUmma
PipelineAsyncUmma
PipelineUmmaAsync
PipelineClcFetchAsync
PipelineTmaMultiConsumersAsync
PipelineProducer
PipelineConsumer
```

分层看：

```text
helpers.py:
    基础 enum、mbarrier/named barrier/fence、PipelineState、同步 helper

sm90.py:
    Hopper 常用 pipeline，如 PipelineAsync、PipelineCpAsync、PipelineTmaAsync、PipelineTmaStore

sm100.py:
    Blackwell 常用 pipeline，如 PipelineTmaUmma、PipelineUmmaAsync、PipelineAsyncUmma
```

### 小 case：只看 Hopper TMA

Hopper GEMM/FA 主路径通常只需要：

```python
producer_group = CooperativeGroup(Agent.Thread, 1)
consumer_group = CooperativeGroup(Agent.Thread, num_mma_warps)

pipeline_k = PipelineTmaAsync.create(
    barrier_storage=smem_barriers,
    num_stages=2,
    producer_group=producer_group,
    consumer_group=consumer_group,
    tx_count=bytes_per_k_tile,
)
```

这里 `PipelineTmaAsync` 会创建：

```text
full mbarrier array: 2 个，TMA transaction barrier
empty mbarrier array: 2 个，consumer release barrier
```

## 4. `Agent`：谁参与同步

源码位置：

```text
helpers.py::Agent
```

定义：

```python
class Agent(enum.Enum):
    Thread = enum.auto()
    Warp = enum.auto()
    ThreadBlock = enum.auto()
    ThreadBlockCluster = enum.auto()
```

语义：

| Agent | 含义 |
| --- | --- |
| `Thread` | 任意 N 个线程参与同步 |
| `Warp` | 一个 32-thread warp，当前 pipeline 实现中较少直接使用 |
| `ThreadBlock` | 整个 CTA/block |
| `ThreadBlockCluster` | 整个 CTA cluster |

注意：`CooperativeGroup` 当前实际支持的主要是 `Agent.Thread`。`Agent.ThreadBlock` 和 `Agent.ThreadBlockCluster` 在 `CooperativeGroup` 构造中会抛 `NotImplementedError`，但 `agent_sync` 支持 `ThreadBlock` 和 `ThreadBlockCluster`。

### 小 case：block 级初始化同步

pipeline 创建后，如果 `defer_sync=False`，通常会：

```python
cute.arch.mbarrier_init_fence()
agent_sync(Agent.ThreadBlock)
```

含义：

```text
warp0 初始化 mbarrier array
mbarrier_init_fence 保证初始化可见
整个 CTA __syncthreads()
```

如果是 cluster pipeline，则可能使用：

```python
agent_sync(Agent.ThreadBlockCluster, is_relaxed=True)
```

## 5. `CooperativeGroup`：参与者数量和 arrival count

源码位置：

```text
helpers.py::CooperativeGroup
```

构造：

```python
CooperativeGroup(agent: Agent, size: int = 1)
```

`size` 的关键作用是作为 mbarrier 的 arrival count：

```python
self.arrive_count = self.cg.size
cute.arch.mbarrier_init(barrier, self.arrive_count)
```

这意味着：

```text
CooperativeGroup(Agent.Thread, 128)
```

通常表示：

```text
这个 barrier 需要 128 个参与线程 arrive 才完成
```

但是对 TMA/UMMA 这类特殊 op，要按 pipeline 实现理解：

- TMA full barrier 的完成主要由 transaction bytes 决定。
- TMA issuing thread 常传 `size=1`，表示一个 elected thread 负责设置 transaction barrier。
- consumer group 的 size 常用于 empty barrier 的 arrival count，或用于 UMMA/TMEM release 语义。

### 小 case：TMA producer group

```python
tma_thread = CooperativeGroup(Agent.Thread, 1)
```

这不是“一个 warp”，而是：

```text
一个 elected thread 负责 producer acquire，并设置 TMA transaction barrier。
```

### 小 case：cp.async producer group

```python
load_threads = CooperativeGroup(Agent.Thread, 128)
```

含义是：

```text
128 个 producer threads 协同发 cp.async。
```

这些线程共同参与 pipeline 同步。

## 6. `PipelineOp`：同步对象代表哪类硬件操作

源码位置：

```text
helpers.py::PipelineOp
```

定义：

```python
class PipelineOp(enum.Enum):
    AsyncThread = enum.auto()
    TCGen05Mma = enum.auto()
    TmaLoad = enum.auto()
    ClcLoad = enum.auto()
    TmaStore = enum.auto()
    Composite = enum.auto()
    AsyncLoad = enum.auto()
```

`PipelineOp` 决定 `MbarrierArray.arrive()` 到底调用什么底层指令。

在 `MbarrierArray.arrive` 中：

```text
AsyncThread -> mbarrier_arrive
TmaLoad     -> mbarrier_arrive_and_expect_tx(tx_count)
AsyncLoad   -> cp_async_mbarrier_arrive_noinc
TCGen05Mma  -> tcgen05.commit(mbarrier)
ClcLoad     -> mbarrier_arrive_and_expect_tx_with_dst
```

### 小 case：同样叫 arrive，语义不同

普通 async thread：

```python
sync_object_full.arrive(stage, None)
```

下降为：

```text
mbarrier.arrive(full[stage])
```

TMA load：

```python
sync_object_full.arrive(stage, None)
```

下降为：

```text
mbarrier.arrive.expect_tx(full[stage], tx_count)
```

UMMA：

```python
sync_object_full.arrive(stage, mask, cta_group)
```

下降为：

```text
tcgen05.commit(mbarrier, mask, cta_group)
```

所以 `PipelineOp` 是 pipeline 连接硬件语义的关键开关。

## 7. `SyncObject`：同步对象的统一接口

源码位置：

```text
helpers.py::SyncObject
```

抽象接口：

```python
arrive()
wait()
arrive_and_wait()
arrive_and_drop()
get_barrier()
max()
```

实际实现包括：

```text
MbarrierArray
NamedBarrier
TmaStoreFence
```

`PipelineAsync` 只依赖 `SyncObject` 接口，不需要知道底层是 mbarrier 还是 TMA store fence。

### 小 case：PipelineTmaStore 不用 mbarrier

TMA store pipeline 的 sync object 是：

```text
TmaStoreFence
```

它的 `arrive()` 是：

```text
cp_async_bulk_commit_group
```

它的 `wait()` 是：

```text
cp_async_bulk_wait_group
```

所以它也能被当成 `SyncObject`，但不是 mbarrier。

## 8. `MbarrierArray`：pipeline 的核心同步对象

源码位置：

```text
helpers.py::MbarrierArray
```

构造参数：

```python
MbarrierArray(
    barrier_storage: cute.Pointer,
    num_stages: int,
    agent: tuple[PipelineOp, CooperativeGroup],
    tx_count: int = 0,
)
```

核心字段：

```python
self.barrier_storage = barrier_storage
self.tx_count = tx_count
self.num_stages = num_stages
self.op_type, self.cg = agent
self.arrive_count = self.cg.size
self.mbarrier_base = self.barrier_storage
```

初始化：

```python
for index in range(self.num_stages):
    cute.arch.mbarrier_init(self.get_barrier(index), self.arrive_count)
```

初始化只由 warp 0 执行：

```python
warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
if warp_idx == 0:
    mbarrier_init(...)
```

### 8.1 `get_barrier(index)`

```python
return self.mbarrier_base + index
```

如果 `num_stages=2`，则：

```text
barrier_storage + 0 -> stage 0 barrier
barrier_storage + 1 -> stage 1 barrier
```

### 8.2 `wait(index, phase)`

```python
cute.arch.mbarrier_wait(self.get_barrier(index), phase)
```

这就是：

```text
等第 index 个 mbarrier 的当前 phase 完成
```

### 8.3 `try_wait(index, phase)`

```python
cute.arch.mbarrier_try_wait(self.get_barrier(index), phase)
```

非阻塞检查。常见用法：

```python
token = pipeline.consumer_try_wait(state)
pipeline.consumer_wait(state, token)
```

如果 try 已经成功，wait 可以跳过实际阻塞路径。

### 8.4 `arrive(index, dst, cta_group)`

`arrive` 会根据 `PipelineOp` 分发：

```text
AsyncThread:
    mbarrier_arrive

TmaLoad:
    mbarrier_arrive_and_expect_tx(tx_count)

AsyncLoad:
    cp_async_mbarrier_arrive_noinc

TCGen05Mma:
    tcgen05.commit
```

### 小 case：2-stage 普通 producer/consumer

构造：

```python
full = MbarrierArray(ptr_full, 2, (PipelineOp.AsyncThread, CooperativeGroup(Agent.Thread, 1)))
empty = MbarrierArray(ptr_empty, 2, (PipelineOp.AsyncThread, CooperativeGroup(Agent.Thread, 1)))
```

状态变化：

```text
producer_acquire:
    empty.wait(index, phase)

producer_commit:
    full.arrive(index)

consumer_wait:
    full.wait(index, phase)

consumer_release:
    empty.arrive(index)
```

## 9. `NamedBarrier`：硬件 named barrier

源码位置：

```text
helpers.py::NamedBarrier
```

字段：

```python
barrier_id
num_threads
```

接口：

```python
arrive()
arrive_unaligned()
wait()
wait_unaligned()
arrive_and_wait()
sync()
```

它使用 CUDA/PTX named barrier：

```text
barrier_arrive(barrier_id, number_of_threads)
barrier(barrier_id, number_of_threads)
```

注意源码注释：

```text
NamedBarriers do not have a standalone wait like mbarriers,
only an arrive_and_wait.
```

也就是说 named barrier 更像：

```text
一批线程在某个固定 barrier_id 上会合
```

它不适合 TMA transaction bytes。

### 小 case：两个 warp 同步

假设 warp0 做 load，warp1 做 compute，想在某点会合：

```python
bar = NamedBarrier(barrier_id=1, num_threads=64)

# warp0 + warp1 都执行
bar.arrive_and_wait()
```

这和 mbarrier pipeline 不同。mbarrier 可以只让 producer arrive、consumer wait；named barrier 的 wait 本身也会 arrive。

## 10. `TmaStoreFence`：TMA store fence pipeline

源码位置：

```text
helpers.py::TmaStoreFence
```

用于 multi-stage epilogue TMA store。

关键方法：

```python
arrive():
    cute.arch.cp_async_bulk_commit_group()

wait():
    cute.arch.cp_async_bulk_wait_group(self.num_stages - 1, read=True)

tail():
    cute.arch.cp_async_bulk_wait_group(0, read=True)
```

这里没有 mbarrier，因为 TMA store 的完成等待走 bulk async group commit/wait。

### 小 case：epilogue TMA store

```python
store_pipeline = PipelineTmaStore.create(
    num_stages=2,
    producer_group=CooperativeGroup(Agent.Thread, 1),
)

store_pipeline.producer_acquire()
copy(tma_store_atom, sO_stage0, gO_tile)
store_pipeline.producer_commit()
```

可以理解为：

```text
producer_acquire -> 等前面的 TMA store group 留出槽位
producer_commit  -> commit 当前 TMA store group
```

## 11. `PipelineState` 和 `PipelineUserType`

源码位置：

```text
helpers.py::PipelineState
helpers.py::PipelineUserType
helpers.py::make_pipeline_state
```

`PipelineUserType` 通常有：

```text
Producer
Consumer
```

`make_pipeline_state` 用它创建初始状态。

`PipelineState` 的核心字段是：

```text
index: 当前 stage
phase: 当前 phase bit
count/stages: 用于 advance 后计算下一轮
```

对 `num_stages=2`：

```text
advance 次数   index   phase
0             0       0
1             1       0
2             0       1
3             1       1
4             0       0
```

### 小 case：consumer 读取 4 个 K blocks

```python
state = make_pipeline_state(PipelineUserType.Consumer, 2)

for i in range(4):
    print(state.index, state.phase)
    state.advance()
```

逻辑输出：

```text
0, 0
1, 0
0, 1
1, 1
```

对应：

```text
读 sK[0] 的第 0 轮
读 sK[1] 的第 0 轮
读 sK[0] 的第 1 轮
读 sK[1] 的第 1 轮
```

## 12. `PipelineAsync`：通用 full/empty pipeline

源码位置：

```text
sm90.py::PipelineAsync
```

创建：

```python
PipelineAsync.create(
    num_stages,
    producer_group,
    consumer_group,
    barrier_storage,
    producer_mask=None,
    consumer_mask=None,
    defer_sync=False,
)
```

内部创建两个 sync object：

```python
sync_object_full = _make_sync_object(
    barrier_storage.align(min_align=8),
    num_stages,
    producer,
)

sync_object_empty = _make_sync_object(
    barrier_storage.align(min_align=8) + num_stages,
    num_stages,
    consumer,
)
```

也就是说 mbarrier storage 布局是：

```text
barrier_storage + 0             -> full[0]
barrier_storage + 1             -> full[1]
...
barrier_storage + num_stages    -> empty[0]
barrier_storage + num_stages+1  -> empty[1]
...
```

核心 API：

```python
producer_acquire(state)
producer_try_acquire(state)
producer_commit(state)
consumer_wait(state)
consumer_try_wait(state)
consumer_release(state)
producer_get_barrier(state)
consumer_get_barrier(state)
producer_tail(state)
make_producer()
make_consumer()
make_participants()
```

### 12.1 `producer_acquire`

```python
self.sync_object_empty.wait(state.index, state.phase)
```

含义：

```text
等这个 stage 空出来，producer 才能写。
```

### 12.2 `producer_commit`

```python
self.sync_object_full.arrive(state.index, self.producer_mask)
```

含义：

```text
producer 写完了，通知 consumer 这个 stage full。
```

### 12.3 `consumer_wait`

```python
self.sync_object_full.wait(state.index, state.phase)
```

含义：

```text
等 producer 写完这个 stage。
```

### 12.4 `consumer_release`

```python
self.sync_object_empty.arrive(state.index, self.consumer_mask)
```

含义：

```text
consumer 读完了，通知 producer 这个 stage empty。
```

### 小 case：普通 async producer 写 shared

```python
producer_group = CooperativeGroup(Agent.Thread, 1)
consumer_group = CooperativeGroup(Agent.Thread, 1)

pipe = PipelineAsync.create(
    barrier_storage=mbar_ptr,
    num_stages=2,
    producer_group=producer_group,
    consumer_group=consumer_group,
)

p_state = make_pipeline_state(PipelineUserType.Producer, 2)
c_state = make_pipeline_state(PipelineUserType.Consumer, 2)

# producer
pipe.producer_acquire(p_state)
write_smem(stage=p_state.index)
pipe.producer_commit(p_state)
p_state.advance()

# consumer
pipe.consumer_wait(c_state)
read_smem(stage=c_state.index)
pipe.consumer_release(c_state)
c_state.advance()
```

## 13. `PipelineCpAsync`：cp.async producer pipeline

源码位置：

```text
sm90.py::PipelineCpAsync
```

它继承 `PipelineAsync`，但创建时 producer op 是：

```python
producer_type = PipelineOp.AsyncLoad
consumer_type = PipelineOp.AsyncThread
```

所以 full sync object 的 `arrive()` 会走：

```python
cute.arch.cp_async_mbarrier_arrive_noinc(...)
```

典型用途：

```text
非 TMA load path
paged KV page_size != tile_n 时手工拼 shared tile
Ampere/SM80 风格 cp.async pipeline
```

### 小 case：paged KV fallback

producer：

```python
pipe.producer_acquire(state)
paged_kv_manager.load_KV(block, sK[..., state.index], "K")
cute.arch.cp_async_commit_group()
pipe.producer_commit(state)
state.advance()
```

consumer：

```python
pipe.consumer_wait(state)
qk_wgmma(sK[..., state.index])
pipe.consumer_release(state)
state.advance()
```

和 TMA 的区别：

```text
cp.async:
    多个线程发很多 cp.async
    commit_group 后 producer_commit 标记 full

TMA:
    一个 elected thread 发 TMA descriptor
    full 由 transaction bytes 完成
```

## 14. `PipelineTmaAsync`：Hopper TMA load pipeline

源码位置：

```text
sm90.py::PipelineTmaAsync
```

创建：

```python
PipelineTmaAsync.create(
    num_stages,
    producer_group,
    consumer_group,
    tx_count,
    barrier_storage,
    cta_layout_vmnk=None,
    tidx=None,
    mcast_mode_mn=(1, 1),
    defer_sync=False,
)
```

关键点：

```python
producer_type = PipelineOp.TmaLoad
consumer_type = PipelineOp.AsyncThread
```

full barrier 是 TMA transaction barrier：

```text
mbarrier.arrive.expect_tx(tx_count)
```

### 14.1 `tx_count`

`tx_count` 是一次 TMA copy 预期写入 shared memory 的字节数。

例如：

```text
K tile shape = (tile_n=128, head_dim=128)
dtype = bf16 = 2 bytes

tx_count = 128 * 128 * 2 = 32768 bytes
```

TMA copy 完成这 32768 bytes 后，full barrier 才 ready。

### 14.2 `producer_acquire`

源码语义：

```python
empty.wait(state.index, state.phase)
full.arrive(state.index, producer_mask)
```

因为 full 的 op type 是 `TmaLoad`，所以 `full.arrive` 实际做：

```text
mbarrier.arrive.expect_tx(full[state.index], tx_count)
```

然后用户发 TMA：

```python
copy(tma_atom.with(full_barrier), gmem_tile, smem_tile)
```

### 14.3 `producer_commit`

源码：

```python
pass
```

原因：

```text
TMA full 不是由 producer_commit 完成的。
TMA copy 硬件完成 tx_count bytes 后，full barrier 完成。
```

### 14.4 `consumer_release`

TMA pipeline 的 `consumer_release` 带 `is_signalling_thread`：

```python
if self.is_signalling_thread:
    empty.arrive(state.index, consumer_mask)
```

cluster/multicast 情况下，不是所有线程都应该 signal empty。`init_empty_barrier_arrive_signal` 会根据 cluster layout 和 multicast mode 计算：

```text
哪个线程向哪个 CTA rank 的 empty barrier arrive
```

### 小 case：Hopper K pipeline

```python
tma_thread = CooperativeGroup(Agent.Thread, 1)
mma_warps = CooperativeGroup(Agent.Thread, 4)

pipeline_k = PipelineTmaAsync.create(
    barrier_storage=mbar_ptr,
    num_stages=2,
    producer_group=tma_thread,
    consumer_group=mma_warps,
    tx_count=32768,
)
```

producer：

```python
state = make_pipeline_state(PipelineUserType.Producer, 2)

pipeline_k.producer_acquire(state)
barrier = pipeline_k.producer_get_barrier(state)
copy(tma_atom_k.with(barrier), gK_tile, sK[..., state.index])
pipeline_k.producer_commit(state)  # noop
state.advance()
```

consumer：

```python
state = make_pipeline_state(PipelineUserType.Consumer, 2)

pipeline_k.consumer_wait(state)
wgmma_qk(sQ, sK[..., state.index])
pipeline_k.consumer_release(state)
state.advance()
```

## 15. `PipelineTmaStore`：TMA store pipeline

源码位置：

```text
sm90.py::PipelineTmaStore
```

用途：

```text
epilogue 中 SMEM -> GMEM 的 TMA store
```

创建：

```python
PipelineTmaStore.create(
    num_stages,
    producer_group,
)
```

它没有 consumer agent。方法：

```python
producer_acquire():
    TmaStoreFence.wait()

producer_commit():
    TmaStoreFence.arrive()

producer_tail():
    TmaStoreFence.tail()
```

### 小 case：O tile TMA store

```python
store_pipe = PipelineTmaStore.create(
    num_stages=2,
    producer_group=CooperativeGroup(Agent.Thread, 1),
)

store_pipe.producer_acquire()
copy(tma_atom_o, sO_tile, gO_tile)
store_pipe.producer_commit()
```

含义：

```text
等之前的 TMA store group 留出队列空间
发起当前 TMA store
commit 当前 bulk async group
```

## 16. `PipelineOrder`：多组顺序控制

源码位置：

```text
sm90.py::PipelineOrder
```

用途：

```text
多个 group 需要按固定顺序通过若干 stage
```

它内部也使用 sync object 和 `PipelineState`，但目标不是 full/empty producer/consumer，而是：

```text
group 0 完成后 group 1 才继续
group 1 完成后 group 2 才继续
...
```

源码注释给的概念是：

```python
pipeline_order = PipelineOrder.create(
    barrier_storage=smem_ptr,
    depth=2,
    length=3,
    group_id=0,
    producer_group=producer_warp,
)

for stage in range(num_stages):
    pipeline_order.wait()
    # process current stage
    pipeline_order.arrive()
```

### 小 case：3 个 warpgroup 顺序做 epilogue 子任务

```text
WG0: load accumulator
WG1: apply epilogue op
WG2: store
```

如果需要严格顺序，可以用 `PipelineOrder` 让每个 group 在对应 stage 上等待前一个 group。

## 17. SM100 pipeline：TMA/UMMA/TMEM

源码位置：

```text
sm100.py
```

Blackwell/SM100 中，Tensor Core 指令是 `tcgen05.mma`，accumulator 在 TMEM。pipeline 需要覆盖更多组合：

```text
TMA producer -> UMMA consumer
AsyncThread producer -> UMMA consumer
UMMA producer -> AsyncThread consumer
```

### 17.1 `PipelineTmaUmma`

用途：

```text
TMA load A/B -> UMMA/tcgen05.mma consume SMEM
```

创建时：

```python
producer_type = PipelineOp.TmaLoad
consumer_type = PipelineOp.TCGen05Mma
```

full：

```text
TMA transaction barrier
```

empty：

```text
UMMA consumer release，通过 tcgen05 commit/mbarrier 语义
```

重要字段：

```python
is_leader_cta
cta_group
producer_mask
consumer_mask
```

2CTA/cluster 场景下，只有 leader CTA 设置某些 TMA transaction barrier，UMMA release 需要指定 CTA group 和 mask。

### 小 case：Blackwell A/B mainloop

```python
ab_producer, ab_consumer = PipelineTmaUmma.create(
    barrier_storage=ab_mbar,
    num_stages=4,
    producer_group=CooperativeGroup(Agent.Thread, 1),
    consumer_group=CooperativeGroup(Agent.Thread, num_mma_warps),
    tx_count=bytes_A + bytes_B,
    cta_layout_vmnk=cluster_layout,
).make_participants()
```

语义：

```text
producer acquire stage
TMA load A/B into SMEM
UMMA waits full
tcgen05.mma reads SMEM
UMMA release empty
```

### 17.2 `PipelineAsyncUmma`

用途：

```text
普通 async-thread producer -> UMMA consumer
```

例如某些 fused input 或非 TMA 路径中，producer 不是 TMA，而是线程生成/写入某个 buffer，然后 UMMA 消费。

创建时：

```python
producer_type = PipelineOp.AsyncThread
consumer_type = PipelineOp.TCGen05Mma
```

2CTA 情况下会计算：

```text
producer_mask: leading CTA rank
consumer_mask: peer CTA mask
cta_group: ONE or TWO
```

### 17.3 `PipelineUmmaAsync`

用途：

```text
UMMA/tcgen05.mma producer -> async-thread consumer
```

典型是：

```text
tcgen05.mma writes TMEM accumulator
epilogue threads wait accumulator ready
epilogue reads TMEM -> RMEM/SMEM/GMEM
```

创建时：

```python
producer_type = PipelineOp.TCGen05Mma
consumer_type = PipelineOp.AsyncThread
```

`producer_commit` 会走：

```text
tcgen05.commit(mbarrier)
```

### 小 case：Blackwell accumulator pipeline

```python
acc_pipeline = PipelineUmmaAsync.create(
    barrier_storage=acc_mbar,
    num_stages=1,
    producer_group=CooperativeGroup(Agent.Thread, 1),
    consumer_group=CooperativeGroup(Agent.Thread, threads_per_cta),
    cta_layout_vmnk=cluster_layout,
)
```

语义：

```text
UMMA producer:
    tcgen05.mma writes TMEM
    tcgen05.commit(acc_mbar)

epilogue consumer:
    wait acc_mbar
    copy TMEM -> RMEM
    store output
```

## 18. 组件选择表

| 场景 | 组件 | producer | consumer | 关键同步 |
| --- | --- | --- | --- | --- |
| 普通 async thread 写 SMEM，thread 读 | `PipelineAsync` | `AsyncThread` | `AsyncThread` | mbarrier full/empty |
| cp.async 写 SMEM，thread/WGMMA 读 | `PipelineCpAsync` | `AsyncLoad` | `AsyncThread` | cp.async mbarrier |
| Hopper TMA load 写 SMEM，WGMMA 读 | `PipelineTmaAsync` | `TmaLoad` | `AsyncThread` | transaction barrier bytes |
| Hopper/SM90 TMA store | `PipelineTmaStore` | `TmaStore` | 无 | bulk commit/wait group |
| Blackwell TMA load A/B，UMMA 读 | `PipelineTmaUmma` | `TmaLoad` | `TCGen05Mma` | TMA tx + UMMA release |
| Blackwell thread producer，UMMA 读 | `PipelineAsyncUmma` | `AsyncThread` | `TCGen05Mma` | mbarrier + tcgen05 release |
| Blackwell UMMA 写 TMEM，thread epilogue 读 | `PipelineUmmaAsync` | `TCGen05Mma` | `AsyncThread` | tcgen05.commit |
| 多组顺序执行 | `PipelineOrder` | group order | group order | ordered mbarrier |

## 19. 在 FlashAttention SM90 中的映射

`flash_fwd_sm90.py` 中：

```python
from cutlass import pipeline
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from flash_attn.cute import pipeline as pipeline_custom
```

这里有两层：

```text
cutlass.pipeline:
    原生组件、enum、state、CooperativeGroup

pipeline_custom:
    FlashAttention 对原生 pipeline 的薄 wrapper
```

FA 创建 group：

```python
ThreadCooperativeGroup = partial(pipeline.CooperativeGroup, pipeline.Agent.Thread)
tma_warp = ThreadCooperativeGroup(1)
load_threads = ThreadCooperativeGroup(self.num_threads_per_warp_group)
mma_warps = ThreadCooperativeGroup(self.num_mma_threads // cute.arch.WARP_SIZE)
```

如果 Q/K/V 走 TMA：

```python
pipeline_q = pipeline_custom.PipelineTmaAsync.create(
    barrier_storage=mbar_ptr_Q,
    num_stages=1,
    producer_group=tma_warp,
    consumer_group=mma_warps,
    tx_count=self.tma_copy_bytes["Q"],
    defer_sync=True,
)

pipeline_k = pipeline_custom.PipelineTmaAsync.create(
    barrier_storage=storage.mbar_ptr_K.data_ptr(),
    num_stages=self.num_stages,
    producer_group=tma_warp,
    consumer_group=mma_warps,
    tx_count=self.tma_copy_bytes["K"],
    defer_sync=True,
)
```

如果 fallback 到 cp.async：

```python
pipeline_q = pipeline_custom.PipelineCpAsync.create(
    barrier_storage=mbar_ptr_Q,
    num_stages=1,
    producer_group=load_threads,
    consumer_group=mma_warps,
    defer_sync=True,
    elect_one_release=True,
    syncwarp_before_release=False,
)
```

### 19.1 Q pipeline

Q 对一个 work tile 通常 single-stage：

```text
pipeline_q:
    num_stages = 1
```

producer：

```python
pipeline_q.producer_acquire_w_index_phase(0, q_producer_phase)
load_Q(...)
pipeline_q.producer_commit_w_index(0)
q_producer_phase ^= 1
```

consumer：

```python
pipeline_q.consumer_wait_w_index_phase(0, q_consumer_phase)
# WGMMA QK uses sQ
pipeline_q.consumer_release_w_index(0)
q_consumer_phase ^= 1
```

因为只有一个 stage，所以 index 固定为 0，靠 phase 翻转区分不同 work tile。

### 19.2 K/V pipeline

K/V streaming over n_block，需要多 stage：

```python
kv_producer_state = pipeline.make_pipeline_state(
    pipeline.PipelineUserType.Producer, self.num_stages
)

kv_consumer_state = pipeline.make_pipeline_state(
    pipeline.PipelineUserType.Consumer, self.num_stages
)
```

producer：

```python
pipeline_k.producer_acquire(kv_producer_state)
load_K(block=n_block, producer_state=kv_producer_state)

pipeline_v.producer_acquire(kv_producer_state)
load_V(block=n_block, producer_state=kv_producer_state)

kv_producer_state.advance()
```

consumer：

```python
pipeline_k.consumer_wait(kv_consumer_state)
qk_wgmma(...)
pipeline_k.consumer_release(kv_consumer_state)

pipeline_v.consumer_wait(kv_consumer_state)
pv_wgmma(...)
pipeline_v.consumer_release(kv_consumer_state)

kv_consumer_state.advance()
```

### 19.3 `pipeline_custom` 为什么存在

`flash_attn.cute.pipeline.py` 对原生类做了几类增强：

1. `PipelineStateSimple`

   用单个 `phase_index` 表达 `index/phase`，更适合 tight loop。

2. `_w_index_phase` / `_w_index`

   给 single-stage Q pipeline 直接传 index/phase，避免构造完整 `PipelineState`。

3. `elect_one_release`

   cp.async fallback 中，release empty 时只让 elected thread arrive。

4. `extra_tx_count`

   TMA producer acquire 可动态增加 transaction bytes。

5. SM100 leader CTA 保护

   `PipelineTmaUmma` 中只有 leader CTA 设置 transaction barrier。

## 20. 常见错误和调试心法

### 20.1 arrival count 和实际 arrive 次数不匹配

现象：

```text
kernel hang
consumer_wait 永远等不到
producer_acquire 永远等不到 empty
```

原因：

```text
CooperativeGroup.size 设置为 128，
但实际只有 1 个线程 arrive。
```

或反过来：

```text
size 设置为 1，
但 128 个线程都 arrive，phase 被污染。
```

检查：

```text
producer_group / consumer_group 的 size
producer_mask / consumer_mask
是否用了 elect_one
```

### 20.2 phase 没有正确 advance

现象：

```text
第一轮正确，第二轮 hang 或读旧数据
```

原因：

```text
state.advance() 漏掉
producer 和 consumer phase 不一致
single-stage pipeline 没有翻 q_producer_phase/q_consumer_phase
```

检查：

```text
每次 producer commit 后是否 advance
每次 consumer release 后是否 advance
single-stage index=0 时 phase 是否 xor 1
```

### 20.3 TMA tx_count 错误

现象：

```text
consumer_wait 过早通过 -> 读未写完 SMEM
consumer_wait 永远不通过 -> tx_count 等不到
```

原因：

```text
tx_count 小于实际 TMA bytes
tx_count 大于实际 TMA bytes
```

检查：

```text
tile shape
dtype bytes
TMA multicast / 2SM 是否影响 transaction bytes
是否有 extra_tx_count
```

### 20.4 TMA producer_commit 的误解

TMA pipeline 中：

```python
producer_commit(state)
```

通常是 noop。不能把它理解成“标记 TMA 已完成”。TMA 的完成来自：

```text
mbarrier.arrive.expect_tx(tx_count)
TMA copy with barrier
硬件完成 tx_count bytes
```

### 20.5 NamedBarrier 和 mbarrier 混用

NamedBarrier 适合：

```text
固定线程集合会合
```

mbarrier pipeline 适合：

```text
producer arrive, consumer wait
TMA transaction bytes
stage/phase circular buffer
```

不要用 NamedBarrier 表达 TMA full/empty。

## 总结

`cutlass.pipeline` 的核心不是一个复杂框架，而是把下面这套固定模式封装起来：

```text
full mbarrier array
empty mbarrier array
PipelineState(index, phase)
producer_acquire / producer_commit
consumer_wait / consumer_release
```

Hopper 中最重要的是：

```text
PipelineTmaAsync = TMA load -> SMEM -> WGMMA consumer
```

Blackwell 中进一步扩展为：

```text
PipelineTmaUmma  = TMA load -> UMMA consumer
PipelineUmmaAsync = UMMA producer -> epilogue consumer
```

理解每个组件时，始终回到三个问题：

```text
1. 谁是 producer？
2. 谁是 consumer？
3. full/empty 分别由谁 signal、由谁 wait？
```

只要这三个问题清楚，`Agent`、`CooperativeGroup`、`PipelineOp`、`MbarrierArray`、`PipelineState` 的作用就能串起来。
