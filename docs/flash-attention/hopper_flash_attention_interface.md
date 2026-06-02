# Hopper FlashAttention 上层接口链路

本文只围绕 Hopper / SM90 forward FlashAttention 的上层调用、参数整理、tile 配置、JIT 编译缓存和 kernel launch 之前的封装逻辑。kernel 内部实现细节见 `hopper_flash_attention_kernel.md`。

源码位置基于仓库相对路径：

- 测试入口：`flash-attention/tests/cute/test_flash_attn.py`
- Python/CuTe 接口：`flash-attention/flash_attn/cute/interface.py`
- SM90 forward kernel 类：`flash-attention/flash_attn/cute/flash_fwd_sm90.py`
- forward 基类：`flash-attention/flash_attn/cute/flash_fwd.py`

## 1. 从测试到 public API

`tests/cute/test_flash_attn.py::test_flash_attn_output` 是比较适合顺藤摸瓜的入口。测试大体流程是：

1. 构造 `q, k, v`，shape 通常是 dense 格式：
   - `q`: `(batch_size, seqlen_q, nheads_q, headdim)`
   - `k`: `(batch_size, seqlen_k, nheads_k, headdim)`
   - `v`: `(batch_size, seqlen_k, nheads_k, headdim_v)`
2. 调用 high-level API：
   - `flash_attn_func(q, k, v, ..., causal=..., window_size=..., return_lse=...)`
3. 用 `attention_ref` 或 PyTorch reference 计算期望输出。
4. 比较 output、LSE、误差阈值。

Hopper 特殊点：

- `test_flash_attn_output` 对 SM90 会跳过 split-KV forward，因为 SM90 路径当前不是通过 split-KV 主路径处理。
- 真正进入 Hopper CuTe kernel 的 forward 入口是 `flash_attn_func -> _flash_attn_fwd`。

## 2. `flash_attn_func` 做什么

`flash_attn_func` 是用户直接调用的 dense QKV 接口。它主要负责把用户层参数转成 `_flash_attn_fwd` 可以消费的形式。

典型输入：

```python
out, lse = flash_attn_func(
    q,
    k,
    v,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    softcap=0.0,
    return_lse=True,
)
```

这层的职责不是写 kernel，而是统一语义：

- 判断是否 causal/local attention。
- 处理 softmax scale 默认值，通常是 `1 / sqrt(head_dim)`。
- 支持 dense、varlen、paged KV、GQA/MQA、可选 aux tensor。
- 将 attention mask、score modifier、window size 等信息继续传给 `_flash_attn_fwd`。

## 3. `_flash_attn_fwd` 是核心分发器

`interface.py::_flash_attn_fwd` 是 forward 的核心调度函数。它做四类事情：

1. 校验输入和 layout。
2. 根据 GPU 架构选择 SM90、SM100/SM120 等实现。
3. 为当前问题规模选择 Hopper tile 配置。
4. 构造 `FlashAttentionForwardSm90`，走 CuTe JIT compile cache，然后 launch。

### 3.1 架构选择

代码通过 `_get_device_arch()` 获取当前设备架构，也可以通过环境变量 `FLASH_ATTENTION_ARCH` 覆盖。

对 Hopper：

```python
if arch // 10 == 9:
    fa_fwd = FlashAttentionForwardSm90(...)
```

Blackwell 路径会进入 `FlashAttentionForwardSm100` 等类，不在本文展开。

### 3.2 head dim 校验

SM90 forward 对 head dim 的约束大致是：

- `8 <= head_dim <= 256`
- `8 <= head_dim_v <= 256`
- head dim 需要满足底层 copy / MMA 对齐要求。

基类会把 head dim pad 到 16 的倍数：

```python
self.tile_hdim = ceil_div(head_dim, 16) * 16
self.tile_hdimv = ceil_div(head_dim_v, 16) * 16
```

所以 kernel 内部的 tile 以 `tile_hdim` 和 `tile_hdimv` 为准，尾部越界通过 predicate/mask 处理。

## 4. Hopper tile size 选择

Hopper forward tile 配置来自 `_tile_size_fwd_sm90(head_dim, head_dim_v, is_causal, is_local, sparse_block_size_q)`。

返回值是：

```python
FwdConfig(
    m_block_size,
    n_block_size,
    mma_pv_is_rs,
    intra_wg_overlap,
)
```

含义：

- `m_block_size`: Q 方向 tile M，也就是每个 CTA 处理多少 query row。
- `n_block_size`: K/V 方向 tile N，也就是每轮 mainloop 读多少 key/value row。
- `mma_pv_is_rs`: P 矩阵是否留在寄存器中直接参与 PV WGMMA。
- `intra_wg_overlap`: 是否在同一 warpgroup 内交叠 QK 和 PV。

源码里的完整决策逻辑可以写成：

```python
if head_dim <= 64:
    if sparse_block_size_q is not None and sparse_block_size_q % 192 != 0:
        return FwdConfig(128, 128, True, True)
    return FwdConfig(192, 128, True, True)
elif head_dim <= 96:
    if sparse_block_size_q is not None and sparse_block_size_q % 192 != 0:
        return FwdConfig(128, 128, False, True)
    if is_causal or is_local:
        return FwdConfig(192, 128, False, True)
    else:
        return FwdConfig(192, 144, False, True)
elif head_dim <= 128:
    return FwdConfig(128, 128, True, True)
elif head_dim <= 192:
    tile_n = 96 if is_local else (128 if head_dim_v <= 128 else 112)
    return FwdConfig(128, tile_n, True, True)
else:
    tile_n = 64 if is_local else 80
    return FwdConfig(128, tile_n, True, True)
```

更详细的表格如下：

| 条件 | `tile_m` | `tile_n` | `mma_pv_is_rs` | `intra_wg_overlap` | 主要原因 |
| --- | --- | --- | --- | --- | --- |
| `head_dim <= 64`，非 sparse 限制 | 192 | 128 | True | True | headdim 小，shared/reg 压力低，可以用 3 个 MMA WG 扩大 M，提高每个 CTA 的 Q rows 工作量。 |
| `head_dim <= 64`，`sparse_block_size_q` 不能被 192 整除 | 128 | 128 | True | True | block sparse 的 Q block 粒度要求 `tile_m` 能整除 sparse Q block，只能退回 128。 |
| `64 < head_dim <= 96`，causal/local | 192 | 128 | False | True | 仍然能承受 3 个 MMA WG，但 P 留寄存器代价过高，走 shared P；causal/local 用更保守的 N=128。 |
| `64 < head_dim <= 96`，非 causal/local | 192 | 144 | False | True | 没有复杂 mask 时可以把 N 拉到 144，增加每次 K/V tile 的工作量。 |
| `64 < head_dim <= 96`，sparse Q block 不兼容 192 | 128 | 128 | False | True | 同样受 sparse Q block 粒度约束，退回 M=128。 |
| `96 < head_dim <= 128` | 128 | 128 | True | True | headdim 到 128 后，3 个 WG 的寄存器/accumulator 压力不划算，改成 2 个 MMA WG。 |
| `128 < head_dim <= 192`，local | 128 | 96 | True | True | local attention 有窗口边界和更多 mask 区域，缩小 N 降低无效/边界工作和资源压力。 |
| `128 < head_dim <= 192`，非 local 且 `head_dim_v <= 128` | 128 | 128 | True | True | V 维不大，PV accumulator 和 shared V 压力可控，保留 N=128。 |
| `128 < head_dim <= 192`，非 local 且 `head_dim_v > 128` | 128 | 112 | True | True | V 维更大，PV 和 shared V 压力上升，把 N 从 128 降到 112。 |
| `192 < head_dim <= 256`，local | 128 | 64 | True | True | headdim 最大且 local mask 复杂，N 需要明显缩小。 |
| `192 < head_dim <= 256`，非 local | 128 | 80 | True | True | 非 local 可以比 local 稍大，但 headdim=256 时 N=128 的资源压力过高。 |

这里的核心不是单纯追求更大的 tile，而是在四类资源之间取平衡：

- `tile_m` 决定一个 CTA 处理多少 Q row，也决定 consumer warpgroup 数量。`tile_m=128` 对应 2 个 MMA WG，block 通常 384 threads；`tile_m=192` 对应 3 个 MMA WG，block 通常 512 threads。
- `tile_n` 决定每轮 mainloop 搬多少 K/V row。`tile_n` 越大，QK 一轮覆盖的 score 列越多，循环次数更少；但 `sK/sV` shared memory、TMA payload、softmax fragment、P fragment 都会变大。
- `head_dim` 决定 QK 的 K 维长度，也直接放大 `sQ/sK` 和 WGMMA K-loop 成本。
- `head_dim_v` 决定 PV 的输出宽度和 `sV/acc_O` 压力。`head_dim_v > 128` 时即使 `head_dim <= 192`，也会把 `tile_n` 从 128 降到 112。

### 4.1 为什么小 headdim 可以用 `M=192`

Hopper 这里的 tiled MMA 是按 64 行 Q 为一个 warpgroup 粒度组织的：

```python
atom_layout_mnk = (tile_m // 64, 1, 1)
```

所以：

```text
tile_m=128 -> atom_layout_mnk=(2,1,1) -> 2 个 MMA warpgroup
tile_m=192 -> atom_layout_mnk=(3,1,1) -> 3 个 MMA warpgroup
```

当 `head_dim <= 64` 或部分 `head_dim <= 96` 场景，单个 Q row 的 K 维较短，`sQ/sK` 占用、QK accumulator、softmax 中间量都比较小。此时增加一个 consumer warpgroup，扩大到 `M=192`，可以让一个 CTA 一次处理更多 Q row，减少 CTA 数量和调度开销，同时提高单 CTA 的 Tensor Core 工作量。

但当 `head_dim >= 128` 后，单行 Q/K 的数据量和 accumulator 压力明显上升。继续用 `M=192` 会让 3 个 consumer warpgroup 同时持有更多 accumulator、softmax state 和 P/O fragment，寄存器预算会被压得很紧，所以源码退回 `M=128`。

### 4.2 为什么 `head_dim <= 96` 反而 `mma_pv_is_rs=False`

`mma_pv_is_rs` 的意思是 PV 的 A operand，也就是 softmax 后的 `P`，是否留在 register 里直接参与 WGMMA：

```text
True:
    S -> softmax -> P(register) -> PV WGMMA

False:
    S -> softmax -> P(shared memory) -> PV WGMMA
```

直觉上 register 路径少了一次 shared memory 往返，应该更快。但源码注释里明确说 Python/CuTe SM90 kernel 对 `head_dim <= 96` 的 `192x` tile 来说，RS 路径性能很差，因此选择 `noRS + overlap`。

原因可以从资源压力理解：

- `tile_m=192` 意味着 3 个 MMA WG。
- `tile_n=128/144` 意味着每个 WG 的 score/P tile 是 `64 x 128` 或 `64 x 144`。
- 如果 P 还留在寄存器中，consumer 还要同时保留 `acc_O`、softmax max/sum、score fragment、P fragment。
- 对 `head_dim=96` 这类不上不下的形状，register pressure 和调度压力可能比 shared memory 往返更贵。

所以这里把 P 放到 shared memory，不是因为 shared memory 更理想，而是为了释放寄存器，让 3-WG、`M=192` 的整体吞吐更好。

### 4.3 为什么非 causal 的 `head_dim <= 96` 可以用 `N=144`

`tile_n` 增大会带来两个直接收益：

- K/V mainloop 轮数减少。
- 每次 QK/PV WGMMA 覆盖更多列，Tensor Core 工作量更饱满。

但是 causal/local 需要处理更多边界：

```text
causal: 当前 Q row 只能看见它左侧/当前位置的 K
local: 当前 Q row 只能看见窗口内 K
```

这些 mask 会让某些 N block 只有部分元素有效。`N` 越大，一个 block 内混入的无效区域和边界判断越多。对 causal/local，源码用 `N=128`；对非 causal/local，没有这类复杂可见性边界，可以用 `N=144` 增加工作量。

### 4.4 为什么大 headdim 逐步缩小 `N`

这里更应该从 FA3 Hopper forward 的流水来理解，而不是只看 shared memory 容量。SM90 forward 的一个 CTA 大致是：

```text
producer warpgroup:
    Q tile 只加载一次
    K/V tile 按 N block 流式加载，K/V 都是 2-stage pipeline

consumer warpgroups:
    对每个 N block:
        wait K(stage)
        QK WGMMA: Q[tile_m, headdim] @ K[tile_n, headdim]^T
        score_mod / mask / online softmax
        P 准备在 register 或 shared
        wait V(stage)
        PV WGMMA: P[tile_m, tile_n] @ V[tile_n, headdim_v]
```

开启 `intra_wg_overlap=True` 后，中间稳定态不是简单的顺序执行，而是尽量做成：

```text
producer:
    load K[next]
    load V[current]

consumer:
    QK(next) overlaps PV(current)
```

所以 `tile_n` 不是越大越好。它会同时放大流水线里的几个阶段：

- K 的 TMA payload 是 `tile_n x head_dim`。`head_dim` 越大，同样的 `tile_n` 会让 K load 时间更长，也占用更多 K pipeline stage 空间。
- QK WGMMA 的输出 score tile 是 `64 x tile_n` per consumer WG，K-loop 长度是 `head_dim`。`head_dim` 和 `tile_n` 同时变大时，QK 阶段会变重。
- softmax 处理的是同一个 `64 x tile_n` score tile。`tile_n` 越大，每行 max/sum 更新、mask、score fragment、P fragment 都越大。
- V 的 TMA payload 是 `tile_n x head_dim_v`，PV WGMMA 的 reduction 维也是 `tile_n`。如果 `head_dim_v` 也大，PV 阶段会明显变重。
- `intra_wg_overlap` 需要 QK(next) 和 PV(current) 尽量形成稳定交叠。如果 `tile_n` 过大，其中一个阶段过长，就会把另一阶段拖成等待，流水线空泡变多。

因此大 headdim 时缩小 `N` 的核心原因是：让 FA3 的 K/V 双 stage 流水、QK/PV overlap、softmax 中间片段和 shared/register 工作集保持在可调度范围内。

对应到源码分支：

```text
head_dim <= 128:
    N=128
    QK 的 K-loop 还不算太长，score/P tile 也能承受。

128 < head_dim <= 192:
    local: N=96
    non-local 且 head_dim_v <= 128: N=128
    non-local 且 head_dim_v > 128: N=112

192 < head_dim <= 256:
    local: N=64
    non-local: N=80
```

这里 `head_dim_v` 单独影响 `N`，是因为 PV 流水的压力主要来自 V 和 O：

```text
V load:      tile_n x head_dim_v
PV output:   64 x head_dim_v per consumer WG
PV reduction: tile_n
```

所以在 `head_dim <= 192` 但 `head_dim_v > 128` 时，QK 侧还可以接受较大的 N，但 PV 侧会变重，于是源码把 `N=128` 降到 `N=112`。

local attention 则会进一步缩小 `N`。原因不是 local 一定计算更少，而是 local window 让很多 N block 处在边界区域：

```text
左窗口边界 block
中间完全可见 block
右窗口/causal 边界 block
```

如果 `N` 太大，一个 N block 里会混入更多无效列或边界列；producer 仍然要加载整块 K/V，consumer 仍然要处理更大的 score tile，再通过 mask 抹掉无效部分。缩小 `N` 可以提高 local/window 场景下的流水粒度，减少边界 block 的浪费。

所以这段逻辑不是简单的 “headdim 大导致 shared memory 不够”，而是 FA3 的整条流水线在变重：

```text
K TMA 更重
QK WGMMA 更重
softmax/P fragment 更大
V TMA 和 PV WGMMA 可能更重
QK/PV overlap 更难平衡
local/window 边界浪费更明显
```

缩小 `tile_n` 是为了让每个 N block 的搬运、计算、softmax 和 overlap 都维持在 Hopper 上比较稳定的节奏里。

### 4.5 `sparse_block_size_q` 为什么会影响 `tile_m`

block sparse 场景下，Q 方向有自己的稀疏 block 粒度。源码注释说：

```text
When sparse_block_size_q is set, tile_m must divide it.
```

所以如果原本想用 `tile_m=192`，但 `sparse_block_size_q % 192 != 0`，这个 tile 不能和 sparse Q block 对齐。此时必须退回：

```text
M=128, N=128
```

这不是性能优先，而是 layout/scheduler 正确性约束优先。

### 4.6 `intra_wg_overlap=True` 的作用

所有 SM90 分支都返回 `intra_wg_overlap=True`。它表示 consumer 端尽量把相邻 N block 的 QK 和 PV 重叠：

```text
QK(next block) overlaps PV(current block)
```

这要求 producer 端 K/V load 也错位安排：

```text
load K[next]
load V[current]
```

好处是减少 WGMMA pipeline 空泡。FlashAttention 每个 N block 都要做两类 GEMM：

```text
QK: Q @ K^T
PV: P @ V
```

如果严格顺序执行，QK 完成、softmax 完成、PV 完成之后才进入下一块，Tensor Core 和 memory pipeline 更容易互相等待。开启 overlap 后，下一块的 QK 可以和当前块的 PV 有更好的交叠。

### 4.7 选择逻辑的整体心智模型

可以把 SM90 tile 选择理解成下面这条规则：

```text
headdim 小:
    用更大的 M=192，让更多 Q rows 进一个 CTA。
    N 保持 128，非 causal 的 hdim<=96 可放到 144。

headdim 中等:
    回到 M=128，避免 3 个 WG 的寄存器压力。
    N 根据 V 维和 local mask 在 96/112/128 之间调节。

headdim 大:
    M 仍为 128，但 N 必须缩到 64/80。
    否则 shared memory 和 accumulator 压力过高。
```

所以这段代码不是一个简单的经验表，而是 Hopper 上 `TMA shared memory 容量 + WGMMA warpgroup 数 + register pressure + mask 区域复杂度 + block sparse 对齐约束` 共同作用后的结果。

## 5. 构造 `FlashAttentionForwardSm90`

Hopper 对象构造大致如下：

```python
fa_fwd = FlashAttentionForwardSm90(
    dtype,
    head_dim,
    head_dim_v,
    qhead_per_kvhead,
    is_causal=causal,
    is_local=local,
    pack_gqa=pack_gqa,
    tile_m=tile_m,
    tile_n=tile_n,
    num_stages=2,
    num_threads=num_threads,
    Q_in_regs=False,
    intra_wg_overlap=intra_wg_overlap,
    mma_pv_is_rs=mma_pv_is_rs,
    mask_mod=mask_mod,
    score_mod=score_mod,
    has_aux_tensors=aux_tensors is not None,
    q_subtile_factor=q_subtile_factor,
    paged_kv_non_tma=page_size not in [None, tile_n],
)
```

关键参数：

- `num_stages=2`: Hopper forward 的 K/V pipeline 默认双 stage。
- `Q_in_regs=False`: Q 先进入 shared memory，再由 WGMMA 使用。
- `pack_gqa`: GQA/MQA 场景下是否把多个 Q head pack 到 M 维处理。
- `paged_kv_non_tma`: paged KV 的 page size 如果不等于 `tile_n`，K/V 不能直接走 TMA tile，需要回落到 cp.async 路径。

## 6. compile key 和 JIT cache

`_flash_attn_fwd.compile_cache` 以 compile key 缓存 CuTe JIT 编译产物。compile key 会包含影响 kernel 代码生成的静态信息，例如：

- dtype。
- `head_dim`、`head_dim_v`。
- `qhead_per_kvhead`。
- causal/local。
- score/mask modifier 的 hash。
- block sparsity 相关静态参数。
- 是否返回 LSE。
- dense/varlen/paged KV。
- `tile_m`、`tile_n`、`num_threads`、`num_stages`。
- `pack_gqa`。
- `arch`。
- paged KV 是否非 TMA。
- `q_subtile_factor`。
- `mma_pv_is_rs`。
- `intra_wg_overlap`。
- scheduler 类型。

这个 cache 很重要：同一类 shape 和静态配置只编译一次，后续调用直接执行 compiled kernel。

## 7. CuTe tensor 封装

进入 `cute.compile` 之前，接口层会把 PyTorch tensor 转成 CuTe tensor/memref。这里有几个重要 layout 转换。

dense Q/O 常见逻辑布局：

- 用户视角：`(batch, seqlen_q, head, dim)`
- kernel 内部经常按局部 tile 访问：`(seqlen_q, dim, head, batch)` 或等价转置 layout

K/V 类似：

- 用户视角：`(batch, seqlen_k, head_kv, dim)`
- kernel 内部按照 `N x headdim` 的 tile 访问 K，按照 `N x headdim_v` 访问 V。

LSE：

- dense 常见逻辑是 `(batch, head, seqlen_q)`，kernel 内部转成适合每个 Q row 写回的 layout。

这些 layout 转换不是数据拷贝，主要是 CuTe memref stride/view 的重解释。

## 8. `FlashAttentionForwardSm90.__call__` 是进入 kernel 前的核心

`FlashAttentionForwardSm90.__call__` 是 Hopper forward 在进入 `@cute.kernel` 之前最重要的一层。它不是普通的 Python wrapper，而是 CuTe JIT 下的 launch-configuration 函数：它把 `_flash_attn_fwd` 已经决定好的静态配置，进一步转成 kernel 需要的 layout、copy atom、TMA tensor、tiled MMA、scheduler 参数、grid/block 和 kernel 参数列表。

可以把 `__call__` 看成这条边界：

```text
_flash_attn_fwd:
    选择实现类、tile_m/tile_n、compile key、编译缓存

FlashAttentionForwardSm90.__call__:
    把这次调用具体化为 SM90 kernel launch

FlashAttentionForwardSm90.kernel:
    真正的 device-side producer/consumer 执行逻辑
```

`__call__` 的总体顺序是：

```text
1. 类型检查和 varlen 标记
2. 对 Q/K/V/O/LSE 做 layout view 转换
3. 创建 QK/PV tiled MMA
4. 根据 tiled MMA 推导线程数、warpgroup 数、寄存器预算
5. 判断 TMA/cp.async 路径和 overlap/barrier 策略
6. 生成 shared memory layout 和 SharedStorage 类型
7. 处理 pack GQA 的 logical layout
8. 创建 TMA copy op、TMA atom 和 TMA tensor descriptor
9. 选择 tile scheduler，生成 scheduler 参数和 grid
10. 计算 softmax scale、window 参数、fastdiv 参数
11. 调用 `self.kernel(...).launch(...)`
```

### 8.1 函数签名说明

`__call__` 的主要参数包括：

```python
def __call__(
    self,
    mQ,
    mK,
    mV,
    mO,
    mLSE,
    softmax_scale,
    mCuSeqlensQ=None,
    mCuSeqlensK=None,
    mSeqUsedQ=None,
    mSeqUsedK=None,
    mPageTable=None,
    window_size_left=None,
    window_size_right=None,
    learnable_sink=None,
    blocksparse_tensors=None,
    aux_tensors=None,
    stream=None,
)
```

这些参数已经是 CuTe tensor/memref，不再是原始 PyTorch tensor。它们来自 `_flash_attn_fwd` 里对 tensor 的封装。

输入形态大致有三类：

- dense：`mQ/mK/mV/mO` 都带 batch 维。
- varlen：通过 `mCuSeqlensQ/mCuSeqlensK` 或 `mSeqUsedQ/mSeqUsedK` 描述每个 batch 的真实长度。
- paged KV：通过 `mPageTable` 把 logical K/V block 映射到 page storage。

`__call__` 不会真的执行 attention，它只决定这次 launch 要怎么执行。

### 8.2 类型检查和 varlen 标记

第一步：

```python
self._check_type(...)
self.varlen_q = mCuSeqlensQ is not None or mSeqUsedQ is not None
```

`_check_type` 校验 Q/K/V/O/LSE/cu_seqlens/seq_used 的 dtype 是否符合 kernel 假设。`varlen_q` 是后面两个地方的重要分支：

- Q/O/LSE 的 layout transpose 方式。
- tile scheduler 选择 `SingleTileVarlenScheduler` 还是 dense scheduler。

### 8.3 对 Q/K/V/O/LSE 做 layout view 转换

接着：

```python
mQ, mK, mV, mO = [assume_tensor_aligned(t) for t in (mQ, mK, mV, mO)]
```

这一步给 CuTe 编译器一个对齐假设，后续 TMA/cp.async/vectorized copy 都依赖对齐信息。

然后是 layout select：

```python
QO_layout_transpose = [1, 3, 2, 0] if mCuSeqlensQ is None else [0, 2, 1]
mQ, mO = [layout_utils.select(t, QO_layout_transpose) for t in (mQ, mO)]

KV_layout_transpose = [1, 3, 2, 0] if mCuSeqlensK is None else [0, 2, 1]
mK, mV = [layout_utils.select(t, KV_layout_transpose) for t in (mK, mV)]

LSE_layout_transpose = [2, 1, 0] if mCuSeqlensQ is None else [1, 0]
mLSE = layout_utils.select(mLSE, LSE_layout_transpose) if mLSE is not None else None
```

dense 情况下，原始逻辑通常是：

```text
Q/O: batch, seqlen_q, head, dim
K/V: batch, seqlen_k, head_kv, dim
LSE: batch, head, seqlen_q
```

select 后，kernel 更容易按下面的方式切 tile：

```text
Q/O: seqlen_q, dim, head, batch
K/V: seqlen_k, dim, head_kv, batch
LSE: seqlen_q, head, batch
```

varlen 情况下没有普通 dense batch 维，select 变成：

```text
Q/O: total_q, dim, head
K/V: total_k, dim, head_kv
LSE: total_q, head
```

这一步非常关键：后面的 TMA atom、local tile、scheduler 都假设 Q/K/V 的 M/N 维在第 0 维，head_dim 在第 1 维，head 在后面的维度。

### 8.4 创建 QK/PV 两个 tiled MMA

`__call__` 调用：

```python
tiled_mma_qk, tiled_mma_pv = self._get_tiled_mma()
```

SM90 forward 有两个 WGMMA：

```text
QK: Q[tile_m, head_dim] @ K[tile_n, head_dim]^T -> S[tile_m, tile_n]
PV: P[tile_m, tile_n] @ V[tile_n, head_dim_v]   -> O[tile_m, head_dim_v]
```

对应源码：

```python
tiled_mma_qk = make_trivial_tiled_mma(
    dtype, dtype,
    OperandMajorMode.K,
    OperandMajorMode.K,
    Float32,
    atom_layout_mnk=(tile_m // 64, 1, 1),
    tiler_mn=(64, tile_n),
)

tiled_mma_pv = make_trivial_tiled_mma(
    dtype, dtype,
    OperandMajorMode.K,
    OperandMajorMode.MN,
    Float32,
    atom_layout_mnk=(tile_m // 64, 1, 1),
    tiler_mn=(64, tile_hdimv),
    a_source=RMEM if mma_pv_is_rs else SMEM,
)
```

几点要注意：

- `atom_layout_mnk=(tile_m // 64, 1, 1)` 表示沿 M 方向复制 WGMMA warpgroup tile。
- QK 的 `tiler_mn=(64, tile_n)`，每个 consumer warpgroup 一次负责 64 行 Q 和 `tile_n` 列 K。
- PV 的 `tiler_mn=(64, tile_hdimv)`，每个 consumer warpgroup 一次负责 64 行 Q 和 `tile_hdimv` 列 O。
- `mma_pv_is_rs=True` 时，PV 的 A operand P 来自 register；否则来自 shared memory `sP`。

### 8.5 从 tiled MMA 推导线程组织

创建 tiled MMA 后，`__call__` 用它反推 consumer 线程数：

```python
self.num_mma_threads = tiled_mma_qk.size
self.num_threads_per_warp_group = 128
self.num_wg_mma = self.num_mma_threads // 128
assert self.num_wg_mma in [1, 2, 3]
self.num_threads = 128 * (self.num_wg_mma + 1)
```

这里多出来的 `+1` 是 producer warpgroup：

```text
1 个 producer WG: 加载 Q/K/V
N 个 consumer WG: 做 QK、softmax、PV、epilogue
```

常见情况：

| `tile_m` | `atom_layout_mnk` | consumer WG | producer WG | total threads |
| --- | --- | --- | --- | --- |
| 128 | `(2,1,1)` | 2 | 1 | 384 |
| 192 | `(3,1,1)` | 3 | 1 | 512 |

然后设置不同阶段需要的线程数：

```python
self.num_producer_threads = 32
self.num_Q_load_threads = 128
self.num_epilogue_threads = self.num_mma_threads
```

这里容易误解：`num_producer_threads=32` 主要对应 TMA 发起侧的实际 producer 线程粒度；如果 Q 或 KV 不走 TMA，而是 cp.async，`num_Q_load_threads=128` 和 load group 会让整个 producer warpgroup 参与搬运。

### 8.6 寄存器预算和 producer-consumer 资源分配

`__call__` 继续设置：

```python
self.num_mma_regs, self.num_producer_regs = {
    1: (256, 56),
    2: (240, 24),
    3: (160, 32),
}[self.num_wg_mma]
```

含义：

```text
consumer warpgroup:
    setmaxregister_increase(num_mma_regs)

producer warpgroup:
    setmaxregister_decrease(num_producer_regs)
```

虽然真正的 `setmaxregister` 在 `kernel` 里执行，但预算是在 `__call__` 决定的。

如果 2 个 MMA WG 且 Q 或 KV 有非 TMA 路径：

```python
if self.num_wg_mma == 2 and (not self.use_tma_Q or not self.use_tma_KV):
    self.num_mma_regs, self.num_producer_regs = 224, 40
```

原因是 cp.async load 比 TMA producer 需要更多普通线程参与和更多寄存器，不能把 producer 压得太低，所以从 consumer 让出一部分寄存器。

### 8.7 block sparsity、scheduler barrier、TMA 开关

`__call__` 在 launch 前还会固定几个重要布尔值：

```python
self.use_block_sparsity = blocksparse_tensors is not None

self.use_scheduler_barrier = (
    (self.num_wg_mma >= 2 and self.tile_hdim <= 128)
    if self.intra_wg_overlap
    else (self.num_wg_mma == 2)
)

self.use_tma_Q = self.arch >= Arch.sm_90 and not (
    self.pack_gqa and self.tile_m % self.qhead_per_kvhead != 0
)
self.use_tma_O = self.use_tma_Q
```

`use_scheduler_barrier` 是 intra-wg overlap 里协调 consumer warpgroup 节奏的开关。`tile_hdim <= 128` 时 QK/PV 更容易做紧密 overlap，因此用 scheduler barrier。

`use_tma_Q` 受 pack GQA 影响：如果 pack GQA 后 `tile_m` 不能被 `qhead_per_kvhead` 整除，Q 的 tile 不是规则 TMA tile，就不能直接用普通 TMA 描述。`use_tma_O` 跟 Q 保持一致，因为 O 的 pack GQA layout 和 Q 对称。

`use_tma_KV` 在对象构造时已经由 `paged_kv_non_tma` 决定。paged KV 的 `page_size != tile_n` 时，KV 需要走非 TMA 路径。

### 8.8 `rescale_O_before_gemm`

还有一个细节：

```python
self.rescale_O_before_gemm = self.tile_hdimv > 128 and self.intra_wg_overlap
```

online softmax 每处理一个 N block 都可能需要把旧的 `acc_O` 乘以 `row_scale`。当 `head_dim_v > 128` 且开启 overlap 时，`acc_O` fragment 更大，PV 也更重。这里决定 rescale O 的位置，以便在 QK/PV overlap 的调度中减少等待和寄存器压力。

### 8.9 生成 shared memory layout

`__call__` 调用：

```python
self._setup_attributes()
```

这个函数来自 forward base，会准备 cp.async fallback 的 tiled copy atom，以及一些基础 copy layout。

随后 SM90 路径重新生成 shared memory layout：

```python
self.sQ_layout, self.sK_layout, self.sV_layout, self.sO_layout = [
    sm90_utils.make_smem_layout(mX.element_type, LayoutEnum.ROW_MAJOR, shape, stage)
    for mX, shape, stage in [
        (mQ, (tile_m, tile_hdim), None),
        (mK, (tile_n, tile_hdim), num_stages),
        (mV, (tile_n, tile_hdimv), num_stages),
        (mO, (tile_m, tile_hdimv), None),
    ]
]
```

shape 含义：

```text
sQ: tile_m x tile_hdim
sK: tile_n x tile_hdim x num_stages
sV: tile_n x tile_hdimv x num_stages
sO: tile_m x tile_hdimv
```

如果 `mma_pv_is_rs=False`，还会创建：

```python
self.sP_layout = sm90_utils.make_smem_layout(
    mV.element_type,
    LayoutEnum.ROW_MAJOR,
    (tile_m, tile_n),
)
```

也就是 softmax 后的 P 要落 shared memory，供 PV WGMMA 读取。

最后：

```python
SharedStorage = self._get_shared_storage_cls()
```

这会根据 `Q_in_regs` 和 `sP_layout` 生成真正的 shared storage 类型，里面包含 `sQ/sK/sV/sP` 以及 TMA/cp.async pipeline 用的 mbarrier storage。

### 8.10 pack GQA layout

在创建 TMA atom 之前，`__call__` 会处理 pack GQA：

```python
mQ_og, mO_og = mQ, mO
if self.pack_gqa:
    nheads_kv = mK.shape[2]
    mQ = pack_gqa_layout(mQ, qhead_per_kvhead, nheads_kv, head_idx=2)
    mO = pack_gqa_layout(mO, qhead_per_kvhead, nheads_kv, head_idx=2)
    if mLSE is not None:
        mLSE = pack_gqa_layout(mLSE, qhead_per_kvhead, nheads_kv, head_idx=1)
```

pack GQA 的目的，是把多个 Q head 对同一个 KV head 的关系折叠到更适合 tile scheduler 和 TMA 描述的 layout 中。注意 `mQ_og/mO_og` 会保留下来，因为某些 TMA atom 需要基于原始 tensor layout 创建 descriptor。

## 9. `__call__` 里的 TMA、cp.async 和 kernel 参数准备

这一节仍然是在 `__call__` 内部，不是 kernel 内部。它的核心任务是为 Q/K/V/O 创建 copy 描述，并决定传给 kernel 的到底是 TMA tensor 还是普通 tensor。

### 9.1 创建 TMA copy op

Hopper TMA 主路径先定义三个 bulk copy op：

```python
gmem_tiled_copy_Q = cpasync.CopyBulkTensorTileG2SOp()
gmem_tiled_copy_KV = cpasync.CopyBulkTensorTileG2SOp()
gmem_tiled_copy_O = cpasync.CopyBulkTensorTileS2GOp()
```

含义：

```text
Q/K/V: global tensor tile -> shared memory tile
O:     shared memory tile -> global tensor tile
```

这里的 `cpasync` 命名容易让人混淆。对 Hopper TMA 来说，它对应的是 bulk tensor copy 能力；而非 TMA fallback 才是普通 cp.async tiled copy。

### 9.2 计算 TMA transaction bytes

`__call__` 还会为 TMA pipeline 记录每个 tile 的 bytes：

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

这个值后面在 `kernel` 里创建 `PipelineTmaAsync` 时作为 `tx_count`：

```text
pipeline_q tx_count = Q tile bytes
pipeline_k tx_count = K tile bytes
pipeline_v tx_count = V tile bytes
```

TMA mbarrier 需要知道一次 transaction 期望到达多少 bytes，consumer 才能正确等待 full barrier。

### 9.3 创建 TMA atom 和 TMA tensor

普通路径用：

```python
cpasync.make_tiled_tma_atom
```

pack GQA 路径用：

```python
make_packgqa_tiled_tma_atom
```

先解释 `cpasync.make_tiled_tma_atom` 本身。它的函数签名可以简化成：

```python
make_tiled_tma_atom(
    op,
    gmem_tensor,
    smem_layout,
    cta_tiler,
    num_multicast=1,
)
```

它的输入分别是：

- `op`: TMA copy 操作类型，例如 `CopyBulkTensorTileG2SOp()` 或 `CopyBulkTensorTileS2GOp()`。
- `gmem_tensor`: global memory 里的 CuTe tensor view，带有 shape/stride/layout 信息。
- `smem_layout`: shared memory tile 的 layout，告诉 TMA 数据到 shared 后应该按什么 swizzle/layout 存。
- `cta_tiler`: CTA 级别一次 TMA 搬运的逻辑 tile 形状，例如 `(tile_m, tile_hdim)` 或 `(tile_n, tile_hdim)`。
- `num_multicast`: multicast factor。这里 FlashAttention forward 基本传 `1`，也就是不做 multicast。

它做的核心事情有三步：

```text
1. 检查 smem_layout 和 cta_tiler 的 rank 是否匹配。
   smem_layout 可以是非 staged，也可以比 cta_tiler 多一个 stage 维。

2. 如果 smem_layout 带 stage 维，只取前面和 cta_tiler 对应的维度。
   例如 sK_layout 是 tile_n x tile_hdim x num_stages，
   真正的 TMA atom 只描述单个 stage 的 tile_n x tile_hdim。

3. 用 gmem tensor 的 identity layout 和 cta_tiler 组合出 cta_v_map，
   再调用底层 NVGPU IR 创建 TMA atom 和 TMA tensor。
```

返回值可以解包为：

```python
tma_atom, tma_tensor = cpasync.make_tiled_tma_atom(...)
```

二者含义不同：

- `tma_atom`: 描述“怎么 copy”的 copy atom。它绑定了 TMA op、TMA descriptor trait、shared memory layout 和 tile copy 规则。kernel 里发起 TMA load/store 时需要它。
- `tma_tensor`: 描述“从 global memory 的哪里 copy”的 TMA tensor view。它把普通 CuTe tensor 的逻辑坐标转换成 TMA 单元消费的坐标，内部包含 TMA layout 需要的 basis stride 信息。后面 `local_tile` 或 `tma_get_copy_fn` 会基于它切出某个 CTA 的 global tile。

所以 `make_tiled_tma_atom` 不是简单创建一个 copy 函数，而是同时完成：

```text
global tensor layout
    + CTA tile shape
    + shared memory layout/swizzle
    + TMA copy op
    -> TMA copy atom + TMA-compatible tensor view
```

这也是为什么它必须在 `__call__` 里做：这些信息都属于 launch 前确定的静态结构，kernel 内部只消费生成好的 atom/tensor。

Q 的 TMA 创建：

```python
if self.use_tma_Q:
    tma_atom_Q, tma_tensor_Q = make_tiled_tma_atom_fn(
        gmem_tiled_copy_Q,
        mQ_og if self.pack_gqa else mQ,
        self.sQ_layout,
        (tile_m, tile_hdim),
    )
```

K/V 的 TMA 创建：

```python
if self.use_tma_KV:
    tma_atom_K, tma_tensor_K = cpasync.make_tiled_tma_atom(
        gmem_tiled_copy_KV,
        mK,
        cute.select(self.sK_layout, mode=[0, 1]),
        (tile_n, tile_hdim),
        1,
    )
    tma_atom_V, tma_tensor_V = cpasync.make_tiled_tma_atom(
        gmem_tiled_copy_KV,
        mV,
        cute.select(self.sV_layout, mode=[0, 1]),
        (tile_n, tile_hdimv),
        1,
    )
```

K/V 只 select layout 的 `[0, 1]`，因为 TMA atom 描述的是单个 stage 的 2D tile：

```text
K stage tile: tile_n x tile_hdim
V stage tile: tile_n x tile_hdimv
```

stage 维由 pipeline state 在 kernel 里选择。

换句话说，K/V 的 shared layout 真实形状是：

```text
sK: tile_n x tile_hdim  x num_stages
sV: tile_n x tile_hdimv x num_stages
```

但是一次 TMA transaction 只负责把一个 K 或 V tile 搬进某一个 pipeline stage：

```text
TMA atom 描述: tile_n x tile_hdim
pipeline state 决定: 写入 stage 0 还是 stage 1
```

如果把 stage 维也放进 TMA atom 的 `cta_tiler`，语义就会变成一次 TMA copy 覆盖多个 pipeline stage，这和 double-buffer pipeline 的设计相反。

O 的 TMA 创建：

```python
if self.use_tma_O:
    mO_tma = mO_og if self.pack_gqa else mO
    if self.varlen_q:
        mO_tma = create_ragged_tensor_for_tma(mO_tma, ragged_dim=0, ptr_shift=True)
    tma_atom_O, tma_tensor_O = make_tiled_tma_atom_fn(
        gmem_tiled_copy_O,
        mO_tma,
        self.sO_layout,
        (tile_m, tile_hdimv),
    )
```

varlen Q 的 O store 需要 ragged tensor，因为每个 batch 的 Q 起点不是规则 dense stride。

pack GQA 路径要多一层 `make_packgqa_tiled_tma_atom`，原因是直接 pack 成：

```text
((qhead_per_kvhead, seqlen), headdim, nheads_kv, batch)
```

会让 TMA 维度变多。wrapper 先把 head 和 seqlen group 到一起，保持普通 TMA 维度，例如：

```text
(seqlen, d, nheads, b)
  -> ((nheads, seqlen), d, b)
```

然后用普通 `cpasync.make_tiled_tma_atom` 创建 TMA atom，最后再把返回的 `tma_tensor` unpack 回 pack GQA 需要的逻辑 layout：

```text
((nheads, seqlen), d, b)
  -> ((qhead_per_kvhead, seqlen), d, nheads_kv, b)
```

这层 wrapper 的目的不是改变 copy 语义，而是在 pack GQA 场景下仍然让 TMA descriptor 保持可表达、维度不过度膨胀。

### 9.4 TMA 路径和 cp.async fallback 在 `__call__` 的边界

`__call__` 不直接执行 cp.async。它只通过传参决定 kernel 里面走哪条路径：

```python
self.kernel(
    tma_tensor_Q if self.use_tma_Q else mQ,
    tma_tensor_K if self.use_tma_KV else mK,
    tma_tensor_V if self.use_tma_KV else mV,
    tma_tensor_O if self.use_tma_O else mO,
    ...
    tma_atom_Q,
    tma_atom_K,
    tma_atom_V,
    tma_atom_O,
    ...
    self.gmem_tiled_copy_Q,
    self.gmem_tiled_copy_K,
    self.gmem_tiled_copy_V,
    self.gmem_tiled_copy_O,
)
```

如果某个 operand 走 TMA：

```text
传入 tma_tensor_X + tma_atom_X
kernel 中创建 PipelineTmaAsync
```

如果某个 operand 不走 TMA：

```text
传入普通 mX tensor + None tma_atom
kernel 中使用 PipelineCpAsync 和 gmem_tiled_copy_X
```

所以 `__call__` 是 TMA/cp.async 分岔的真正位置。

### 9.5 softmax scale、window 和 fastdiv

launch 前还会准备几个 runtime 参数：

```python
softmax_scale_log2, softmax_scale = utils.compute_softmax_scale_log2(
    softmax_scale, self.score_mod
)

window_size_left = Int32(window_size_left) if window_size_left is not None else None
window_size_right = Int32(window_size_right) if window_size_right is not None else None

fastdiv_mods = utils.compute_fastdiv_mods(
    mQ, mK, qhead_per_kvhead, pack_gqa, aux_tensors, mPageTable
)
```

`softmax_scale_log2` 是为了 kernel 内部用 `exp2` 路径做 online softmax。`fastdiv_mods` 是把一些除法/取模预处理成 fast divmod 参数，减少 device 端 scheduler 和 index 计算开销。

## 10. `__call__` 里的 tile scheduler 和 launch

tile scheduler 也在 `__call__` 里决定。它决定 grid 中每个 CTA 处理哪个 `(m_block, head, batch)`。

### 10.1 scheduler 类型选择

源码逻辑：

```python
if mCuSeqlensQ is not None or mSeqUsedQ is not None:
    TileScheduler = SingleTileVarlenScheduler
else:
    TileScheduler = (
        SingleTileScheduler
        if not self.is_causal or self.is_local
        else SingleTileLPTScheduler
    )
```

含义：

- `SingleTileVarlenScheduler`: varlen Q 或 seqUsedQ，需要根据 cu_seqlens/used length 找每个 batch 的有效 Q blocks。
- `SingleTileScheduler`: dense 非 causal，或 local 路径。每个 tile 的工作量比较常规。
- `SingleTileLPTScheduler`: dense causal 路径。LPT 通常是 longest-processing-time 思路，用于 causal 下不同 M block 的有效 K 范围不同、工作量不均衡的情况。

这里有个容易忽略的点：local 虽然也有 mask，但源码选择的是 `SingleTileScheduler`，不是 `SingleTileLPTScheduler`。原因是 local window 让每个 Q block 的 K 范围更接近固定窗口，而不是像全 causal 那样前后 block 工作量差异很大。

### 10.2 scheduler 参数

`__call__` 构造：

```python
tile_sched_args = TileSchedulerArguments(
    ceil_div(size(mQ.shape[0]), tile_m),
    size(mQ.shape[2]),
    size(mQ.shape[3]) if mCuSeqlensQ is None else size(mCuSeqlensQ.shape[0] - 1),
    1,
    size(mK.shape[0]) if mPageTable is None else mK.shape[0] * mPageTable.shape[1],
    mQ.shape[1],
    mV.shape[1],
    total_q=...,
    tile_shape_mn=(tile_m, tile_n),
    mCuSeqlensQ=mCuSeqlensQ,
    mSeqUsedQ=mSeqUsedQ,
    qhead_per_kvhead_packgqa=...,
    element_size=dtype.width // 8,
    is_persistent=False,
    lpt=self.is_causal or self.is_local,
)
```

几个关键字段：

- `ceil_div(size(mQ.shape[0]), tile_m)`: Q 方向 M blocks 数。
- `size(mQ.shape[2])`: head 数。pack GQA 后这里可能是 KV head 视角的 head 维。
- batch 数：dense 用 `mQ.shape[3]`，varlen 用 `mCuSeqlensQ.shape[0] - 1`。
- `num_splits=1`: Hopper forward 这个路径不做 split-KV。
- K 总长度：paged KV 时是 `num_pages * page_size` 的 logical 上限。
- `mQ.shape[1]` 和 `mV.shape[1]`: 分别对应 Q/K 的 headdim 和 V 的 headdimv。
- `tile_shape_mn=(tile_m, tile_n)`: scheduler 需要知道 tile 粒度，才能计算 block index。
- `qhead_per_kvhead_packgqa`: pack GQA 时用于把 Q head/KV head 的关系映射进 tile。
- `element_size`: 用于某些 scheduler 或 TMA/predicate 相关计算。
- `is_persistent=False`: 当前 SM90 forward 不是 persistent scheduler。
- `lpt=self.is_causal or self.is_local`: 告诉 scheduler 参数层这个 attention 是否有非矩形可见区域。

然后：

```python
tile_sched_params = TileScheduler.to_underlying_arguments(tile_sched_args)
grid_dim = TileScheduler.get_grid_shape(tile_sched_params)
```

`tile_sched_params` 会被传进 kernel。kernel 内部 producer 和 consumer 都用同一份 scheduler 参数，保证 load 和 compute 处理同一个 work tile。

### 10.3 kernel 参数列表的结构

`__call__` 最后调用：

```python
self.kernel(...).launch(
    grid=grid_dim,
    block=[self.num_threads, 1, 1],
    stream=stream,
    min_blocks_per_mp=1,
)
```

传给 kernel 的参数可以分为几组：

```text
1. global/TMA tensors:
   mQ, mK, mV, mO, mLSE

2. varlen/paged metadata:
   mCuSeqlensQ, mCuSeqlensK, mSeqUsedQ, mSeqUsedK, mPageTable

3. TMA atoms:
   tma_atom_Q, tma_atom_K, tma_atom_V, tma_atom_O

4. attention runtime scalars:
   softmax_scale_log2, softmax_scale
   window_size_left, window_size_right
   learnable_sink

5. sparse/aux metadata:
   blocksparse_tensors, aux_tensors, fastdiv_mods

6. shared memory layouts:
   sQ_layout, sK_layout, sV_layout, sO_layout, sP_layout

7. cp.async fallback tiled copies:
   gmem_tiled_copy_Q/K/V/O

8. WGMMA descriptors:
   tiled_mma_qk, tiled_mma_pv

9. scheduler:
   tile_sched_params, TileScheduler

10. shared storage type:
   SharedStorage
```

这也是为什么 `__call__` 很重要：device kernel 本身不再重新推导这些静态结构，它只消费 `__call__` 已经准备好的对象。

### 10.4 launch 配置

最终 launch：

```python
grid = grid_dim
block = [self.num_threads, 1, 1]
min_blocks_per_mp = 1
```

`block` 通常是：

```text
tile_m=128 -> 384 threads
tile_m=192 -> 512 threads
```

`min_blocks_per_mp=1` 是合理的：Hopper FA3 forward 的每个 CTA 使用大量 shared memory、TMA barrier、consumer accumulator 和 warpgroup 资源，本来就不是追求一个 SM 上塞很多 CTA，而是让一个 CTA 内部的 producer-consumer pipeline 跑满。

### 10.5 `__call__` 到 `kernel` 的边界总结

进入 kernel 之前，`__call__` 已经决定了这些事情：

```text
tensor view:
    Q/K/V/O/LSE 的逻辑维度顺序

compute:
    QK tiled MMA
    PV tiled MMA
    consumer warpgroup 数

threads/registers:
    block threads
    producer/consumer register budget

memory:
    shared memory layout
    SharedStorage 类型
    TMA atom / TMA tensor
    cp.async fallback tiled copy

scheduling:
    TileScheduler 类型
    tile_sched_params
    grid_dim

runtime scalars:
    softmax_scale_log2
    window size
    fastdiv mods
```

所以读 Hopper FlashAttention forward 时，`__call__` 是必须重点读的部分。它不是简单 launch，而是把 Python/CuTe 的高层 attention 配置固化成 SM90 kernel 可执行形态的地方。

## 11. 上层链路总流程图

```text
test_flash_attn_output
  |
  | 构造 q/k/v, causal/local/GQA/varlen 参数
  v
flash_attn_func
  |
  | 统一 public API 语义
  | 处理 softmax_scale, window_size, return_lse 等
  v
_flash_attn_fwd
  |
  | 校验 dtype/shape/head_dim
  | 判断 dense / varlen / paged KV / GQA
  | 获取 arch
  v
arch == SM90
  |
  | _tile_size_fwd_sm90
  | 选择 tile_m, tile_n, mma_pv_is_rs, intra_wg_overlap
  v
构造 FlashAttentionForwardSm90
  |
  | 生成 compile_key
  | PyTorch tensor -> CuTe tensor/memref
  v
cute.compile 或 compile_cache 命中
  |
  | FlashAttentionForwardSm90.__call__
  | 创建 TMA descriptor, tiled_mma, scheduler, grid/block
  v
FlashAttentionForwardSm90.kernel
```

## 12. 读源码时的建议顺序

建议按这个顺序看：

1. `tests/cute/test_flash_attn.py::test_flash_attn_output`
2. `interface.py::flash_attn_func`
3. `interface.py::_flash_attn_fwd`
4. `interface.py::_tile_size_fwd_sm90`
5. `flash_fwd.py::FlashAttentionForwardBase.__init__`
6. `flash_fwd_sm90.py::FlashAttentionForwardSm90.__call__`
7. `flash_fwd_sm90.py::FlashAttentionForwardSm90.kernel`

看完这份文档后，进入第二份 kernel 文档会更顺：第二份默认你已经知道 `tile_m/tile_n/num_stages/num_threads` 是从哪里来的。
