# Hopper FlashAttention 上层接口链路

本文围绕 FlashAttention CuTe Hopper / SM90 forward 路径，解释从 public API 到 `FlashAttentionForwardSm90.kernel.launch` 之前发生的事情。重点不是 kernel 内部的 QK、softmax、PV 细节，而是 kernel 之前如何把一个高层 attention 调用，配置成 Hopper 上可以高效执行的 launch。

## 目录

- [1. 总体逻辑](#1-总体逻辑)
- [2. Public API 到 `_flash_attn_fwd`](#2-public-api-到-_flash_attn_fwd)

- [3. SM90 tile 配置](#3-sm90-tile-配置)

- [4. compile key 和 JIT cache](#4-compile-key-和-jit-cache)
- [5. 构造 `FlashAttentionForwardSm90`](#5-构造-flashattentionforwardsm90)
- [6. CuTe tensor 封装](#6-cute-tensor-封装)
- [7. `FlashAttentionForwardSm90.__call__`](#7-flashattentionforwardsm90__call__)
- [8. kernel 入口如何消费这些配置](#8-kernel-入口如何消费这些配置)
- [9. Paged KV 的 TMA 路径和 cp.async fallback](#9-paged-kv-的-tma-路径和-cpasync-fallback)

- [10. Case Study: Qwen2.5-7B-Instruct](#10-case-study-qwen2.5-7B-instruct)
- [11. 源码阅读指南](#11-源码阅读指南)
- [12. 结论](#12-结论)

文中源码路径基于本地仓库：

```text
flash-attention-src/
  flash_attn/cute/interface.py
  flash_attn/cute/flash_fwd.py
  flash_attn/cute/flash_fwd_sm90.py
  flash_attn/cute/tile_scheduler.py
  flash_attn/cute/paged_kv.py
  flash_attn/cute/block_info.py
  flash_attn/cute/seqlen_info.py
  flash_attn/cute/mask.py
  tests/cute/test_flash_attn.py
```

## 1. 总体逻辑

FlashAttention 的上层接口链路可以分成三层：

```text
用户 / 模型语义层
    q/k/v, causal, local, GQA, varlen, paged KV, softmax scale

_flash_attn_fwd 调度层
    shape 校验、dtype 校验、arch 选择、tile 选择、compile key、JIT cache

FlashAttentionForwardSm90.__call__ launch 配置层
    layout view、MMA、TMA descriptor、shared memory layout、scheduler、grid/block
```

kernel 前的工作不是“计算 attention”，而是把动态、宽泛的 Python 调用固化成一个 SM90 kernel 可以直接消费的执行计划：

```text
API 参数
  -> shape / dtype / mask / GQA 语义
  -> SM90 tile_m / tile_n / WGMMA 形态
  -> shared memory layout / TMA copy atom / scheduler params
  -> kernel launch
```

收益主要来自四点：

```text
1. 静态特化
   head_dim、tile、causal/local、TMA/cp.async 等进入 compile key，
   JIT 可以生成更专门的代码。

2. 减少 kernel 内动态推导
   layout、scheduler、MMA、TMA descriptor 在 __call__ 阶段确定。

3. 更好利用 Hopper 硬件
   WGMMA 做 QK/PV，TMA 做大块 global/shared copy，
   producer-consumer pipeline 让搬运和计算重叠。

4. 适配真实推理形态
   GQA、paged KV、varlen、cache seqlens 等 metadata 在 kernel 前整理好。
```

## 2. Public API 到 `_flash_attn_fwd`

### 2.1 dense API 和 varlen / paged API

普通 dense attention 可以走：

```python
flash_attn_func(q, k, v, causal=True, ...)
```

源码入口：

```python
# interface.py
def flash_attn_func(...):
    return FlashAttnFunc.apply(...)
```

带 paged KV cache 的真实推理场景通常走：

```python
flash_attn_varlen_func(
    q,
    k_cache_paged,
    v_cache_paged,
    seqused_k=cache_seqlens,
    page_table=page_table,
    causal=True,
)
```

源码入口：

```python
# interface.py
def flash_attn_varlen_func(...):
    return FlashAttnVarlenFunc.apply(...)
```

`FlashAttnVarlenFunc.forward` 最终调用：

```python
out, lse = _flash_attn_fwd(
    q,
    k,
    v,
    cu_seqlens_q=cu_seqlens_q,
    cu_seqlens_k=cu_seqlens_k,
    seqused_q=seqused_q,
    seqused_k=seqused_k,
    page_table=page_table,
    causal=causal,
    window_size_left=window_size[0],
    window_size_right=window_size[1],
    ...
)
```

所以 `_flash_attn_fwd` 是 forward 的核心 host-side 分发器。

### 2.2 `_flash_attn_fwd` 的输入整理

源码位置：

```python
# interface.py
def _flash_attn_fwd(
    q,
    k,
    v,
    qv=None,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    seqused_q=None,
    seqused_k=None,
    max_seqlen_q=None,
    max_seqlen_k=None,
    min_seqlen_k=None,
    page_table=None,
    softmax_scale=None,
    causal=False,
    ...
):
```

第一步是把输入变成 contiguous-friendly：

```python
q, k, v, qv = [maybe_contiguous(t) for t in (q, k, v, qv)]
```

然后解析 Q 的 shape：

```python
q_shape = q.shape if q is not None else qv.shape
num_head, head_dim = q_shape[-2:]

if cu_seqlens_q is None:
    batch_size, seqlen_q = q_shape[:2]
    total_q = batch_size * seqlen_q
else:
    batch_size = cu_seqlens_q.shape[0] - 1
    seqlen_q = None
    total_q = q_shape[0]
```

如果带 paged KV：

```python
if page_table is not None:
    assert cu_seqlens_k is None
    assert page_table.dtype == torch.int32
    assert page_table.stride(-1) == 1
    max_num_pages_per_seq = page_table.shape[1]
    assert page_table.shape == (batch_size, max_num_pages_per_seq)
    num_pages, page_size = v.shape[:2]
    seqlen_k = num_pages * page_size
else:
    num_pages, page_size = None, None
    seqlen_k = v.shape[-3]
```

这里有一个容易误解的点：`seqlen_k = num_pages * page_size` 是 cache storage 的静态容量视角，不一定等于当前请求真实使用的 KV 长度。真实长度通常通过 `seqused_k` 传入，并在 kernel 里由 `SeqlenInfoQK` 读取。

paged KV 的 shape 校验：

```python
if page_table is not None:
    assert k is None or k.shape == (num_pages, page_size, num_head_kv, head_dim)
    assert v.shape == (num_pages, page_size, num_head_kv, head_dim_v)
```

非 paged dense 的 shape 则是：

```python
assert k is None or k.shape == (batch_size, seqlen_k, num_head_kv, head_dim)
assert v.shape == (batch_size, seqlen_k, num_head_kv, head_dim_v)
```

### 2.3 dtype、head dim、GQA、softmax scale

dtype 校验：

```python
assert v.dtype in [
    torch.float16,
    torch.bfloat16,
    torch.float8_e4m3fn,
    torch.float8_e5m2,
]
```

head 数关系：

```python
assert num_head % num_head_kv == 0
qhead_per_kvhead = num_head // num_head_kv

if pack_gqa is None:
    pack_gqa = qhead_per_kvhead > 1
```

这就是 GQA/MQA 的来源。例如 Qwen2.5-7B 是 28 个 Q heads、4 个 KV heads：

```text
qhead_per_kvhead = 28 / 4 = 7
pack_gqa = True
```

head dim 校验：

```python
alignment = 16 // v.element_size()
if arch // 10 not in [8, 12]:
    _validate_head_dims(head_dim, head_dim_v, arch // 10, alignment)
```

bf16/fp16 是 2 bytes，所以 `alignment=8`。SM90 forward 支持大致范围是 `8 <= head_dim <= 256`，并满足底层 copy/MMA 对齐要求。

softmax scale 默认值：

```python
if softmax_scale is None:
    softmax_scale = 1.0 / math.sqrt(head_dim)
```

### 2.4 causal/local/window 解析

源码调用：

```python
causal, local, window_size_left, window_size_right =
    _resolve_causal_local_window(causal, window_size_left, window_size_right, mask_mod)
```

这里把用户层的 causal/window/mask_mod 语义统一成 kernel 能使用的状态。

decoder-only LLM 的 prefill、chunked prefill、decode 都是 causal。区别是：

```text
普通 prefill:
    seqlen_q = seqlen_k = prompt_len

chunked prefill:
    seqlen_q = current_chunk_len
    seqlen_k = cached_prefix_len + current_chunk_len

decode:
    seqlen_q = 1
    seqlen_k = past_len + 1
```

chunked prefill 里的 causal offset 不是单独传一个 `prefix_len`，而是通过：

```text
seqlen_k - seqlen_q
```

在 block range 和 mask 计算中体现。

## 3. SM90 tile 配置

### 3.1 `_tile_size_fwd_sm90`

源码：

```python
@dataclass(frozen=True)
class FwdConfig:
    m_block_size: int
    n_block_size: int
    mma_pv_is_rs: bool
    intra_wg_overlap: bool

def _tile_size_fwd_sm90(head_dim, head_dim_v, is_causal, is_local, sparse_block_size_q=None):
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

含义：

```text
tile_m:
    每个 CTA 处理多少 Q rows。

tile_n:
    每轮 mainloop 处理多少 K/V rows。

mma_pv_is_rs:
    softmax 后的 P 是否留在 register，直接作为 PV WGMMA 的 A operand。

intra_wg_overlap:
    consumer 端是否尝试让 QK(next) 和 PV(current) 重叠。
```

### 3.2 为什么 tile 选择不是越大越好

`tile_m` 直接决定 consumer warpgroup 数：

```python
atom_layout_mnk = (tile_m // 64, 1, 1)
```

所以：

```text
tile_m=128 -> 2 个 consumer WG
tile_m=192 -> 3 个 consumer WG
```

小 head_dim 时，Q/K 每行短，accumulator 和 shared/register 压力较低，可以用 `M=192` 扩大单 CTA 工作量。

head_dim 到 128 或更大时，QK accumulator、softmax state、P/O fragment、shared memory footprint 都变重，继续用 3 个 consumer WG 不一定划算，所以回到 `M=128`。

`tile_n` 控制每轮读多少 K/V rows。它越大：

```text
K TMA payload 更大
V TMA payload 更大
QK score tile 更大
softmax fragment 更大
PV reduction 维更大
```

head_dim/head_dim_v 越大，`tile_n` 通常越需要缩小，避免 pipeline 失衡、寄存器压力过高、shared memory 过重。

## 4. compile key 和 JIT cache

源码构造 compile key：

```python
compile_key = (
    dtype,
    head_dim,
    head_dim_v,
    qhead_per_kvhead,
    causal,
    score_mod_hash,
    mask_mod_hash,
    use_block_sparsity,
    block_sparse_broadcast_pattern,
    aux_tensor_metadata,
    lse is None,
    cu_seqlens_q is None,
    cu_seqlens_k is None,
    seqused_q is None,
    seqused_k is None,
    page_table is not None,
    window_size_left is not None,
    window_size_right is not None,
    learnable_sink is not None,
    q_descale is not None,
    k_descale is not None,
    v_descale is not None,
    ...
    tile_m,
    tile_n,
    q_stage,
    num_threads,
    is_split_kv,
    pack_gqa,
    arch,
    page_size not in [None, tile_n],
    use_2cta_instrs,
    q_subtile_factor,
    mma_pv_is_rs,
    intra_wg_overlap,
    use_clc_scheduler,
    ...
)
```

这个 key 的原则是：凡是会改变 kernel 代码结构的东西，都要进入 key。

例如：

```text
head_dim/head_dim_v:
    影响 WGMMA K 维、shared layout、predicate。

tile_m/tile_n:
    影响 CTA tile、shared memory、MMA atom、scheduler。

page_size not in [None, tile_n]:
    决定 paged KV 是 TMA path 还是 cp.async path。

pack_gqa:
    改变 Q/O/LSE layout 和 head 映射。

causal/local:
    改变 block range、mask、scheduler。

mma_pv_is_rs:
    改变 P 的来源，register source 或 shared source。
```

compile cache 命中时，后续相同静态配置的调用不需要重新编译：

```python
_flash_attn_fwd.compile_cache = get_jit_cache("fwd")
```

## 5. 构造 `FlashAttentionForwardSm90`

SM90 分支：

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

基类 `FlashAttentionForwardBase.__init__` 做静态属性：

```python
self.dtype = dtype
self.tile_hdim = ceil(head_dim / 16) * 16
self.tile_hdimv = ceil(head_dim_v / 16) * 16
self.check_hdim_oob = head_dim != self.tile_hdim
self.check_hdim_v_oob = head_dim_v != self.tile_hdimv
self.qhead_per_kvhead = qhead_per_kvhead
self.is_causal = is_causal
self.is_local = is_local
self.pack_gqa = pack_gqa
self.tile_m = tile_m
self.tile_n = tile_n
self.num_threads = num_threads
self.num_stages = num_stages
self.Q_in_regs = Q_in_regs
self.score_mod = score_mod
self.mask_mod = mask_mod
self.arch = BaseDSL._get_dsl().get_arch_enum()
```

SM90 子类补充：

```python
self.intra_wg_overlap = intra_wg_overlap
self.mma_pv_is_rs = mma_pv_is_rs
self.buffer_align_bytes = 1024
self.use_tma_KV = not paged_kv_non_tma
self.cluster_shape_mn = (1, 1)
```

注意：

```text
page_size == tile_n
    -> paged_kv_non_tma=False
    -> use_tma_KV=True

page_size != tile_n
    -> paged_kv_non_tma=True
    -> use_tma_KV=False
```

## 6. CuTe tensor 封装

compile cache miss 时，PyTorch tensor 被转成 CuTe tensor：

```python
page_table_tensor = (
    to_cute_tensor(page_table, assumed_align=4, leading_dim=1)
    if page_table is not None
    else None
)

q_tensor, k_tensor, v_tensor, o_tensor = [
    to_cute_tensor(t) for t in (q, k, v, out)
]
```

这一步保留 pointer、shape、stride、dtype 信息，供 CuTe JIT 在 `__call__` 中生成 layout、TMA、MMA、scheduler。

然后：

```python
_flash_attn_fwd.compile_cache[compile_key] = cute.compile(
    fa_fwd,
    q_tensor,
    k_tensor,
    v_tensor,
    o_tensor,
    lse_tensor,
    softmax_scale,
    cu_seqlens_q_tensor,
    cu_seqlens_k_tensor,
    seqused_q_tensor,
    seqused_k_tensor,
    page_table_tensor,
    window_size_left,
    window_size_right,
    learnable_sink_tensor,
    sparse_tensors,
    cute_aux_tensors,
    current_stream,
    options="--enable-tvm-ffi",
)
```

`cute.compile` 编译的是 `fa_fwd.__call__` 这个 JIT launch configuration 函数，而不是只编译 device kernel。

## 7. `FlashAttentionForwardSm90.__call__`

`__call__` 是进入 kernel 前最重要的函数。它把 `_flash_attn_fwd` 的静态决策转成 kernel launch 所需的具体对象。

### 7.1 类型检查和 layout select

源码：

```python
self._check_type(...)
self.varlen_q = mCuSeqlensQ is not None or mSeqUsedQ is not None

mQ, mK, mV, mO = [assume_tensor_aligned(t) for t in (mQ, mK, mV, mO)]

QO_layout_transpose = [1, 3, 2, 0] if mCuSeqlensQ is None else [0, 2, 1]
mQ, mO = [layout_utils.select(t, QO_layout_transpose) for t in (mQ, mO)]

KV_layout_transpose = [1, 3, 2, 0] if mCuSeqlensK is None else [0, 2, 1]
mK, mV = [layout_utils.select(t, KV_layout_transpose) for t in (mK, mV)]

LSE_layout_transpose = [2, 1, 0] if mCuSeqlensQ is None else [1, 0]
mLSE = layout_utils.select(mLSE, LSE_layout_transpose) if mLSE is not None else None
```

dense Q/O 从：

```text
[batch, seqlen_q, head, dim]
```

变成：

```text
[seqlen_q, dim, head, batch]
```

paged K/V 原始 shape：

```text
[num_pages, page_size, head_kv, dim]
```

select 后：

```text
[page_size, dim, head_kv, num_pages]
```

这个布局让 tile 访问的前两维直接是：

```text
K/V tile: [tile_n, tile_hdim]
```

### 7.2 创建 QK/PV tiled MMA

源码：

```python
tiled_mma_qk, tiled_mma_pv = self._get_tiled_mma()
```

SM90 QK MMA：

```python
tiled_mma_qk = make_trivial_tiled_mma(
    self.dtype,
    self.dtype,
    OperandMajorMode.K,
    OperandMajorMode.K,
    Float32,
    atom_layout_mnk=(self.tile_m // 64, 1, 1),
    tiler_mn=(64, self.tile_n),
)
```

SM90 PV MMA：

```python
tiled_mma_pv = make_trivial_tiled_mma(
    self.dtype,
    self.dtype,
    OperandMajorMode.K,
    OperandMajorMode.MN,
    Float32,
    atom_layout_mnk=(self.tile_m // 64, 1, 1),
    tiler_mn=(64, self.tile_hdimv),
    a_source=OperandSource.RMEM if self.mma_pv_is_rs else OperandSource.SMEM,
)
```

含义：

```text
QK:
    Q[tile_m, head_dim] @ K[tile_n, head_dim]^T -> S[tile_m, tile_n]

PV:
    P[tile_m, tile_n] @ V[tile_n, head_dim_v] -> O[tile_m, head_dim_v]
```

`atom_layout_mnk=(tile_m // 64, 1, 1)` 表示沿 M 方向复制 warpgroup MMA atom。

### 7.3 线程组织和寄存器预算

源码：

```python
self.num_mma_threads = tiled_mma_qk.size
self.num_threads_per_warp_group = 128
self.num_wg_mma = self.num_mma_threads // 128
self.num_threads = 128 * (self.num_wg_mma + 1)

self.num_producer_threads = 32
self.num_Q_load_threads = 128
self.num_epilogue_threads = self.num_mma_threads

self.num_mma_regs, self.num_producer_regs = {
    1: (256, 56),
    2: (240, 24),
    3: (160, 32),
}[self.num_wg_mma]
```

结构：

```text
1 个 producer warpgroup:
    发起 Q/K/V load，TMA path 下实际主要由一个 warp 发 TMA。

N 个 consumer warpgroups:
    做 QK、softmax、PV、epilogue。
```

如果 `tile_m=128`：

```text
num_wg_mma = 2
num_threads = 384
consumer regs = 240
producer regs = 24
```

如果某些 operand 不能走 TMA，而要 cp.async：

```python
if self.num_wg_mma == 2 and (not self.use_tma_Q or not self.use_tma_KV):
    self.num_mma_regs, self.num_producer_regs = 224, 40
```

原因是 cp.async fallback 需要 producer 线程参与更多普通 copy，producer 需要更多寄存器，consumer 要让出一部分。

### 7.4 TMA 开关

源码：

```python
self.use_tma_Q = self.arch >= Arch.sm_90 and not (
    self.pack_gqa and self.tile_m % self.qhead_per_kvhead != 0
)
self.use_tma_O = self.use_tma_Q
```

`use_tma_KV` 在构造函数里已经由 `paged_kv_non_tma` 决定：

```python
self.use_tma_KV = not paged_kv_non_tma
```

三条路径：

```text
Q/O:
    pack_gqa 后 tile_m 如果不能整除 qhead_per_kvhead，
    Q/O tile 不是普通规则 TMA tile，不能直接 TMA。

K/V non-paged:
    一般可以 TMA。

K/V paged:
    page_size == tile_n 时可以 TMA；
    page_size != tile_n 时走 PagedKVManager + cp.async。
```

### 7.5 shared memory layout

源码：

```python
self.sQ_layout, self.sK_layout, self.sV_layout, self.sO_layout = [
    sm90_utils.make_smem_layout(mX.element_type, LayoutEnum.ROW_MAJOR, shape, stage)
    for mX, shape, stage in [
        (mQ, (self.tile_m, self.tile_hdim), None),
        (mK, (self.tile_n, self.tile_hdim), self.num_stages),
        (mV, (self.tile_n, self.tile_hdimv), self.num_stages),
        (mO, (self.tile_m, self.tile_hdimv), None),
    ]
]

self.sP_layout = None
if not self.mma_pv_is_rs:
    self.sP_layout = sm90_utils.make_smem_layout(
        mV.element_type,
        LayoutEnum.ROW_MAJOR,
        (self.tile_m, self.tile_n),
    )
```

shape：

```text
sQ: tile_m x tile_hdim
sK: tile_n x tile_hdim x num_stages
sV: tile_n x tile_hdimv x num_stages
sO: tile_m x tile_hdimv
sP: tile_m x tile_n, only when mma_pv_is_rs=False
```

`num_stages=2` 表示 K/V double buffering。

SharedStorage 类型动态生成：

```python
SharedStorage = self._get_shared_storage_cls()
```

其中包含：

```text
mbar_ptr_Q
mbar_ptr_K
mbar_ptr_V
sQ
sK
sV
sP
```

这些 mbarrier storage 后面用来构造 TMA/cp.async pipeline。

### 7.6 pack GQA layout

源码：

```python
mQ_og, mO_og = mQ, mO
if self.pack_gqa:
    nheads_kv = mK.shape[2]
    mQ = pack_gqa_layout(mQ, self.qhead_per_kvhead, nheads_kv, head_idx=2)
    mO = pack_gqa_layout(mO, self.qhead_per_kvhead, nheads_kv, head_idx=2)
    if mLSE is not None:
        mLSE = pack_gqa_layout(mLSE, self.qhead_per_kvhead, nheads_kv, head_idx=1)
```

为什么保留 `mQ_og/mO_og`：

```text
pack 后 layout 更适合 scheduler 和 head 映射；
但某些 TMA descriptor 创建仍需要原始 tensor layout 作为基础。
```

### 7.7 TMA copy op、TMA atom、TMA tensor

创建 copy op：

```python
gmem_tiled_copy_Q = cpasync.CopyBulkTensorTileG2SOp()
gmem_tiled_copy_KV = cpasync.CopyBulkTensorTileG2SOp()
gmem_tiled_copy_O = cpasync.CopyBulkTensorTileS2GOp()
```

计算 TMA transaction bytes：

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

这些 bytes 后面传给 `PipelineTmaAsync.create(tx_count=...)`，让 mbarrier 知道一次 transaction 预期完成多少字节。

Q TMA：

```python
if self.use_tma_Q:
    tma_atom_Q, tma_tensor_Q = make_tiled_tma_atom_fn(
        gmem_tiled_copy_Q,
        mQ_og if self.pack_gqa else mQ,
        self.sQ_layout,
        (self.tile_m, self.tile_hdim),
    )
```

K/V TMA：

```python
if self.use_tma_KV:
    tma_atom_K, tma_tensor_K = cpasync.make_tiled_tma_atom(
        gmem_tiled_copy_KV,
        mK,
        cute.select(self.sK_layout, mode=[0, 1]),
        (self.tile_n, self.tile_hdim),
        1,
    )
    tma_atom_V, tma_tensor_V = cpasync.make_tiled_tma_atom(
        gmem_tiled_copy_KV,
        mV,
        cute.select(self.sV_layout, mode=[0, 1]),
        (self.tile_n, self.tile_hdimv),
        1,
    )
```

注意 K/V 只 select shared layout 的 `[0, 1]`，不把 stage 维放进 TMA atom。原因是一次 TMA transaction 只搬一个 K 或 V tile 到某一个 pipeline stage；stage 由 pipeline state 决定。

O TMA：

```python
if self.use_tma_O:
    mO_tma = mO_og if self.pack_gqa else mO
    if self.varlen_q:
        mO_tma = create_ragged_tensor_for_tma(mO_tma, ragged_dim=0, ptr_shift=True)
    tma_atom_O, tma_tensor_O = make_tiled_tma_atom_fn(
        gmem_tiled_copy_O,
        mO_tma,
        self.sO_layout,
        (self.tile_m, self.tile_hdimv),
    )
```

一个关键点：

```text
page_table 不参与 TMA descriptor 创建。
```

TMA descriptor 描述“如何搬一个规则 tile”。paged KV 的 `page_table` 在 kernel load 阶段把 logical n_block 映射成 physical page_idx。

### 7.8 scheduler 参数和 grid

scheduler 选择：

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

```text
SingleTileScheduler:
    dense non-causal 或 local。

SingleTileLPTScheduler:
    dense causal。LPT 用来优先调度工作量更大的后部 Q blocks。

SingleTileVarlenScheduler:
    varlen Q 或 seqused_q，需要根据每个 batch 的有效 Q 长度映射 work tile。
```

参数：

```python
tile_sched_args = TileSchedulerArguments(
    ceil_div(size(mQ.shape[0]), self.tile_m),
    size(mQ.shape[2]),
    size(mQ.shape[3]) if mCuSeqlensQ is None else size(mCuSeqlensQ.shape[0] - 1),
    1,
    size(mK.shape[0]) if mPageTable is None else mK.shape[0] * mPageTable.shape[1],
    mQ.shape[1],
    mV.shape[1],
    total_q=...,
    tile_shape_mn=(self.tile_m, self.tile_n),
    mCuSeqlensQ=mCuSeqlensQ,
    mSeqUsedQ=mSeqUsedQ,
    qhead_per_kvhead_packgqa=self.qhead_per_kvhead if self.pack_gqa else 1,
    element_size=self.dtype.width // 8,
    is_persistent=False,
    lpt=self.is_causal or self.is_local,
)
```

paged KV 下：

```text
mK.shape[0] = page_size
mPageTable.shape[1] = max pages per sequence
seqlen_k_static = page_size * max_pages_per_seq
```

真实 used length 由 `mSeqUsedK` 在 kernel 里的 `SeqlenInfoQK` 读取。

转成底层参数并生成 grid：

```python
tile_sched_params = TileScheduler.to_underlying_arguments(tile_sched_args)
grid_dim = TileScheduler.get_grid_shape(tile_sched_params)
```

`SingleTileLPTScheduler` 的核心逻辑：

```python
def get_grid_shape(params):
    return (params.total_blocks, params.num_splits, Int32(1))

def get_current_work(...):
    ...
    if params.lpt:
        block = params.num_block - 1 - block
```

也就是 causal 下反转 Q block 顺序，先处理后面的重 block。

### 7.9 runtime scalar 和 fastdiv

softmax scale 转换：

```python
softmax_scale_log2, softmax_scale = utils.compute_softmax_scale_log2(
    softmax_scale,
    self.score_mod,
)
```

kernel 内部常用 `exp2` 做 softmax，所以提前准备 log2 scale。

window 参数：

```python
window_size_left = Int32(window_size_left) if window_size_left is not None else None
window_size_right = Int32(window_size_right) if window_size_right is not None else None
```

fast divmod 参数：

```python
fastdiv_mods = utils.compute_fastdiv_mods(
    mQ,
    mK,
    self.qhead_per_kvhead,
    self.pack_gqa,
    aux_tensors,
    mPageTable,
)
```

这类参数用于减少 kernel 里 index mapping 的除法/取模开销。

### 7.10 kernel launch

最后：

```python
self.kernel(
    tma_tensor_Q if self.use_tma_Q else mQ,
    tma_tensor_K if self.use_tma_KV else mK,
    tma_tensor_V if self.use_tma_KV else mV,
    tma_tensor_O if self.use_tma_O else mO,
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
    self.sQ_layout,
    self.sK_layout,
    self.sV_layout,
    self.sO_layout,
    self.sP_layout,
    self.gmem_tiled_copy_Q,
    self.gmem_tiled_copy_K,
    self.gmem_tiled_copy_V,
    self.gmem_tiled_copy_O,
    tiled_mma_qk,
    tiled_mma_pv,
    tile_sched_params,
    TileScheduler,
    SharedStorage,
    aux_tensors,
    fastdiv_mods,
).launch(
    grid=grid_dim,
    block=[self.num_threads, 1, 1],
    stream=stream,
    min_blocks_per_mp=1,
)
```

进入 kernel 前，所有关键配置已经确定：

```text
tensor view
MMA atom
TMA atom / tensor
shared layout
SharedStorage
pipeline transaction bytes
scheduler type / params / grid
block threads
register budget
softmax scale
window / mask metadata
fastdiv metadata
```

## 8. kernel 入口如何消费这些配置

虽然本文重点是 kernel 前，但看 kernel 开头能验证上层配置的用途。

kernel 参数签名包括：

```python
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
    ...
    tiled_mma_qk,
    tiled_mma_pv,
    tile_sched_params,
    TileScheduler,
    SharedStorage,
    ...
):
```

kernel 开头预取 TMA descriptor：

```python
if warp_idx == 0:
    for tma_atom in (tma_atom_Q, tma_atom_K, tma_atom_V, tma_atom_O):
        if tma_atom is not None:
            cpasync.prefetch_descriptor(tma_atom)
```

创建 pipeline：

```python
if self.use_tma_Q:
    pipeline_q = PipelineTmaAsync.create(..., tx_count=self.tma_copy_bytes["Q"])
else:
    pipeline_q = PipelineCpAsync.create(...)

if self.use_tma_KV:
    pipeline_k = PipelineTmaAsync.create(..., tx_count=self.tma_copy_bytes["K"])
    pipeline_v = PipelineTmaAsync.create(..., tx_count=self.tma_copy_bytes["V"])
else:
    pipeline_k = PipelineCpAsync.create(...)
    pipeline_v = PipelineCpAsync.create(...)
```

producer / consumer 分流：

```python
if warp_idx < 4:
    setmaxregister_decrease(self.num_producer_regs)
    self.load(...)
else:
    setmaxregister_increase(self.num_mma_regs)
    self.mma(...)
```

paged TMA 里 page table 使用：

```python
page_idx = (
    mPageTable[batch_idx, n_block]
    if mPageTable is not None and self.use_tma_KV
    else None
)

load_K(block=n_block, producer_state=kv_producer_state, page_idx=page_idx)
```

`load_KV`：

```python
if self.use_tma_KV:
    src_idx = block if page_idx is None else page_idx
    tma_load_fn(src_idx=src_idx, producer_state=producer_state)
else:
    paged_kv_manager.load_KV(block, sX[..., producer_state.index], K_or_V)
    cp_async_commit_group()
pipeline_kv.producer_commit(producer_state)
```

这说明：

```text
TMA descriptor:
    描述如何搬一个规则 tile。

page_table:
    runtime 把 logical n_block 映射为 physical page_idx。

pipeline state:
    决定写入 shared memory 的哪个 stage。
```

## 9. Paged KV 的 TMA 路径和 cp.async fallback

### 9.1 TMA paged KV

条件：

```text
page_size == tile_n
```

host / JIT 配置：

```python
paged_kv_non_tma = page_size not in [None, tile_n]
self.use_tma_KV = not paged_kv_non_tma
```

K/V tensor layout 经过 select 后：

```text
[page_size, dim, head_kv, num_pages]
```

TMA atom 的 tile shape：

```text
K: [tile_n, tile_hdim]
V: [tile_n, tile_hdimv]
```

如果 `page_size == tile_n`，一个 logical N block 正好是一页，kernel 里：

```python
page_idx = mPageTable[batch_idx, n_block]
tma_load_fn(src_idx=page_idx, producer_state=...)
```

这就是 paged TMA 的理想情况。

### 9.2 cp.async fallback paged KV

如果：

```text
page_size != tile_n
```

一个 `tile_n` 可能跨多个 pages，或者一个 page 包含多个 tile。此时很难用单个规则 TMA descriptor 描述整块 K/V tile。

源码走：

```python
paged_kv_manager = PagedKVManager.create(...)
```

`PagedKVManager.create` 会构造普通 cp.async tiled copy：

```python
atom_async_copy = cute.make_copy_atom(
    cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
    dtype,
    num_bits_per_copy=128,
)

gmem_tiled_copy_KV = cute.make_tiled_copy_tv(atom_async_copy, thr_layout, val_layout)
```

加载时先查 page table：

```python
page_idx, page_offset = divmod(row_idx + self.leftpad_k, self.page_size_divmod)
page = self.mPageTable[page_idx] if is_valid else 0
self.tPrPage[i] = page
self.tPrPageOffset[i] = page_offset
```

然后逐行构造 pointer：

```python
x_ptr_i64 = elem_pointer(mX, (page_offset, d_offset, page)).toint()
x_gmem_ptr = cute.make_ptr(...)
```

最后发 cp.async：

```python
cute.copy(
    self.gmem_tiled_copy_KV,
    mX_paged_cur_copy_ki,
    tXsX_k,
    pred=should_load,
)
```

所以 cp.async fallback 的本质是：

```text
每个 tile 内部逐行处理 page boundary；
每行通过 page table 找 physical page 和 page offset；
再用普通 async copy 搬到 shared。
```

相比 TMA，它更灵活，但 overhead 更高。

## 10. Case Study: Qwen2.5-7B-Instruct

### 10.1 模型和输入

以 Qwen2.5-7B-Instruct 风格配置为例：

```text
hidden_size = 3584
num_attention_heads = 28
num_key_value_heads = 4
head_dim = 3584 / 28 = 128
qhead_per_kvhead = 28 / 4 = 7
dtype = bf16
GPU = H100 / SM90
```

真实推理 chunked prefill：

```text
cached prefix = 2048 tokens
current chunk = 1024 tokens
visible KV = 3072 tokens
page_size = 128
```

输入：

```text
q: [1, 1024, 28, 128]
k_cache_paged: [num_physical_pages, 128, 4, 128]
v_cache_paged: [num_physical_pages, 128, 4, 128]
page_table: [1, max_pages_per_seq]
seqused_k: [1] = [3072]
causal = True
```

如果 `max_pages_per_seq=24`，则：

```text
max logical capacity = 24 * 128 = 3072
```

但源码允许 `max_pages_per_seq` 大于真实使用页数；真实长度由 `seqused_k` 决定。

### 10.2 API 调用形态

更贴近源码测试的调用：

```python
out, lse = flash_attn_varlen_func(
    q,
    k_cache_paged,
    v_cache_paged,
    seqused_k=cache_seqlens,
    page_table=page_table,
    causal=True,
    return_lse=False,
)
```

测试里类似：

```python
out, lse, *rest = flash_attn_varlen_func(
    q if not varlen_q else q_unpad,
    k_cache if page_size is None else k_cache_paged,
    v_cache if page_size is None else v_cache_paged,
    seqused_k=cache_seqlens,
    page_table=page_table,
    cu_seqlens_q=cu_seqlens_q,
    causal=causal,
    window_size=window_size,
)
```

### 10.3 `_flash_attn_fwd` 对本 case 的解析

shape：

```python
q_shape = q.shape
num_head, head_dim = q_shape[-2:]  # 28, 128
batch_size, seqlen_q = q_shape[:2] # 1, 1024
total_q = 1024
```

paged KV：

```python
num_pages, page_size = v.shape[:2]
num_head_kv = v.shape[-2]   # 4
head_dim_v = v.shape[-1]    # 128
```

校验：

```python
assert page_table.shape == (batch_size, max_num_pages_per_seq)
assert k.shape == (num_pages, page_size, num_head_kv, head_dim)
assert v.shape == (num_pages, page_size, num_head_kv, head_dim_v)
```

GQA：

```python
qhead_per_kvhead = num_head // num_head_kv
                  = 28 // 4
                  = 7
pack_gqa = True
```

softmax：

```python
softmax_scale = 1 / sqrt(128)
```

### 10.4 tile 选择

调用：

```python
fwd_cfg = _tile_size_fwd_sm90(
    head_dim=128,
    head_dim_v=128,
    is_causal=True,
    is_local=False,
    sparse_block_size_q=None,
)
```

命中：

```python
elif head_dim <= 128:
    return FwdConfig(128, 128, True, True)
```

本 case 得到：

```text
tile_m = 128
tile_n = 128
mma_pv_is_rs = True
intra_wg_overlap = True
```

因为：

```text
page_size = 128
tile_n = 128
```

所以：

```python
paged_kv_non_tma = page_size not in [None, tile_n]
                 = False
```

进而：

```python
self.use_tma_KV = True
```

### 10.5 compile key 中本 case 的关键项

关键 compile key 值：

```text
dtype = BFloat16
head_dim = 128
head_dim_v = 128
qhead_per_kvhead = 7
causal = True
page_table is not None = True
seqused_k is None = False
tile_m = 128
tile_n = 128
pack_gqa = True
arch = SM90
page_size not in [None, tile_n] = False
mma_pv_is_rs = True
intra_wg_overlap = True
```

这会编译或命中一个“SM90 + bf16 + hdim128 + GQA7 + causal + paged TMA KV + 128x128 tile”的 forward kernel。

### 10.6 `FlashAttentionForwardSm90` 静态属性

构造后：

```text
dtype = bf16
tile_hdim = 128
tile_hdimv = 128
check_hdim_oob = False
check_hdim_v_oob = False
qhead_per_kvhead = 7
is_causal = True
is_local = False
pack_gqa = True
tile_m = 128
tile_n = 128
num_stages = 2
Q_in_regs = False
mma_pv_is_rs = True
intra_wg_overlap = True
use_tma_KV = True
```

### 10.7 `__call__` layout 结果

Q：

```text
原始: [1, 1024, 28, 128]
select [1,3,2,0]
结果: [1024, 128, 28, 1]
```

K/V：

```text
原始: [num_pages, 128, 4, 128]
select [1,3,2,0]
结果: [128, 128, 4, num_pages]
```

这就是为什么 paged TMA 可以把第 0 维看成 page 内 token offset，第 3 维看成 physical page index。

### 10.8 MMA 配置

本 case：

```text
tile_m = 128
tile_n = 128
tile_hdim = 128
tile_hdimv = 128
```

QK：

```text
Q tile: [128, 128]
K tile: [128, 128]
S tile: [128, 128]
```

PV：

```text
P tile: [128, 128]
V tile: [128, 128]
O tile: [128, 128]
```

MMA atom：

```text
atom_layout_mnk = (2, 1, 1)
QK tiler_mn = (64, 128)
PV tiler_mn = (64, 128)
PV A source = RMEM
```

所以：

```text
2 个 consumer warpgroup
每个 consumer WG 处理 64 行 Q
P 留在 register 中喂给 PV
```

### 10.9 线程和寄存器

源码推导：

```python
num_wg_mma = 2
num_threads = 128 * (2 + 1) = 384
num_mma_regs = 240
num_producer_regs = 24
```

线程组织：

```text
threads 0..127:
    producer WG

threads 128..255:
    consumer WG 0

threads 256..383:
    consumer WG 1
```

进入 kernel 后：

```python
if warp_idx < 4:
    setmaxregister_decrease(24)
    load(...)
else:
    setmaxregister_increase(240)
    mma(...)
```

### 10.10 shared memory

本 case：

```text
sQ = [128, 128]
sK = [128, 128, 2]
sV = [128, 128, 2]
sO = [128, 128]
sP = None
```

bf16 bytes：

```text
sQ ~= 32 KB
sK ~= 64 KB
sV ~= 64 KB
sO 复用 sQ storage
```

这里不是简单相加决定 occupancy，因为 shared storage 中存在复用、alignment、mbarrier storage、swizzle layout 等因素。但直觉上可以看到：每个 CTA 的资源很重，所以 launch 使用 `min_blocks_per_mp=1` 是合理的。

### 10.11 TMA 配置

TMA transaction bytes：

```text
Q = 128 * 128 * 2 = 32768 bytes
K = 128 * 128 * 2 = 32768 bytes
V = 128 * 128 * 2 = 32768 bytes
```

K TMA atom：

```python
tma_atom_K, tma_tensor_K = cpasync.make_tiled_tma_atom(
    CopyBulkTensorTileG2SOp(),
    mK,
    sK_layout_without_stage,
    (128, 128),
    1,
)
```

V TMA atom：

```python
tma_atom_V, tma_tensor_V = cpasync.make_tiled_tma_atom(
    CopyBulkTensorTileG2SOp(),
    mV,
    sV_layout_without_stage,
    (128, 128),
    1,
)
```

paged KV 的实际 page 映射不在这里做，而是在 kernel load 阶段：

```python
page_idx = mPageTable[batch_idx, n_block]
tma_load_fn(src_idx=page_idx, producer_state=producer_state)
```

对本 case，logical KV length 是 3072，`tile_n=128`：

```text
num logical K blocks = 3072 / 128 = 24
```

每个 `n_block` 对应一个 logical page：

```text
n_block 0  -> page_table[0, 0]
n_block 1  -> page_table[0, 1]
...
n_block 23 -> page_table[0, 23]
```

### 10.12 scheduler 和 causal offset

因为：

```text
causal=True
local=False
Q 是 dense
```

scheduler：

```text
SingleTileLPTScheduler
```

如果暂时不考虑 pack GQA 后的 M/head 细节，从 Q row 角度：

```text
seqlen_q = 1024
tile_m = 128
Q blocks = 8
```

KV：

```text
seqused_k = 3072
tile_n = 128
K blocks = 24
```

`SeqlenInfoQK.create` 会读：

```python
if mSeqUsedK is not None:
    seqlen_k = mSeqUsedK[batch_idx]
```

所以 kernel 内真实 `seqlen_k=3072`。

`BlockInfo.get_n_block_min_max` 对 causal 的上界计算：

```python
n_block_max = ceil_div(seqlen_k, tile_n)

m_idx_max = (m_block + 1) * tile_m
if qhead_per_kvhead_packgqa > 1:
    m_idx_max = ceil_div(m_idx_max, qhead_per_kvhead_packgqa)

n_idx = m_idx_max + seqlen_k - seqlen_q
n_block_max = min(n_block_max, ceil_div(n_idx, tile_n))
```

其中：

```text
seqlen_k - seqlen_q = 3072 - 1024 = 2048
```

这就是 cached prefix 造成的 causal offset。

在不考虑 pack GQA 的简化视角下：

```text
Q block 0: rows 0..127
    max visible K index ~= 2048 + 127

Q block 7: rows 896..1023
    max visible K index ~= 2048 + 1023 = 3071
```

所以后面的 Q blocks 工作量更大。`SingleTileLPTScheduler` 反转 block 顺序：

```python
block = params.num_block - 1 - block
```

优先派发重 block，减少尾部等待。

### 10.13 launch 总结

最终进入 launch：

```text
grid = SingleTileLPTScheduler.get_grid_shape(tile_sched_params)
block = [384, 1, 1]
min_blocks_per_mp = 1
```

kernel 参数中包含：

```text
tma_tensor_Q/K/V/O
tma_atom_Q/K/V/O
mPageTable
mSeqUsedK
softmax_scale_log2
shared layouts
tiled_mma_qk/pv
tile scheduler params
SharedStorage
fastdiv_mods
```

也就是说，本 case 进入 kernel 前完整配置为：

```text
模型语义:
    Qwen-style GQA causal attention

shape:
    q = [1,1024,28,128]
    paged kv = [num_pages,128,4,128]
    used_k = 3072

tile:
    tile_m = 128
    tile_n = 128

MMA:
    QK WGMMA, atom_layout_mnk=(2,1,1), tiler=(64,128)
    PV WGMMA, atom_layout_mnk=(2,1,1), tiler=(64,128), P from register

threads/registers:
    384 threads
    1 producer WG + 2 consumer WGs
    producer regs 24
    consumer regs 240

memory:
    sQ 128x128
    sK 128x128x2
    sV 128x128x2
    sO 128x128
    no sP

TMA:
    Q/K/V/O TMA path enabled
    K/V tile bytes 32 KB each
    page_table maps n_block -> physical page_idx

scheduler:
    SingleTileLPTScheduler
    causal block reversal
    seqlen_k from seqused_k

launch:
    block = 384 threads
    grid = scheduler generated
    min_blocks_per_mp = 1
```

## 11. 源码阅读指南

建议按这个顺序读：

```text
1. tests/cute/test_flash_attn.py
   看 paged KV 测试如何构造 k_cache_paged/v_cache_paged/page_table/cache_seqlens。

2. interface.py::flash_attn_varlen_func
   看 public API 如何进入 autograd wrapper。

3. interface.py::_flash_attn_fwd
   看 shape 校验、paged KV 校验、GQA、softmax scale、tile、compile key。

4. interface.py::_tile_size_fwd_sm90
   看 SM90 tile_m/tile_n/mma_pv_is_rs/intra_wg_overlap 的规则。

5. flash_fwd.py::FlashAttentionForwardBase.__init__
   看 head_dim padding、check_hdim_oob、静态属性。

6. flash_fwd_sm90.py::FlashAttentionForwardSm90.__call__
   重点读 layout select、MMA、寄存器、shared layout、TMA atom、scheduler、launch。

7. tile_scheduler.py::SingleTileLPTScheduler
   看 causal LPT 如何反转 block 顺序。

8. block_info.py::BlockInfo
   看 causal/local 如何计算 n_block_min/n_block_max。

9. paged_kv.py::PagedKVManager
   看 page_size != tile_n 时 cp.async fallback 如何逐行查 page table。

10. flash_fwd_sm90.py::kernel/load/load_KV
   只看 kernel 开头如何消费 __call__ 准备好的 TMA、pipeline、scheduler。
```

## 12. 结论

Hopper FlashAttention forward 在进入 kernel 前做的核心事情是：

```text
把 attention API 调用转成一个 SM90-specialized execution plan。
```

这个 plan 包含：

```text
shape 语义:
    dense/varlen/paged/GQA/causal/local

tile 策略:
    tile_m/tile_n/mma_pv_is_rs/intra_wg_overlap

compute:
    QK/PV WGMMA atom

memory:
    shared layout、TMA atom、TMA transaction bytes、cp.async fallback

resource:
    producer/consumer warpgroup、register budget、num_threads

scheduling:
    TileScheduler 类型、grid、causal LPT、varlen metadata

runtime scalars:
    softmax_scale_log2、window、seqused、page_table、fastdiv
```

对 Qwen2.5 风格的 `head_dim=128`、GQA、chunked prefill、paged KV case，最关键的一条链是：

```text
head_dim=128
  -> tile_m=128, tile_n=128
  -> 2 consumer WGs + 1 producer WG
  -> 384 threads
  -> P stays in registers for PV
  -> K/V double-buffer TMA
  -> page_size=128 时 paged KV TMA-friendly
  -> causal LPT scheduler
  -> page_table 在 kernel load 阶段映射 logical n_block 到 physical page
```

这也是为什么 kernel 前的配置非常多：FlashAttention 的性能来自“少存 attention matrix”只是第一层，更深的一层是它把 tensor layout、WGMMA、TMA、pipeline、scheduler、register/shared memory 都提前编排成适合 Hopper 的执行形态。
