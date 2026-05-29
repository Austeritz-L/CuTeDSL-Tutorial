# Layout 和 Tensor

这一章先解释三个 GEMM kernel 里已经用到的布局代数 API：

- `Gemm/navie_sgemm.py`
- `Gemm/navie_tensorop.py`
- `Gemm/ldsm_tensorop.py`

核心问题是：

```text
逻辑坐标 -> Layout -> 物理 offset -> Tensor engine -> 真实内存值
```

CuTe DSL 里的 tensor 不是一个单纯的 pointer。它更接近：

```text
pointer/engine + layout + element type + memory space
```

所以 `local_tile`、`local_partition`、`partition_A/B/C` 这类 API 大多数时候
不是在搬数据，而是在构造新的 tensor view。真正的数据访问发生在
`cute.copy`、`cute.gemm`、索引读写、load/store 这些操作里。

## 源码入口

建议先看这些文件：

- `cutlass/python/CuTeDSL/cutlass/cute/typing.py`
  - `Layout`
  - `Tensor`
- `cutlass/python/CuTeDSL/cutlass/cute/core.py`
  - `make_layout`
  - `zipped_divide`
  - `local_tile`
  - `local_partition`
  - `dice`
- `cutlass/python/CuTeDSL/cutlass/cute/tensor.py`
  - `make_fragment_like`
  - `make_rmem_tensor_like`
- `cutlass/python/CuTeDSL/cutlass/cute/runtime.py`
  - `from_dlpack`

源码里很多函数都带有 `@dsl_user_op`。这说明它们通常不是在 Python 侧立即计算出一个普通对象，而是在 JIT tracing 的过程中构造 MLIR/CuTe IR。理解这一点很重要，否则很容易把 CuTe DSL 当成普通 Python tensor 库。

## Layout 是什么

`Layout` 描述的是坐标映射。对一个简单 2D layout 来说，可以先这样理解：

```text
layout = shape : stride
offset(m, n) = m * stride_m + n * stride_n
```

比如：

```python
layout = cute.make_layout((16, 8), stride=(8, 1))
```

这是一个 row-major 的 16x8 tile。坐标 `(m, n)` 对应线性 offset：

```text
m * 8 + n
```

当 CuTe 打印：

```text
(16,8):(8,1)
```

可以读成：

```text
shape  = (16, 8)
stride = (8, 1)
```

CuTe layout 还可以是层级化的，比如：

```text
((2,2),1,1):((1,2),0,0)
```

它仍然是 `shape:stride`，只是某个 mode 本身又是一个嵌套结构。
Tensor Core 寄存器 fragment、tiled view、per-lane ownership 经常会出现这种层级 layout。

在源码层面，`typing.py::Layout` 暴露了 `shape` 和 `stride` 属性；
`core.py::make_layout` 负责创建 layout。如果不传 `stride`，CuTe 会构造一个默认的 compact layout。

## Tensor 是什么

`typing.py::Tensor` 的源码注释里有一个很关键的形式化描述：

```text
T(c) = *(E + L(c))
```

其中：

- `E` 是 engine/iterator，也就是数据从哪里开始
- `L` 是 layout，也就是坐标如何映射到 offset
- `c` 是逻辑坐标

所以 tensor 的本质是：

```text
用 layout 把逻辑坐标转成 offset，然后在 engine 上取值
```

这也是为什么 `local_tile` 和 `local_partition` 返回的是 tensor：它们返回的是新的视图，不一定分配新内存。

三个 kernel 里常用的 tensor 属性包括：

- `mA.shape[0]`、`mA.shape[1]`：读取逻辑维度
- `mA.element_type`：读取元素类型，用于构造 copy atom 或 shared memory tensor
- `tensor.type`：编译期类型信息，适合做静态调试

host 侧通过 `from_dlpack` 把 PyTorch tensor 包成 CuTe tensor：

```python
a_tensor = from_dlpack(a, assumed_align=16)
b_tensor = from_dlpack(b, assumed_align=16)
c_tensor = from_dlpack(c, assumed_align=16)
```

进入 JIT 函数后，`mA`、`mB`、`mC` 就是 CuTe DSL 的 tensor view。

## `make_layout`

三个 kernel 里出现了几类 `make_layout`：

```python
thread_layout = cute.make_layout((self.bM, self.bN), stride=(self.bN, 1))
sA_layout = cute.make_layout((self.bM, self.bK), stride=(self.bK, 1))
sB_layout = cute.make_layout((self.bN, self.bK), stride=(self.bK, 1))
g2s_thread_layout = cute.make_layout((64, 2), stride=(2, 1))
g2s_value_layout = cute.make_layout((1, 8))
```

`make_layout` 的作用就是构造一个坐标到 offset 的函数。

在 `navie_sgemm.py` 里：

```python
thread_layout = cute.make_layout((self.bM, self.bN), stride=(self.bN, 1))
```

如果 CTA tile 是 16x16，那么这个 layout 把 256 个 thread id 映射到 C tile：

```text
tid = m * 16 + n
```

也就是每个线程负责一个 `C[m, n]`。

在当前 `ldsm_tensorop.py` 里，GMEM->SMEM 不再手写 `local_partition`
线程布局，而是用 `make_tiled_copy_tv` 构造 tiled copy：

```python
thread_layout = cute.make_layout(
    (self.num_threads // shape_dim_1, shape_dim_1),
    stride=(shape_dim_1, 1),
)
value_layout = cute.make_layout((1, copy_elems))
tiled_g2s_A = cute.make_tiled_copy_tv(copy_atom, thread_layout, value_layout)
```

当前配置是：

```text
CTA tile = (128, 128, 16)
dtype    = fp16
copy     = 128-bit Universal copy
copy_elems = 128 / 16 = 8 fp16
threads  = 128
```

因为 `bK = 16`，所以：

```text
shape_dim_1 = bK / copy_elems = 16 / 8 = 2
thread_layout = (64, 2):(2, 1)
value_layout  = (1, 8):(8, 1)
```

这表示 G2S tiled copy 把一个 `128x16` 的 A/B tile 分给 128 个线程，
每个 copy atom 搬 8 个连续 fp16，也就是 16B。

这里要区分两个概念：

```text
thread layout 的元素数 = 线程数
value layout 描述每条 copy 指令搬多少连续元素
tiled copy 把 thread layout 和 value layout 组合成 (thread,value)->tile 坐标
```

## `local_tile`

`local_tile` 负责从大 tensor 里取出一个 tile。`core.py` 的注释直接给出了等价式：

```text
local_tile(input, tiler, coord) = zipped_divide(input, tiler)[coord]
```

`zipped_divide` 会把 tensor 分成：

```text
(tile modes, rest modes)
```

对 GEMM 来说，我们定义 CTA tiler：

```python
self.cta_tiler = (bM, bN, bK)
```

它对应 GEMM 的三个逻辑 mode：

```text
M, N, K
```

但是每个 operand 只使用其中两个 mode：

```text
C: (M, N)
A: (M, K)
B: (N, K)
```

所以三个 kernel 里都要用 `proj` 把不用的 mode 去掉。

### C 的 CTA tile

```python
gC = cute.local_tile(
    mC,
    tiler=self.cta_tiler,
    coord=(bidx, bidy, None),
    proj=(1, 1, None),
)
```

含义是：

```text
保留 M
保留 N
去掉 K
选择第 (bidx, bidy) 个 C tile
```

得到的 `gC` 是当前 CTA 负责的 C tile，形状近似是 `(bM, bN)`。

### A 的 CTA tile

```python
gA = cute.local_tile(
    mA,
    tiler=self.cta_tiler,
    coord=(bidx, None, k_tile),
    proj=(1, None, 1),
)
```

含义是：

```text
保留 M
去掉 N
保留 K
选择第 bidx 个 M tile
选择第 k_tile 个 K tile
```

得到的 `gA` 是当前 CTA 在当前 K block 要使用的 A tile，形状近似是`(bM, bK)`。

### B 的 CTA tile

```python
gB = cute.local_tile(
    mB,
    tiler=self.cta_tiler,
    coord=(None, bidy, k_tile),
    proj=(None, 1, 1),
)
```

含义是：

```text
去掉 M
保留 N
保留 K
选择第 bidy 个 N tile
选择第 k_tile 个 K tile
```

得到的 `gB` 是当前 CTA 在当前 K block 要使用的 B tile，形状近似是`(bN, bK)`。这里 B 在我们的代码里被看成 `(N, K)`，所以后续参考值也用了：

```python
torch.einsum("mk,nk->mn", a, b)
```

## `proj`

`proj` 是这一章最重要的概念。

CTA tiler 是 3D：

```text
(M, N, K)
```

但是 A/B/C 都是 2D tensor：

```text
C: (M, N)
A: (M, K)
B: (N, K)
```

`proj` 告诉 CuTe：哪些 tiler mode 应该保留，哪些应该被投影掉。可以先用下面的
规则理解：

```text
1 或其他 integer -> 保留这个 mode
None              -> 去掉这个 mode
```

源码层面的机制是 `dice`。`core.py::dice` 的注释是：当 dicer 里对应位置是
integer 时，保留 input 的这个 mode。`local_partition` 里也直接用了：

```python
_cute_ir.local_partition(..., tiler=dice(tiler, proj), ...)
```

因此：

```python
proj=(1, None, 1)
```

可以读成：

```text
保留 mode 0
去掉 mode 1
保留 mode 2
```

在 GEMM 里，这让我们可以用同一个 3D CTA tiler 描述 C、A、B 三种 2D operand。

## `local_partition`

`local_partition` 负责把一个 tensor view 分成某个线程拥有的 slice。源码入口是：

```text
core.py::local_partition(target, tiler, index, proj=1)
```

这里的 `tiler` 经常是 thread layout，`index` 经常是 `tidx`。如果提供了 `proj`，
源码会先做：

```text
dice(tiler, proj)
```

然后再把 `index` 映射到被投影后的 layout 上。

### SIMT scalar SGEMM

在 `navie_sgemm.py` 里：

```python
thread_layout = cute.make_layout((self.bM, self.bN), stride=(self.bN, 1))
tCgC = cute.local_partition(gC, thread_layout, tidx, proj=(1, 1))
tCgA = cute.local_partition(gA, thread_layout, tidx, proj=(1, None))
tCgB = cute.local_partition(gB, thread_layout, tidx, proj=(None, 1))
```

假设 `bM = 16`、`bN = 16`、`bK = 16`。

对 C：

```text
thread_layout = (16,16):(16,1)
tid = m * 16 + n
```

所以每个线程拿到一个 C 元素。

对 A：

```text
proj=(1, None)
```

只保留 thread layout 的 M mode。也就是说，对同一个 `m`，不同 `n` 的线程会拿到
同一行 A：

```text
A[m, 0:bK]
```

对 B：

```text
proj=(None, 1)
```

只保留 thread layout 的 N mode。每个 C 线程会拿到：

```text
B[n, 0:bK]
```

然后：

```python
cute.gemm(mma_atom, tCrC, tCrA, tCrB, tCrC)
```

在这个 scalar universal atom 的例子里，可以近似理解成：

```python
for k in range(bK):
    tCrC[0] += tCrA[k] * tCrB[k]
```

### GMEM 到 SMEM 的 tiled copy partition

在 `ldsm_tensorop.py` 里：

```python
thr_g2s_A = tiled_g2s_A.get_slice(tidx)
thr_g2s_B = tiled_g2s_B.get_slice(tidx)

tAgA = thr_g2s_A.partition_S(gA)
tAsA = thr_g2s_A.partition_D(sA)
tBgB = thr_g2s_B.partition_S(gB)
tBsB = thr_g2s_B.partition_D(sB)
```

这里 source 和 destination 使用同一个 TiledCopy 的 source/destination
partition 规则：

```text
tAgA: 当前线程负责的 A 的 GMEM slice
tAsA: 当前线程负责的 A 的 SMEM slice
tBgB: 当前线程负责的 B 的 GMEM slice
tBsB: 当前线程负责的 B 的 SMEM slice
```

所以后面可以直接：

```python
cute.copy(tiled_g2s_A, tAgA, tAsA)
cute.copy(tiled_g2s_B, tBgB, tBsB)
```

这一步已经是 tiled copy，但 copy atom 仍然是 `CopyUniversalOp`。
也就是说，G2S 仍然是普通 global/shared memory load-store，不是 `cp.async`；
只是线程和值的分配由 TiledCopy 负责。

## `make_fragment_like`

`make_fragment_like(src, dtype)` 会创建一个和 `src` layout 对齐的 register memory
fragment。

在 `navie_sgemm.py` 里：

```python
tCrC = cute.make_fragment_like(tCgC, cutlass.Float32)
tCrA = cute.make_fragment_like(tCgA, cutlass.Float32)
tCrB = cute.make_fragment_like(tCgB, cutlass.Float32)
```

命名习惯可以这样读：

```text
t = thread
C/A/B = operand
g/r/s = global/register/shared
```

所以：

```text
tCgC: 当前线程看到的 C 的 global memory view
tCrC: 当前线程持有的 C 的 register fragment
tCgA: 当前线程看到的 A 的 global memory view
tCrA: 当前线程持有的 A 的 register fragment
```

源码上，`tensor.py::make_fragment_like` 的关键逻辑是：

- 如果输入是 tensor，就调用 `make_rmem_tensor_like`
- 如果输入是 layout，就创建 fragment layout，必要时再包装成 RMEM tensor

因此：

```python
cute.copy(copy_atom_A, tCgA, tCrA)
```

就是从 GMEM view 加载到寄存器 fragment。

## TensorOp 里的 `partition_A/B/C`

Tensor Core 版本里会看到：

```python
thr_mma = tiled_mma.get_slice(tidx)
tCgC = thr_mma.partition_C(gC)
tCgA = thr_mma.partition_A(gA)
tCgB = thr_mma.partition_B(gB)
```

在 ldmatrix 版本里也有：

```python
tCsA = thr_mma.partition_A(sA)
tCsB = thr_mma.partition_B(sB)
```

它们和 `local_partition` 不同。

`local_partition` 是用户显式提供 thread layout。比如你说：

```text
用 (16,16):(16,1) 把 256 个线程映射到 16x16 C tile
```

但是 `partition_A/B/C` 是 MMA atom 根据硬件指令的 fragment contract 生成的
per-lane view。当前 `ldsm_tensorop.py` 使用 tiled m16n8k16 Tensor Core atom；
每个 lane 必须持有规定位置的 A/B/C 寄存器。这个 layout 不是随便设计的，而是由
PTX 指令语义决定。

这一章只需要记住：

```text
local_partition      = 你提供线程布局
thr_mma.partition_*  = MMA atom 提供 per-lane fragment 布局
```

`partition_A/B/C` 的源码和 PTX 对应关系放到 `03_mma_atom_tiledmma.md` 里展开。

## 静态 shape 打印

当前教程代码默认只保留静态打印：

```python
print(f"[DSL INFO] tAgA = {tAgA.type}")
```

Python `print(...)` 是 JIT tracing 阶段的静态打印。它发生在 DSL 构造 IR 的时候，适合看 `tensor.type`、layout、shape、memory space 等编译期信息。因为 `for k_tile`这类循环在 JIT 里可能是符号化构造，所以静态打印经常只出现一次。

当前教程代码默认只保留静态打印，也就是只打印 `.type`、layout、shape 和 memory
space。device 运行时打印会真实执行在 CUDA kernel 里，并且可能解引用 tensor view。
为了让读者专注于形状推导，当前版本不再使用运行时打印。

## API 总结

| API | 作用 | 在当前 kernel 中的用途 |
| --- | --- | --- |
| `from_dlpack` | 把 PyTorch tensor 包成 CuTe tensor | host 侧输入输出 |
| `tensor.shape` | 读取逻辑维度 | 计算 grid 和 K-loop |
| `tensor.element_type` | 读取元素类型 | 构造 copy atom、SMEM tensor |
| `tensor.type` | 查看编译期类型和 layout | 静态调试 |
| `cute.ceil_div` | 向上整除 | 计算 CTA grid |
| `cute.make_layout` | 构造坐标映射 | thread layout、SMEM layout |
| `cute.local_tile` | 取 CTA tile | 构造 `gA`、`gB`、`gC` |
| `proj` | 投影掉不用的 mode | 用 `(M,N,K)` tiler 描述 A/B/C |
| `cute.local_partition` | 取当前线程的 slice | scalar/SIMT 教学 kernel |
| `cute.make_tiled_copy_tv` | 用 thread/value layout 构造 tiled copy | 当前 `ldsm_tensorop.py` 的 G2S |
| `ThrCopy.partition_S/D` | 按 TiledCopy 切 source/destination | 当前 `ldsm_tensorop.py` 的 G2S/S2R |
| `cute.make_fragment_like` | 构造匹配 view 的 RMEM fragment | scalar SGEMM |
| `cute.size` | 获取 tensor/layout 的逻辑元素数 | 静态 shape 推导和 scalar loop |
| `thr_mma.partition_A/B/C` | MMA per-lane view | TensorOp kernel |
| `SmemAllocator.allocate_tensor` | 按 layout 分配 shared memory tensor | ldmatrix kernel |

## 三个 kernel 怎么读

### `navie_sgemm.py`

这是最适合理解 layout algebra 的版本：

```text
global tensor -> CTA tile -> per-thread slice -> register fragment
```

thread layout 是显式的。每个线程负责一个 C 元素、一段 A 行向量、一段 B 行向量。
最后的 GEMM 本质上就是对 K tile 做 dot product。

### `navie_tensorop.py`

CTA tile 仍然用同样的 `local_tile` 方式取出。区别是后面不再由我们手写
thread layout，而是交给 Tensor Core MMA atom：

```text
CTA tile -> MMA per-lane partition -> register fragment
```

这里打印出来的层级 layout 会复杂很多，因为它编码的是
`mma.sync.aligned.m16n8k8` 的 lane 到寄存器 fragment 的映射。

### `ldsm_tensorop.py`

当前这个版本把 CTA tile 扩展到 `(128,128,16)`，并使用 tiled copy/tiled MMA：

```text
GMEM CTA tile -> TiledCopy G2S partition -> SMEM tile
SMEM tensor   -> ldmatrix copy view     -> MMA register fragment
```

GMEM 到 SMEM 这一步使用 `CopyUniversalOp` + `make_tiled_copy_tv`，每条 copy 是
128-bit vectorized Universal copy。SMEM 到 RMEM 使用 `LdMatrix8x8x16bOp(False, 4)`
+ `make_tiled_copy_A/B`，并通过 `retile` 把 ldmatrix 的 destination view 对齐到
MMA register fragment。计算阶段使用 tiled m16n8k16 Tensor Core atom 拼出
`128x128` 的 CTA 结果 tile。
