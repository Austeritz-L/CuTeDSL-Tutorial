# Elementwise Kernel: TV Layout and Predicate Notes

This note explains the two most confusing parts of
`Elementwise/elementwise_mul.py`:

1. why `cute.make_layout_tv(thr_layout, val_layout)` prints a layout that does
   not look like the original `(M, N)` layout we wrote;
2. why the predicate tensor for vectorized copy has shape like `(2, (1,))`
   instead of the element shape `((4, 2))`.

The concrete example used here is fp32:

```python
thr_layout = cute.make_ordered_layout((4, 8), order=(1, 0))
val_layout = cute.make_ordered_layout((2, 4), order=(1, 0))
tiler_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)
```

The printed result is:

```text
thr_layout = (4,8):(8,1)
val_layout = (2,4):(4,1)
tiler_mn   = (8,32)
tv_layout  = ((8,4),(4,2)):((32,2),(8,1))
```

## Human View

When we write:

```text
thr_layout = (4,8)
val_layout = (2,4)
```

we usually read it as:

```text
thread_m = 4, thread_n = 8
value_m  = 2, value_n  = 4
```

So the CTA tile is:

```text
TileM = thread_m * value_m = 4 * 2 = 8
TileN = thread_n * value_n = 8 * 4 = 32
```

That is why `tiler_mn` is:

```text
(8, 32)
```

For a row-major tensor:

```text
W: (seq_len, dim):(dim,1)
```

we want each thread to copy:

```text
4 contiguous columns x 2 rows
```

For fp32, 4 contiguous columns are exactly 16 bytes, so this matches one 128-bit vectorized copy.

## What `make_layout_tv` Actually Does

The Python CuTeDSL implementation is in:

```text
/root/lizhiyuan/cutlass/python/CuTeDSL/cutlass/cute/core.py
```

The important part is:

```python
layout_mn = raked_product(thr_layout, val_layout)
thr_size = size(thr_layout)
val_size = size(val_layout)
tmp = make_layout((thr_size, val_size))
layout_tv = composition(right_inverse(layout_mn), tmp)
tiler_mn = product_each(layout_mn.shape_method())
```

The C++ implementation for tiled copy is equivalent:

```text
/root/lizhiyuan/cutlass/include/cute/atom/copy_atom.hpp
```

```cpp
auto layout_mn = raked_product(thr_layout, val_layout);
auto layout_tv = right_inverse(layout_mn)
                   .with_shape(make_shape(size(thr_layout), size(val_layout)));
auto tiler = product_each(shape(layout_mn));
```

So `make_layout_tv` does not simply concatenate:

```text
((thread_m, thread_n), (value_m, value_n))
```

Instead it does:

```text
1. raked_product(thr_layout, val_layout)
2. right_inverse(layout_mn)
3. reshape the inverse as (thread_id, value_id)
```

## The `raked_product` Step

The C++ source says:

```cpp
auto result = logical_product(append<R>(block), append<R>(tiler));
return zip(get<1>(result), get<0>(result));
```

Here:

```text
block = thr_layout
tiler = val_layout
```

The important part is:

```cpp
zip(get<1>(result), get<0>(result))
```

This puts the value-layout modes before the thread-layout modes in the intermediate layout.

For:

```text
thr = (4,8):(8,1)
val = (2,4):(4,1)
```

a small C++ print test gives:

```text
layout_mn = ((2,4),(4,8)):((128,8),(32,1))
```

This `layout_mn` maps:

```text
(M,N) coordinate -> (thread_id, value_id)
```

But its displayed mode order is already:

```text
((value_m,value_n),(thread_m,thread_n))
```

not:

```text
((thread_m,thread_n),(value_m,value_n))
```

## The `right_inverse` Step

`right_inverse(layout_mn)` computes the inverse map:

```text
(thread_id, value_id) -> (M,N) coordinate
```

The C++ source for `right_inverse` flattens and coalesces the input layout, then sorts modes by stride:

```cpp
// Sort by strides
using Sorted = detail::SortByKey<decltype(filtered_stride), decltype(filtered_seq)>;
```

For:

```text
layout_mn = ((2,4),(4,8)):((128,8),(32,1))
```

flatten the shape and stride:

```text
shape  = 2, 4, 4, 8
stride = 128, 8, 32, 1
```

Sorted by stride:

```text
stride 1   -> thread_n, shape 8
stride 8   -> value_n,  shape 4
stride 32  -> thread_m, shape 4
stride 128 -> value_m,  shape 2
```

After `with_shape((size(thr_layout), size(val_layout)))`, meaning `with_shape((32, 8))`, CuTe prints:

```text
tv_layout = ((8,4),(4,2)):((32,2),(8,1))
```

This is not the same layout as:

```text
((4,8),(2,4)):((64,4),(32,1))
```

The two textual layouts are not mathematically identical. The printed `tv_layout` is the canonical inverse layout produced by `raked_product` and `right_inverse`.

A useful way to read the printed layout is:

```text
((thread_n, thread_m), (value_n, value_m))
```

not:

```text
((thread_m, thread_n), (value_m, value_n))
```

## Why `composition` Makes It Easier to Understand

Raw `tv_layout` maps:

```text
(thread_id, value_id) -> CTA tile logical coordinate
```

It is still tile-local.

The data tensor slice has real memory strides. For example:

```text
blkW = (8,32):(100,1)
```

This means:

```text
offset = row * 100 + col
```

Then:

```python
tidfrgW = cute.composition(blkW, tv_layout)
```

means:

```text
tidfrgW = blkW o tv_layout
```

or:

```text
(thread,value) -> tile coordinate -> global memory offset
```

For the current elementwise kernel, this prints:

```text
tidfrgW = ((8,4),(4,2)):((4,200),(1,100))
```

This is usually the most useful layout to inspect, because its stride is now real global-memory stride:

```text
thread_n extent 8 -> stride 4
thread_m extent 4 -> stride 200
value_n  extent 4 -> stride 1
value_m  extent 2 -> stride 100
```

So each thread gets:

```text
value_n = 4 contiguous columns, stride 1
value_m = 2 rows, stride 100
```

That is exactly the intended access pattern:

```text
4 contiguous fp32 values per vector copy
2 vector copies per thread
```

## Per-Thread Shape

After:

```python
thr_coord = (tidx, None)
thrW = tidfrgW[thr_coord]
thrCrd = tidfrgCrd[thr_coord]
```

the thread mode is fixed and the value mode remains.

The printed result is:

```text
thrW   = ((4,2)):((1,100))
thrCrd = ((4,2)):((1@1,1@0))
```

The shape is not:

```text
(4,2)
```

It is:

```text
((4,2),)
```

That means there is one outer mode, and that mode is nested.

Therefore:

```python
thrCrd.shape[0]     # (4,2)
thrCrd.shape[0][0]  # 4
thrCrd.shape[0][1]  # 2
```

but:

```python
thrCrd.shape[1]
```

is invalid, because the outer rank is only 1.

## What `rmem` Tensor Means

`rmem` means register memory.

For example:

```python
frgW = cute.make_fragment_like(thrW)
```

creates a register fragment with the same logical shape as `thrW`.

Similarly:

```python
frgPred = cute.make_rmem_tensor(pred_shape, cutlass.Boolean)
```

creates a per-thread predicate tensor in registers.

It is not global memory and not shared memory. It is a thread-local register fragment.

## Why Predicate Shape Is `(2, (1,))`

For this kernel:

```text
thrW.shape = ((4,2))
```

Interpret it as:

```text
AtomV = 4
RestV = 2
```

`AtomV = 4` means one vectorized copy moves 4 fp32 elements:

```text
4 * sizeof(float) = 16 bytes = 128 bits
```

`RestV = 2` means the thread has two such vector copies.

The data fragment is element-shaped:

```text
((4,2))
```

But the predicate for `cute.copy` is copy-instruction-shaped, not element-shaped.

So the predicate is:

```text
(RestV, (AtomPred))
```

For this case:

```text
RestV = 2
AtomPred = 1
```

Therefore:

```python
pred_shape = (thrCrd.shape[0][1], (1,))
frgPred = cute.make_rmem_tensor(pred_shape, cutlass.Boolean)
```

prints:

```text
frgPred = (2,(1)):(1,(0))
```

The `(1,)` means:

```text
one predicate controls one vector copy
```

For fp32, that one predicate controls 4 elements. In other words:

```text
1 predicate * 4 fp32 values = one 128-bit copy
```

This is why this is valid:

```python
pred_shape = (thrCrd.shape[0][1], (1,))
```

but this is not valid for vectorized copy:

```python
frgPred = cute.make_rmem_tensor(thrCrd.shape, cutlass.Boolean)
```

The second version would create an element-level predicate:

```text
((4,2))
```

but `CopyUniversalOp` expects a vector-copy-level predicate:

```text
(2,(1))
```

## How Predicate Values Are Filled

The code:

```python
for i in cutlass.range_constexpr(cute.size(frgPred)):
    frgPred[i] = cute.elem_less(thrCrd[i * copy_elems], shape)
```

For fp32:

```text
copy_elems = 4
```

and:

```text
cute.size(frgPred) = 2
```

So it checks:

```text
i = 0 -> thrCrd[0]
i = 1 -> thrCrd[4]
```

Those are the starting coordinates of the two vector copies.

`cute.elem_less(coord, shape)` checks:

```text
coord[0] < shape[0]
coord[1] < shape[1]
```

or:

```text
row < seq_len and col < dim
```

If a vector copy starts outside the valid logical tensor, its predicate is false, and the copy is skipped.

This safely handles residual CTA tiles in the M direction, such as:

```text
seq_len = 511
```

where the final CTA covers rows beyond the valid tensor.

For N-direction tails that are not multiples of the vector width, a scalar tail path is needed, because a vector predicate controls the whole vector copy, not individual elements inside it.

## Relation to TensorOp GEMM Predicate Code

The same idea appears in the CUTLASS CuTeDSL TensorOp GEMM example:

```python
tBpB = cute.make_rmem_tensor(
    cute.make_layout(
        (
            tBsB.shape[0][1],
            cute.size(tBsB, mode=[1]),
            cute.size(tBsB, mode=[2]),
        ),
        stride=(cute.size(tBsB, mode=[1]), 1, 0),
    ),
    cutlass.Boolean,
)
```

The important part is:

```python
tBsB.shape[0][1]
```

This drops the atom-internal element dimension and keeps the number of copy atoms, exactly like our:

```python
thrCrd.shape[0][1]
```

The GEMM code also uses:

```python
stride=(cute.size(tBsB, mode=[1]), 1, 0)
```

The final `0` stride means the K dimension reuses the same predicate values.
That kernel stores predicates for the N dimension and handles K residue with an explicit branch.

The core idea is the same:

```text
predicate tensors are shaped for copy atoms, not for individual elements.
```

## Takeaways

`make_layout_tv` should be understood as:

```text
layout_mn = raked_product(thr_layout, val_layout)
layout_tv = right_inverse(layout_mn).with_shape((num_threads, values_per_thread))
```

It is not a direct textual preservation of:

```text
((thread_m,thread_n),(value_m,value_n))
```

For actual memory access, inspect:

```python
cute.composition(blkW, tv_layout)
```

because that connects tile-local TV mapping to real global-memory strides.

For vectorized copy predicates:

```text
data shape      = ((AtomV, RestV))
predicate shape = (RestV, (1,))
```

For the current fp32 kernel:

```text
data shape      = ((4,2))
predicate shape = (2,(1))
```

The `(1,)` means one bool controls one vector copy, and that vector copy moves 4 fp32 elements.
