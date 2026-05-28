import argparse
import time
from typing import Sequence, Tuple, Type

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


@cute.kernel
def weights_qscale_mul_kernel(
    gW: cute.Tensor,  # tiled weights, shape: ((TileM, TileN), Rest)
    gQ: cute.Tensor,  # tiled q_scale, shape: ((TileM, TileN), Rest)
    gO: cute.Tensor,  # tiled output, shape: ((TileM, TileN), Rest)
    tv_layout: cute.Layout,
    gate_scale: cutlass.Float32,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()

    blk_coord = ((None, None), (bidx, bidy))
    blkW = gW[blk_coord]
    blkQ = gQ[blk_coord]
    blkO = gO[blk_coord]

    # print("[DSL INFO] CTA tensors:")
    # print(f"[DSL INFO]   blkW   = {blkW.type}")
    # print(f"[DSL INFO]   blkQ   = {blkQ.type}")
    # print(f"[DSL INFO]   blkO   = {blkO.type}")

    tidfrgW = cute.composition(blkW, tv_layout)
    tidfrgQ = cute.composition(blkQ, tv_layout)
    tidfrgO = cute.composition(blkO, tv_layout)

    thr_coord = (tidx, cute.repeat_like(None, tidfrgQ[1]))
    thrW = tidfrgW[thr_coord]
    thrQ = tidfrgQ[thr_coord]
    thrO = tidfrgO[thr_coord]

    # print("[DSL INFO] Per-thread tensors:")
    # print(f"[DSL INFO]   thrW   = {thrW.type}")
    # print(f"[DSL INFO]   thrQ   = {thrQ.type}")
    # print(f"[DSL INFO]   thrO   = {thrO.type}")

    copy_atom_load_w = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gW.element_type)
    copy_atom_load = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gQ.element_type)
    copy_atom_store = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gO.element_type)

    frgW = cute.make_fragment_like(thrW)
    frgQ = cute.make_fragment_like(thrQ)
    frgO = cute.make_fragment_like(thrO)
    cute.copy(copy_atom_load_w, thrW, frgW)
    cute.copy(copy_atom_load, thrQ, frgQ)

    for i in cutlass.range_constexpr(cute.size(frgO)):
        frgO[i] = frgW[i] * frgQ[i] * gate_scale

    cute.copy(copy_atom_store, frgO, thrO)


@cute.jit
def weights_qscale_mul(
    mW: cute.Tensor,
    mQ: cute.Tensor,
    mO: cute.Tensor,
    gate_scale: cutlass.Float32,
    stream: cuda.CUstream,
):
    """Compute out[seq, dim] = weights[seq, dim] * q_scale[seq, dim] * gate_scale.

    This is the pure 2D version of:

        weights * q_scale * gate_scale

    This version intentionally has no residue predicate so that the copies stay
    vectorized. Use tile-friendly shapes; otherwise the last CTA can read/write
    out of bounds.
    """

    dtype = mQ.element_type
    copy_bits = 128
    vector_size = copy_bits // dtype.width

    # Map threads along the contiguous dim dimension for coalesced vectorized
    # loads/stores. For fp32 this creates a CTA tile of (16, 32), matching the
    # default tutorial shape.
    thr_layout = cute.make_ordered_layout((4, 8), order=(1, 0))
    val_layout = cute.make_ordered_layout((4, vector_size), order=(1, 0))
    tiler_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)

    # print("[DSL INFO] Input tensors:")
    # print(f"[DSL INFO]   mW = {mW.type}")
    # print(f"[DSL INFO]   mQ = {mQ.type}")
    # print(f"[DSL INFO]   mO = {mO.type}")
    # print("[DSL INFO] Tiling parameters:")
    # print(f"[DSL INFO]   tiler_mn = {tiler_mn}")
    # print(f"[DSL INFO]   tv_layout = {tv_layout}")

    gW = cute.zipped_divide(mW, tiler_mn)
    gQ = cute.zipped_divide(mQ, tiler_mn)
    gO = cute.zipped_divide(mO, tiler_mn)

    # print("[DSL INFO] Tiled tensors:")
    # print(f"[DSL INFO]   gW = {gW.type}")
    # print(f"[DSL INFO]   gQ = {gQ.type}")
    # print(f"[DSL INFO]   gO = {gO.type}")

    weights_qscale_mul_kernel.set_name_prefix("weights_qscale_mul_kernel")
    weights_qscale_mul_kernel(gW, gQ, gO, tv_layout, gate_scale).launch(
        grid=cute.product_each(gO.shape[1]),
        block=[cute.size(tv_layout, mode=[0]), 1, 1],
        stream=stream,
    )


def parse_shape(value: str) -> Tuple[int, int]:
    items = tuple(int(x.strip()) for x in value.split(","))
    if len(items) != 2:
        raise argparse.ArgumentTypeError("expected seq_len,dim")
    return items


def parse_int_list(value: str) -> Tuple[int, ...]:
    items = tuple(int(x.strip()) for x in value.split(",") if x.strip())
    if not items:
        raise argparse.ArgumentTypeError("expected comma-separated integers")
    return items


def dtype_from_name(name: str) -> Type[cutlass.Numeric]:
    if name == "float32":
        return cutlass.Float32
    if name == "float16":
        return cutlass.Float16
    raise argparse.ArgumentTypeError("dtype must be float32 or float16")


def benchmark_cuda_events(fn, warmup_iterations: int, iterations: int) -> float:
    import torch

    for _ in range(warmup_iterations):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000.0 / iterations


def run_case(
    shape: Tuple[int, int] = (1024, 32),
    dtype: Type[cutlass.Numeric] = cutlass.Float32,
    gate_scale: float = 1.25,
    warmup_iterations: int = 10,
    iterations: int = 100,
    skip_ref_check: bool = False,
):
    import torch
    import cutlass.torch as cutlass_torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required to run this example")

    seq_len, dim = shape

    torch_dtype = cutlass_torch.dtype(dtype)
    tile_m = 16
    tile_n = 8 * (128 // dtype.width)
    if seq_len % tile_m != 0 or dim % tile_n != 0:
        raise ValueError(
            "This vectorized no-predicate version requires "
            f"seq_len % {tile_m} == 0 and dim % {tile_n} == 0, "
            f"but got seq_len={seq_len}, dim={dim}. "
            "Use a tile-friendly shape or add predicates."
        )

    torch.manual_seed(0)
    weights = torch.randn((seq_len, dim), device="cuda", dtype=torch_dtype)
    q_scale = torch.randn((seq_len, dim, 1), device="cuda", dtype=torch_dtype)
    out = torch.empty_like(q_scale)

    # Keep the user-facing operation as weights.unsqueeze(-1) * q_scale, while
    # presenting the trivial last dimension as a 2D view to the CuTeDSL kernel.
    q_scale_2d = q_scale.squeeze(-1)
    out_2d = out.squeeze(-1)

    weights_tensor = from_dlpack(weights, assumed_align=16)
    q_tensor = from_dlpack(q_scale_2d, assumed_align=16)
    out_tensor = from_dlpack(out_2d, assumed_align=16)

    torch_stream = torch.cuda.current_stream()
    current_stream = cuda.CUstream(torch_stream.cuda_stream)

    print(f"\n=== Case: seq_len={seq_len}, dim={dim} ===")
    print("Compiling CuTeDSL elementwise mul kernel...")
    start_time = time.time()
    compiled_fn = cute.compile[cute.GenerateLineInfo(True)](
        weights_qscale_mul,
        weights_tensor,
        q_tensor,
        out_tensor,
        gate_scale,
        current_stream,
    )
    print(f"Compilation time: {time.time() - start_time:.4f} seconds")

    def run_cutedsl():
        compiled_fn(
            weights_tensor,
            q_tensor,
            out_tensor,
            gate_scale,
            current_stream,
        )

    @torch.compile
    def run_torch():
        return weights.unsqueeze(-1) * q_scale * gate_scale

    if not skip_ref_check:
        run_cutedsl()
        torch.cuda.synchronize()
        ref = run_torch()
        if dtype is cutlass.Float16:
            torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)
        else:
            torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)
        print("Correctness check: PASS")

    cutedsl_us = benchmark_cuda_events(run_cutedsl, warmup_iterations, iterations)
    torch_us = benchmark_cuda_events(run_torch, warmup_iterations, iterations)

    bytes_per_elem = dtype.width // 8
    # CuTeDSL reads q_scale + weights and writes out. Weight reuse through cache is
    # shape-dependent, so this is a simple logical traffic estimate.
    cutedsl_bytes = (q_scale.numel() * 2 + weights.numel()) * bytes_per_elem
    torch_bytes_min = (q_scale.numel() * 3 + weights.numel()) * bytes_per_elem

    print("Benchmark:")
    print(f"Shape: weights=({seq_len}, {dim}), q_scale=({seq_len}, {dim}, 1)")
    print(f"Dtype: {torch_dtype}, gate_scale={gate_scale}")
    print(f"CuTeDSL fused kernel: {cutedsl_us:.3f} us")
    print(f"PyTorch eager expr:   {torch_us:.3f} us")
    print(f"Speedup:             {torch_us / cutedsl_us:.2f}x")
    print(
        "Estimated CuTeDSL logical bandwidth: "
        f"{cutedsl_bytes / (cutedsl_us / 1e6) / 1e9:.2f} GB/s"
    )
    print(
        "Estimated PyTorch min logical bandwidth: "
        f"{torch_bytes_min / (torch_us / 1e6) / 1e9:.2f} GB/s"
    )

    return {
        "seq_len": seq_len,
        "dim": dim,
        "cutedsl_us": cutedsl_us,
        "torch_us": torch_us,
        "speedup": torch_us / cutedsl_us,
    }


def run(
    shapes: Sequence[Tuple[int, int]],
    dtype: Type[cutlass.Numeric] = cutlass.Float32,
    gate_scale: float = 1.25,
    warmup_iterations: int = 10,
    iterations: int = 100,
    skip_ref_check: bool = False,
):
    results = []
    for shape in shapes:
        results.append(
            run_case(
                shape=shape,
                dtype=dtype,
                gate_scale=gate_scale,
                warmup_iterations=warmup_iterations,
                iterations=iterations,
                skip_ref_check=skip_ref_check,
            )
        )

    print("\n=== Summary ===")
    print("seq_len,dim,cutedsl_us,torch_us,speedup")
    for result in results:
        print(
            f"{result['seq_len']},{result['dim']},"
            f"{result['cutedsl_us']:.3f},"
            f"{result['torch_us']:.3f},"
            f"{result['speedup']:.2f}x"
        )
    print("\nNote: PyTorch eager measures the literal expression and may launch more than one kernel.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shape",
        type=parse_shape,
        default=None,
        help="Run one case only, formatted as seq_len,dim. By default run the benchmark suite.",
    )
    parser.add_argument(
        "--seq-lens",
        type=parse_int_list,
        default=(256, 512, 2048, 8192),
        help="Comma-separated seq_len values for the benchmark suite.",
    )
    parser.add_argument(
        "--dims",
        type=parse_int_list,
        default=(32, 128),
        help="Comma-separated dim values for the benchmark suite.",
    )
    parser.add_argument("--dtype", type=dtype_from_name, default=cutlass.Float32)
    parser.add_argument("--gate-scale", type=float, default=1.25)
    parser.add_argument("--warmup-iterations", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--skip-ref-check", action="store_true")
    args = parser.parse_args()

    shapes = [args.shape] if args.shape is not None else [
        (seq_len, dim) for seq_len in args.seq_lens for dim in args.dims
    ]
    run(
        shapes=shapes,
        dtype=args.dtype,
        gate_scale=args.gate_scale,
        warmup_iterations=args.warmup_iterations,
        iterations=args.iterations,
        skip_ref_check=args.skip_ref_check,
    )
