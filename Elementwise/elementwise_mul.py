import argparse
import time
from pathlib import Path
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
    cO: cute.Tensor,  # tiled coordinate tensor, shape: ((TileM, TileN), Rest)
    shape: cute.Shape,
    copy_elems: cutlass.Constexpr,
    tv_layout: cute.Layout,
    gate_scale: cutlass.Float32,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()

    blk_coord = ((None, None), (bidx, bidy))
    blkW = gW[blk_coord]
    blkQ = gQ[blk_coord]
    blkO = gO[blk_coord]
    blkCrd = cO[blk_coord]

    print("[DSL INFO] CTA tensors:")
    print(f"[DSL INFO]   blkW   = {blkW.type}")
    print(f"[DSL INFO]   blkQ   = {blkQ.type}")
    print(f"[DSL INFO]   blkO   = {blkO.type}")
    print(f"[DSL INFO]   blkCrd = {blkCrd.type}")

    tidfrgW = cute.composition(blkW, tv_layout)
    tidfrgQ = cute.composition(blkQ, tv_layout)
    tidfrgO = cute.composition(blkO, tv_layout)
    tidfrgCrd = cute.composition(blkCrd, tv_layout)

    print("[DSL INFO] Thread-value tensors:")
    print(f"[DSL INFO]   tidfrgW   = {tidfrgW.type}")
    print(f"[DSL INFO]   tidfrgQ   = {tidfrgQ.type}")
    print(f"[DSL INFO]   tidfrgO   = {tidfrgO.type}")
    print(f"[DSL INFO]   tidfrgCrd = {tidfrgCrd.type}")

    thr_coord = (tidx, None)
    thrW = tidfrgW[thr_coord]
    thrQ = tidfrgQ[thr_coord]
    thrO = tidfrgO[thr_coord]
    thrCrd = tidfrgCrd[thr_coord]

    print("[DSL INFO] Per-thread tensors:")
    print(f"[DSL INFO]   thr_coord = {thr_coord}")
    print(f"[DSL INFO]   thrW      = {thrW.type}")
    print(f"[DSL INFO]   thrQ      = {thrQ.type}")
    print(f"[DSL INFO]   thrO      = {thrO.type}")
    print(f"[DSL INFO]   thrCrd    = {thrCrd.type}")

    copy_atom_load_w = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gW.element_type)
    copy_atom_load = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gQ.element_type)
    copy_atom_store = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gO.element_type)

    frgW = cute.make_fragment_like(thrW)
    frgQ = cute.make_fragment_like(thrQ)
    frgO = cute.make_fragment_like(thrO)

    # Build a per-thread predicate for out-of-bounds threads in the residual tile.
    # thrCrd.shape[0][1] = 2, for 2 times 128bits copy.
    # (1) for the 1 * 4 * sizeof(cutlass.Float32) = 128bits load/store.
    pred_shape = (thrCrd.shape[0][1], (1,))
    frgPred = cute.make_rmem_tensor(pred_shape, cutlass.Boolean)

    print("[DSL INFO] Register fragments:")
    print(f"[DSL INFO]   frgW      = {frgW.type}")
    print(f"[DSL INFO]   frgQ      = {frgQ.type}")
    print(f"[DSL INFO]   frgO      = {frgO.type}")
    print(f"[DSL INFO]   frgPred  = {frgPred.type}")

    for i in cutlass.range_constexpr(cute.size(frgPred)):
        frgPred[i] = cute.elem_less(thrCrd[i * copy_elems], shape)

    cute.copy(copy_atom_load_w, thrW, frgW, pred=frgPred)
    cute.copy(copy_atom_load, thrQ, frgQ, pred=frgPred)

    for i in cutlass.range_constexpr(cute.size(frgO)):
        frgO[i] = frgW[i] * frgQ[i] * gate_scale

    cute.copy(copy_atom_store, frgO, thrO, pred=frgPred)


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

    A coordinate tensor builds the residue predicate, so non-multiple seq/dim
    shapes are safe.
    """

    dtype = mQ.element_type
    copy_bits = 128
    vector_size = copy_bits // dtype.width

    # Map threads along the contiguous dim dimension for coalesced vectorized
    # loads/stores. For fp32 this creates a CTA tile of (16, 32), matching the
    # default tutorial shape.
    thr_layout = cute.make_ordered_layout((4, 8), order=(1, 0))
    val_layout = cute.make_ordered_layout((2, vector_size), order=(1, 0))
    tiler_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)

    print("[DSL INFO] Input tensors:")
    print(f"[DSL INFO]   mW = {mW.type}")
    print(f"[DSL INFO]   mQ = {mQ.type}")
    print(f"[DSL INFO]   mO = {mO.type}")
    print("[DSL INFO] Tiling parameters:")
    print(f"[DSL INFO]   dtype       = {dtype}")
    print(f"[DSL INFO]   vector_size = {vector_size}")
    print(f"[DSL INFO]   thr_layout  = {thr_layout}")
    print(f"[DSL INFO]   val_layout  = {val_layout}")
    print(f"[DSL INFO]   tiler_mn    = {tiler_mn}")
    # not       ((4,8),(2,4)):((64,4),(32,1))
    # actually  ((8,4),(4,2)):((32,2),(8,1))
    print(f"[DSL INFO]   tv_layout   = {tv_layout}")

    gW = cute.zipped_divide(mW, tiler_mn)
    gQ = cute.zipped_divide(mQ, tiler_mn)
    gO = cute.zipped_divide(mO, tiler_mn)
    idO = cute.make_identity_tensor(mO.shape)
    cO = cute.zipped_divide(idO, tiler=tiler_mn)

    print("[DSL INFO] Tiled tensors:")
    print(f"[DSL INFO]   gW = {gW.type}")
    print(f"[DSL INFO]   gQ = {gQ.type}")
    print(f"[DSL INFO]   gO = {gO.type}")
    print(f"[DSL INFO]   cO = {cO.type}")

    weights_qscale_mul_kernel.set_name_prefix("weights_qscale_mul_kernel")
    weights_qscale_mul_kernel(
        gW, gQ, gO, cO, mO.shape, vector_size, tv_layout, gate_scale
    ).launch(
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


def export_chrome_trace(
    torch,
    run_cutedsl,
    run_torch_compiled,
    trace_path: Path,
    warmup_iterations: int,
    profile_iterations: int,
):
    from torch.profiler import ProfilerActivity, profile, record_function

    for _ in range(warmup_iterations):
        run_cutedsl()
        run_torch_compiled()
    torch.cuda.synchronize()

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof:
        for _ in range(profile_iterations):
            with record_function("cutedsl_weights_qscale_mul"):
                run_cutedsl()
            with record_function("torch_compile_weights_qscale_mul"):
                run_torch_compiled()

    torch.cuda.synchronize()
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    prof.export_chrome_trace(str(trace_path))

    print(f"Chrome trace: {trace_path}")
    print("Profiler CUDA time summary:")
    print(
        prof.key_averages().table(
            sort_by="cuda_time_total",
            row_limit=12,
        )
    )


def run_case(
    shape: Tuple[int, int] = (1024, 32),
    dtype: Type[cutlass.Numeric] = cutlass.Float32,
    gate_scale: float = 1.25,
    warmup_iterations: int = 10,
    iterations: int = 100,
    skip_ref_check: bool = False,
    profile_trace_dir: str | None = None,
    profile_iterations: int = 10,
):
    import torch
    import cutlass.torch as cutlass_torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required to run this example")

    seq_len, dim = shape

    torch_dtype = cutlass_torch.dtype(dtype)
    vector_size = 128 // dtype.width
    if dim % vector_size != 0:
        raise ValueError(
            "This vectorized predicate version requires "
            f"dim % {vector_size} == 0, but got dim={dim}. "
            "Use a vector-friendly dim or switch the copy atom to scalar for full dim residue."
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

    def run_torch_eager():
        return weights.unsqueeze(-1) * q_scale * gate_scale

    run_torch_compiled = torch.compile(run_torch_eager)
    run_torch_compiled()
    torch.cuda.synchronize()

    if not skip_ref_check:
        run_cutedsl()
        torch.cuda.synchronize()
        ref = run_torch_compiled()
        if dtype is cutlass.Float16:
            torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)
        else:
            torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)
        print("Correctness check: PASS")

    cutedsl_us = benchmark_cuda_events(run_cutedsl, warmup_iterations, iterations)
    torch_us = benchmark_cuda_events(run_torch_compiled, warmup_iterations, iterations)

    if profile_trace_dir is not None:
        trace_path = (
            Path(profile_trace_dir)
            / f"elementwise_mul_seq{seq_len}_dim{dim}_{torch_dtype}.json"
        )
        export_chrome_trace(
            torch,
            run_cutedsl,
            run_torch_compiled,
            trace_path,
            warmup_iterations,
            profile_iterations,
        )

    bytes_per_elem = dtype.width // 8
    # CuTeDSL reads q_scale + weights and writes out. Weight reuse through cache is
    # shape-dependent, so this is a simple logical traffic estimate.
    cutedsl_bytes = (q_scale.numel() * 2 + weights.numel()) * bytes_per_elem
    torch_bytes_min = (q_scale.numel() * 3 + weights.numel()) * bytes_per_elem

    print("Benchmark:")
    print(f"Shape: weights=({seq_len}, {dim}), q_scale=({seq_len}, {dim}, 1)")
    print(f"Dtype: {torch_dtype}, gate_scale={gate_scale}")
    print(f"CuTeDSL fused kernel: {cutedsl_us:.3f} us")
    print(f"torch.compile expr:   {torch_us:.3f} us")
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
    profile_trace_dir: str | None = None,
    profile_iterations: int = 10,
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
                profile_trace_dir=profile_trace_dir,
                profile_iterations=profile_iterations,
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
    print("\nNote: torch.compile is warmed up before correctness, benchmark, and profiler collection.")


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
        default=[511],
        help="Comma-separated seq_len values for the benchmark suite.",
    )
    parser.add_argument(
        "--dims",
        type=parse_int_list,
        default=[100],
        help="Comma-separated dim values for the benchmark suite.",
    )
    parser.add_argument("--dtype", type=dtype_from_name, default=cutlass.Float32)
    parser.add_argument("--gate-scale", type=float, default=1.25)
    parser.add_argument("--warmup-iterations", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--skip-ref-check", action="store_true")
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Export a torch.profiler Chrome trace for each benchmark case.",
    )
    parser.add_argument(
        "--profile-dir",
        default="traces",
        help="Directory for torch.profiler Chrome trace JSON files.",
    )
    parser.add_argument(
        "--profile-iterations",
        type=int,
        default=10,
        help="Number of CuTeDSL/torch.compile iterations recorded in each trace.",
    )
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
        profile_trace_dir=args.profile_dir if args.profile else None,
        profile_iterations=args.profile_iterations,
    )
