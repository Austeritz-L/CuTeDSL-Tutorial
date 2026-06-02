import argparse
import math
import time
from typing import Tuple

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


class NaiveFlashAttention:
    """Scaffold for a teaching FlashAttention forward kernel.

    Tensor layout follows the CUTLASS CuTeDSL FA2 example:
    - Q: (batch, seqlen_q, num_heads, head_dim), contiguous in head_dim
    - K: (batch, seqlen_k, num_heads, head_dim), contiguous in head_dim
    - V: (batch, seqlen_k, num_heads, head_dim), contiguous in head_dim
    - O: (batch, seqlen_q, num_heads, head_dim), contiguous in head_dim

    This file intentionally leaves the kernel body as a shape-printing scaffold.
    Fill `kernel()` with the actual algorithm:

        S = Q @ K^T
        P = softmax(S * softmax_scale)
        O = P @ V

    A first naive version can use one CTA per `(batch, head, q-block)` and keep
    the implementation scalar/simple before introducing tensor cores, online
    softmax, cp.async, swizzle, or pipelining.
    """

    def __init__(
        self,
        q_block_size: int = 16,
        k_block_size: int = 16,
        head_dim: int = 64,
        num_threads: int = 128,
        is_causal: bool = False,
    ):
        self.q_block_size = q_block_size
        self.k_block_size = k_block_size
        self.head_dim = head_dim
        self.num_threads = num_threads
        self.is_causal = is_causal

        assert q_block_size > 0
        assert k_block_size > 0
        assert head_dim > 0
        assert num_threads > 0

    @staticmethod
    def can_implement(
        dtype: cutlass.Numeric,
        head_dim: int,
        q_block_size: int,
        k_block_size: int,
        num_threads: int,
    ) -> bool:
        if dtype not in (cutlass.Float16, cutlass.BFloat16, cutlass.Float32):
            return False
        if head_dim <= 0 or q_block_size <= 0 or k_block_size <= 0:
            return False
        if num_threads <= 0:
            return False
        return True

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        softmax_scale: cutlass.Float32,
        stream: cuda.CUstream = cuda.CUstream(cuda.CUstream_flags.CU_STREAM_DEFAULT),
    ):
        if cutlass.const_expr(
            not (
                mQ.element_type == mK.element_type
                and mQ.element_type == mV.element_type
                and mQ.element_type == mO.element_type
            )
        ):
            raise TypeError("Q, K, V, and O must have the same dtype in this scaffold")

        batch = mQ.shape[0]
        seqlen_q = mQ.shape[1]
        num_heads = mQ.shape[2]
        seqlen_k = mK.shape[1]

        print("[DSL INFO] NaiveFlashAttention launch setup:")
        print(f"[DSL INFO]   Q = {mQ.type}")
        print(f"[DSL INFO]   K = {mK.type}")
        print(f"[DSL INFO]   V = {mV.type}")
        print(f"[DSL INFO]   O = {mO.type}")
        print(f"[DSL INFO]   q_block_size = {self.q_block_size}")
        print(f"[DSL INFO]   k_block_size = {self.k_block_size}")
        print(f"[DSL INFO]   head_dim     = {self.head_dim}")
        print(f"[DSL INFO]   num_threads  = {self.num_threads}")
        print(f"[DSL INFO]   is_causal    = {self.is_causal}")

        grid = (
            cute.ceil_div(seqlen_q, self.q_block_size),
            batch * num_heads,
            1,
        )

        self.kernel(
            mQ,
            mK,
            mV,
            mO,
            softmax_scale,
        ).launch(
            grid=grid,
            block=(self.num_threads, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        softmax_scale: cutlass.Float32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        q_tile_idx, batch_head_idx, _ = cute.arch.block_idx()

        seqlen_q = mQ.shape[1]
        seqlen_k = mK.shape[1]
        num_heads = mQ.shape[2]
        batch_idx = batch_head_idx // num_heads
        head_idx = batch_head_idx - batch_idx * num_heads
        q_start = q_tile_idx * self.q_block_size

        # TODO: implement the naive FlashAttention kernel here.
        #
        # Suggested first implementation:
        # 1. Map each CTA to one `(batch, head, q_block)` tile.
        # 2. For each q row in the block, compute scores over all K rows:
        #       score[q, k] = dot(Q[b, q, h, :], K[b, k, h, :]) * scale
        # 3. Apply causal mask when `self.is_causal`.
        # 4. Compute row-wise softmax.
        # 5. Accumulate output:
        #       O[b, q, h, d] = sum_k softmax(score[q, k]) * V[b, k, h, d]
        #
        # Keep the first version scalar and readable. After it is correct, we
        # can replace the QK/PV dot products with tiled MMA and use online
        # softmax to avoid materializing the full score matrix.


def attention_ref(
    q,
    k,
    v,
    softmax_scale: float,
    is_causal: bool,
):
    # PyTorch SDPA expects (batch, heads, sequence, head_dim).
    q_ref = q.permute(0, 2, 1, 3)
    k_ref = k.permute(0, 2, 1, 3)
    v_ref = v.permute(0, 2, 1, 3)
    out = torch_scaled_dot_product_attention(
        q_ref,
        k_ref,
        v_ref,
        softmax_scale,
        is_causal,
    )
    return out.permute(0, 2, 1, 3).contiguous()


def torch_scaled_dot_product_attention(
    q,
    k,
    v,
    softmax_scale: float,
    is_causal: bool,
):
    import torch

    return torch.nn.functional.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=is_causal,
        scale=softmax_scale,
    )


def run(
    dtype: cutlass.Numeric = cutlass.Float16,
    batch_size: int = 1,
    seqlen_q: int = 128,
    seqlen_k: int = 128,
    num_heads: int = 4,
    head_dim: int = 64,
    q_block_size: int = 16,
    k_block_size: int = 16,
    num_threads: int = 128,
    softmax_scale: float | None = None,
    is_causal: bool = False,
    run_kernel: bool = False,
    skip_ref_check: bool = False,
):
    import torch
    import cutlass.torch as cutlass_torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required to run this example")

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)

    if not NaiveFlashAttention.can_implement(
        dtype, head_dim, q_block_size, k_block_size, num_threads
    ):
        raise TypeError(
            "Unsupported naive attention scaffold config: "
            f"{dtype=}, {head_dim=}, {q_block_size=}, {k_block_size=}, {num_threads=}"
        )

    torch_dtype = cutlass_torch.dtype(dtype)
    torch.manual_seed(0)
    q = torch.randn(
        (batch_size, seqlen_q, num_heads, head_dim),
        device="cuda",
        dtype=torch_dtype,
    )
    k = torch.randn(
        (batch_size, seqlen_k, num_heads, head_dim),
        device="cuda",
        dtype=torch_dtype,
    )
    v = torch.randn(
        (batch_size, seqlen_k, num_heads, head_dim),
        device="cuda",
        dtype=torch_dtype,
    )
    o = torch.empty(
        (batch_size, seqlen_q, num_heads, head_dim),
        device="cuda",
        dtype=torch_dtype,
    )

    q_tensor = from_dlpack(q, assumed_align=16)
    k_tensor = from_dlpack(k, assumed_align=16)
    v_tensor = from_dlpack(v, assumed_align=16)
    o_tensor = from_dlpack(o, assumed_align=16)

    torch_stream = torch.cuda.current_stream()
    current_stream = cuda.CUstream(torch_stream.cuda_stream)

    attention = NaiveFlashAttention(
        q_block_size=q_block_size,
        k_block_size=k_block_size,
        head_dim=head_dim,
        num_threads=num_threads,
        is_causal=is_causal,
    )

    print("Compiling naive FlashAttention scaffold...")
    print(f"  dtype: {dtype}")
    print(f"  batch_size: {batch_size}")
    print(f"  seqlen_q: {seqlen_q}")
    print(f"  seqlen_k: {seqlen_k}")
    print(f"  num_heads: {num_heads}")
    print(f"  head_dim: {head_dim}")
    print(f"  q_block_size: {q_block_size}")
    print(f"  k_block_size: {k_block_size}")
    print(f"  num_threads: {num_threads}")
    print(f"  softmax_scale: {softmax_scale}")
    print(f"  is_causal: {is_causal}")
    print(f"  run_kernel: {run_kernel}")
    start = time.time()
    compiled_attention = cute.compile[cute.GenerateLineInfo](
        attention,
        q_tensor,
        k_tensor,
        v_tensor,
        o_tensor,
        cutlass.Float32(softmax_scale),
        stream=current_stream,
    )
    print(f"Compilation time: {time.time() - start:.4f} seconds")

    ref = None
    if not skip_ref_check:
        ref = attention_ref(q, k, v, softmax_scale, is_causal)
        print("Reference output computed with torch scaled_dot_product_attention.")

    if not run_kernel:
        print("Kernel execution skipped. Fill kernel(), then run with --run_kernel.")
        return

    compiled_attention(
        q_tensor,
        k_tensor,
        v_tensor,
        o_tensor,
        cutlass.Float32(softmax_scale),
        current_stream,
    )
    torch.cuda.synchronize()

    if not skip_ref_check:
        torch.testing.assert_close(o, ref, atol=2e-2, rtol=2e-2)
        print("PASS")
    else:
        print("Kernel executed; reference check skipped.")


def parse_triplet(value: str) -> Tuple[int, int, int]:
    items = tuple(int(x.strip()) for x in value.split(","))
    if len(items) != 3:
        raise argparse.ArgumentTypeError("expected three comma-separated integers")
    return items


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Naive FlashAttention scaffold")
    parser.add_argument("--dtype", type=cutlass.dtype, default=cutlass.Float16)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seqlen_q", type=int, default=128)
    parser.add_argument("--seqlen_k", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--head_dim", type=int, default=64)
    parser.add_argument("--q_block_size", type=int, default=16)
    parser.add_argument("--k_block_size", type=int, default=16)
    parser.add_argument("--num_threads", type=int, default=128)
    parser.add_argument("--softmax_scale", type=float, default=None)
    parser.add_argument("--is_causal", action="store_true")
    parser.add_argument(
        "--run_kernel",
        action="store_true",
        help="Run the kernel body. The scaffold kernel is intentionally TODO.",
    )
    parser.add_argument("--skip_ref_check", action="store_true")
    args = parser.parse_args()

    run(
        dtype=args.dtype,
        batch_size=args.batch_size,
        seqlen_q=args.seqlen_q,
        seqlen_k=args.seqlen_k,
        num_heads=args.num_heads,
        head_dim=args.head_dim,
        q_block_size=args.q_block_size,
        k_block_size=args.k_block_size,
        num_threads=args.num_threads,
        softmax_scale=args.softmax_scale,
        is_causal=args.is_causal,
        run_kernel=args.run_kernel,
        skip_ref_check=args.skip_ref_check,
    )
