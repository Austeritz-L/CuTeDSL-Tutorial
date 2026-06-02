import argparse
import os
import sys
import time
from typing import Tuple

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


class NaiveSGemm:
    """A minimal tiled FP32 GEMM: C[M, N] = A[M, K] * B[N, K].

    This is intentionally simple for tutorial use:
    - one CTA computes one (BLK_M, BLK_N) output tile
    - one thread computes one C element
    - A/B tiles are copied through shared memory with CopyUniversalOp
    - accumulation uses MmaUniversalOp(Float32)
    - M, N, K are expected to be exact multiples of the tile sizes
    """

    def __init__(self, cta_tiler: Tuple[int, int, int] = (16, 16, 16)):
        self.cta_tiler = cta_tiler
        self.bM, self.bN, self.bK = cta_tiler
        self.num_threads = self.bM * self.bN

    @cute.jit
    def __call__(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mC: cute.Tensor,
        stream: cuda.CUstream = cuda.CUstream(cuda.CUstream_flags.CU_STREAM_DEFAULT),
    ):
        grid = (
            cute.ceil_div(mC.shape[0], self.bM),
            cute.ceil_div(mC.shape[1], self.bN),
            1,
        )

        self.kernel(
            mA, mB, mC, 
        ).launch(
            grid=grid,
            block=(self.num_threads, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mC: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, _ = cute.arch.block_idx()

        thread_layout = cute.make_layout((self.bM, self.bN), stride=(self.bN, 1))

        copy_atom_A = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), mA.element_type)
        copy_atom_B = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), mB.element_type)
        copy_atom_C = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), mC.element_type)
        
        mma_atom = cute.make_mma_atom(cute.nvgpu.MmaUniversalOp(cutlass.Float32))

        gC = cute.local_tile(
            mC, tiler=self.cta_tiler, coord=(bidx, bidy, None), proj=(1, 1, None)
        )

        # One CTA computes one C tile. local_partition maps each thread to
        # one element of that C tile because thread_layout is (BLK_M, BLK_N).
        tCgC = cute.local_partition(gC, thread_layout, tidx, proj=(1, 1))
        tCrC = cute.make_fragment_like(tCgC, cutlass.Float32)
        tCrC.fill(0.0)

        k_tile_count = mA.shape[1] // self.bK

        for k_tile in range(k_tile_count):
            gA = cute.local_tile(
                mA,
                tiler=self.cta_tiler,
                coord=(bidx, None, k_tile),
                proj=(1, None, 1),
            )
            gB = cute.local_tile(
                mB,
                tiler=self.cta_tiler,
                coord=(None, bidy, k_tile),
                proj=(None, 1, 1),
            )

            # Project the same thread layout onto A's M mode and B's N mode.
            # For a single C(m, n), the thread gets A(m, 0:BLK_K) and
            # B(n, 0:BLK_K), then the universal MMA atom computes the dot.
            tCgA = cute.local_partition(gA, thread_layout, tidx, proj=(1, None))
            tCgB = cute.local_partition(gB, thread_layout, tidx, proj=(None, 1))

            tCrA = cute.make_fragment_like(tCgA, cutlass.Float32)
            tCrB = cute.make_fragment_like(tCgB, cutlass.Float32)
            cute.copy(copy_atom_A, tCgA, tCrA)
            cute.copy(copy_atom_B, tCgB, tCrB)

            # for k in range(cute.size(tCrA)):
            #     tCrC[0] += tCrA[k] * tCrB[k]
            cute.gemm(mma_atom, tCrC, tCrA, tCrB, tCrC)

        cute.copy(copy_atom_C, tCrC, tCgC)


def run(
    mnk: Tuple[int, int, int] = (128, 128, 128),
    tile: Tuple[int, int, int] = (16, 16, 16),
):
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required to run this example")

    M, N, K = mnk
    bM, bN, bK = tile
    if M % bM != 0 or N % bN != 0 or K % bK != 0:
        raise ValueError("This first naive version requires M, N, K to be tile multiples")

    torch.manual_seed(0)
    a = torch.randn((M, K), device="cuda", dtype=torch.float32)
    b = torch.randn((N, K), device="cuda", dtype=torch.float32)
    c = torch.empty((M, N), device="cuda", dtype=torch.float32)

    a_tensor = from_dlpack(a, assumed_align=16)
    b_tensor = from_dlpack(b, assumed_align=16)
    c_tensor = from_dlpack(c, assumed_align=16)

    torch_stream = torch.cuda.current_stream()
    current_stream = cuda.CUstream(torch_stream.cuda_stream)

    gemm = NaiveSGemm(tile)
    print("Compiling naive SGEMM...")
    start = time.time()
    compiled_gemm = cute.compile[cute.GenerateLineInfo](
        gemm, a_tensor, b_tensor, c_tensor, stream=current_stream
    )
    print(f"Compilation time: {time.time() - start:.4f} seconds")

    compiled_gemm(a_tensor, b_tensor, c_tensor)
    torch.cuda.synchronize()

    ref = torch.einsum("mk,nk->mn", a, b)
    torch.testing.assert_close(c, ref, atol=1e-3, rtol=1e-5)
    print("PASS")


def parse_triplet(value: str) -> Tuple[int, int, int]:
    items = tuple(int(x.strip()) for x in value.split(","))
    if len(items) != 3:
        raise argparse.ArgumentTypeError("expected three comma-separated integers")
    return items


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mnk", type=parse_triplet, default=(128, 128, 128))
    parser.add_argument("--tile", type=parse_triplet, default=(16, 64, 32))
    args = parser.parse_args()
    run(args.mnk, args.tile)
