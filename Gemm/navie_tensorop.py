import argparse
import time
from typing import Tuple

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


class NaiveTensorOpGemm:
    """A minimal Ampere Tensor Core GEMM using one m16n8k8 MMA atom per CTA.

    This is a teaching kernel for understanding Tensor Core register fragments:
    - one CTA has exactly one warp, i.e. 32 threads
    - one CTA computes one C tile with shape (16, 8)
    - each K step consumes one A/B tile with K = 8
    - A/B are copied from global memory directly to register fragments
    - cute.gemm emits the warp-level mma.sync.aligned.m16n8k8 instruction

    Logical tensors:
    - A: (M, K), row-major, dtype fp16
    - B: (N, K), row-major in this tensor view, dtype fp16
    - C: (M, N), row-major, dtype fp32
    """

    def __init__(self, cta_tiler: Tuple[int, int, int] = (16, 8, 8)):
        self.cta_tiler = cta_tiler
        self.bM, self.bN, self.bK = cta_tiler
        assert self.cta_tiler == (16, 8, 8), "this example is fixed to m16n8k8"
        self.num_threads = 32

    @cute.jit
    def __call__(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mC: cute.Tensor,
        stream: cuda.CUstream = cuda.CUstream(cuda.CUstream_flags.CU_STREAM_DEFAULT),
    ):
        mma_op = cute.nvgpu.warp.MmaF16BF16Op(
            cutlass.Float16,
            cutlass.Float32,
            (16, 8, 8),
        )

        # atom_layout=(1,1,1) means no tiling above the hardware atom:
        # this CTA/warp is exactly one m16n8k8 MMA atom.
        tiled_mma = cute.make_tiled_mma(mma_op, cute.make_layout((1, 1, 1)))
        print("[DSL INFO] naive tensorop MMA setup:")
        print(f"[DSL INFO]   cta_tiler      = {self.cta_tiler}")
        print("[DSL INFO]   mma_inst_shape = (16, 8, 8)")
        print("[DSL INFO]   atom_layout    = (1, 1, 1)")
        print(f"[DSL INFO]   tiled_mma      = {tiled_mma}")

        grid = (
            cute.ceil_div(mC.shape[0], self.bM),
            cute.ceil_div(mC.shape[1], self.bN),
            1,
        )

        self.kernel(mA, mB, mC, tiled_mma).launch(
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
        tiled_mma: cute.TiledMma,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, _ = cute.arch.block_idx()

        thr_mma = tiled_mma.get_slice(tidx)

        gC = cute.local_tile(
            mC,
            tiler=self.cta_tiler,
            coord=(bidx, bidy, None),
            proj=(1, 1, None),
        )

        # partition_C returns the per-lane accumulator layout required by
        # mma.sync.m16n8k8. For f32 accumulation each lane owns four f32 C regs.
        tCgC = thr_mma.partition_C(gC)
        tCrC = tiled_mma.make_fragment_C(tCgC)
        tCrC.fill(0.0)

        print("[DSL INFO] naive tensorop C partition:")
        print(f"[DSL INFO]   thr_mma = {thr_mma}")
        print(f"[DSL INFO]   gC      = {gC.type}")
        print(f"[DSL INFO]   tCgC    = {tCgC.type}")
        print(f"[DSL INFO]   tCrC    = {tCrC.type}")

        copy_A = cute.make_copy_atom(
            cute.nvgpu.CopyG2ROp(),
            mA.element_type,
            num_bits_per_copy=mA.element_type.width,
        )
        copy_B = cute.make_copy_atom(
            cute.nvgpu.CopyG2ROp(),
            mB.element_type,
            num_bits_per_copy=mB.element_type.width,
        )
        copy_C = cute.make_copy_atom(
            cute.nvgpu.CopyR2GOp(),
            mC.element_type,
            num_bits_per_copy=mC.element_type.width,
        )

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

            # These partitions are the register-fragment layout bridge:
            # gA/gB are logical CTA tiles, while tCgA/tCgB are per-lane views
            # matching PTX mma.sync.m16n8k8 A/B register requirements.
            tCgA = thr_mma.partition_A(gA)
            tCgB = thr_mma.partition_B(gB)

            tCrA = tiled_mma.make_fragment_A(tCgA)
            tCrB = tiled_mma.make_fragment_B(tCgB)

            cute.copy(copy_A, tCgA, tCrA)
            cute.copy(copy_B, tCgB, tCrB)

            print("[DSL INFO] naive tensorop A/B fragments:")
            print(f"[DSL INFO]   gA   = {gA.type}")
            print(f"[DSL INFO]   gB   = {gB.type}")
            print(f"[DSL INFO]   tCgA = {tCgA.type}")
            print(f"[DSL INFO]   tCgB = {tCgB.type}")
            print(f"[DSL INFO]   tCrA = {tCrA.type}")
            print(f"[DSL INFO]   tCrB = {tCrB.type}")

            cute.gemm(tiled_mma, tCrC, tCrA, tCrB, tCrC)

        cute.copy(copy_C, tCrC, tCgC)


def run(mnk: Tuple[int, int, int] = (128, 128, 128)):
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required to run this example")

    M, N, K = mnk
    if M % 16 != 0 or N % 8 != 0 or K % 8 != 0:
        raise ValueError("m16n8k8 example requires M%16 == 0, N%8 == 0, K%8 == 0")

    torch.manual_seed(0)
    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((N, K), device="cuda", dtype=torch.float16)
    c = torch.empty((M, N), device="cuda", dtype=torch.float32)

    a_tensor = from_dlpack(a, assumed_align=16)
    b_tensor = from_dlpack(b, assumed_align=16)
    c_tensor = from_dlpack(c, assumed_align=16)

    torch_stream = torch.cuda.current_stream()
    current_stream = cuda.CUstream(torch_stream.cuda_stream)

    gemm = NaiveTensorOpGemm()
    print("Compiling naive m16n8k8 TensorOp GEMM...")
    start = time.time()
    compiled_gemm = cute.compile[cute.GenerateLineInfo](
        gemm, a_tensor, b_tensor, c_tensor, stream=current_stream
    )
    print(f"Compilation time: {time.time() - start:.4f} seconds")

    compiled_gemm(a_tensor, b_tensor, c_tensor)
    torch.cuda.synchronize()

    ref = torch.einsum("mk,nk->mn", a.float(), b.float())
    torch.testing.assert_close(c, ref, atol=2e-2, rtol=2e-2)
    print("PASS")


def parse_triplet(value: str) -> Tuple[int, int, int]:
    items = tuple(int(x.strip()) for x in value.split(","))
    if len(items) != 3:
        raise argparse.ArgumentTypeError("expected three comma-separated integers")
    return items


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mnk", type=parse_triplet, default=(128, 128, 128))
    args = parser.parse_args()
    run(args.mnk)
