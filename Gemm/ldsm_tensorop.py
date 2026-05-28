import argparse
import time
from typing import Tuple

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.utils
from cutlass.cute.runtime import from_dlpack


class LDSMTensorOpGemm:
    """A minimal Ampere Tensor Core GEMM using one m16n8k8 MMA atom per CTA.

    This is a teaching kernel for understanding Tensor Core register fragments:
    - one CTA has exactly one warp, i.e. 32 threads
    - one CTA computes one C tile with shape (16, 8)
    - each K step consumes one A/B tile with K = 8
    - A/B are copied from global memory to shared memory (Universal copy with scalar thread slices)
    - A/B are copied from shared memory to register fragments with ldmatrix (LdMatrix8x8x16bOp)
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
        debug: cutlass.Constexpr = True,
        stream: cuda.CUstream = cuda.CUstream(cuda.CUstream_flags.CU_STREAM_DEFAULT),
    ):
        mma_op = cute.nvgpu.warp.MmaF16BF16Op(
            cutlass.Float16,
            cutlass.Float32,
            (16, 8, 8),
        )

        sA_layout = cute.make_layout((self.bM, self.bK), stride=(self.bK, 1))
        sB_layout = cute.make_layout((self.bN, self.bK), stride=(self.bK, 1))

        # atom_layout=(1,1,1) means no tiling above the hardware atom:
        # this CTA/warp is exactly one m16n8k8 MMA atom.
        tiled_mma = cute.make_tiled_mma(mma_op, cute.make_layout((1, 1, 1)))

        grid = (
            cute.ceil_div(mC.shape[0], self.bM),
            cute.ceil_div(mC.shape[1], self.bN),
            1,
        )

        self.kernel(mA, mB, mC, sA_layout, sB_layout, tiled_mma, debug).launch(
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
        sA_layout: cute.Layout,
        sB_layout: cute.Layout,
        tiled_mma: cute.TiledMma,
        debug: cutlass.Constexpr,
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

        if debug:
            if bidx == 0 and bidy == 0 and tidx == 0:
                cute.printf("=== ldsm tensorop m16n8k8 debug ===")
                cute.printf("CTA tile C shape: 16 x 8, K tile: 8, threads: 32")
                cute.printf("lane {} owns {} C accumulator registers", tidx, cute.size(tCrC))
                cute.printf("===tCgC and tCrC after initialization===")
                cute.print_tensor(tCgC)
                cute.print_tensor(tCrC)

        smem = cutlass.utils.SmemAllocator()
        sA = smem.allocate_tensor(mA.element_type, sA_layout, 16)
        sB = smem.allocate_tensor(mB.element_type, sB_layout, 16)

        # Copy atoms for loading A/B from global memory to shared memory.
        # Keep this path deliberately simple: each thread owns a small slice
        # of the CTA tile via local_partition and copies it with a scalar atom.
        g2S_copy_A = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            mA.element_type,
            num_bits_per_copy=mA.element_type.width,
        )
        g2S_copy_B = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            mB.element_type,
            num_bits_per_copy=mB.element_type.width,
        )
        r2G_copy_C = cute.make_copy_atom(
            cute.nvgpu.CopyR2GOp(),
            mC.element_type,
            num_bits_per_copy=mC.element_type.width,
        )

        # 32 threads cooperatively fill A(16,8) and B(8,8) shared-memory tiles.
        g2s_thr_layout_A = cute.make_layout((self.bM, 2), stride=(2, 1))
        g2s_thr_layout_B = cute.make_layout((self.bN, 4), stride=(4, 1))

        # ldmatrix atoms for loading A/B from shared memory to register fragments.
        # m16n8k8 consumes:
        #   A: 16x8 fp16 = two 8x8 matrices -> ldmatrix.x2
        #   B:  8x8 fp16 = one  8x8 matrix  -> ldmatrix.x1
        # make_tiled_copy_A/B binds the ldmatrix copy layout to tiled_mma's
        # PTX register-fragment layout; retile() below gives the RMEM view
        # that ldmatrix writes into.
        s2R_copy_A = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(False, 2),
            mA.element_type,
        )
        s2R_copy_B = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(False, 1),
            mB.element_type,
        )
        tiled_s2r_A = cute.make_tiled_copy_A(s2R_copy_A, tiled_mma)
        tiled_s2r_B = cute.make_tiled_copy_B(s2R_copy_B, tiled_mma)
        thr_s2r_A = tiled_s2r_A.get_slice(tidx)
        thr_s2r_B = tiled_s2r_B.get_slice(tidx)

        tCsA = thr_mma.partition_A(sA)
        tCsB = thr_mma.partition_B(sB)
        tCrA = tiled_mma.make_fragment_A(tCsA)
        tCrB = tiled_mma.make_fragment_B(tCsB)

        tCsA_copy_view = thr_s2r_A.partition_S(sA)
        tCrA_copy_view = thr_s2r_A.retile(tCrA)
        tCsB_copy_view = thr_s2r_B.partition_S(sB)
        tCrB_copy_view = thr_s2r_B.retile(tCrB)

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

            tAgA = cute.local_partition(gA, g2s_thr_layout_A, tidx)
            tAsA = cute.local_partition(sA, g2s_thr_layout_A, tidx)
            tBgB = cute.local_partition(gB, g2s_thr_layout_B, tidx)
            tBsB = cute.local_partition(sB, g2s_thr_layout_B, tidx)

            # static print
            if debug:
                print("[DSL INFO] GMEM->SMEM per-thread partition types:")
                print(f"[DSL INFO]   tAgA = {tAgA.type}")
                print(f"[DSL INFO]   tAsA = {tAsA.type}")
                print(f"[DSL INFO]   tBgB = {tBgB.type}")
                print(f"[DSL INFO]   tBsB = {tBsB.type}")

            cute.copy(g2S_copy_A, tAgA, tAsA)
            cute.copy(g2S_copy_B, tBgB, tBsB)
            cute.arch.sync_threads()

            cute.copy(tiled_s2r_A, tCsA_copy_view, tCrA_copy_view)
            cute.copy(tiled_s2r_B, tCsB_copy_view, tCrB_copy_view)

            if debug:
                if bidx == 0 and bidy == 0 and k_tile == 0 and tidx == 0:
                    cute.printf("===gmem->smem copied data in shared memory===")
                    cute.print_tensor(tAsA)
                    cute.print_tensor(tBsB)
                    cute.printf("===ldmatrix smem views and rmem fragments===")
                    cute.print_tensor(tCrA)
                    cute.print_tensor(tCrB)

            cute.gemm(tiled_mma, tCrC, tCrA, tCrB, tCrC)
            cute.arch.sync_threads()

            if debug:
                if bidx == 0 and bidy == 0 and k_tile == 0 and tidx == 0:
                    cute.printf("lane {} C fragment after one mma.sync:", tidx)
                    cute.print_tensor(tCrC)

        cute.copy(r2G_copy_C, tCrC, tCgC)


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

    gemm = LDSMTensorOpGemm()
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
