import argparse
import time
from typing import Tuple

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.utils
from cutlass.cute.runtime import from_dlpack


class LDSMTensorOpGemm:
    """Ampere Tensor Core GEMM using a 128x128x16 CTA tile.

    This keeps the data movement easy to inspect:
    - CTA tile is (M,N,K) = (128,128,16)
    - G2S uses tiled CopyUniversalOp with 128-bit vector copies
    - S2R uses ldmatrix via make_tiled_copy_A/B
    - compute uses tiled MMA over m16n8k16 Tensor Core atoms

    Logical tensors:
    - A: (M, K), row-major, dtype fp16
    - B: (N, K), row-major in this tensor view, dtype fp16
    - C: (M, N), row-major, dtype fp32
    """

    def __init__(
        self,
        cta_tiler: Tuple[int, int, int] = (128, 128, 16),
        atom_layout_mnk: Tuple[int, int, int] = (2, 2, 1),
    ):
        self.cta_tiler = cta_tiler
        self.bM, self.bN, self.bK = cta_tiler
        self.atom_layout_mnk = atom_layout_mnk
        self.mma_inst_shape = (16, 8, 16)
        atom_lay_M, atom_lay_N, atom_lay_K = atom_layout_mnk
        mmaM, mmaN, mmaK = self.mma_inst_shape

        assert self.cta_tiler == (128, 128, 16), "this example is fixed to 128x128x16"
        assert atom_lay_K == 1, "this simple example keeps atom_layout K fixed to 1"
        assert self.bM % (atom_lay_M * mmaM) == 0
        assert self.bN % (atom_lay_N * mmaN) == 0
        assert self.bK % mmaK == 0

        self.num_threads = atom_lay_M * atom_lay_N * atom_lay_K * 32

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
            self.mma_inst_shape,
        )

        sA_layout = cute.make_layout((self.bM, self.bK), stride=(self.bK, 1))
        sB_layout = cute.make_layout((self.bN, self.bK), stride=(self.bK, 1))

        permutation_mnk = (
            self.atom_layout_mnk[0] * self.mma_inst_shape[0],
            self.atom_layout_mnk[1] * self.mma_inst_shape[1] * 2,
            self.atom_layout_mnk[2] * self.mma_inst_shape[2],
        )
        tiled_mma = cute.make_tiled_mma(
            mma_op,
            cute.make_layout(self.atom_layout_mnk),
            permutation_mnk=permutation_mnk,
        )
        print("[DSL INFO] CTA and MMA setup:")
        print(f"[DSL INFO]   cta_tiler       = {self.cta_tiler}")
        print(f"[DSL INFO]   mma_inst_shape  = {self.mma_inst_shape}")
        print(f"[DSL INFO]   atom_layout_mnk = {self.atom_layout_mnk}")
        print(f"[DSL INFO]   permutation_mnk = {permutation_mnk}")
        print(f"[DSL INFO]   tiled_mma       = {tiled_mma}")
        print(f"[DSL INFO]   sA_layout       = {sA_layout}")
        print(f"[DSL INFO]   sB_layout       = {sB_layout}")

        g2s_copy_bits = 128
        g2S_copy_A = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            mA.element_type,
            num_bits_per_copy=g2s_copy_bits,
        )
        g2S_copy_B = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            mB.element_type,
            num_bits_per_copy=g2s_copy_bits,
        )
        tiled_g2s_A = self._make_gmem_tiled_copy_AB(
            g2S_copy_A, mA.element_type, g2s_copy_bits
        )
        tiled_g2s_B = self._make_gmem_tiled_copy_AB(
            g2S_copy_B, mB.element_type, g2s_copy_bits
        )
        print("[DSL INFO] G2S tiled copies:")
        print(f"[DSL INFO]   tiled_g2s_A = {tiled_g2s_A}")
        print(f"[DSL INFO]   tiled_g2s_B = {tiled_g2s_B}")

        grid = (
            cute.ceil_div(mC.shape[0], self.bM),
            cute.ceil_div(mC.shape[1], self.bN),
            1,
        )

        self.kernel(
            mA,
            mB,
            mC,
            sA_layout,
            sB_layout,
            tiled_g2s_A,
            tiled_g2s_B,
            tiled_mma,
        ).launch(
            grid=grid,
            block=(self.num_threads, 1, 1),
            stream=stream,
        )

    def _make_gmem_tiled_copy_AB(self, copy_atom, dtype, copy_bits: cutlass.Constexpr):
        copy_elems = copy_bits // dtype.width
        shape_dim_1 = cute.size(self.bK) // copy_elems
        thread_layout = cute.make_layout(
            (self.num_threads // shape_dim_1, shape_dim_1),
            stride=(shape_dim_1, 1),
        )
        value_layout = cute.make_layout((1, copy_elems))
        return cute.make_tiled_copy_tv(copy_atom, thread_layout, value_layout)

    @cute.kernel
    def kernel(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mC: cute.Tensor,
        sA_layout: cute.Layout,
        sB_layout: cute.Layout,
        tiled_g2s_A: cute.TiledCopy,
        tiled_g2s_B: cute.TiledCopy,
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

        # partition_C returns the per-lane accumulator layout required by the
        # tiled m16n8k16 MMA atoms.
        tCgC = thr_mma.partition_C(gC)
        tCrC = tiled_mma.make_fragment_C(tCgC)
        tCrC.fill(0.0)

        print("[DSL INFO] MMA C partition:")
        print(f"[DSL INFO]   thr_mma = {thr_mma}")
        print(f"[DSL INFO]   gC      = {gC.type}")
        print(f"[DSL INFO]   tCgC    = {tCgC.type}")
        print(f"[DSL INFO]   tCrC    = {tCrC.type}")

        smem = cutlass.utils.SmemAllocator()
        sA = smem.allocate_tensor(mA.element_type, sA_layout, 16)
        sB = smem.allocate_tensor(mB.element_type, sB_layout, 16)

        r2G_copy_C = cute.make_copy_atom(
            cute.nvgpu.CopyR2GOp(),
            mC.element_type,
            num_bits_per_copy=mC.element_type.width,
        )

        thr_g2s_A = tiled_g2s_A.get_slice(tidx)
        thr_g2s_B = tiled_g2s_B.get_slice(tidx)

        # ldmatrix atoms for loading A/B from shared memory to register fragments.
        # m16n8k16 uses ldmatrix.x4 fragments for both A and B in this tiled MMA.
        # make_tiled_copy_A/B binds the ldmatrix copy layout to tiled_mma's
        # PTX register-fragment layout; retile() below gives the RMEM view
        # that ldmatrix writes into.
        s2R_copy_A = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(False, 4),
            mA.element_type,
        )
        s2R_copy_B = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(False, 4),
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
        print("[DSL INFO] SMEM and S2R ldmatrix views:")
        print(f"[DSL INFO]   sA              = {sA.type}")
        print(f"[DSL INFO]   sB              = {sB.type}")
        print(f"[DSL INFO]   tiled_s2r_A     = {tiled_s2r_A}")
        print(f"[DSL INFO]   tiled_s2r_B     = {tiled_s2r_B}")
        print(f"[DSL INFO]   tCsA            = {tCsA.type}")
        print(f"[DSL INFO]   tCsB            = {tCsB.type}")
        print(f"[DSL INFO]   tCrA            = {tCrA.type}")
        print(f"[DSL INFO]   tCrB            = {tCrB.type}")
        print(f"[DSL INFO]   tCsA_copy_view = {tCsA_copy_view.type}")
        print(f"[DSL INFO]   tCrA_copy_view = {tCrA_copy_view.type}")
        print(f"[DSL INFO]   tCsB_copy_view = {tCsB_copy_view.type}")
        print(f"[DSL INFO]   tCrB_copy_view = {tCrB_copy_view.type}")

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

            tAgA = thr_g2s_A.partition_S(gA)
            tAsA = thr_g2s_A.partition_D(sA)
            tBgB = thr_g2s_B.partition_S(gB)
            tBsB = thr_g2s_B.partition_D(sB)

            print("[DSL INFO] GMEM->SMEM tiled-copy partition types:")
            print(f"[DSL INFO]   gA   = {gA.type}")
            print(f"[DSL INFO]   gB   = {gB.type}")
            print(f"[DSL INFO]   tAgA = {tAgA.type}")
            print(f"[DSL INFO]   tAsA = {tAsA.type}")
            print(f"[DSL INFO]   tBgB = {tBgB.type}")
            print(f"[DSL INFO]   tBsB = {tBsB.type}")

            cute.copy(tiled_g2s_A, tAgA, tAsA)
            cute.copy(tiled_g2s_B, tBgB, tBsB)
            cute.arch.sync_threads()

            cute.copy(tiled_s2r_A, tCsA_copy_view, tCrA_copy_view)
            cute.copy(tiled_s2r_B, tCsB_copy_view, tCrB_copy_view)

            cute.gemm(tiled_mma, tCrC, tCrA, tCrB, tCrC)
            cute.arch.sync_threads()

        cute.copy(r2G_copy_C, tCrC, tCgC)


def run(mnk: Tuple[int, int, int] = (128, 128, 128)):
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required to run this example")

    M, N, K = mnk
    if M % 128 != 0 or N % 128 != 0 or K % 16 != 0:
        raise ValueError(
            "128x128x16 tiled TensorOp example requires "
            "M%128 == 0, N%128 == 0, and K%16 == 0"
        )

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
    print("Compiling 128x128x16 tiled ldmatrix TensorOp GEMM...")
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
    parser.add_argument("--mnk", type=parse_triplet, default=(1024, 1024, 128))
    args = parser.parse_args()
    run(args.mnk)
