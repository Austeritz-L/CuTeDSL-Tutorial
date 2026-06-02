import argparse
import time
from typing import Tuple

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


class SplitKTensorOpGemm:
    """Scaffold for split-K TensorOp GEMM on Ampere.

    Split-K parallelizes the K dimension across multiple CTAs:

        partial[split, M, N] = A[:, K_split] @ B[:, K_split]^T
        C[M, N] = sum_split partial[split, M, N]

    This file intentionally does not implement the kernels. It only prepares the
    same TensorOp/tiled-copy objects used by `ldsm_tensorop.py`, launches a
    partial GEMM kernel over `(M tile, N tile, split_k)`, and then launches a
    reduction kernel. Fill `compute_partial_kernel()` and `reduce_kernel()`.

    To keep the partial workspace simple, it is represented as a 2-D matrix:

        partial_c.shape = (split_k_slices * M, N)

    Split `s` writes row `m` of C to row `s * M + m` in `partial_c`.
    """

    def __init__(
        self,
        cta_tiler: Tuple[int, int, int] = (128, 128, 16),
        atom_layout_mnk: Tuple[int, int, int] = (2, 2, 1),
        split_k_slices: int = 2,
        reduce_tiler: Tuple[int, int, int] = (16, 16, 1),
    ):
        self.cta_tiler = cta_tiler
        self.bM, self.bN, self.bK = cta_tiler
        self.atom_layout_mnk = atom_layout_mnk
        self.split_k_slices = split_k_slices
        self.reduce_tiler = reduce_tiler
        self.reduce_m, self.reduce_n, _ = reduce_tiler
        self.mma_inst_shape = (16, 8, 16)

        atom_lay_M, atom_lay_N, atom_lay_K = atom_layout_mnk
        mmaM, mmaN, mmaK = self.mma_inst_shape

        assert self.cta_tiler == (128, 128, 16), "this scaffold is fixed to 128x128x16"
        assert split_k_slices > 0
        assert atom_lay_K == 1, "this scaffold keeps atom_layout K fixed to 1"
        assert self.bM % (atom_lay_M * mmaM) == 0
        assert self.bN % (atom_lay_N * mmaN) == 0
        assert self.bK % mmaK == 0
        assert self.reduce_m * self.reduce_n <= 1024

        self.num_threads = atom_lay_M * atom_lay_N * atom_lay_K * 32
        self.reduce_threads = self.reduce_m * self.reduce_n

    @cute.jit
    def __call__(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mC: cute.Tensor,
        mPartialC: cute.Tensor,
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

        grid_m = cute.ceil_div(mC.shape[0], self.bM)
        grid_n = cute.ceil_div(mC.shape[1], self.bN)
        total_k_tiles = mA.shape[1] // self.bK
        split_k_tiles = total_k_tiles // self.split_k_slices

        print("[DSL INFO] SplitK TensorOp GEMM scaffold:")
        print(f"[DSL INFO]   A               = {mA.type}")
        print(f"[DSL INFO]   B               = {mB.type}")
        print(f"[DSL INFO]   C               = {mC.type}")
        print(f"[DSL INFO]   partial C       = {mPartialC.type}")
        print(f"[DSL INFO]   cta_tiler       = {self.cta_tiler}")
        print(f"[DSL INFO]   mma_inst_shape  = {self.mma_inst_shape}")
        print(f"[DSL INFO]   atom_layout_mnk = {self.atom_layout_mnk}")
        print(f"[DSL INFO]   permutation_mnk = {permutation_mnk}")
        print(f"[DSL INFO]   split_k_slices  = {self.split_k_slices}")
        print(f"[DSL INFO]   total_k_tiles   = {total_k_tiles}")
        print(f"[DSL INFO]   split_k_tiles   = {split_k_tiles}")
        print(f"[DSL INFO]   grid_m          = {grid_m}")
        print(f"[DSL INFO]   grid_n          = {grid_n}")
        print(f"[DSL INFO]   sA_layout       = {sA_layout}")
        print(f"[DSL INFO]   sB_layout       = {sB_layout}")
        print(f"[DSL INFO]   tiled_mma       = {tiled_mma}")
        print(f"[DSL INFO]   tiled_g2s_A     = {tiled_g2s_A}")
        print(f"[DSL INFO]   tiled_g2s_B     = {tiled_g2s_B}")

        self.compute_partial_kernel(
            mA,
            mB,
            mPartialC,
            sA_layout,
            sB_layout,
            tiled_g2s_A,
            tiled_g2s_B,
            tiled_mma,
            grid_m,
            split_k_tiles,
        ).launch(
            grid=(grid_m, grid_n, self.split_k_slices),
            block=(self.num_threads, 1, 1),
            stream=stream,
        )

        self.reduce_kernel(mPartialC, mC, grid_m).launch(
            grid=(
                cute.ceil_div(mC.shape[0], self.reduce_m),
                cute.ceil_div(mC.shape[1], self.reduce_n),
                1,
            ),
            block=(self.reduce_threads, 1, 1),
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
    def compute_partial_kernel(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mPartialC: cute.Tensor,
        sA_layout: cute.Layout,
        sB_layout: cute.Layout,
        tiled_g2s_A: cute.TiledCopy,
        tiled_g2s_B: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
        grid_m: cutlass.Int32,
        split_k_tiles: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, split_idx = cute.arch.block_idx()

        print("[DSL INFO] SplitK partial kernel TODO:")
        print(f"[DSL INFO]   tidx          = {tidx}")
        print(f"[DSL INFO]   bidx          = {bidx}")
        print(f"[DSL INFO]   bidy          = {bidy}")
        print(f"[DSL INFO]   split_idx     = {split_idx}")
        print(f"[DSL INFO]   grid_m        = {grid_m}")
        print(f"[DSL INFO]   split_k_tiles = {split_k_tiles}")
        print(f"[DSL INFO]   sA_layout     = {sA_layout}")
        print(f"[DSL INFO]   sB_layout     = {sB_layout}")
        print(f"[DSL INFO]   tiled_g2s_A   = {tiled_g2s_A}")
        print(f"[DSL INFO]   tiled_g2s_B   = {tiled_g2s_B}")
        print(f"[DSL INFO]   tiled_mma     = {tiled_mma}")

        # TODO: implement split-K partial GEMM.
        #
        # Suggested structure:
        # 1. partial_m_tile = split_idx * grid_m + bidx
        # 2. gPartialC = local_tile(mPartialC, ..., coord=(partial_m_tile, bidy, None))
        # 3. Use tiled_mma.get_slice(tidx) and partition_C(gPartialC)
        # 4. Allocate sA/sB with sA_layout/sB_layout
        # 5. Build ldmatrix S2R copy views
        # 6. k_tile_begin = split_idx * split_k_tiles
        #    k_tile_end   = k_tile_begin + split_k_tiles
        # 7. Loop k_tile in this range:
        #       GMEM A/B -> SMEM A/B -> ldmatrix -> cute.gemm
        # 8. Store tCrC into the partial C tile.

    @cute.kernel
    def reduce_kernel(
        self,
        mPartialC: cute.Tensor,
        mC: cute.Tensor,
        grid_m: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, _ = cute.arch.block_idx()

        print("[DSL INFO] SplitK reduction kernel TODO:")
        print(f"[DSL INFO]   tidx            = {tidx}")
        print(f"[DSL INFO]   bidx            = {bidx}")
        print(f"[DSL INFO]   bidy            = {bidy}")
        print(f"[DSL INFO]   grid_m          = {grid_m}")
        print(f"[DSL INFO]   reduce_tiler    = {self.reduce_tiler}")
        print(f"[DSL INFO]   split_k_slices  = {self.split_k_slices}")
        print(f"[DSL INFO]   partial C       = {mPartialC.type}")
        print(f"[DSL INFO]   C               = {mC.type}")

        # TODO: implement split-K reduction.
        #
        # Suggested structure:
        # 1. Map one thread to one output element in a small C tile.
        # 2. For split_idx in range_constexpr(self.split_k_slices):
        #       partial_m_tile = split_idx * grid_m + bidx
        #       load partial C element
        #       accumulate into fp32 register
        # 3. Store the accumulated value to final C.


def run(
    mnk: Tuple[int, int, int] = (128, 128, 128),
    split_k_slices: int = 2,
    run_kernel: bool = False,
    skip_ref_check: bool = False,
):
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required to run this example")

    M, N, K = mnk
    if M % 128 != 0 or N % 128 != 0 or K % 16 != 0:
        raise ValueError(
            "SplitK TensorOp example requires M%128 == 0, N%128 == 0, K%16 == 0"
        )
    if split_k_slices <= 0:
        raise ValueError("split_k_slices must be positive")
    if (K // 16) % split_k_slices != 0:
        raise ValueError("(K / 16) must be divisible by split_k_slices")

    torch.manual_seed(0)
    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((N, K), device="cuda", dtype=torch.float16)
    c = torch.empty((M, N), device="cuda", dtype=torch.float32)
    partial_c = torch.empty((split_k_slices * M, N), device="cuda", dtype=torch.float32)

    a_tensor = from_dlpack(a, assumed_align=16)
    b_tensor = from_dlpack(b, assumed_align=16)
    c_tensor = from_dlpack(c, assumed_align=16)
    partial_c_tensor = from_dlpack(partial_c, assumed_align=16)

    torch_stream = torch.cuda.current_stream()
    current_stream = cuda.CUstream(torch_stream.cuda_stream)

    gemm = SplitKTensorOpGemm(split_k_slices=split_k_slices)
    print("Compiling split-K TensorOp GEMM scaffold...")
    print(f"  mnk: {mnk}")
    print(f"  split_k_slices: {split_k_slices}")
    print(f"  run_kernel: {run_kernel}")
    start = time.time()
    compiled_gemm = cute.compile[cute.GenerateLineInfo](
        gemm,
        a_tensor,
        b_tensor,
        c_tensor,
        partial_c_tensor,
        stream=current_stream,
    )
    print(f"Compilation time: {time.time() - start:.4f} seconds")

    ref = None
    if not skip_ref_check:
        ref = torch.einsum("mk,nk->mn", a.float(), b.float())
        print("Reference output computed with torch.einsum.")

    if not run_kernel:
        print("Kernel execution skipped. Fill kernels, then run with --run_kernel.")
        return

    compiled_gemm(a_tensor, b_tensor, c_tensor, partial_c_tensor)
    torch.cuda.synchronize()

    if not skip_ref_check:
        torch.testing.assert_close(c, ref, atol=2e-2, rtol=2e-2)
        print("PASS")
    else:
        print("Kernel executed; reference check skipped.")


def parse_triplet(value: str) -> Tuple[int, int, int]:
    items = tuple(int(x.strip()) for x in value.split(","))
    if len(items) != 3:
        raise argparse.ArgumentTypeError("expected three comma-separated integers")
    return items


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mnk", type=parse_triplet, default=(1024, 1024, 128))
    parser.add_argument("--split_k_slices", type=int, default=2)
    parser.add_argument("--run_kernel", action="store_true")
    parser.add_argument("--skip_ref_check", action="store_true")
    args = parser.parse_args()
    run(args.mnk, args.split_k_slices, args.run_kernel, args.skip_ref_check)
