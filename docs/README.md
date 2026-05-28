# CuTe DSL Source Notes

This directory records source-level notes for understanding CuTe DSL from the bottom up.

The goal is to connect four layers:

```text
Layout algebra
  -> Tensor views
  -> Copy/MMA atoms
  -> Tiled copy / tiled MMA
  -> Generated GPU instructions
```

## Reading Order

1. [Layout and Tensor](01_layout_tensor.md)
2. [Copy Atom and Tiled Copy](02_copy_atom_tiledcopy.md)
3. [MMA Atom and Tiled MMA](03_mma_atom_tiledmma.md)
4. [ldmatrix to MMA Fragment](04_ldmatrix_to_mma_fragment.md)
5. [Naive TensorOp GEMM Walkthrough](05_navie_tensorop_walkthrough.md)

## Local Source Map

The notes mainly reference the local CUTLASS checkout:

```text
/root/lizhiyuan/cutlass/python/CuTeDSL/cutlass/cute
```

Important entry points:

- `core.py`: layout construction, tiling, partitioning
- `typing.py`: `Layout`, `Tensor`, and type wrappers
- `atom.py`: `MmaAtom`, `TiledMma`, `CopyAtom`, `TiledCopy`
- `algorithm.py`: `cute.copy` and `cute.gemm`
- `nvgpu/warp/mma.py`: Ampere warp-level MMA ops
- `nvgpu/warp/copy.py`: `ldmatrix` copy ops

