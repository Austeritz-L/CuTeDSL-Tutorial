#include <iostream>
#include <cstdio>

// Use Thrust to handle host/device allocations
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

// Cutlass includes
#include <cutlass/half.h>                       // F16 data type
#include <cutlass/util/print_error.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/cluster_launch.hpp>

// CuTe includes
#include <cute/tensor.hpp>                      // CuTe tensor implementation
#include <cute/arch/cluster_sm90.hpp>           // CuTe functions for querying the details of cluster launched
#include <cute/numeric/integral_constant.hpp>   // Compile time in constants such as _1, _256 etc.
#include <cute/algorithm/cooperative_copy.hpp>  // Auto vectorized copy operation
#include <cute/arch/tmem_allocator_sm100.hpp>   // TMEM allocator for SM100

using namespace cute;

// The shared memory buffers for A and B matrices.
template <class TypeA,
          class ASmemLayout>    
struct SharedStorage
{
  alignas(128) cute::ArrayEngine<TypeA, cute::cosize_v<ASmemLayout>> A;
  CUTE_DEVICE constexpr auto tensor_sA() { return make_tensor(make_smem_ptr(A.begin()), ASmemLayout{}); }
};

// The device kernel
template <class SharedStorage,
          class ATensor,
          class CTA_Tiler,
          class Copy_Tiler,
          class Tiled_S2r_Copy>
__global__ static
void
test_copy_device(ATensor mA,
            CTA_Tiler cta_tiler,
            Copy_Tiler copy_tiler,
            Tiled_S2r_Copy tiled_s2r_copy)
{
    auto cta_coord = make_coord(blockIdx.x, blockIdx.y);
    Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, _1>{});
    // if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
    //   print("gA:"); print(gA); print("\n"); // gA:gmem_ptr[32b](0x7e969f000000) o (_128,_128):(256,_1)
    // }
    Tensor gA_tiles = tiled_divide(gA, copy_tiler);
    // if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
    //   print("gA_tiles:"); print(gA_tiles); print("\n"); // gA_tiles:gmem_ptr[32b](0x726c8b000000) o ((_128,_32),_1,_4):((256,_1),_0,_32)
    // }

    // smem
    extern __shared__ char shared_memory[];
    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(shared_memory);

    Tensor smem_buffA = shared_storage.tensor_sA(); 

    //copy
    auto thr_s2r_copy = tiled_s2r_copy.get_thread_slice(threadIdx.x);
    Tensor tSsA = thr_s2r_copy.partition_S(smem_buffA);
    Tensor tSrA = make_fragment_like(tSsA);
    using RegType = typename decltype(tSrA)::value_type;

    for (int k_block = 0; k_block < size<2>(gA_tiles); ++k_block) {
      Tensor gA_tile = gA_tiles(make_coord(_, _), 0, k_block);

      // Gmem -> Smem
      cooperative_copy<128>(threadIdx.x, gA_tile, smem_buffA);
      __syncthreads();

      // Smem -> Reg
      copy(tiled_s2r_copy, tSsA, tSrA);
      #pragma unroll
      for (int i = 0; i < size(tSrA); ++i) {
        tSrA(i) = tSrA(i) + RegType(1);
      }
      // Reg -> Gmem
      Tensor tDgA = thr_s2r_copy.partition_D(gA_tile);
      copy(tiled_s2r_copy, tSrA, tDgA);
      __syncthreads();
    }
  
}

template <class TypeA, class LayoutA>
void test_copy(TypeA* device_ptr_A, LayoutA layout_A)
{
    Tensor mA = make_tensor(make_gmem_ptr(device_ptr_A), layout_A); 
    auto R = shape<0>(layout_A);
    auto C = shape<1>(layout_A);
    // smem
    auto copy_tiler = make_shape(Int<128>{}, Int<32>{});
    // auto sA_layout = make_layout(
    //     copy_tiler,
    //     make_stride(Int<32>{}, Int<1>{})
    // );
    auto sA_layout = composition(
      Swizzle<3, 4, 3>{},
      make_layout(
          copy_tiler,
          make_stride(Int<32>{}, Int<1>{})
      )
    );
    using SMEMStorage = SharedStorage<TypeA, decltype(sA_layout)>;
    // copy
    using S2RAtom = cute::Copy_Atom<cute::AutoVectorizingCopy, TypeA>;
    auto tiled_s2r_copy = cute::make_tiled_copy(
      S2RAtom{},
      Layout<Shape<_128, _1>>{},
      Layout<Shape<_1, _32>>{}
    );
    
    // launch params
    auto cta_tiler = make_shape(Int<128>{}, Int<128>{});
    auto cta_layout = tiled_divide(mA, cta_tiler);
    // print("cta_layout:\t"); print(cta_layout); print("\n"); //cta_layout:     gmem_ptr[32b](0x767d73000000) o ((_128,_128),2,2):((256,_1),32768,_128)
    auto cluster_shape = make_shape(Int<1>{}, Int<1>{}, Int<1>{});
    dim3 dimBlock(128);
    dim3 dimCluster(size<0>(cluster_shape), size<1>(cluster_shape), size<2>(cluster_shape));
    dim3 dimGrid(size<1>(cta_layout), size<2>(cta_layout));
    int  smemBytes = sizeof(SMEMStorage);
    auto* kernel_ptr = &test_copy_device<SMEMStorage,
                                  decltype(mA), decltype(cta_tiler), decltype(copy_tiler), decltype(tiled_s2r_copy)>;
    // Set kernel attributes (set SMEM)
    CUTE_CHECK_ERROR(cudaFuncSetAttribute(kernel_ptr,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        smemBytes));
    cutlass::ClusterLaunchParams params = {dimGrid, dimBlock, dimCluster, smemBytes};
    cutlass::Status status = cutlass::launch_kernel_on_cluster(params, (void const*) kernel_ptr,
                                                            mA,
                                                            cta_tiler,
                                                            copy_tiler,
                                                            tiled_s2r_copy);
    CUTE_CHECK_LAST();
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "Error: Failed at kernel Launch" << std::endl;
    }
}


int main(int argc, char** argv)
{
  cudaDeviceProp props;
  int current_device_id;
  cudaGetDevice(&current_device_id);
  cudaGetDeviceProperties(&props, current_device_id);
  cudaError_t error = cudaGetDeviceProperties(&props, 0);
  if (error != cudaSuccess) {
    std::cerr << "cudaGetDeviceProperties() returned an error: " << cudaGetErrorString(error) << std::endl;
    return -1;
  }

  int R = 256;
  int C = 256;

  // RxC K-major Row-Major
  Layout layout = make_layout(make_shape (R, C),
                                make_stride(C, Int<1>{}));  
  using TypeA = float;
  thrust::host_vector<TypeA>   host(R * C);
  Tensor host_tensor = make_tensor(host.data(), layout);
  print("host_tensor:\t"); print(host_tensor); print("\n"); 

  for (int i=0; i<R; i++) {
    for (int j=0; j<C; j++) {
        host_tensor(i,j) = TypeA(i*C + j);
    }
  }

  // Copy tensor from host memory to device memory
  thrust::device_vector<TypeA> device = host;

  test_copy(device.data().get(), layout);

  // Copy tensor from device memory to host memory
  host = device;
  host_tensor = make_tensor(host.data(), layout);

  thrust::host_vector<TypeA> host_reference(R*C);
  Tensor host_reference_tensor = make_tensor(host_reference.data(), layout);
  for (int i=0; i<R; i++) {
    for (int j=0; j<C; j++) {
        host_reference_tensor(i,j) = TypeA(i*C + j + 1);
    }
  }

  ////////////////////////////////////////////////////////////
  //
  // Compare results
  //
  ////////////////////////////////////////////////////////////
  TypeA rel_err = 0.f;
  for (int i=0; i<R; i++) {
    for (int j=0; j<C; j++) {
        rel_err += std::abs(host_reference_tensor(i,j) - host_tensor(i,j)) ;
    }
  }
  
  bool success = rel_err <= 0.0;
  std::cout << "Execution is " << ((success) ? "successful." : "failed. rel_err:") << rel_err << std::endl;

  // Warmup
  for (int i = 0; i < 5; ++i) {
    test_copy(device.data().get(), layout);
  }
  cudaDeviceSynchronize();
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  int iters = 100;
  for (int i = 0; i < iters; ++i) {
    test_copy(device.data().get(), layout);
  }

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float elapsed_ms = 0.0f;
  cudaEventElapsedTime(&elapsed_ms, start, stop);

  std::cout << "Average time: "
            << elapsed_ms / iters
            << " ms" << std::endl;

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return 0;
}
