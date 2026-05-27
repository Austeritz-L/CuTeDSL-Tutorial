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


template <class TypeA,
          class ASmemLayout>    
struct SharedStorage
{
  alignas(128) cute::ArrayEngine<TypeA, cute::cosize_v<ASmemLayout>> A;
  CUTE_DEVICE constexpr auto tensor_sA() { return make_tensor(make_smem_ptr(A.begin()), ASmemLayout{}); }
};

// The device kernel
template <class ATensor,
          class SharedStorage,
          class CTA_Tiler,
          class TypeA,
          class Thread_LayoutA,
          class Thread_LayoutB>
__global__ static
void
test_transpose_device(ATensor mA,
            CTA_Tiler cta_tiler,
            TypeA* mB_ptr,
            Thread_LayoutA thread_layoutA,
            Thread_LayoutB thread_layoutB)
{
   bool need_print = (threadIdx.x == 0) && (blockIdx.x == 0) && (blockIdx.y == 0);
   auto R = size<0>(mA);
   auto C = size<1>(mA);
   auto cta_R = size<0>(cta_tiler);
   auto cta_C = size<1>(cta_tiler);
   auto cta_coord_A = make_coord(blockIdx.x, blockIdx.y);
   auto cta_coord_B = make_coord(blockIdx.y, blockIdx.x);
   auto cta_tiler_T = make_shape(cta_C, cta_R);
   auto mB_layout = make_layout(make_shape(C, R),
                                make_stride(R, Int<1>{})
                                );
   Tensor mB = make_tensor(make_gmem_ptr(mB_ptr), mB_layout);

   Tensor gA = local_tile(mA, cta_tiler, cta_coord_A, Step<_1, _1>{});
   Tensor gB = local_tile(mB, cta_tiler_T, cta_coord_B, Step<_1, _1>{});

   extern __shared__ char shared_memory[];
   SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(shared_memory);
   Tensor smem_buff = shared_storage.tensor_sA(); 
   auto smem_buff_T_layout = make_layout(make_shape(size<1>(smem_buff), size<0>(smem_buff)),
                                          make_stride(stride<1>(smem_buff), stride<0>(smem_buff))
                                        );
   Tensor smem_buff_T = make_tensor(smem_buff.data(), smem_buff_T_layout);

   Tensor thr_tile_A = local_partition(gA, thread_layoutA, threadIdx.x);
   Tensor thr_tile_B = local_partition(gB, thread_layoutB, threadIdx.x);
   Tensor thr_smem_buff = local_partition(smem_buff, thread_layoutA, threadIdx.x);
   Tensor thr_smem_buff_T = local_partition(smem_buff_T, thread_layoutB, threadIdx.x);
   // Tensor tSrA = make_fragment_like(thr_tile_A); // create reg 
  //  if (need_print) {
  //     print("gA:\t"); print(gA); print("\n"); 
  //     print("gB:\t"); print(gB); print("\n"); 
  //     print("thr_tile_A:\t"); print(thr_tile_A); print("\n"); 
  //     print("tthr_smem_buff:\t"); print(thr_smem_buff); print("\n");
  //     print("thr_smem_buff_T:\t"); print(thr_smem_buff_T); print("\n");
  //     //print("tSrA:\t"); print(tSrA); print("\n");
  //  }
   copy(thr_tile_A, thr_smem_buff);
   __syncthreads();
   copy(thr_smem_buff_T, thr_tile_B); 
}

template <class TypeA, class LayoutA>
void test_transpose(TypeA const* device_ptr_A, LayoutA layout_A, TypeA* device_ptr_B)
{
    Tensor mA = make_tensor(make_gmem_ptr(device_ptr_A), layout_A); 
    auto R = shape<0>(layout_A);
    auto C = shape<1>(layout_A);
    // launch params
    auto cta_tiler = make_shape(Int<128>{}, Int<64>{});
    auto cta_layout = tiled_divide(mA, cta_tiler);
    auto thr_layoutA =
      make_layout(make_shape(Int<4>{}, Int<32>{}), make_stride(Int<32>{}, Int<1>{}));
    auto thr_layoutB =
      make_layout(make_shape(Int<4>{}, Int<32>{}), make_stride(Int<32>{}, Int<1>{}));
    auto smem_Layout = make_layout(cta_tiler, make_stride(Int<65>{}, Int<1>{})); // padding
    using SMEMStorage = SharedStorage<TypeA, decltype(smem_Layout)>; 
    auto cluster_shape = make_shape(Int<1>{}, Int<1>{}, Int<1>{});
    dim3 dimBlock(128);
    dim3 dimCluster(size<0>(cluster_shape), size<1>(cluster_shape), size<2>(cluster_shape));
    dim3 dimGrid(size<1>(cta_layout), size<2>(cta_layout));
    int  smemBytes = sizeof(SMEMStorage);
    auto* kernel_ptr = &test_transpose_device<decltype(mA), SMEMStorage, decltype(cta_tiler), TypeA, decltype(thr_layoutA), decltype(thr_layoutB)>;
    // Set kernel attributes (set SMEM)
    CUTE_CHECK_ERROR(cudaFuncSetAttribute(kernel_ptr,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        smemBytes)
                                      );
    cutlass::ClusterLaunchParams params = {dimGrid, dimBlock, dimCluster, smemBytes};
    cutlass::Status status = cutlass::launch_kernel_on_cluster(params, (void const*) kernel_ptr,
                                                            mA,
                                                            cta_tiler,
                                                            device_ptr_B,
                                                            thr_layoutA,
                                                            thr_layoutB);
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

  int R = 4096;
  int C = 4096;

  // RxC K-major Row-Major
  Layout layout = make_layout(make_shape (R, C),
                                make_stride(C, Int<1>{}));  
  Layout layout_T = make_layout(make_shape (C, R),
                                make_stride(R, Int<1>{}));  
  using TypeA = float;
  thrust::host_vector<TypeA>   host_A(R * C);
  Tensor host_tensor_A = make_tensor(host_A.data(), layout);

  for (int i=0; i<R; i++) {
    for (int j=0; j<C; j++) {
        host_tensor_A(i,j) = TypeA(i*C + j);
    }
  }

  // Copy tensor from host memory to device memory
  thrust::device_vector<TypeA> device_A = host_A;
  thrust::device_vector<TypeA> device_B(R * C);

  test_transpose(device_A.data().get(), layout, device_B.data().get());

  // Copy tensor from device memory to host memory
  thrust::host_vector<TypeA> host_B = device_B;
  Tensor host_tensor_B = make_tensor(host_B.data(), layout_T);

  thrust::host_vector<TypeA> host_reference(R*C);
  Tensor host_reference_tensor = make_tensor(host_reference.data(), layout_T);
  for (int i=0; i<R; i++) {
    for (int j=0; j<C; j++) {
        host_reference_tensor(j,i) = TypeA(i*C + j);
    }
  }

  ////////////////////////////////////////////////////////////
  //
  // Compare results
  //
  ////////////////////////////////////////////////////////////
  TypeA rel_err = 0.f;
  for (int i=0; i<C; i++) {
    for (int j=0; j<R; j++) {
        rel_err += std::abs(host_reference_tensor(i,j) - host_tensor_B(i,j)) ;
    }
  }
  
  bool success = rel_err <= 0.0;
  std::cout << "Execution is " << ((success) ? "successful." : "failed. rel_err:") << rel_err << std::endl;

  // Warmup
  for (int i = 0; i < 5; ++i) {
    test_transpose(device_A.data().get(), layout, device_B.data().get());
  }
  cudaDeviceSynchronize();
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  int iters = 100;
  for (int i = 0; i < iters; ++i) {
    test_transpose(device_A.data().get(), layout, device_B.data().get());
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
