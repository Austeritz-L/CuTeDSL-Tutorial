#include <iostream>
#include <cstdio>

// Use Thrust to handle host/device allocations
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

// Cutlass includes
#include <cutlass/half.h> // F16 data type
#include <cutlass/util/print_error.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/cluster_launch.hpp>

// CuTe includes
#include <cute/tensor.hpp>                     // CuTe tensor implementation
#include <cute/arch/cluster_sm90.hpp>          // CuTe functions for querying the details of cluster launched
#include <cute/numeric/integral_constant.hpp>  // Compile time in constants such as _1, _256 etc.
#include <cute/algorithm/cooperative_copy.hpp> // Auto vectorized copy operation
#include <cute/arch/tmem_allocator_sm100.hpp>  // TMEM allocator for SM100

using namespace cute;

template <class TypeA,
          class SmemLayout, // full layout, e.g. (64,32,3)
          class StageLayout, // one-stage layout, e.g. (64,32)
          class StageLayout_T,
          int Stages>
struct SharedStorage
{
    alignas(16) cute::uint64_t tma_barrier[Stages];
    alignas(16) cute::uint64_t free_barrier[Stages];
    alignas(128) cute::ArrayEngine<TypeA, cosize_v<SmemLayout>> A;

    CUTE_DEVICE constexpr auto smem_ptr_stage(int s) {
        TypeA* ptr = A.begin() + s * cute::cosize_v<StageLayout>;
        return make_smem_ptr(ptr);
    }

    template <class SmemPtr>
    CUTE_DEVICE constexpr auto tensor_sA_stage_from_ptr(SmemPtr ptr) {
        return make_tensor(ptr, StageLayout{});
    }

    template <class SmemPtr>
    CUTE_DEVICE constexpr auto tensor_sA_stage_T_from_ptr(SmemPtr ptr) {
        return make_tensor(ptr, StageLayout_T{});
    }
};

// The device kernel
template <class ATensor,
          class ATensor_TMA,
          class TMA_Atom_A,
          class SharedStorage,
          class CTA_Tiler,
          class Stage_Tiler,
          class TypeA,
          class Thread_LayoutB,
          int Stages>
__global__ static void
test_transpose_device(ATensor mA,
                      ATensor_TMA mA_tma,
                      CUTE_GRID_CONSTANT TMA_Atom_A const tma_atom_A,
                      CTA_Tiler cta_tiler,
                      Stage_Tiler stage_tiler,
                      TypeA *mB_ptr,
                      Thread_LayoutB thread_layoutB)
{
    // bool need_print = (threadIdx.x == 0) && (blockIdx.x == 0) && (blockIdx.y == 0);
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int warp_num = blockDim.x / 32;
    int lane_id = tid & 31;
    auto R = size<0>(mA);
    auto C = size<1>(mA);
    auto stage_R = size<0>(stage_tiler);
    auto stage_C = size<1>(stage_tiler);
    auto cta_coord_A = make_coord(blockIdx.x, _);
    auto cta_coord_B = make_coord(_, blockIdx.x);
    auto stage_tiler_T = make_shape(stage_C, stage_R);
    auto mB_layout = make_layout(make_shape(C, R),
                                 make_stride(R, Int<1>{}));
    Tensor mB = make_tensor(make_gmem_ptr(mB_ptr), mB_layout);

    Tensor gA = local_tile(mA_tma, stage_tiler, cta_coord_A, Step<_1, _1>{}); // ArithTuple(_0,0) o (_128,_64,64):(_1@1,_1@0,_64@0)
    Tensor gB = local_tile(mB, stage_tiler_T, cta_coord_B, Step<_1, _1>{}); // gmem_ptr[32b](0x7f3a52000000) o (_64,_128,64)
    // smem initialize
    extern __shared__ char shared_memory[];
    SharedStorage &shared_storage = *reinterpret_cast<SharedStorage *>(shared_memory);
    // uint32_t elect_one_thr = elect_one_sync();
    // if (elect_one_thr) {
    //     for (int i = warp_id; i<Stages; i+=warp_num) {
    //         initialize_barrier(shared_storage.tma_barrier[i], /* num_threads */ 1);
    //         initialize_barrier(shared_storage.free_barrier[i], /* num_threads */ 1);
    //         arrive_barrier(shared_storage.free_barrier[i]); // all stages should be free at first
    //     }
    // }
    if (tid == 0) {
        #pragma unroll
        for (int s = 0; s < Stages; ++s) {
            initialize_barrier(shared_storage.tma_barrier[s], 1);
            initialize_barrier(shared_storage.free_barrier[s], 1);
            arrive_barrier(shared_storage.free_barrier[s]); // initial free
        }
    }
    __syncthreads();
    // cache smem pointer & initialize phase bits
    using SmemPtr = decltype(shared_storage.smem_ptr_stage(0));
    SmemPtr smem_ptrs[Stages];
    int phase_bits[Stages];
    #pragma unroll
    for (int i=0; i<Stages; i++) {
        smem_ptrs[i] = shared_storage.smem_ptr_stage(i);
        phase_bits[i] = 0;
    }
    
    // producer & consumer
    bool is_producer = warp_id < (warp_num / 2);
    // producer
    if (is_producer) {
        int producer_tid = tid;
        int producer_stage = 0;
        // one thread launch TMA
        if (producer_tid == 0) {
            for (int i=0; i<size<2>(gA); i++) {
                if (producer_stage >= Stages) {
                    producer_stage = 0;
                }
                Tensor sA_stage = shared_storage.tensor_sA_stage_from_ptr(smem_ptrs[producer_stage]);
                Tensor gA_stage = gA(_, _, i);
                auto [tAgA, tAsA] = tma_partition(
                    tma_atom_A,
                    Int<0>{}, Layout<_1>{},
                    group_modes<0,2>(sA_stage),
                    group_modes<0,2>(gA_stage)
                );
                int tma_transaction_bytes = int(size(tAsA)) * sizeof(TypeA);
                // wait stage smem slot to be free
                wait_barrier(
                    shared_storage.free_barrier[producer_stage],
                    phase_bits[producer_stage]
                );
                phase_bits[producer_stage] ^= 1;
                // TMA launch
                set_barrier_transaction_bytes(shared_storage.tma_barrier[producer_stage], tma_transaction_bytes);
                copy(tma_atom_A.with(shared_storage.tma_barrier[producer_stage]),
                    tAgA,
                    tAsA
                );
                producer_stage++;
            }   
        }
    }
    // consumer
    if (!is_producer) {
        int consumer_tid = tid - ((warp_num / 2) * 32);
        int consumer_stage = 0;

        for (int i=0; i<size<2>(gB); i++) {
            if (consumer_stage >= Stages) {
                    consumer_stage = 0;
            }

            if (consumer_tid == 0) {
                wait_barrier(
                    shared_storage.tma_barrier[consumer_stage],
                    phase_bits[consumer_stage]
                );
                phase_bits[consumer_stage] ^= 1;
            }
            asm volatile("bar.sync %0, %1;" :: "r"(1), "r"(64) : "memory");
            Tensor sA_T_stage = shared_storage.tensor_sA_stage_T_from_ptr(smem_ptrs[consumer_stage]);
            Tensor gB_stage = gB(_, _, i);
            Tensor thr_tile_gB_stage = local_partition(gB_stage, thread_layoutB, consumer_tid);
            Tensor thr_tile_sA_T_stage = local_partition(sA_T_stage, thread_layoutB, consumer_tid);
            copy(thr_tile_sA_T_stage, thr_tile_gB_stage);
            asm volatile("bar.sync %0, %1;" :: "r"(1), "r"(64) : "memory");
            if (consumer_tid == 0) {
                arrive_barrier(shared_storage.free_barrier[consumer_stage]);
            }
            consumer_stage++;
        }
    }
}

template <class TypeA, class LayoutA>
void test_transpose(TypeA const *device_ptr_A, LayoutA layout_A, TypeA *device_ptr_B)
{
    Tensor mA = make_tensor(make_gmem_ptr(device_ptr_A), layout_A);
    auto R = shape<0>(layout_A);
    auto C = shape<1>(layout_A);
    constexpr int stage = 3;
    // launch params
    auto stage_tiler = make_shape(Int<128>{}, Int<64>{});
    auto stage_tiler_T = make_shape(size<1>(stage_tiler), size<0>(stage_tiler));
    auto stage_layout = make_layout(
        stage_tiler,
        GenRowMajor{}
    );
    static constexpr int StageElems = cute::cosize_v<decltype(stage_layout)>;
    auto stage_layout_T = make_layout(
        stage_tiler_T,
        GenColMajor{}
    );
    auto smem_layout = make_layout(
        make_shape(size<0>(stage_tiler), size<1>(stage_tiler), Int<stage>{}),
        make_stride(stride<0>(stage_layout), 
                    stride<1>(stage_layout), 
                    Int<StageElems>{})
                );
    auto cta_tiler = make_shape(Int<128>{}, C);
    auto cta_layout = tiled_divide(mA, cta_tiler);
    auto thr_layoutB =
        make_layout(make_shape(Int<2>{}, Int<32>{}), make_stride(Int<32>{}, Int<1>{}));
    
    using SMEMStorage = SharedStorage<TypeA, 
        decltype(smem_layout), 
        decltype(stage_layout), 
        decltype(stage_layout_T), 
        stage>;
    // TMA
    Copy_Atom tma_atom_A = make_tma_atom(
        SM90_TMA_LOAD{},
        mA,
        stage_layout,
        stage_tiler);
    Tensor mA_tma = tma_atom_A.get_tma_tensor(shape(mA)); // TMA's view of mA
    auto cluster_shape = make_shape(Int<1>{}, Int<1>{}, Int<1>{});
    dim3 dimBlock(128);
    dim3 dimCluster(size<0>(cluster_shape), size<1>(cluster_shape), size<2>(cluster_shape));
    dim3 dimGrid(size<1>(cta_layout), 1, 1);
    int smemBytes = sizeof(SMEMStorage);
    auto *kernel_ptr = &test_transpose_device<decltype(mA), 
                                                decltype(mA_tma), 
                                                decltype(tma_atom_A), 
                                                SMEMStorage, 
                                                decltype(cta_tiler), 
                                                decltype(stage_tiler), 
                                                TypeA, 
                                                decltype(thr_layoutB),
                                                stage>;
    // Set kernel attributes (set SMEM)
    CUTE_CHECK_ERROR(cudaFuncSetAttribute(kernel_ptr,
                                          cudaFuncAttributeMaxDynamicSharedMemorySize,
                                          smemBytes));
    cutlass::ClusterLaunchParams params = {dimGrid, dimBlock, dimCluster, smemBytes};
    cutlass::Status status = cutlass::launch_kernel_on_cluster(params, (void const *)kernel_ptr,
                                                               mA,
                                                               mA_tma,
                                                               tma_atom_A,
                                                               cta_tiler,
                                                               stage_tiler,
                                                               device_ptr_B,
                                                               thr_layoutB);
    CUTE_CHECK_LAST();
    if (status != cutlass::Status::kSuccess)
    {
        std::cerr << "Error: Failed at kernel Launch" << std::endl;
    }
}

int main(int argc, char **argv)
{
    cudaDeviceProp props;
    int current_device_id;
    cudaGetDevice(&current_device_id);
    cudaGetDeviceProperties(&props, current_device_id);
    cudaError_t error = cudaGetDeviceProperties(&props, 0);
    if (error != cudaSuccess)
    {
        std::cerr << "cudaGetDeviceProperties() returned an error: " << cudaGetErrorString(error) << std::endl;
        return -1;
    }

    int R = 4096;
    int C = 4096;

    // RxC K-major Row-Major
    Layout layout = make_layout(make_shape(R, C),
                                make_stride(C, Int<1>{}));
    Layout layout_T = make_layout(make_shape(C, R),
                                  make_stride(R, Int<1>{}));
    using TypeA = float;
    thrust::host_vector<TypeA> host_A(R * C);
    Tensor host_tensor_A = make_tensor(host_A.data(), layout);

    for (int i = 0; i < R; i++)
    {
        for (int j = 0; j < C; j++)
        {
            host_tensor_A(i, j) = TypeA(i * C + j);
        }
    }

    // Copy tensor from host memory to device memory
    thrust::device_vector<TypeA> device_A = host_A;
    thrust::device_vector<TypeA> device_B(R * C);

    test_transpose(device_A.data().get(), layout, device_B.data().get());

    // Copy tensor from device memory to host memory
    thrust::host_vector<TypeA> host_B = device_B;
    Tensor host_tensor_B = make_tensor(host_B.data(), layout_T);

    thrust::host_vector<TypeA> host_reference(R * C);
    Tensor host_reference_tensor = make_tensor(host_reference.data(), layout_T);
    for (int i = 0; i < R; i++)
    {
        for (int j = 0; j < C; j++)
        {
            host_reference_tensor(j, i) = TypeA(i * C + j);
        }
    }

    ////////////////////////////////////////////////////////////
    //
    // Compare results
    //
    ////////////////////////////////////////////////////////////
    TypeA rel_err = 0.f;
    for (int i = 0; i < C; i++)
    {
        for (int j = 0; j < R; j++)
        {
            rel_err += std::abs(host_reference_tensor(i, j) - host_tensor_B(i, j));
        }
    }

    bool success = rel_err <= 0.0;
    std::cout << "Execution is " << ((success) ? "successful." : "failed. rel_err:") << rel_err << std::endl;

    // // Warmup
    // for (int i = 0; i < 5; ++i)
    // {
    //     test_transpose(device_A.data().get(), layout, device_B.data().get());
    // }
    // cudaDeviceSynchronize();
    // cudaEvent_t start, stop;
    // cudaEventCreate(&start);
    // cudaEventCreate(&stop);

    // cudaEventRecord(start);
    // int iters = 100;
    // for (int i = 0; i < iters; ++i)
    // {
    //     test_transpose(device_A.data().get(), layout, device_B.data().get());
    // }

    // cudaEventRecord(stop);
    // cudaEventSynchronize(stop);
    // float elapsed_ms = 0.0f;
    // cudaEventElapsedTime(&elapsed_ms, start, stop);

    // std::cout << "Average time: "
    //           << elapsed_ms / iters
    //           << " ms" << std::endl;

    // cudaEventDestroy(start);
    // cudaEventDestroy(stop);

    return 0;
}
