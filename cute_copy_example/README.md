# Cute copy 小例子

流程： 将一个矩阵切块，搬运到SMEM, 再搬运到寄存器，最后存回GMEM, 展示不同内存的初始化，切割，与复制操作，这里只用到一个矩阵A, 读A再写回到A

# 开头include

```plain
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
```
 
# main 数据准备

在host侧初始化A与它的layout

```plain
  int R = 256;
  int C = 256;

  // RxC K-major Row-Major
  Layout layout = make_layout(make_shape (R, C), make_stride(C, Int<1>{})); 
  using TypeA = float;
  thrust::host_vector<TypeA>   host(R * C); // 这里会malloc分配host内存
  Tensor host_tensor = make_tensor(host.data(), layout);
```
 
随便填点数进去

```plain
  for (int i=0; i<R; i++) {
    for (int j=0; j<C; j++) {
        host_tensor(i,j) = TypeA(i*C + j);
    }
  }
```
 
把host侧的矩阵复制到GPU侧

```plain
thrust::device_vector<TypeA> device = host;
```
 
调用host侧辅助函数，**注意要用.data().get()**

```plain
test_copy(device.data().get(), layout); // .data()不是raw pointer, 还得加个.get()
```
 
# SMEM封装

这里ASmemLayout被放入模板，所以被视为编译常量，cute::cosize_v<ASmemLayout>会根据layout自动算出A的大小

```plain
template <class TypeA,
          class ASmemLayout>    
struct SharedStorage
{
  alignas(128) cute::ArrayEngine<TypeA, cute::cosize_v<ASmemLayout>> A;
  CUTE_DEVICE constexpr auto tensor_sA() { return make_tensor(make_smem_ptr(A.begin()), ASmemLayout{}); }
};
```
 
# host辅助发射函数

为了让数据类型灵活+为每一个layout单独编译， 定义模板

```plain
template <class TypeA, class LayoutA>
void test_copy(TypeA* device_ptr_A, LayoutA layout_A)
{
}
```
 
建立矩阵A与它的shape

```plain
    Tensor mA = make_tensor(make_gmem_ptr(device_ptr_A), layout_A); 
    auto R = shape<0>(layout_A);
    auto C = shape<1>(layout_A);
```
 
希望一个cta负责128 x 128的子矩阵,并使用tiled_divide算出需要发射多少个cta,这里tile divide可以看到返回值包括切出来的tile大小+数量((_128,_128),2,2)，它已经算出来了要发射2x2个cta

```plain
 auto cta_tiler = make_shape(Int<128>{}, Int<128>{});
 auto cta_layout = tiled_divide(mA, cta_tiler);
// print("cta_layout:\t"); print(cta_layout); print("\n"); //cta_layout:     gmem_ptr[32b](0x767d73000000) o ((_128,_128),2,2):((256,_1),32768,_128)
```
 
在向SMEM搬运的时候，每次搬运128x32的子矩阵

```plain
auto copy_tiler = make_shape(Int<128>{}, Int<32>{});
```
 
使用smem的时候加上swizzle(创建layout不再是shape + stride, 而是用composition,可以加个swizzle进去),并定义一个上面的SMEM类

```plain
    auto sA_layout = composition(
      Swizzle<3, 4, 3>{},
      make_layout(
          copy_tiler,
          make_stride(Int<32>{}, Int<1>{})
      )

   using SMEMStorage = SharedStorage<TypeA, decltype(sA_layout)>;
```
 
定义copy类型与方式，试试向量化copy （普通copy可以跳过这步），下面的layout为thread layout 与 value layout

thread layout决定线程布局， <128, 1>代表每个线程负责一行 （严重conflict）

value layout负责每个线程负责拷贝元素数量，<1, 32> 对应一个线程负责一行32列个元素

****注意这里Layout的维度要和拷贝的张量维度一致，这里都是2维**

```plain
    using S2RAtom = cute::Copy_Atom<cute::AutoVectorizingCopy, TypeA>;
    auto tiled_s2r_copy = cute::make_tiled_copy(
      S2RAtom{},
      Layout<Shape<_128, _1>>{},
      Layout<Shape<_1, _32>>{}
    );
```
 
定义cluster_shape (协作cta, 核函数发射需要) 与其他发射参数

```plain
    auto cluster_shape = make_shape(Int<1>{}, Int<1>{}, Int<1>{}); // 不用协作cta
    dim3 dimBlock(128);
    dim3 dimCluster(size<0>(cluster_shape), size<1>(cluster_shape), size<2>(cluster_shape));
    dim3 dimGrid(size<1>(cta_layout), size<2>(cta_layout));
    int  smemBytes = sizeof(SMEMStorage);
    auto* kernel_ptr = &test_copy_device<SMEMStorage,
                                  decltype(mA), decltype(cta_tiler), decltype(copy_tiler), decltype(tiled_s2r_copy)>;
    cutlass::ClusterLaunchParams params = {dimGrid, dimBlock, dimCluster, smemBytes};
```
 
发射

```plain
    cutlass::Status status = cutlass::launch_kernel_on_cluster(params, (void const*) kernel_ptr,
                                                            mA,
                                                            cta_tiler,
                                                            copy_tiler,
                                                            tiled_s2r_copy);
```
 
# GPU侧函数

模板传SMEM类，需要拷贝到张量类型，CTA的切割形状，每次SMEM拷贝切割形状，拷贝类型（刚刚创建的向量化拷贝）

```plain
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
```
 
创建SMEM并拿到buffer对应tensor

```plain
    extern __shared__ char shared_memory[];
    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(shared_memory);

    Tensor smem_buffA = shared_storage.tensor_sA(); 
```
 
拿到本cta需要处理的子矩阵,可以看到local tile切出来了一个128x128的块，与之前cta块定义大小相符

```plain
    auto cta_coord = make_coord(blockIdx.x, blockIdx.y);
    Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, _1>{}); 
// gA:gmem_ptr[32b](0x7e969f000000) o (_128,_128):(256,_1)
```
 
之前定义每次copy 128x32的tile, 所以把128x128子矩阵继续切分,可以看到它切成了一个128x32x1x4的tile array,一会loop就遍历后两个维度

```plain
Tensor gA_tiles = tiled_divide(gA, copy_tiler); 
// gA_tiles:gmem_ptr[32b](0x726c8b000000) o ((_128,_32),_1,_4):((256,_1),_0,_32)
```
 
用host侧传进来的copy切分SMEM buffer, 让每个线程得到负责位置索引

```plain
    auto thr_s2r_copy = tiled_s2r_copy.get_thread_slice(threadIdx.x);
    Tensor tSsA = thr_s2r_copy.partition_S(smem_buffA);
```
 
声明并创建一个与tSsA形状相同的寄存器空间

```plain
Tensor tSrA = make_fragment_like(tSsA);
```
 
loop 遍历gA_tiles后两个维度，由于倒数第二个维度是1，所以简化成一个loop

```plain
for (int k_block = 0; k_block < size<2>(gA_tiles); ++k_block) {
}
```
 
拿到本次loop需要处理的tile,并把数据从Gmem -> Smem，**注意make_coord(_, _)表示选择全部，copy需要dst和src的逻辑shape一致**

```plain
for (int k_block = 0; k_block < size<2>(gA_tiles); ++k_block) {
  Tensor gA_tile = gA_tiles(make_coord(_, _), 0, k_block);
  cooperative_copy<128>(threadIdx.x, gA_tile, smem_buffA);
  __syncthreads();
}
```
 
Smem -> Reg

```plain
for (int k_block = 0; k_block < size<2>(gA_tiles); ++k_block) {
  Tensor gA_tile = gA_tiles(make_coord(_, _), 0, k_block);
  cooperative_copy<128>(threadIdx.x, gA_tile, smem_buffA);
  __syncthreads();
  copy(tiled_s2r_copy, tSsA, tSrA);
}
```
 
Reg -> Gmem **注意这里直接复用了之前创建的Smem->Reg的copy,严格来说应该为reg->gmem单独创建一个copy**

```plain
for (int k_block = 0; k_block < size<2>(gA_tiles); ++k_block) {
  Tensor gA_tile = gA_tiles(make_coord(_, _), 0, k_block);
  cooperative_copy<128>(threadIdx.x, gA_tile, smem_buffA);
  __syncthreads();
  copy(tiled_s2r_copy, tSsA, tSrA);
  Tensor tDgA = thr_s2r_copy.partition_D(gA_tile);
  copy(tiled_s2r_copy, tSrA, tDgA);
  __syncthreads();
}
```


# 编译指令

```plain
nvcc -std=c++17 \
  -gencode=arch=compute_103a,code=sm_103a \
  -DCUTLASS_ARCH_MMA_SM100_SUPPORTED \
  -DCUTE_ARCH_TCGEN05_TMEM_ENABLED \
  -I/cutlass-main/include \
  -I/cutlass-main/tools/util/include \
  -I/cutlass-main/examples/cute/tutorial \
  -I/cutlass-main/examples/cute/tutorial/blackwell \
  test_copy.cu \
  -o test_copy
```
 

