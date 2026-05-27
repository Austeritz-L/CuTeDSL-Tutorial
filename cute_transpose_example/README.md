# Cute transpose 小例子

目标：把源矩阵A转置后写入B

分为多个阶段, 意在了解cute使用方式

第一阶段仅使用Gmem -> Smem -> Gmem 转置

第二阶段加入Smem帮助转置

第三阶段加入swizzle解决bank conflict

第四阶段 使用TMA从Gmem -> Smem

第五阶段 使用TMA与多轮流水

# 基础认知

### Layout = logical shape + stride

Layout可以被看做一个offset计算器，给layout一个logical shape内的索引，layout会返回一个offset,如

```plain
auto off = layout(make_coord(row, col));
```
 
我们可以永远把logical shape想象成row major的视图，而不必管物理层的数据排布，因为stride是一个翻译器，将logical shape翻译成物理地址。 当讨论张量布局的时候，应该至少包括三类信息：逻辑形状，步长，与物理地址排布。

在cute里，print一个layout会得到像(_128,_128):(256,_1)一样的东西，冒号前为logical shape, 后面为stride.后续layout也会用这种方式表示。

### Copy 使用方式

copy 一个src 到dst时，只需要保证他们的逻辑视图一样，copy会使用src与dst相同的逻辑索引，使用src和dst各自的stride写入物理地址。**copy, tile_divide, local_tile,local_partition,它们的切分与索引使用的均为逻辑视图与逻辑索引**。 我们可以利用这一点，保证src 与 dst logcial shape相同, stride不同完成转置等操作。

### transpose定义

transpose在cute里可以有两种定义，

**第一种：**物理地址转置，需要数据搬运

假设A, B为两块大小一样的矩阵，我们需要物理转置A写到B,假设A的layout为**(R,C):(C,1)**

我们只需要设置B的layout为**(R,C):(1,R)**便可以通过copy 操作完整转置。

举个具体例子，copy会选择一个A里面的i,j坐标，用步长计算offset: offset = i*C +j, 我们期望的转置后的物理坐标offset为i+j*R

copy的目标B坐标不变，仍然是i,j,因为copy需要保证src与dst的逻辑视图一样，所以copy的目标B写入offset为i+j*R,这与期望一致，完成转置。

**第二种：**同一片物理地址的视图转换，无需数据搬运

假设A为**(R,C):(C,1)**

新建一个layout **(C,R):(1,C)**

这样使用逻辑索引遍历的时候就变成转置矩阵，但实际物理地址中的数据没有任何变化

所以：**逻辑形状不变，变换stride->物理转置； 逻辑形状与stride都交换->物理不变,视图转置**

# 阶段一 Gmem -> Reg -> Gem

这里省略host侧main准备数据，详情看copy小例子

### host侧发射辅助函数

在辅助发射函数内准备tensorA, 准备一个cta tile负责大小，与local_tile用到的thread layout

```plain
    Tensor mA = make_tensor(make_gmem_ptr(device_ptr_A), layout_A); 
    auto R = shape<0>(layout_A);
    auto C = shape<1>(layout_A);
    auto cta_tiler = make_shape(Int<128>{}, Int<64>{});
    auto thr_layout =
        make_layout(make_shape(Int<16>{}, Int<8>{}), make_stride(Int<8>{}, Int<1>{}));
```
 
算发射多少cta+发射

```plain
    auto cta_layout = tiled_divide(mA, cta_tiler);
    auto cluster_shape = make_shape(Int<1>{}, Int<1>{}, Int<1>{});
    dim3 dimBlock(128);
    dim3 dimCluster(size<0>(cluster_shape), size<1>(cluster_shape), size<2>(cluster_shape));
    dim3 dimGrid(size<1>(cta_layout), size<2>(cta_layout));
    int  smemBytes = 0;
    auto* kernel_ptr = &test_transpose_device<decltype(mA), decltype(cta_tiler), TypeA, decltype(thr_layout)>;
    // Set kernel attributes (set SMEM)
    CUTE_CHECK_ERROR(cudaFuncSetAttribute(kernel_ptr,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        smemBytes));
    cutlass::ClusterLaunchParams params = {dimGrid, dimBlock, dimCluster, smemBytes};
    cutlass::Status status = cutlass::launch_kernel_on_cluster(params, (void const*) kernel_ptr,
                                                            mA,
                                                            cta_tiler,
                                                            device_ptr_B,
                                                            thr_layout);
```
 
### Device侧算子

基于TensorA layout创建它的物理地址转置layout,用物理地址转置layout创建TensorB

```plain
   auto R = size<0>(mA);
   auto C = size<1>(mA);
   auto layout_AT = make_layout(make_shape(R, C), 
                                            make_stride(Int<1>{}, R));
   Tensor mB = make_tensor(make_gmem_ptr(mB_ptr), layout_AT);
```
 
计算本cta应该处理那块tile

**注意这里用的cta_coord是一样的，这是因为local_tile切割的是逻辑视图，所以拿到的tile的逻辑索引集合相同；但因为stride不一样，可以算一下发现同样的逻辑地址，A与B的物理地址互为物理地址转置关系，所以用一样的cta_coord拿tile即可**

```plain
   auto cta_coord_A = make_coord(blockIdx.x, blockIdx.y);
   Tensor gA = local_tile(mA, cta_tiler, cta_coord_A, Step<_1, _1>{}); // 使用的cta_coord一样 
   Tensor gB = local_tile(mB, cta_tiler, cta_coord_A, Step<_1, _1>{});
```
 
计算每个线程负责哪几个元素

**注意，因为mA,mB逻辑shape一样，所以local_tile切出来的逻辑shape也一样，用同样的thread_layout切出来的thread view tile逻辑shape当然也是一样的**

```plain
   Tensor thr_tile_A = local_partition(gA, thread_layout, threadIdx.x);
   Tensor thr_tile_B = local_partition(gB, thread_layout, threadIdx.x);
```
 
创建寄存器空间

```plain
   Tensor tSrA = make_fragment_like(thr_tile_A); // create reg 
```
 
拷贝操作,这里能推导除dst, reg, src三者逻辑shape全部一样，所以拷贝成功。

```plain
   copy(thr_tile_A, tSrA);
   copy(tSrA, thr_tile_B); 
```
 
# 第二阶段 Gmem -> Smem -> Gmem

### Smem struct 创建

```plain
template <class TypeA,
          class ASmemLayout>    
struct SharedStorage
{
  alignas(128) cute::ArrayEngine<TypeA, cute::cosize_v<ASmemLayout>> A;
  CUTE_DEVICE constexpr auto tensor_sA() { return make_tensor(make_smem_ptr(A.begin()), ASmemLayout{}); }
};
```
 
### Device侧算子

**大致流程：(R,C) row major A读取 -> (R,C) row major Smem写入 -> (C,R) col major Smem读取 -> (C,R) row major B写入**

取mA的shape与本cta负责块的shape

```plain
   auto R = size<0>(mA);
   auto C = size<1>(mA);
   auto cta_R = size<0>(cta_tiler);
   auto cta_C = size<1>(cta_tiler);
```
 
取本cta计算块的坐标,**注意这里与上面的坐标不一样，因为最后会以row major的形式写入，与mA row major相同，所以在逻辑布局就需要转置坐标**

```plain
   auto cta_coord_A = make_coord(blockIdx.x, blockIdx.y);
   auto cta_coord_B = make_coord(blockIdx.y, blockIdx.x); // transpose coordinate
```
 
同样的，用来切割mB的cta tiler也需要转置,并创建mB** (row major CxR)**

```plain
auto cta_tiler_T = make_shape(cta_C, cta_R);
auto mB_layout = make_layout(make_shape(C, R),
                                make_stride(R, Int<1>{})
                                );
Tensor mB = make_tensor(make_gmem_ptr(mB_ptr), mB_layout);
```
 
切本cta负责的块，**注意这里切出来的gA与gB逻辑形状不一样，非正方形tile会导致直接copy出错，**当然直接copy的结果也不会是转置

```plain
   Tensor gA = local_tile(mA, cta_tiler, cta_coord_A, Step<_1, _1>{});
   Tensor gB = local_tile(mB, cta_tiler_T, cta_coord_B, Step<_1, _1>{});
```
 
创建Smem,与拿出一个layout **(R,C) row major **解释这块Smem

```plain
   extern __shared__ char shared_memory[];
   SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(shared_memory);
   Tensor smem_buff = shared_storage.tensor_sA(); 
```
 
创建一个transposed layout **(C,R) row major **来以另一种方式解释Smem,(这个转置视图应该被封装在smem sturct里面并在辅助发射函数内准备好)

```plain
   auto smem_buff_T_layout = make_layout(make_shape(size<1>(smem_buff), size<0>(smem_buff)),
                                          make_stride(stride<1>(smem_buff), stride<0>(smem_buff))
                                        );
   Tensor smem_buff_T = make_tensor(smem_buff.data(), smem_buff_T_layout);
```
 
到这里可以仔细分析一下Smem的两种视图与用同一个thread layout访问时的布局

**假设:** Smem R=128, C=32 row major; thread layout 4, 32, row major

那么这里的访问方式是刚好一个warp读取一行的32个数据，这32个物理地址连续，每个warp会读8行

**假设:** Smem R=32, C=128 col major; thread layout 仍然是 4, 32, row major

那么这里的访问变成4个warp一起读一行的128个数据，并且128个数据不连续，每个数据间隔32步长，这刚好是我们想要的在smem RxC row major 视角下的warp连续读一列Smem，最后合并访存存入gmem.



切割成线程视图tile, **这里thread_layoutA与thread_layoutB在本例中完全相同**，但为后续调优与语义理解，分成两个不同变量

```plain
   Tensor thr_tile_A = local_partition(gA, thread_layoutA, threadIdx.x);
   Tensor thr_tile_B = local_partition(gB, thread_layoutB, threadIdx.x);
   Tensor thr_smem_buff = local_partition(smem_buff, thread_layoutA, threadIdx.x);
   Tensor thr_smem_buff_T = local_partition(smem_buff_T, thread_layoutB, threadIdx.x);
```
 
最后拷贝,**注意实际转置发生在第二个拷贝**。

```plain
   copy(thr_tile_A, thr_smem_buff);
   __syncthreads();
   copy(thr_smem_buff_T, thr_tile_B); 
```
 
# 第三阶段 Swizzle 解决bank conflict

### Smem struct 与辅助发射函数改动

把Smem转置layout也放进struct, 并在辅助发射函数里面构造Smem 转置 layout,

为了演示与计算方便，Smem大小改成64x64

使用Swizzle<6,0,6>

```plain
template <class TypeA,
          class ASmemLayout,
          class ASmemLayout_T>    
struct SharedStorage
{
  alignas(128) cute::ArrayEngine<TypeA, cute::cosize_v<ASmemLayout>> A;
  CUTE_DEVICE constexpr auto tensor_sA() { return make_tensor(make_smem_ptr(A.begin()), ASmemLayout{}); }
  CUTE_DEVICE constexpr auto tensor_sA_T() { return make_tensor(make_smem_ptr(A.begin()), ASmemLayout_T{}); }
};

// 辅助发射函数内
    auto cta_tiler = make_shape(Int<64>{}, Int<64>{});
    auto smem_base = make_layout(cta_tiler, make_stride(Int<64>{}, Int<1>{}));
    auto smem_Layout = composition(Swizzle<6,0,6>{}, smem_base);
    auto smem_base_T = make_layout(cta_tiler_T, make_stride(Int<1>{}, Int<64>{}));
    auto smem_Layout_T = composition(Swizzle<6,0,6>{}, smem_base_T);
    using SMEMStorage = SharedStorage<TypeA, decltype(smem_Layout), decltype(smem_Layout_T)>; 
```
 
thread layout被改成2x64,其他代码均无变化

```plain
    auto thr_layoutA =
      make_layout(make_shape(Int<2>{}, Int<64>{}), make_stride(Int<64>{}, Int<1>{}));
    auto thr_layoutB =
      make_layout(make_shape(Int<2>{}, Int<64>{}), make_stride(Int<64>{}, Int<1>{}));
```
 
### Swizzle理解

Swizzle<x,y,z>表示int BBits, int MBase, int SShift = BBits

其中MBase表示以2^MBase元素为一组作为permute单位，这里我们用的是单个float,所以MBase=0

BBits表示以2^BBits个元素组为一个周期，在周期内做permute，在我们设置的Smem里面，一行有64个float,刚刚以一个float为单位，所以BBits=6

SShift表示一种permute会被重复多少次，我们不需要重复的pattern,所以和BBits设置为一样即可，所以SShift=6

看一下Swizzle后的布局

![image](assets/resources/vSjLYuW51LAYfJqBmxA_bSckulMlBUl2TJrfIJiC24w.png)

### bank conflict free证明

##### 写入

设warp写入的行为row, row \\in \[0,63\]

 1.首先证明:设col_new = row ^ col, 任意两个col1, col2 \\in \[0,63\], col1!=col2, col1_new != col2_new

```plain
假设 col1_new = col2_new
(col1^row）= (col2^row)
(col1^row)^row = (col2^row)^row
col1 = col2 与 col1!=col2 冲突
可得 col1_new != col2_new
```
 
2.其次证明col1_new&31 != col2_new&31

先看第一个warp col \\in \[0,31\]

```plain
(row^col)&31 = (row&31)^(col&31) = (row&31)^col
row&31 \in [0,31]
刚刚已经证明row ^ col1 != row ^ col2， 所以对任意两个不同col,(row^col1)&31 !=(row^col2)&31 无bank conflict
```
 
再看第二个warp col \\in \[32,63\]

```plain
(row^col)&31 = (row&31)^(col&31)
row&31 \in [0,31]
col%31 \in [0,31] 且 col1!=col2 -> col1%31 != col2%31 (就是去掉最高位一个bit,这里不做证明)
后面证明与上面情况完全一样， 无bank conflict
```
 
##### 读出

变成col一样，row都不一样，与写入证明过程一致

##### 函数打印bank

```plain
template <class Layout>
void print_warp_banks(Layout layout, const char* name, int fixed_row, bool transpose_read) {
    std::cout << "\n" << name << "\n";

    int hist[32] = {0};

    for (int lane = 0; lane < 32; ++lane) {
        int row = fixed_row;
        int col = lane;

        auto off = layout(make_coord(row, col));
        int bank = int(off) & 31;

        hist[bank]++;

        std::cout
            << "lane=" << lane
            << " coord=(" << row << "," << col << ")"
            << " off=" << int(off)
            << " bank=" << bank
            << "\n";
    }

    std::cout << "hist:";
    for (int b = 0; b < 32; ++b) {
        if (hist[b]) std::cout << " bank" << b << "=" << hist[b];
    }
    std::cout << "\n";
}

    print_warp_banks(smem_Layout,   "swizzle normal write", 0, false);
    print_warp_banks(smem_Layout_T, "swizzle transpose read", 0, true);
```
 
通过打印，写入与读出均无bank conflict



# 第四阶段 加入TMA

### Shared Storage Struct 改动

需要加入tma_barrier供cta所有线程查询一次tma搬运状态

```plain
template <class TypeA,
          class ASmemLayout>
struct SharedStorage
{
    alignas(16) cute::uint64_t tma_barrier; // <---------加入tma_barrier
    alignas(128) cute::ArrayEngine<TypeA, cute::cosize_v<ASmemLayout>> A;
    CUTE_DEVICE constexpr auto tensor_sA() { return make_tensor(make_smem_ptr(A.begin()), ASmemLayout{}); }
};
```
 
### host侧 辅助发射函数改动

创建tma copy atom, 它决定一次tma搬运的src layout是什么样，dst layout是什么样，一次搬运tile大小

**注意smem_Layout元素数必须和cta_tiler元素数一致，mA是全局大矩阵，元素数不必与cta_tiler一致，但逻辑维度需要与cta_tiler一致**

```plain
    Copy_Atom tma_atom_A = make_tma_atom(
        SM90_TMA_LOAD{},
        mA,
        smem_Layout,
        cta_tiler);
```
 
生成TMA视角下的mA，并把它当作输入矩阵传入device侧函数

```plain
Tensor mA_tma = tma_atom_A.get_tma_tensor(shape(mA)); // TMA's view of mA

auto *kernel_ptr = &test_transpose_device<decltype(mA), decltype(mA_tma), decltype(tma_atom_A), SMEMStorage, decltype(cta_tiler), TypeA, decltype(thr_layoutB)>;

cutlass::Status status = cutlass::launch_kernel_on_cluster(params, (void const *)kernel_ptr,
                                                               mA,
                                                               mA_tma,
                                                               tma_atom_A,
                                                               cta_tiler,
                                                               device_ptr_B,
                                                               thr_layoutB);
```
 
### device 算子改动

用传进来的tma_atom_A与切好的mA_tma（gA, 一块拷贝的src,与上面host侧定义的一次搬运大小相同），和smem_buff生成TMA的src和dst

**注意如果不用cluster TMA broadcast, 第二和第三个参数就填Int<0>{}, Layout<_1>{}，具体意义还需研究**

**注意partition输入的src与dst需要展平成一维，这里使用group_modes展开，否则会导致illegal memory access**

```plain
    auto [tAgA, tAsA] = tma_partition(tma_atom_A,
                                      Int<0>{}, Layout<_1>{},
                                      group_modes<0,2>(smem_buff),
                                      group_modes<0,2>(gA));
```
 
一个线程初始化tma barrier

initialize_barrier里面第二个输入为等待的线程数量，具体哪一步作为等待尚不明确(大概是发起拷贝的那行指令)，其对应ptx为

 asm  volatile  ( "mbarrier.init.shared::cta.b64  [ %0 ] , %1; \ n"

     ::  "r" (smem_int_ptr),

        "r" (thread_count));

初始化一个tma phase bit. 这个bit用来与tma_barrier内部的一个phase bit做对比，如果相同则表明tma搬运完成，同时tma barrier内部bit翻转，具体流程可以被视为：

**第一次搬运前：**

tma_barrier_phase_bit: 1

expected_phase_bit: 0

**第一次搬运完成后：**tma phase bit 翻转，与expected phase bit 一致，barrier放行

tma_barrier_phase_bit: 0

expected_phase_bit: 0

**第二次搬运前：**手动把expected_phase_bit翻转，作为下一次搬运的期望值

tma_barrier_phase_bit: 0

expected_phase_bit: 1

```plain
    uint32_t elect_one_thr = cute::elect_one_sync();
    uint32_t elect_one_warp = (threadIdx.x / 32 == 0);
    if (elect_one_warp && elect_one_thr)
    {
        cute::initialize_barrier(shared_storage.tma_barrier, /* num_threads */ 1);
    }
    int tma_barrier_phase_bit = 0;
```
 
同步cta内所有线程，以免其他线程使用未初始化的tma barrier作为wait barrier输入进行轮询导致未定义行为

让一个线程设置本次搬运数据大小+发起tma搬运

所有线程等待搬运完成

翻转expected phase bit (一次搬运可以不做)

```plain
    __syncthreads();
    int tma_transaction_bytes = sizeof(make_tensor_like(tAsA));
    if (elect_one_warp && elect_one_thr)
    {
        cute::set_barrier_transaction_bytes(shared_storage.tma_barrier, tma_transaction_bytes);
        copy(tma_atom_A.with(shared_storage.tma_barrier),
             tAgA,
             tAsA);
    }
    cute::wait_barrier(shared_storage.tma_barrier,
                       tma_barrier_phase_bit);
    tma_barrier_phase_bit ^= 1;
```
 

