# CSC CUDA 编程课程 — 练习指南

## 概述

本仓库包含 CSC（芬兰 IT 科学中心）"CUDA 编程入门"课程的 CUDA C/C++ 练习。顶层 README 指向 CSC 2017 年的课程页面，说明仓库包含课程练习。`course-material` 中的 README 标明课程于 2017 年 5 月 17-19 日在 CSC 举办，规划了从 GPU 入门编程、工具使用、性能优化、统一内存、动态并行、多 GPU 编程到 MPI+CUDA 的三天进阶路线。

这是 CUDA C/C++ 课程，而非 CUDA Fortran。练习源码使用 `.cu`、`.cpp`、`.c` 和 `.h` 文件、CUDA Runtime API、Jacobi 规约中的 Thrust 库、以及最终 Ping-Pong 练习中的 MPI。多数练习目录包含 `solution/` 子目录，提供了完成版本代码以展示预期实现模式。`demo/` 目录包含一个 stencil 优化示例，`course-material/` 包含课程说明 README 和 `intro-to-cuda-csc.pdf`。

## 仓库结构

```text
csc-CUDA-c/
|-- README.md
|-- course-material/
|   |-- README.md
|   `-- intro-to-cuda-csc.pdf
|-- demo/
|   `-- stencil code/
|       `-- stencil.cu
|-- exercises/
|   |-- README.md
|   |-- compiling/
|   |-- first-cuda-program/
|   |-- profile-nvvp/
|   |-- profile-nvprof/
|   |-- debug-cuda-memcheck/
|   |-- debug-nsight/
|   |-- error-checking/
|   |-- jacobi/
|   |-- jacobi-timing-events/
|   |-- simple-streams/
|   |-- kernel-optimization/
|   |-- first-unified-memory-program/
|   |-- unified-memory-streams/
|   |-- dynamic-sync/
|   |-- dynamic-mandelbrot/
|   |-- multi-GPU/
|   `-- mpi-ping-add-pong/
`-- Docs/
    `-- CUDA-Exercises-Guide.md
```

## 课程材料

`course-material/README.md` 是课程介绍与运行指南。给出了三天日程安排，说明了本地工作站和 Taito-GPU 的使用方式，列出了重要的 `nvcc` 编译选项：

```text
nvcc -arch=sm_30 <source>.cu
-lineinfo
-Xptxas="-v"
-Xptxas="-dlcm=ca"
-use_fast_math
-rdc=true
-lcudart
```

课程环境使用教室的 Quadro K600 GPU，配备 CUDA SDK 8.0，Taito-GPU 提交命令如：

```bash
srun -n1 -pgpu --gres=gpu:1 ./my_program
```

课程讲义为 `course-material/intro-to-cuda-csc.pdf`。讲义强调 CUDA Runtime API 编程（非底层 Driver API），覆盖了与练习相同的主题路线：Host/Device 内存、Kernel 与线程层级、调试/性能分析工具、分支与同步、Stream 与 Event、Kernel 优化、统一内存、动态并行、多 GPU 编程和 CUDA-aware MPI。

## Demo：Stencil 代码

目录：`demo/stencil code/`

该 Demo 不是练习，但非常重要——它展示了一个围绕 1D Stencil 的较大型优化案例研究。比较了基准全局内存、只读缓存加载、常量内存系数和共享内存分块等多种实现。

关键文件：

- `stencil.cu`：完整 CUDA 程序，包含多种 stencil kernel、`std::chrono` 计时、手动验证、常量内存、只读缓存加载、共享内存、Host 页锁定内存和统一内存。

涉及概念：

- `__constant__` 内存用于小系数数组。
- `__ldg()` 只读缓存加载。
- `__shared__` 内存分块缓冲与 `__syncthreads()`。
- `cudaMemcpyToSymbol()` 用于常量内存初始化。
- `cudaMallocHost()` 页锁定 Host 内存。
- `cudaMallocManaged()` 统一内存。

代表性代码：

```cuda
__constant__ float const_stencilWeight[21];

__global__ void stencilConst2(float *src, float *dst, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    idx += 11;
    if (idx >= size)
        return;
    float out = 0;
    #pragma unroll
    for(int i = -10;i < 10; i++)
        out += __ldg(&src[idx+i]) * const_stencilWeight[i+10];
    dst[idx] = out;
}

cudaMemcpyToSymbol(const_stencilWeight, weights, sizeof(float)*21);
getErrorCuda((stencilConst2<<<blocks, blockSize>>>(a, b, 10240000)));
```

独特之处：这是一个性能演示，而非填空式骨架。将一个基准程序压缩了多种内存层级话题。

## 练习详解

### 编译

目录：`exercises/compiling/`

教授概念：

- 使用 `nvcc` 编译 CUDA C 程序。
- 查询 CUDA 设备。
- 启动最小的 kernel。
- 同步以显示设备端 `printf()` 输出。

关键文件：

- `README.md`：要求学生在本地工作站和 Taito-GPU 上编译运行 `hello.cu`。
- `hello.cu`：完整程序，无需修改。

练习任务：在两个目标平台上编译并运行提供的 CUDA 程序。

关键 API 模式：

```cuda
__global__ void hello()
{
  printf("Greetings from your GPU\n");
}

cudaGetDeviceCount(&count);
cudaGetDevice(&device);
hello<<<1,1>>>();
cudaDeviceSynchronize();
```

独特之处：这是环境冒烟测试。引入了 kernel 启动语法，但不涉及内存分配或错误处理。

### 第一个 CUDA 程序

目录：`exercises/first-cuda-program/`

教授概念：

- 设备内存分配与释放。
- 将结果从设备拷贝回主机。
- 计算全局线程索引。
- 防止 kernel 越界启动。

关键文件：

- `README.md`：列出五项 TODO：内存分配、释放、kernel 索引、启动、数据拷贝。
- `set.cu`：带 TODO 的骨架代码。
- `solution/set.cu`：完成版本。

练习任务：实现一个 kernel 使 `A[i] = i`（`N = 128`），分配 `d_A`，启动 kernel，拷贝数据回 Host，释放设备内存。

关键 API 模式：

```cuda
__global__ void set(int *A, int N)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N)
    A[idx] = idx;
}

cudaMalloc((void**)&d_A, N * sizeof(int));
set<<<2, 64>>>(d_A, N);
cudaMemcpy(h_A, d_A, N * sizeof(int), cudaMemcpyDeviceToHost);
cudaFree((void*)d_A);
```

独特之处：这是第一个完整的 CUDA 内存管理工作流，但有意省略了错误检查，使基本机制更清晰。

### NVVP 性能分析

目录：`exercises/profile-nvvp/`

教授概念：

- 使用 NVIDIA Visual Profiler（`nvvp`）进行可视化性能分析。
- 编译时使用 `-lineinfo`。
- 理解数据传输和 kernel 执行时间。

关键文件：

- `README.md`：说明如何用 `nvcc -lineinfo main.cu` 编译并创建 NVVP 分析会话。
- `main.cu`：简单的拷贝 kernel 加 Host-Device 传输。

练习任务：分析程序，检查 Host→Device 和 Device→Host 拷贝时间，以及 kernel 执行时间。

关键 API 模式：

```cuda
__global__ void copyKernel(int *src, int *dst)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    dst[idx] = src[idx];
}

cudaMalloc(&a_dev, sizeof(int)*128*100);
cudaMemcpy(a_dev, a, sizeof(int)*1000, cudaMemcpyHostToDevice);
copyKernel<<<100, 128>>>(a_dev, b_dev);
cudaDeviceSynchronize();
cudaMemcpy(b, b_dev, sizeof(int)*1000, cudaMemcpyDeviceToHost);
```

独特之处：工具优先。代码刻意简单，使学生能专注于 profiler 时间线解读。

### NVPROF 性能分析

目录：`exercises/profile-nvprof/`

教授概念：

- 使用 `nvprof` 进行命令行性能分析。
- 比较多组 kernel 启动。
- 区分拷贝大小、启动范围和运行时间测量。

关键文件：

- `README.md`：要求学生编译并通过 `nvprof` 运行，找出数据传输和 kernel 时间。
- `main.cu`：scale kernel 以不同启动配置和指针偏移多次调用。

练习任务：使用 `nvprof` 报告数据传输时间和 kernel 运行时间；可选地推理浮点操作计数。

关键 API 模式：

```cuda
__global__ void scaleKernel(float *src, float *dst, float scale)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    dst[idx] = src[idx] * scale;
}

scaleKernel<<<100, 128>>>(a_dev, b_dev, 4.0f);
scaleKernel<<<50, 128>>>(a_dev, b_dev, 4.0f);
scaleKernel<<<50, 128>>>(a_dev+50, b_dev+50, 4.0f);
```

独特之处：展示同一 kernel 如何根据启动几何和参数的不同，在 profiling 中呈现为不同事件。

### cuda-memcheck 调试

目录：`exercises/debug-cuda-memcheck/`

教授概念：

- 查找设备端越界访问。
- 理解闭区间 vs 开区间边界检查的区别。
- 对小程序使用 cuda-memcheck。

关键文件：

- `README.md`：要求学生诊断为什么数组拷贝不正常。
- `main.cu`：含 bug 的版本。
- `solution/main.cu`：修正了边界检查。

练习任务：编译程序，在 cuda-memcheck 下运行，识别出一个线程在数组之外读写了一个元素，修复边界守卫。

Bug 与修复：

```cuda
// Bug：
if(idx > size)
    return;

// 修复：
if(idx >= size)
    return;
```

独特之处：kernel 启动创建的线程远超 1000 个（`100 * 128`），因此正确性完全依赖边界条件。

### Nsight 调试

目录：`exercises/debug-nsight/`

教授概念：

- 将 CUDA 项目导入 Nsight。
- 在调试器下运行。
- 使用断点和线程/Block 检查。
- 检查每个线程的内存值。

关键文件：

- `README.md`：Nsight 导入/调试指令，询问 block 3 的 thread 56、全局 thread 824 读取了什么值。
- `src/Debug_nsight.cu`：随机输入初始化和一个 scale kernel。

练习任务：使用 Nsight 在 kernel 内部停止并检查指定线程的值。

关键 API 模式：

```cuda
unsigned idx = blockIdx.x*blockDim.x+threadIdx.x;
if (idx >= count)
    return;

const float in = dataIn[idx];
dataIn[idx] = in * scale;

int threadsPerBlock = 256;
int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
scaleKernel<<<blocksPerGrid, threadsPerBlock>>>(devA, devB, 2, size);
```

独特之处：面向交互式调试，而非代码实现。

### 错误检查

目录：`exercises/error-checking/`

教授概念：

- 用宏封装 CUDA Runtime 调用。
- 使用 `cudaGetLastError()` 检查 kernel 启动/运行时错误。
- 为什么显式同步能更早捕获异步错误。
- 诊断无效 Block 大小和无效 Host/Device 指针使用。

关键文件：

- `README.md`：要求学生添加内存操作和 kernel 启动，然后尝试各种错误场景。
- `error-test.cu`：向量加法骨架。
- `error_checks.h`：`CUDA_CHECK` 和 `CHECK_ERROR_MSG` 宏。
- `solution/error-test.cu`：含检查的完整向量加法。

练习任务：使用提供的宏完成内存分配、拷贝、启动、同步、数据取回和清理。

关键 API 模式：

```cuda
#define CUDA_CHECK(errarg)   __checkErrorFunc(errarg, __FILE__, __LINE__)
#define CHECK_ERROR_MSG(errstr) __checkErrMsgFunc(errstr, __FILE__, __LINE__)

CUDA_CHECK( cudaMalloc((void**)&dA, sizeof(double)*N) );
CUDA_CHECK( cudaMemcpy(dA, hA, sizeof(double)*N, cudaMemcpyHostToDevice) );

grid.x = (N + ThreadsInBlock - 1) / ThreadsInBlock;
threads.x = ThreadsInBlock;
vector_add<<<grid, threads>>>(dC, dA, dB, N);

cudaDeviceSynchronize();
CHECK_ERROR_MSG("vector_add kernel");
```

独特之处：形式化了一个安全模式，后续练习大量复用。

### Jacobi 迭代

目录：`exercises/jacobi/`

教授概念：

- 将 2D 有限差分模板从 CPU 移植到 GPU。
- 2D Grid 和 2D 线程 Block。
- 数值 kernel 的边界守卫。
- 比较 CPU 和 GPU 结果。
- 使用 Thrust 规约进行收敛检查。

关键文件：

- `README.md`：要求学生实现 `sweepGPU`，尝试不同 grid 和 block 大小，用 NVVP 查看性能。
- `jacobi.cu`：CPU 和 GPU Jacobi 程序骨架。
- `jacobi.h`：用 Thrust `transform_reduce` 实现的 `diffGPU()`。
- `error_checks.h`：错误检查宏。
- `solution/jacobi.cu`：完成的 GPU 版本。

练习任务：实现与 `sweepCPU` 等价的 GPU sweep，每次迭代调用两次，取回结果，释放设备内存，验证平均差值。

关键 API 模式：

```cuda
__global__
void sweepGPU(double *phi, const double *phiPrev, const double *source,
              double h2, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int index = i + j*N;

    if (i > 0 && j > 0 && i < N-1 && j < N-1)
        phi[index] = 0.25 * (phiPrev[(i-1) + j*N] +
                             phiPrev[(i+1) + j*N] +
                             phiPrev[i + (j-1)*N] +
                             phiPrev[i + (j+1)*N] -
                             h2 * source[index]);
}

dim3 dimBlock(blocksize, blocksize);
dim3 dimGrid((N + blocksize - 1) / blocksize,
             (N + blocksize - 1) / blocksize);
sweepGPU<<<dimGrid, dimBlock>>>(phiPrev_d, phi_d, source_d, h*h, N);
```

独特之处：第一个真正的数值求解器，第一个正确性是数值性而非简单整数等价的练习。

### Jacobi 计时与 Event

目录：`exercises/jacobi-timing-events/`

教授概念：

- CUDA Event 创建、记录、同步、计算耗时和清理。
- 将 GPU 计时与 CPU 墙钟计时分离。
- 复用之前的 kernel 并改进测量质量。

关键文件：

- `README.md`：要求学生修改之前的 Jacobi 练习，用 CUDA Event 计时 GPU 工作。
- `jacobi.cu`：带 Event TODO 的 Jacobi 骨架。
- `solution/jacobi.cu`：完成 Event 计时的版本。
- `jacobi.h`：Thrust 收敛规约。
- `error_checks.h`：错误检查宏。

练习任务：添加 Event 初始化，记录开始和结束，在停止 Event 上同步，计算耗时毫秒数，打印秒数，销毁 Event。

关键 API 模式：

```cuda
cudaEvent_t start, stop;
CUDA_CHECK( cudaEventCreate(&start) );
CUDA_CHECK( cudaEventCreate(&stop) );

CUDA_CHECK( cudaEventRecord(start, 0) );
// 重复 sweepGPU 启动和 diffGPU 调用
CUDA_CHECK( cudaEventRecord(stop, 0) );
CUDA_CHECK( cudaEventSynchronize(stop) );
CUDA_CHECK( cudaEventElapsedTime(&gputime, start, stop) );

CUDA_CHECK( cudaEventDestroy(start) );
CUDA_CHECK( cudaEventDestroy(stop) );
```

独特之处：教授了后续 Stream 和多 GPU 练习中也会用到的计时原语。

### 简单 Stream

目录：`exercises/simple-streams/`

教授概念：

- 页锁定 Host 内存。
- CUDA Stream 创建与销毁。
- 异步拷贝。
- 向特定 Stream 中启动 kernel。
- 将单次向量操作分解到多个 Stream。
- 比较默认 Stream 与 1、2、4、8 个 Stream 的计时。

关键文件：

- `README.md`：描述测试数组加法 kernel 在多 Stream 下的表现。
- `streams.cu`：带 TODO 的骨架，涉及 pinned 分配、异步拷贝、Stream kernel 启动、异步拷回和 Host 清理。
- `solution/streams.cu`：完成的 Stream 实现。
- `error_checks.h`：错误检查宏。

练习任务：分配 pinned Host 内存，创建 Stream，拆分向量，对每段使用 `cudaMemcpyAsync()` 和 Stream kernel 启动，用 Event 测量，释放 pinned 内存。

关键 API 模式：

```cuda
CUDA_CHECK( cudaMallocHost((void**)&hA, sizeof(double) * N) );
CUDA_CHECK( cudaStreamCreate(&s[i].strm) );

CUDA_CHECK( cudaMemcpyAsync(&dA[sidx], &hA[sidx],
                            sizeof(double) * slen,
                            cudaMemcpyHostToDevice, s[i].strm) );

vector_add<<<grid, threads, 0, s[i].strm>>>(&dC[sidx], &dA[sidx],
                                            &dB[sidx], slen, iterations);

CUDA_CHECK( cudaMemcpyAsync(&hC[sidx], &dC[sidx],
                            sizeof(double) * slen,
                            cudaMemcpyDeviceToHost, s[i].strm) );
```

独特之处：第一个显式的重叠/并发练习，必须有 pinned Host 内存才能实现有意义的异步传输行为。

### Kernel 优化

目录：`exercises/kernel-optimization/`

教授概念：

- 分析一个非平凡的射线行进渲染器。
- 使用 NVVP/nvprof 指标识别性能瓶颈。
- 浮点精度陷阱。
- 分支发散和启动几何。
- 内存合并/索引错误。
- 使用 `__ldg()` 的只读缓存加载。
- 针对跨步访问模式考虑共享内存。

关键文件：

- `README.md`：描述了一个分形体渲染器，要求学生用 NVVP 识别问题，提供 Taito-GPU 上 `nvprof` 收集指标和时间线的命令。
- `main.cu`：包含 `computeUV`、`trace`、`shade`、`globalIllumination` 和 `downsample` kernel 的渲染器。
- `util.h`：`Vec3`、`getErrorCuda`、PNG 写入、clamp/辅助函数。
- `solution/main.cu`：带注释的修复版本。
- `solution/util.h`：与骨架相同的辅助结构。

练习任务：分析程序，识别主要性能问题，可选地修复部分问题。解决方案添加了关于意外双精度提升、更优启动几何、x/y 索引和只读加载的注释和修改。

关键 API 和优化模式：

```cuda
// 骨架中意外使用双精度：
float uvx = (tan(3.14159265 / 4.0)) * (2.0*x - widthFloat) / widthFloat;

// 解决方案改用浮点运算：
float uvx = (tanf(3.14159265f / 4.0f)) *
            (2.0f*x - widthFloat) / widthFloat;
```

```cuda
// 骨架 trace 启动配置：
threadsTrace.x = 1024;
threadsTrace.y = 1;

// 解决方案启动配置：
threadsTrace.x = 8;
threadsTrace.y = 8;
```

```cuda
// 解决方案 downsample 中使用只读缓存：
outR += __ldg(&rawR[((inY+i)*width*scale)+(inX+j)]);
outG += __ldg(&rawG[((inY+i)*width*scale)+(inX+j)]);
outB += __ldg(&rawB[((inY+i)*width*scale)+(inX+j)]);
```

独特之处：这是一个性能分析和诊断练习，而非直接的填空式 TODO。代码刻意划分为多个 kernel，便于学生定位瓶颈。

### 第一个统一内存程序

目录：`exercises/first-unified-memory-program/`

教授概念：

- 将显式 Host/Device 内存代码移植到统一内存。
- 访问托管内存前进行同步。
- 使用 `cudaMallocManaged()` 简化拷贝密集型示例。

关键文件：

- `README.md`：要求学生将之前的 `set.cu` 移植到统一内存。
- `set.cu`：显式 `cudaMalloc()`/`cudaMemcpy()` 版本。
- `solution/set.cu`：托管内存版本。

练习任务：用一次托管分配替代分离的 Host/Device 分配，移除显式数据拷回，kernel 完成后同步，释放托管分配。

关键 API 模式：

```cuda
int *A;
cudaMallocManaged((void**)&A, N * sizeof(int));

set<<<2, 64>>>(A, N);
cudaDeviceSynchronize();

for (int i = 0; i < N; i++)
  printf("%i ", A[i]);

cudaFree((void*)A);
```

独特之处：用不同内存模型重复第一个 CUDA 编程任务，使显式 vs 托管的对比清晰明了。

### 统一内存与 Stream

目录：`exercises/unified-memory-streams/`

教授概念：

- 将托管内存与 Stream 结合使用。
- 为独立的托管内存工作按 Stream 分配。
- 将托管分配关联到 Stream。
- 在 CPU 读取托管数据前同步 Stream。

关键文件：

- `README.md`：要求学生将托管内存工作拆分到多个 Stream。
- `streams.cu`：带 TODO 的骨架（托管分配、Stream 关联、CPU 访问前同步、清理）。
- `solution/streams.cu`：完成版本。
- `error_checks.h`：错误检查宏。

练习任务：为每个 Stream 的 A、B、C 数组使用托管内存分配，关联到对应 Stream，在该 Stream 中启动 kernel，CPU 读取前同步，释放所有分配。

关键 API 模式：

```cuda
CUDA_CHECK( cudaMallocManaged((void**)&(s[i].A), sizeof(double) * s[i].len) );
CUDA_CHECK( cudaMallocManaged((void**)&(s[i].B), sizeof(double) * s[i].len) );
CUDA_CHECK( cudaMallocManaged((void**)&(s[i].C), sizeof(double) * s[i].len) );

CUDA_CHECK( cudaStreamAttachMemAsync(s[i].strm, s[i].A) );
CUDA_CHECK( cudaStreamAttachMemAsync(s[i].strm, s[i].B) );
CUDA_CHECK( cudaStreamAttachMemAsync(s[i].strm, s[i].C) );

vector_add<<<grid, threads, 0, s[i].strm>>>(s[i].C, s[i].A, s[i].B,
                                            slen, iterations);
cudaStreamSynchronize(s[i].strm);
```

独特之处：与 `simple-streams` 不同，此版本没有显式的 Host-Device 拷贝；练习聚焦于托管内存的所有权和 Stream 关联。

### 动态并行同步

目录：`exercises/dynamic-sync/`

教授概念：

- 设备端 kernel 启动。
- 从设备代码同步子 kernel。
- 一个线程启动工作后的 Block 级别同步。
- `__device__` 全局数据。

关键文件：

- `README.md`：询问为什么程序打印不出 0 到 96 的值，以及如何修复。
- `simple.cu`：父 kernel 启动子 kernel 但未正确等待的骨架。
- `solution/simple.cu`：添加了设备端 `cudaDeviceSynchronize()` 和 `__syncthreads()`。

练习任务：确保所有父线程在子 kernel `k2` 填充 `data[]` 之后再读取。

关键 API 模式：

```cuda
__device__ int data[blocksize];

__global__ void k1(void)
{
    int idx = threadIdx.x;

    if (idx == 0) {
        k2<<<1, blocksize>>>();
        cudaDeviceSynchronize();
    }
    __syncthreads();

    printf("Thread %i has value %i\n", idx, data[idx]);
}
```

独特之处：引入了动态并行和设备端 CUDA Runtime 调用，在典型动态并行设置中编译需要可重定位设备代码支持。

### 动态 Mandelbrot

目录：`exercises/dynamic-mandelbrot/`

教授概念：

- 递归工作负载中的动态并行。
- Mariani-Silver Mandelbrot 细分算法。
- 从 kernel 内部启动设备端子 kernel。
- 子 kernel 启动后的设备端错误检查。
- CUDA Event 计时。
- 使用可重定位设备代码编译。
- 可选的 PNG 输出。

关键文件：

- `README.md`：描述了递归 Mariani-Silver 算法，要求学生运行和分析代码，然后修改 `MAX_DEPTH` 重新运行。
- `Makefile`：使用 `--relocatable-device-code=true`、`-arch=sm_35`、OpenMP 编译选项和 `-lpng` 构建；支持 `WRITE_PNG=no`。
- `mandelbrot.cu`：动态并行 Mandelbrot 实现。
- `pngwriter.c` 和 `pngwriter.h`：PNG 输出辅助。
- `error_checks.h`：Host/Device 兼容的错误检查宏。
- `solution/mandelbrot.cu`：将 `MAX_DEPTH` 从 3 改为 8。

练习任务：分析提供的动态 Mandelbrot 代码，研究递归深度如何影响性能，特别地将 `MAX_DEPTH` 改为 8。

关键 API 模式：

```cuda
__global__ void mandelbrot_block_k(int *iter_counts, int w, int h,
        complex cmin, complex cmax, int x0, int y0, int d, int depth)
{
    int comm_iter_count = border_iterations(w, h, cmin, cmax, x0, y0, d);

    if(threadIdx.x == 0 && threadIdx.y == 0) {
        if(comm_iter_count != DIFF_ITER_COUNT) {
            iter_fill_k<<<grid, bs>>>(iter_counts, w, x0, y0, d,
                                      comm_iter_count);
        } else if(depth + 1 < MAX_DEPTH && d / SUBDIV > MIN_SIZE) {
            mandelbrot_block_k<<<grid, bs>>>(iter_counts, w, h, cmin, cmax,
                                             x0, y0, d / SUBDIV, depth + 1);
        } else {
            mandelbrot_pixel_k<<<grid, bs>>>(iter_counts, w, h, cmin, cmax,
                                             x0, y0, d);
        }
        CHECK_ERROR_MSG("mandelbrot_block_k");
    }
}
```

构建模式：

```make
mandelbrot: mandelbrot.cu pngwriter.o
	nvcc -O3 -arch=sm_35 $(DFLAGS) --relocatable-device-code=true \
		-Xcompiler -Wall -Xcompiler -fopenmp $^ -o $@ -lpng
```

独特之处：最直接的动态并行工作负载，使用递归设备端 kernel 启动而非 Host 驱动的迭代。

### 多 GPU

目录：`exercises/multi-GPU/`

教授概念：

- 枚举 CUDA 设备。
- 使用 `cudaSetDevice()` 选择活动设备。
- 将向量拆分到多个 GPU。
- 为每个设备分配内存。
- 为每个设备创建一个 Stream。
- 多 GPU 上的异步传输和 kernel 启动。
- 页锁定 Host 内存用于并发传输。
- 使用 CUDA Event 计时多 GPU 工作。

关键文件：

- `README.md`：要求学生将向量加法拆分到两个或更多设备，并在可视化 Profiler 中验证重叠。
- `main.cu`：双 GPU 向量加法骨架。
- `solution/ex5.cu`：完整的双 GPU 解决方案。
- `error_checks.h`：错误检查宏。

练习任务：要求两个设备，分配 pinned Host 数组，将 N 拆分为两个 `Decomp` 区域，在每个设备上分配内存和 Stream，异步拷贝每段数据，在对应 Stream 中启动分段 kernel，拷回数据，同步 Stream 并清理。

关键 API 模式：

```cuda
CUDA_CHECK( cudaGetDeviceCount(&devicecount) );
CUDA_CHECK( cudaSetDevice(i) );
CUDA_CHECK( cudaMalloc((void**)&dA[i], sizeof(double) * dec[i].len) );
CUDA_CHECK( cudaStreamCreate(&(strm[i])) );

CUDA_CHECK( cudaMemcpyAsync(dA[i], &hA[dec[i].start],
                            sizeof(double) * dec[i].len,
                            cudaMemcpyHostToDevice, strm[i]) );

vector_add<<<grid, threads, 0, strm[i]>>>(dC[i], dA[i], dB[i], dec[i].len);

CUDA_CHECK( cudaMemcpyAsync(&hC[dec[i].start], dC[i],
                            sizeof(double) * dec[i].len,
                            cudaMemcpyDeviceToHost, strm[i]) );

CUDA_CHECK( cudaStreamSynchronize(strm[i]) );
```

独特之处：从单个 GPU 上的 Stream 级别并发推进到跨多个 GPU 的设备级别并发。

### MPI Ping Add Pong

目录：`exercises/mpi-ping-add-pong/`

教授概念：

- 将 MPI 和 CUDA Runtime 代码结合。
- 基于节点本地 MPI Rank 选择 GPU 设备。
- 将 CUDA kernel 与 MPI C++ Host 代码分开编译。
- 直接使用设备指针进行 CUDA-aware MPI。
- 手动 GPU→Host→MPI→Host→GPU 中转回退方案。
- 页锁定 Host 内存与 MPI/CUDA 传输。

关键文件：

- `README.md`：要求每个 Rank 选择自己的 GPU，完成 CUDA-aware MPI，可选地实现手动 Host 中转传输。
- `Makefile`：用 `nvcc` 编译 `.cu`，用 `mpicxx` 编译 `.cpp`，然后用 `mpicxx` 和 `-lcudart` 链接。
- `add_kernel.cu`：CUDA kernel 和 C++ 包装函数。
- `pingpong.cpp`：MPI 程序骨架。
- `solution/pingpong.cpp`：完整的 CUDA-aware 和手动传输路径。
- `error_checks.h`：错误检查宏。

练习任务：恰好运行两个 MPI Rank，按节点本地 Rank 选择设备，分配 pinned Host 内存和设备内存，使用设备指针实现直接 CUDA-aware MPI 发送/接收，可选地实现 Host 中转版本。

关键 API 模式：

```cpp
MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0,
                    MPI_INFO_NULL, &intranodecomm);
MPI_Comm_rank(intranodecomm, nodeRank);
CUDA_CHECK( cudaGetDeviceCount(devCount) );
CUDA_CHECK( cudaSetDevice(noderank) );
```

```cpp
// CUDA-aware MPI 路径
if (rank == 0) {
    MPI_Send(dA, N, MPI_DOUBLE, 1, 11, MPI_COMM_WORLD);
    MPI_Recv(dA, N, MPI_DOUBLE, 1, 12, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
} else {
    MPI_Recv(dA, N, MPI_DOUBLE, 0, 11, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    call_kernel(dA, N, grid, tib);
    MPI_Send(dA, N, MPI_DOUBLE, 0, 12, MPI_COMM_WORLD);
}
```

```cuda
// add_kernel.cu
void call_kernel(double *data, int N, int blocksize, int tib)
{
    add_kernel<<<blocksize, tib>>> (data, N);
}
```

独特之处：最终分布式内存练习，也是唯一一个将 CUDA 与 MPI 集成的练习。

## 学习进阶路线

推荐顺序：

1. `compiling` — 验证 CUDA 工具链和运行时访问。
2. `first-cuda-program` — 学习 kernel 索引和显式设备内存。
3. `profile-nvvp` 和 `profile-nvprof` — 在代码仍简单时学习 Profiler 工作流。
4. `debug-cuda-memcheck` 和 `debug-nsight` — 学习内存检查和交互式调试。
5. `error-checking` — 添加健壮的运行时和启动错误处理。
6. `jacobi` — 移植真正的 2D 数值方法。
7. `jacobi-timing-events` — 正确测量 GPU 工作时间。
8. `simple-streams` — 用 Stream 重叠拷贝和计算。
9. `kernel-optimization` — 用 Profiling 证据推理性能。
10. `first-unified-memory-program` — 用托管内存重新审视简单内存管理。
11. `unified-memory-streams` — 结合托管内存和 Stream。
12. `dynamic-sync` — 理解设备端启动和同步。
13. `dynamic-mandelbrot` — 分析递归动态并行。
14. `multi-GPU` — 将一个问题拆分到多个 GPU。
15. `mpi-ping-add-pong` — 将 CUDA 与 MPI 及 CUDA-aware 通信结合。

## CUDA 概念掌握清单

完成课程练习后，你应当能够：

- 使用 `nvcc` 编译 CUDA C/C++，掌握架构选项、`-lineinfo`，在需要时使用可重定位设备代码。
- 编写 `__global__` kernel 并用 `<<<grid, block>>>` 启动。
- 计算 1D 和 2D 全局线程索引。
- 在 kernel 中防止越界内存访问。
- 分配、拷贝和释放显式设备内存。
- 使用托管内存，并知道何时 CPU 访问需要同步。
- 使用页锁定 Host 内存进行异步传输工作流。
- 一致地检查 CUDA Runtime 和 kernel 错误。
- 使用可视化和命令行工具进行 CUDA 应用性能分析。
- 调试 CUDA 内存错误和检查 kernel 线程。
- 使用 CUDA Event 测量 GPU 端耗时。
- 使用 CUDA Stream 实现并发拷贝/计算流水线。
- 使用 Thrust 设备规约作为迭代求解器的一部分。
- 识别分支发散、内存索引问题、float/double 提升和缓存/只读访问优化机会。
- 在 Stencil 风格 kernel 中使用常量内存、共享内存和只读缓存模式。
- 从 kernel 中启动 kernel 并正确同步动态并行。
- 使用 `cudaSetDevice()` 将工作拆分到多个 GPU。
- 将 CUDA kernel 与 MPI Host 代码集成，包括 CUDA-aware MPI 传输。

## 速查表：使用的 CUDA API 和语言特性

内存管理：

```cuda
cudaMalloc((void**)&ptr, bytes);
cudaFree((void*)ptr);
cudaMallocHost((void**)&host_ptr, bytes);
cudaFreeHost((void*)host_ptr);
cudaMallocManaged((void**)&ptr, bytes);
cudaMemset(ptr, value, bytes);
cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice);
cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost);
cudaMemcpy(dst, src, bytes, cudaMemcpyDefault);
cudaMemcpyAsync(dst, src, bytes, kind, stream);
cudaMemcpyToSymbol(symbol, src, bytes);
cudaStreamAttachMemAsync(stream, ptr);
```

执行和同步：

```cuda
kernel<<<grid, block>>>(args);
kernel<<<grid, block, shared_bytes, stream>>>(args);
cudaDeviceSynchronize();
cudaThreadSynchronize();      // 旧代码中使用的遗留调用
cudaStreamSynchronize(stream);
__syncthreads();
```

设备和 Stream 管理：

```cuda
cudaGetDeviceCount(&count);
cudaGetDevice(&device);
cudaSetDevice(device);
cudaGetDeviceProperties(&prop, device);
cudaStreamCreate(&stream);
cudaStreamDestroy(stream);
```

计时和错误：

```cuda
cudaEventCreate(&event);
cudaEventRecord(event, stream_or_0);
cudaEventSynchronize(event);
cudaEventElapsedTime(&milliseconds, start, stop);
cudaEventDestroy(event);
cudaGetLastError();
cudaPeekAtLastError();
cudaGetErrorString(error);
```

CUDA 语言特性：

```cuda
__global__ void kernel(...);
__device__ int data[N];
__host__ __device__ float helper(...);
__shared__ float tile[...];
__constant__ float coeffs[...];
__ldg(&value);
threadIdx.x; threadIdx.y;
blockIdx.x; blockIdx.y;
blockDim.x; blockDim.y;
gridDim.x;
```

CUDA/MPI 练习中使用的 MPI 调用：

```cpp
MPI_Init(&argc, &argv);
MPI_Comm_rank(MPI_COMM_WORLD, &rank);
MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0,
                    MPI_INFO_NULL, &intranodecomm);
MPI_Send(buffer, N, MPI_DOUBLE, peer, tag, MPI_COMM_WORLD);
MPI_Recv(buffer, N, MPI_DOUBLE, peer, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
MPI_Wtime();
MPI_Finalize();
```

## 关于 Solution 的说明

多个目录包含 `solution/` 子目录。它们有助于检查预期实现，但同时也是学习材料的一部分，因为它们展示了课程后续反复使用的惯用模式：

- `first-cuda-program/solution`：最小显式内存模式。
- `error-checking/solution`：可复用的错误检查模式。
- `jacobi/solution` 和 `jacobi-timing-events/solution`：数值 kernel + 计时演进。
- `simple-streams/solution`：pinned 内存和异步 Stream 模式。
- `unified-memory-streams/solution`：与 Stream 关联的托管内存。
- `dynamic-sync/solution`：子 kernel 同步。
- `dynamic-mandelbrot/solution`：更深的动态并行递归深度。
- `multi-GPU/solution`：每设备 Stream 和异步分段。
- `mpi-ping-add-pong/solution`：CUDA-aware MPI 和手动 Host 中转传输。
