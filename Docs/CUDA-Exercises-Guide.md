# CSC CUDA Programming Course — Exercise Guide

## Overview

This repository contains the CUDA C/C++ exercises for CSC - IT Center for
Science's "Introduction to CUDA programming" course. The top-level README
points to CSC's 2017 course page and states that the repository contains the
course exercises. The course-material README places the course at CSC on
17-19 May 2017 and lays out a three-day progression from introductory GPU
programming through tools, optimization, unified memory, dynamic parallelism,
multi-GPU programming, and MPI with CUDA.

This is a CUDA C/C++ course, not a CUDA Fortran course. The exercise sources
use `.cu`, `.cpp`, `.c`, and `.h` files, CUDA runtime APIs, Thrust in the
Jacobi reductions, and MPI in the final ping-pong exercise. Many exercise
directories include a `solution/` subdirectory with completed code that shows
the intended implementation pattern. The `demo/` directory contains a stencil
optimization example, and `course-material/` contains the lecture README and
`intro-to-cuda-csc.pdf`.

## Repository Structure

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

## Course Material

`course-material/README.md` is the course description and run guide. It gives
the three-day schedule, explains local workstation and Taito-GPU usage, and
lists important `nvcc` flags:

```text
nvcc -arch=sm_30 <source>.cu
-lineinfo
-Xptxas="-v"
-Xptxas="-dlcm=ca"
-use_fast_math
-rdc=true
-lcudart
```

The course environment described in the README used classroom Quadro K600 GPUs
with CUDA SDK 8.0 and Taito-GPU batch execution such as:

```bash
srun -n1 -pgpu --gres=gpu:1 ./my_program
```

The lecture slides are stored as `course-material/intro-to-cuda-csc.pdf`.
The slides emphasize CUDA Runtime API programming, not the lower-level driver
API, and cover the same path reflected by the exercises: host/device memory,
kernels and thread hierarchy, debugging/profiling tools, branching and
synchronization, streams and events, kernel optimization, unified memory,
dynamic parallelism, multi-GPU programming, and CUDA-aware MPI.

## Demo: Stencil Code

Directory: `demo/stencil code/`

The demo is not listed as an exercise, but it is important because it shows a
larger optimization case study around a 1D stencil. It compares baseline global
memory, read-only cache loads, constant memory coefficients, and shared-memory
tiling.

Key files:

- `stencil.cu`: complete CUDA program with several stencil kernels, timing via
  `std::chrono`, manual verification, constant memory, read-only cache loads,
  shared memory, host pinned memory, and managed memory.

Concepts shown:

- `__constant__` memory for small coefficient arrays.
- `__ldg()` read-only cache loads.
- `__shared__` memory tile buffering and `__syncthreads()`.
- `cudaMemcpyToSymbol()` for constant memory initialization.
- Pinned host memory with `cudaMallocHost()`.
- Managed memory with `cudaMallocManaged()`.

Representative code:

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

What makes it unique: this is a performance demonstration, not a fill-in
skeleton. It compresses many memory hierarchy topics into one benchmark.

## Exercises

### Compiling

Directory: `exercises/compiling/`

Concepts taught:

- Compiling a CUDA C program with `nvcc`.
- Querying CUDA devices.
- Launching the smallest possible kernel.
- Synchronizing so device-side `printf()` output appears.

Key files:

- `README.md`: asks the student to compile and run `hello.cu` on local
  workstations and Taito-GPU.
- `hello.cu`: complete program; no modifications required.

Exercise task: compile and run the provided CUDA program on both target
platforms.

Key API pattern:

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

What makes it unique: this is the environment smoke test. It introduces kernel
launch syntax without memory allocation or error handling.

### First CUDA Program

Directory: `exercises/first-cuda-program/`

Concepts taught:

- Device memory allocation and freeing.
- Copying results from device to host.
- Computing a global thread index.
- Guarding kernels against over-launch.

Key files:

- `README.md`: lists five TODOs for allocation, free, kernel indexing, launch,
  and copy-back.
- `set.cu`: skeleton with TODOs.
- `solution/set.cu`: completed version.

Exercise task: implement a kernel that fills `A[i] = i` for `N = 128`, allocate
`d_A`, launch the kernel, copy the data back, and free device memory.

Key API pattern:

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

What makes it unique: this is the first full CUDA memory-management workflow,
but intentionally omits error checking so the basic mechanics are visible.

### Profiling With NVVP

Directory: `exercises/profile-nvvp/`

Concepts taught:

- Visual profiling with NVIDIA Visual Profiler (`nvvp`).
- Using `-lineinfo` during compilation.
- Interpreting transfer and kernel execution time.

Key files:

- `README.md`: instructions for compiling with `nvcc -lineinfo main.cu` and
  creating an NVVP profiling session.
- `main.cu`: simple copy kernel plus host-device transfers.

Exercise task: profile the program, inspect host-to-device and device-to-host
copy timings, and inspect kernel execution time.

Key API pattern:

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

What makes it unique: it is tool-first. The code is deliberately simple so the
student can focus on profiler timeline interpretation.

### Profiling With NVPROF

Directory: `exercises/profile-nvprof/`

Concepts taught:

- Command-line profiling with `nvprof`.
- Comparing multiple kernel launches.
- Distinguishing copy sizes, launch ranges, and runtime measurements.

Key files:

- `README.md`: asks the student to compile, run through `nvprof`, and find data
  transfer and kernel times.
- `main.cu`: scale kernel called several times with different launch grids and
  pointer offsets.

Exercise task: use `nvprof` to report data transfer times and kernel runtimes;
optionally reason about floating point operation counts.

Key API pattern:

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

What makes it unique: it shows how the same kernel can appear as separate
profile events depending on launch geometry and arguments.

### Debugging With cuda-memcheck

Directory: `exercises/debug-cuda-memcheck/`

Concepts taught:

- Finding device out-of-bounds accesses.
- Understanding inclusive vs exclusive bounds checks.
- Using cuda-memcheck on an otherwise small program.

Key files:

- `README.md`: asks the student to diagnose why an array copy does not work.
- `main.cu`: buggy version.
- `solution/main.cu`: corrected bounds check.

Exercise task: compile the program, run it under cuda-memcheck, identify that
one thread reads/writes one element beyond the arrays, and fix the guard.

Bug and fix:

```cuda
// Bug:
if(idx > size)
    return;

// Fix:
if(idx >= size)
    return;
```

What makes it unique: the kernel launch creates far more than 1000 threads
(`100 * 128`), so correctness depends entirely on the boundary condition.

### Debugging With Nsight

Directory: `exercises/debug-nsight/`

Concepts taught:

- Importing a CUDA project into Nsight.
- Running under the debugger.
- Using breakpoints and thread/block inspection.
- Inspecting per-thread memory values.

Key files:

- `README.md`: Nsight import/debugging instructions and asks what value thread
  824, thread 56 of block 3, reads.
- `src/Debug_nsight.cu`: random input initialization and a scale kernel.

Exercise task: use Nsight to stop inside the kernel and inspect a selected
thread's value.

Key API pattern:

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

What makes it unique: it is interactive debugging-oriented rather than
implementation-oriented.

### Error Checking

Directory: `exercises/error-checking/`

Concepts taught:

- Wrapping CUDA runtime calls with a macro.
- Checking kernel launch/runtime errors with `cudaGetLastError()`.
- Why explicit synchronization catches asynchronous errors earlier.
- Diagnosing invalid block sizes and invalid host/device pointer usage.

Key files:

- `README.md`: asks the student to add memory operations and kernel launch, then
  experiment with error cases.
- `error-test.cu`: skeleton vector add.
- `error_checks.h`: `CUDA_CHECK` and `CHECK_ERROR_MSG`.
- `solution/error-test.cu`: complete vector add with checks.

Exercise task: complete allocations, copies, launch, synchronization, copy-back,
and cleanup using the provided macros.

Key API pattern:

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

What makes it unique: it formalizes a safety pattern reused by many later
exercises.

### Jacobi Iteration

Directory: `exercises/jacobi/`

Concepts taught:

- Porting a 2D finite-difference stencil from CPU to GPU.
- 2D grids and 2D thread blocks.
- Boundary guards for numerical kernels.
- Comparing CPU and GPU results.
- Thrust reductions for convergence checks.

Key files:

- `README.md`: asks the student to implement `sweepGPU`, experiment with grid
  and block sizes, and inspect performance with NVVP.
- `jacobi.cu`: skeleton CPU and GPU Jacobi program.
- `jacobi.h`: `diffGPU()` implemented with Thrust `transform_reduce`.
- `error_checks.h`: error-check macros.
- `solution/jacobi.cu`: completed GPU version.

Exercise task: implement the GPU sweep equivalent to `sweepCPU`, call it twice
per iteration, copy back results, free device memory, and verify average
difference.

Key API pattern:

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

What makes it unique: it is the first realistic numerical solver and the first
exercise where correctness is numerical rather than just integer equality.

### Jacobi Timing With Events

Directory: `exercises/jacobi-timing-events/`

Concepts taught:

- CUDA event creation, recording, synchronization, elapsed time, and cleanup.
- Separating GPU timing from CPU wall-clock timing.
- Reusing a previous kernel while improving measurement quality.

Key files:

- `README.md`: asks the student to modify the previous Jacobi exercise to time
  GPU work with CUDA events.
- `jacobi.cu`: Jacobi skeleton with event TODOs.
- `solution/jacobi.cu`: completed event timing.
- `jacobi.h`: Thrust convergence reduction.
- `error_checks.h`: error-check macros.

Exercise task: add event initialization, record start and stop, synchronize on
the stop event, compute elapsed milliseconds, print seconds, and destroy the
events.

Key API pattern:

```cuda
cudaEvent_t start, stop;
CUDA_CHECK( cudaEventCreate(&start) );
CUDA_CHECK( cudaEventCreate(&stop) );

CUDA_CHECK( cudaEventRecord(start, 0) );
// repeated sweepGPU launches and diffGPU calls
CUDA_CHECK( cudaEventRecord(stop, 0) );
CUDA_CHECK( cudaEventSynchronize(stop) );
CUDA_CHECK( cudaEventElapsedTime(&gputime, start, stop) );

CUDA_CHECK( cudaEventDestroy(start) );
CUDA_CHECK( cudaEventDestroy(stop) );
```

What makes it unique: it teaches the timing primitive that later stream and
multi-GPU exercises also use.

### Simple Streams

Directory: `exercises/simple-streams/`

Concepts taught:

- Page-locked host memory.
- CUDA stream creation and destruction.
- Asynchronous copies.
- Launching kernels into specific streams.
- Decomposing one vector operation across multiple streams.
- Comparing default-stream timing to 1, 2, 4, and 8 streams.

Key files:

- `README.md`: describes testing multiple streams for an array addition kernel.
- `streams.cu`: skeleton with TODOs for pinned allocation, async copies, stream
  kernel launch, async copy-back, and host cleanup.
- `solution/streams.cu`: completed stream implementation.
- `error_checks.h`: error-check macros.

Exercise task: allocate pinned host memory, create streams, split the vector,
use `cudaMemcpyAsync()` and stream kernel launches for each segment, measure
with events, and free pinned memory.

Key API pattern:

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

What makes it unique: it is the first explicit overlap/concurrency exercise and
requires pinned host memory for meaningful asynchronous transfer behavior.

### Kernel Optimization

Directory: `exercises/kernel-optimization/`

Concepts taught:

- Profiling a nontrivial ray-marching renderer.
- Identifying performance bottlenecks with NVVP/nvprof metrics.
- Floating point precision pitfalls.
- Branch divergence and launch geometry.
- Memory coalescing/indexing mistakes.
- Read-only cache loads with `__ldg()`.
- Considering shared memory for strided access patterns.

Key files:

- `README.md`: describes a fractal volume ray tracer and asks students to use
  NVVP to identify issues, with Taito-GPU `nvprof` commands for metrics and
  timeline collection.
- `main.cu`: renderer with `computeUV`, `trace`, `shade`,
  `globalIllumination`, and `downsample` kernels.
- `util.h`: `Vec3`, `getErrorCuda`, PNG writer, clamp/helpers.
- `solution/main.cu`: annotated/fixed version.
- `solution/util.h`: same helper structure as skeleton.

Exercise task: profile the program, identify the main performance issues, and
optionally fix some. The solution adds comments and changes around accidental
double precision, better launch geometry, x/y indexing, and read-only loads.

Key API and optimization patterns:

```cuda
// Accidental double precision in skeleton:
float uvx = (tan(3.14159265 / 4.0)) * (2.0*x - widthFloat) / widthFloat;

// Solution uses float math:
float uvx = (tanf(3.14159265f / 4.0f)) *
            (2.0f*x - widthFloat) / widthFloat;
```

```cuda
// Skeleton trace launch:
threadsTrace.x = 1024;
threadsTrace.y = 1;

// Solution launch:
threadsTrace.x = 8;
threadsTrace.y = 8;
```

```cuda
// Solution read-only cache in downsample:
outR += __ldg(&rawR[((inY+i)*width*scale)+(inX+j)]);
outG += __ldg(&rawG[((inY+i)*width*scale)+(inX+j)]);
outB += __ldg(&rawB[((inY+i)*width*scale)+(inX+j)]);
```

What makes it unique: it is a profiling and diagnosis exercise rather than a
straight TODO completion. The code is intentionally split into kernels so
students can locate bottlenecks.

### First Unified Memory Program

Directory: `exercises/first-unified-memory-program/`

Concepts taught:

- Porting explicit host/device memory code to unified memory.
- Synchronizing before host access to managed memory.
- Simplifying copy-heavy examples with `cudaMallocManaged()`.

Key files:

- `README.md`: asks students to port the earlier `set.cu` to unified memory.
- `set.cu`: explicit `cudaMalloc()`/`cudaMemcpy()` version.
- `solution/set.cu`: managed-memory version.

Exercise task: replace separate host/device allocations with one managed
allocation, remove explicit copy-back, synchronize after kernel completion, and
free the managed allocation.

Key API pattern:

```cuda
int *A;
cudaMallocManaged((void**)&A, N * sizeof(int));

set<<<2, 64>>>(A, N);
cudaDeviceSynchronize();

for (int i = 0; i < N; i++)
  printf("%i ", A[i]);

cudaFree((void*)A);
```

What makes it unique: it repeats the first CUDA programming task with a
different memory model, making the explicit-vs-managed contrast clear.

### Unified Memory With Streams

Directory: `exercises/unified-memory-streams/`

Concepts taught:

- Combining managed memory and streams.
- Per-stream allocations for independent managed-memory work.
- Attaching managed allocations to streams.
- Synchronizing streams before CPU reads managed results.

Key files:

- `README.md`: asks students to split managed-memory work across multiple
  streams.
- `streams.cu`: skeleton with TODOs for managed allocation, stream attachment,
  synchronization before CPU access, and cleanup.
- `solution/streams.cu`: completed version.
- `error_checks.h`: error-check macros.

Exercise task: allocate each stream's `A`, `B`, and `C` arrays with managed
memory, attach them to the stream, launch the kernel in that stream, synchronize
before reading on the CPU, and free the allocations.

Key API pattern:

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

What makes it unique: unlike `simple-streams`, this version has no explicit
host-device copies; the exercise is about managed-memory ownership and stream
association.

### Dynamic Parallelism Synchronization

Directory: `exercises/dynamic-sync/`

Concepts taught:

- Device-side kernel launches.
- Synchronizing child kernels from device code.
- Block-level synchronization after one thread launches work.
- `__device__` global data.

Key files:

- `README.md`: asks why the program does not print values 0 to 96 and how to
  fix it.
- `simple.cu`: skeleton where parent kernel launches child kernel but does not
  wait correctly.
- `solution/simple.cu`: adds device-side `cudaDeviceSynchronize()` and
  `__syncthreads()`.

Exercise task: make sure all parent threads read `data[]` after child kernel
`k2` has filled it.

Key API pattern:

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

What makes it unique: it introduces dynamic parallelism with device-side CUDA
runtime calls, so it requires relocatable device code support when compiled in
typical dynamic-parallelism settings.

### Dynamic Mandelbrot

Directory: `exercises/dynamic-mandelbrot/`

Concepts taught:

- Dynamic parallelism in a recursive workload.
- Mariani-Silver Mandelbrot subdivision.
- Device-side child kernel launches from within a kernel.
- Device-side error checking after child launches.
- CUDA event timing.
- Building with relocatable device code.
- Optional PNG output.

Key files:

- `README.md`: describes the recursive Mariani-Silver algorithm and asks
  students to run and profile the code, then change `MAX_DEPTH` and rerun.
- `Makefile`: builds with `--relocatable-device-code=true`, `-arch=sm_35`,
  OpenMP compiler flags, and `-lpng`; supports `WRITE_PNG=no`.
- `mandelbrot.cu`: dynamic-parallelism Mandelbrot implementation.
- `pngwriter.c` and `pngwriter.h`: PNG output helper.
- `error_checks.h`: host/device-capable error-check macros.
- `solution/mandelbrot.cu`: changes `MAX_DEPTH` from 3 to 8.

Exercise task: profile the provided dynamic Mandelbrot code, study how
recursion depth affects performance, and specifically change `MAX_DEPTH` to 8.

Key API pattern:

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

Build pattern:

```make
mandelbrot: mandelbrot.cu pngwriter.o
	nvcc -O3 -arch=sm_35 $(DFLAGS) --relocatable-device-code=true \
		-Xcompiler -Wall -Xcompiler -fopenmp $^ -o $@ -lpng
```

What makes it unique: it is the most direct dynamic parallelism workload and
uses recursive device-side kernel launches rather than host-driven iteration.

### Multi-GPU

Directory: `exercises/multi-GPU/`

Concepts taught:

- Enumerating CUDA devices.
- Selecting active devices with `cudaSetDevice()`.
- Splitting a vector across GPUs.
- Allocating per-device memory.
- Creating one stream per device.
- Asynchronous transfers and kernel launches on multiple GPUs.
- Pinned host memory for concurrent transfers.
- Timing multi-GPU work with CUDA events.

Key files:

- `README.md`: asks students to split vector add across two or more devices and
  verify overlap in the visual profiler.
- `main.cu`: skeleton for a two-GPU vector add.
- `solution/ex5.cu`: complete two-GPU solution.
- `error_checks.h`: error-check macros.

Exercise task: require two devices, allocate pinned host arrays, split `N` into
two `Decomp` regions, allocate memory and streams on each device, copy each
slice asynchronously, launch the slice kernel in the matching stream, copy back,
synchronize streams, and clean up.

Key API pattern:

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

What makes it unique: it moves from stream-level concurrency on one GPU to
device-level concurrency across multiple GPUs.

### MPI Ping Add Pong

Directory: `exercises/mpi-ping-add-pong/`

Concepts taught:

- Combining MPI and CUDA runtime code.
- Selecting GPU devices based on node-local MPI rank.
- Compiling CUDA kernels separately from MPI C++ host code.
- CUDA-aware MPI using device pointers directly.
- Manual GPU-to-host-to-MPI-to-host-to-GPU fallback.
- Pinned host memory with MPI/CUDA transfers.

Key files:

- `README.md`: asks each rank to select its own GPU, complete CUDA-aware MPI,
  and optionally implement manual host-staged transfer.
- `Makefile`: compiles `.cu` with `nvcc`, `.cpp` with `mpicxx`, then links with
  `mpicxx` and `-lcudart`.
- `add_kernel.cu`: CUDA kernel and C++ wrapper function.
- `pingpong.cpp`: MPI program skeleton.
- `solution/pingpong.cpp`: complete CUDA-aware and manual transfer paths.
- `error_checks.h`: error-check macros.

Exercise task: run exactly two MPI ranks, choose devices by node-local rank,
allocate pinned host memory and device memory, implement direct CUDA-aware MPI
send/receive using device pointers, and optionally implement the host-staged
version.

Key API pattern:

```cpp
MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0,
                    MPI_INFO_NULL, &intranodecomm);
MPI_Comm_rank(intranodecomm, nodeRank);
CUDA_CHECK( cudaGetDeviceCount(devCount) );
CUDA_CHECK( cudaSetDevice(noderank) );
```

```cpp
// CUDA-aware MPI path
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

What makes it unique: it is the final distributed-memory exercise and the only
one that integrates CUDA with MPI.

## Learning Progression

Recommended order:

1. `compiling` - verify CUDA toolchain and runtime access.
2. `first-cuda-program` - learn kernel indexing plus explicit device memory.
3. `profile-nvvp` and `profile-nvprof` - learn profiler workflows while code is
   still simple.
4. `debug-cuda-memcheck` and `debug-nsight` - learn memory-check and interactive
   debugging.
5. `error-checking` - add robust runtime and launch error handling.
6. `jacobi` - port a real 2D numerical method.
7. `jacobi-timing-events` - measure GPU work correctly.
8. `simple-streams` - overlap copies and kernels with streams.
9. `kernel-optimization` - use profiling evidence to reason about performance.
10. `first-unified-memory-program` - revisit simple memory management with
    managed memory.
11. `unified-memory-streams` - combine managed memory and streams.
12. `dynamic-sync` - understand device-side launches and synchronization.
13. `dynamic-mandelbrot` - profile recursive dynamic parallelism.
14. `multi-GPU` - split one problem over multiple GPUs.
15. `mpi-ping-add-pong` - combine CUDA with MPI and CUDA-aware communication.

## CUDA Concepts Mastery Checklist

After completing the course exercises, you should be able to:

- Compile CUDA C/C++ with `nvcc`, architecture flags, `-lineinfo`, and
  relocatable device code when needed.
- Write `__global__` kernels and launch them with `<<<grid, block>>>`.
- Compute 1D and 2D global thread indices.
- Guard kernels against out-of-bounds memory accesses.
- Allocate, copy, and free explicit device memory.
- Use managed memory and know when CPU access requires synchronization.
- Use pinned host memory for asynchronous transfer workflows.
- Check CUDA runtime and kernel errors consistently.
- Profile CUDA applications with visual and command-line tools.
- Debug CUDA memory errors and inspect kernel threads.
- Use CUDA events for GPU-side elapsed-time measurement.
- Use CUDA streams for concurrent copy/compute pipelines.
- Use Thrust device reductions as part of an iterative solver.
- Recognize branch divergence, memory indexing problems, float/double
  promotion, and cache/read-only access opportunities.
- Use constant memory, shared memory, and read-only cache patterns in stencil
  style kernels.
- Launch kernels from kernels and synchronize dynamic parallelism correctly.
- Split work across multiple GPUs with `cudaSetDevice()`.
- Integrate CUDA kernels with MPI host code, including CUDA-aware MPI transfers.

## Quick Reference: CUDA API Calls and Language Features Used

Memory management:

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

Execution and synchronization:

```cuda
kernel<<<grid, block>>>(args);
kernel<<<grid, block, shared_bytes, stream>>>(args);
cudaDeviceSynchronize();
cudaThreadSynchronize();      // legacy call used in older code
cudaStreamSynchronize(stream);
__syncthreads();
```

Device and stream management:

```cuda
cudaGetDeviceCount(&count);
cudaGetDevice(&device);
cudaSetDevice(device);
cudaGetDeviceProperties(&prop, device);
cudaStreamCreate(&stream);
cudaStreamDestroy(stream);
```

Timing and errors:

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

CUDA language features:

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

MPI calls used in the CUDA/MPI exercise:

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

## Notes on Solutions

Several directories include `solution/` subdirectories. They are useful for
checking the intended implementation, but they are also part of the learning
material because they show idiomatic patterns used repeatedly later in the
course:

- `first-cuda-program/solution`: minimal explicit memory pattern.
- `error-checking/solution`: reusable error-check pattern.
- `jacobi/solution` and `jacobi-timing-events/solution`: numerical kernel plus
  timing evolution.
- `simple-streams/solution`: pinned memory and async stream pattern.
- `unified-memory-streams/solution`: managed memory attached to streams.
- `dynamic-sync/solution`: child-kernel synchronization.
- `dynamic-mandelbrot/solution`: deeper dynamic-parallel recursion depth.
- `multi-GPU/solution`: per-device streams and asynchronous slices.
- `mpi-ping-add-pong/solution`: CUDA-aware MPI and manual host-staged transfer.
