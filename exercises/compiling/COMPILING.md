# How CUDA Compilation Works

CUDA programs are heterogeneous programs: one source file can contain code for
the CPU and code for the GPU. A file such as `hello.cu` is not compiled in the
same way as a normal C or C++ file, because different parts of the program run
on different processors.

The CUDA compiler driver, `nvcc`, coordinates this process. It separates host
code from device code, sends each part to the correct compiler, packages the GPU
code, and links everything into a normal executable.

## CPU + GPU Heterogeneous Compilation

A typical CUDA source file contains both:

- Host code: ordinary C or C++ code that runs on the CPU.
- Device code: CUDA kernels and device functions that run on the GPU.

For example:

```cpp
__global__ void hello()
{
  printf("Greetings from your GPU\n");
}

int main()
{
  hello<<<1, 1>>>();
  cudaDeviceSynchronize();
}
```

The `main()` function is host code. It runs on the CPU and launches work on the
GPU. The `hello()` function is device code because it is marked with
`__global__`, meaning it is a kernel that can be launched from the host and
executed on the GPU.

These two parts require different compilation paths:

- The CPU code is compiled by a host compiler such as `gcc` or `g++`.
- The GPU code is compiled by NVIDIA's device compiler into GPU instructions.
- The compiled host and device parts are linked together into one executable.

The final program looks like a normal executable, but it contains embedded GPU
code that the CUDA runtime and driver can load onto the GPU when a kernel is
launched.

## NVCC Compilation Flow

`nvcc` is a compiler driver. It does not simply compile everything by itself.
Instead, it manages several compilation tools and steps.

At a high level, the flow is:

1. `nvcc` preprocesses the `.cu` file.
2. `nvcc` separates host code from device code.
3. Host code is passed to the system C/C++ compiler.
4. Device code is compiled to PTX, NVIDIA's intermediate GPU assembly language.
5. PTX is compiled to SASS, the real machine code executed by a specific GPU.
6. Device code is packaged into the object file or executable.
7. Host object files, device code, and CUDA libraries are linked together.

The flow can be pictured as:

```text
hello.cu
  |
  |-- host code --> gcc/g++ ------------> host object code
  |
  |-- device code --> CUDA compiler
                       |
                       |-- PTX intermediate code
                       |
                       |-- SASS / cubin GPU machine code
  |
  +-- link host code + device code + CUDA runtime --> executable
```

### Host Compilation

Host code is ordinary C or C++ code. On Linux, `nvcc` usually invokes `gcc` or
`g++` to compile it. This means that normal C++ syntax, headers, and libraries
are handled by the host compiler.

For example, code such as this is compiled for the CPU:

```cpp
int main(void)
{
  cudaGetDeviceCount(&count);
  hello<<<1, 1>>>();
  cudaDeviceSynchronize();
}
```

The kernel launch syntax, `hello<<<1, 1>>>()`, is CUDA-specific, so `nvcc`
translates it before passing the resulting host code to the host compiler.

### Device Compilation

Device code includes kernels and functions marked with CUDA qualifiers such as:

```cpp
__global__
__device__
```

This code is compiled for the GPU. The usual path is:

```text
CUDA C++ device code -> PTX -> SASS / cubin
```

PTX is a virtual GPU instruction set. It is not tied to one exact GPU model.
SASS is the final machine code for a specific GPU architecture, such as
`sm_75` or `sm_80`.

### Fat Binaries

A CUDA executable can contain GPU code for more than one architecture. This is
called a fat binary.

For example, one executable can include code for both:

- `sm_75`, for Turing GPUs.
- `sm_80`, for Ampere GPUs.

At runtime, the CUDA driver selects the best matching code for the GPU in the
machine. This is useful when the same executable needs to run on different
systems.

A fat binary may also include PTX. If exact SASS code for the current GPU is not
available, the CUDA driver may JIT-compile PTX into machine code at runtime.

## Key NVCC Flags

### `-o`: Choose the Output File

Use `-o` to name the output executable or object file.

```bash
nvcc -o hello hello.cu
```

This creates an executable named `hello`.

Without `-o`, the default output name is usually `a.out`.

### `-arch`: Select a GPU Architecture

The `-arch` flag selects the GPU architecture to compile for.

```bash
nvcc -arch=sm_75 -o hello hello.cu
```

Here, `sm_75` means "generate code for a GPU with compute capability 7.5",
which corresponds to NVIDIA Turing GPUs.

You may also see virtual architectures such as `compute_75`. The difference is:

- `compute_75` describes a virtual PTX target.
- `sm_75` describes a real GPU machine-code target.

For simple programs, `-arch=sm_XX` is often enough.

### `-code`: Select the Final Code Target

The `-code` flag controls which final code target is emitted. It is commonly
used together with `-arch`.

For example:

```bash
nvcc -arch=compute_75 -code=sm_75 -o hello hello.cu
```

This tells `nvcc` to compile using the `compute_75` virtual architecture and
emit real machine code for `sm_75`.

### `-gencode`: Build for Multiple Architectures

The most flexible architecture flag is `-gencode`. It lets you specify multiple
virtual and real architecture pairs.

```bash
nvcc \
  -gencode arch=compute_75,code=sm_75 \
  -gencode arch=compute_80,code=sm_80 \
  -o hello hello.cu
```

This creates a fat binary containing GPU machine code for both `sm_75` and
`sm_80`.

You can also include PTX for forward compatibility:

```bash
nvcc \
  -gencode arch=compute_80,code=sm_80 \
  -gencode arch=compute_80,code=compute_80 \
  -o hello hello.cu
```

The first `-gencode` emits SASS for `sm_80`. The second emits PTX for
`compute_80`, which a newer CUDA driver may JIT-compile for a later GPU.

### `--ptx` or `-ptx`: Emit PTX

Use `--ptx` or `-ptx` to stop after generating PTX.

```bash
nvcc -ptx hello.cu
```

This creates a `.ptx` file instead of a complete executable. PTX is useful for
learning, debugging compilation, and inspecting the virtual GPU instructions
generated from CUDA code.

### `--cubin` or `-cubin`: Emit a Cubin

Use `--cubin` or `-cubin` to generate a CUDA binary object containing GPU
machine code.

```bash
nvcc -arch=sm_75 -cubin hello.cu
```

This creates a `.cubin` file. Unlike PTX, a cubin is specific to a real GPU
architecture.

## Common Compute Capabilities

CUDA architecture names usually appear in two forms:

- `compute_XX`: virtual architecture used for PTX.
- `sm_XX`: real streaming multiprocessor architecture used for SASS.

Common examples are:

| Architecture | GPU generation |
| --- | --- |
| `sm_50`, `sm_52`, `sm_53` | Maxwell |
| `sm_60`, `sm_61`, `sm_62` | Pascal |
| `sm_70` | Volta |
| `sm_75` | Turing |
| `sm_80`, `sm_86` | Ampere |
| `sm_89` | Ada |
| `sm_90` | Hopper |

The best architecture flag depends on the GPU you will run on. If you compile
for an architecture that is too new, the program may not run on older GPUs. If
you compile only for a very old architecture, the program may miss performance
features available on newer GPUs.

## Practical Compilation Examples

The examples below assume you are in this directory and want to compile
`hello.cu`.

### Simple Compilation

```bash
nvcc -o hello hello.cu
```

This asks `nvcc` to compile the CUDA source file and produce an executable named
`hello`.

Run it with:

```bash
./hello
```

### Compile for a Specific GPU Architecture

```bash
nvcc -arch=sm_75 -o hello hello.cu
```

This generates GPU machine code for compute capability 7.5.

### Build a Multi-Architecture Fat Binary

```bash
nvcc \
  -gencode arch=compute_75,code=sm_75 \
  -gencode arch=compute_80,code=sm_80 \
  -o hello hello.cu
```

This embeds GPU code for both `sm_75` and `sm_80` in the executable. At runtime,
the CUDA driver chooses the best match for the installed GPU.

### Generate PTX

```bash
nvcc -ptx hello.cu
```

This produces PTX instead of a complete executable. You can open the generated
`.ptx` file in a text editor to inspect the intermediate GPU code.

### Separate Compilation

You can compile first and link later:

```bash
nvcc -c hello.cu
nvcc -o hello hello.o
```

The first command creates an object file, `hello.o`. The second command links
that object file into an executable. This is the same idea used in larger C or
C++ projects, where many source files are compiled separately and then linked.

## Understanding the Compilation Stages

### 1. Preprocessing

The source file is preprocessed first. This handles `#include`, `#define`, and
conditional compilation such as `#ifdef`.

CUDA programs often include normal C/C++ headers and CUDA headers:

```cpp
#include <cstdio>
#include <cmath>
```

After preprocessing, the compiler sees one expanded translation unit.

### 2. Device Code Compilation

Device code is extracted and compiled by NVIDIA's device compiler.

The usual path is:

```text
CUDA device code -> PTX -> SASS/cubin
```

PTX is an intermediate representation. SASS is real GPU machine code. A cubin
file stores GPU binary code for a particular architecture.

### 3. Host Code Compilation

The host part is compiled by the host compiler. On many Linux systems this is
`gcc` or `g++`.

The host compiler produces ordinary CPU object code, usually in a `.o` file.
The host object code contains calls into the CUDA runtime, including the code
needed to launch kernels.

### 4. Linking

The linker combines:

- Host object code.
- Packaged device code.
- The CUDA runtime library.
- Any other libraries used by the program.

The result is an executable:

```text
host .o + device code + CUDA runtime -> executable
```

### 5. Inspecting the Steps with `--dryrun`

To see what `nvcc` would do internally, use:

```bash
nvcc --dryrun -o hello hello.cu
```

This prints the compilation commands without actually running them. The output
is verbose, but it is useful because it shows that `nvcc` is coordinating many
separate tools rather than acting as a single simple compiler.

## Runtime Considerations

### CUDA Runtime Library

Most CUDA programs use the CUDA runtime API, including calls such as:

```cpp
cudaGetDeviceCount(&count);
cudaDeviceSynchronize();
```

When you compile with `nvcc`, it automatically links the CUDA runtime library
unless you ask it not to. This is one reason it is usually easier to link CUDA
programs with `nvcc` instead of calling `gcc` or `g++` directly.

### Driver Loading and JIT Compilation

When the executable runs and reaches a kernel launch, the CUDA runtime and CUDA
driver load the embedded GPU code.

If the executable contains SASS for the current GPU, the driver can use it
directly. If it contains suitable PTX but not exact SASS for the GPU, the driver
may JIT-compile the PTX into machine code for the installed GPU.

JIT means "just in time": compilation happens at runtime rather than when you
ran `nvcc`.

### Forward Compatibility

PTX provides a form of forward compatibility. A newer CUDA driver can often take
older PTX and compile it for a newer GPU architecture.

For example, an executable might contain:

- SASS for `sm_80`, for current Ampere GPUs.
- PTX for `compute_80`, as a fallback for future GPUs.

On a future GPU, the driver may not find exact SASS, but it may be able to
JIT-compile the PTX.

This does not mean every old CUDA program will automatically use every new GPU
feature. It means PTX can help one executable run across a wider range of GPUs.

## Practical Advice

For small exercises, start with:

```bash
nvcc -o hello hello.cu
```

When you know the GPU architecture, compile for it explicitly:

```bash
nvcc -arch=sm_75 -o hello hello.cu
```

For programs that need to run on several GPU generations, use `-gencode` and
build a fat binary:

```bash
nvcc \
  -gencode arch=compute_75,code=sm_75 \
  -gencode arch=compute_80,code=sm_80 \
  -gencode arch=compute_80,code=compute_80 \
  -o hello hello.cu
```

The main idea is that CUDA compilation is a two-world process: CPU code is built
for the host processor, GPU code is built for one or more GPU architectures, and
`nvcc` packages and links those pieces into one executable.
