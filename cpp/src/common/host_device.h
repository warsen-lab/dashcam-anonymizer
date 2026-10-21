#pragma once

// Marks a function that CUDA kernels call as well as host code (spec 2026-10-20 §4: the mosaic grid is shared by
// the CPU and the GPU): __host__ __device__ under nvcc, nothing for the host compiler, so a VC_WITH_CUDA=OFF build
// compiles the same code as before.
#if defined(__CUDACC__)
#define VC_HOST_DEVICE __host__ __device__
#else
#define VC_HOST_DEVICE
#endif
