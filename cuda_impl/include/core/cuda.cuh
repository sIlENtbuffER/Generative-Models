#ifndef CUDA_CUH
#define CUDA_CUH

#include <stddef.h>
#include <stdio.h>

#define THREADS_PER_BLOCK 256

static inline size_t cuda_blocks(size_t numel) {
    return (numel + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
}

__device__ static void block_reduce_add(float *target, float value) {
    __shared__ float partial[THREADS_PER_BLOCK];

    partial[threadIdx.x] = value;
    __syncthreads();

    for (size_t stride=blockDim.x/2; stride>0; stride/=2) {
        if (threadIdx.x < stride) {
            partial[threadIdx.x] += partial[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) atomicAdd(target, partial[0]);
}

#define CUDA_CHECK(call)                                            \
    do {                                                            \
        cudaError_t err = (call);                                   \
        if (err != cudaSuccess) {                                   \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",            \
                    __FILE__, __LINE__, cudaGetErrorString(err));   \
            return -1;                                              \
        }                                                           \
    } while (0)                                                     
#endif
