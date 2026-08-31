#ifndef CUDA_CUH
#define CUDA_CUH

#include <stddef.h>
#include <stdio.h>

#define THREADS_PER_BLOCK 256

static inline size_t cuda_blocks(size_t numel) {
    return (numel + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
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
