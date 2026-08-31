#include "modules/linear.cuh"
#include "core/cuda.cuh"

#include <math.h>

__global__ static void linear_he_init_kernel(float *data, size_t numel, float scale, uint64_t seed) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) return;
    
    RNG rng;
    rng_seed(&rng, seed, i);
    data[i] = rng_normal(&rng) * scale;
}

__global__ static void linear_b_cal_kernel(float *out, float *b, size_t numel, size_t stride) {
    __shared__ float partial[THREADS_PER_BLOCK];
    
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    partial[threadIdx.x] = 0.0f;
    if (i < numel) {
        partial[threadIdx.x] = b[i % stride];
    }
    __syncthreads();

    for (size_t s=blockDim.x/2; s>0; s/=2) {
        if (threadIdx.x < s) {
            partial[threadIdx.x] += partial[threadIdx.x + s];
        }
        __syncthreads();
    }
    
    if (threadIdx.x == 0) atomicAdd(&out[i], partial[0]);
}

int linear_alloc(Linear *linear, size_t in_dim, size_t out_dim) {
    *linear = (Linear){0};
    linear->in_dim = in_dim;
    linear->out_dim = out_dim;

    if (tensor_alloc_2d(&linear->W, in_dim, out_dim) != 0 || tensor_alloc_1d(&linear->b, out_dim) != 0 || tensor_alloc_2d(&linear->dW, in_dim, out_dim) != 0 || tensor_alloc_1d(&linear->db, out_dim) != 0) {
        linear_free(linear);
        return -1;
    }
    return 0;
}

void linear_free(Linear *linear) {
    tensor_free(&linear->W); 
    tensor_free(&linear->b); 
    tensor_free(&linear->dW); 
    tensor_free(&linear->db);

    *linear = (Linear){0};
}

int linear_he_init(Linear *linear, uint64_t *seed) {
    float scale = sqrtf(2.0f / (float)linear->in_dim);
    linear_he_init_kernel<<<cuda_blocks(linear->W.numel), THREADS_PER_BLOCK>>>(linear->W.data, linear->W.numel, scale, *seed);
    tensor_fill(&linear->b, 0.0f);

    (*seed)++;
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

int linear_forward(const Linear *linear, const DeviceTensor *x, DeviceTensor *output) {
    if (tensor_matmul(x, &linear->W, output) != 0) return -1;
    linear_b_cal_kernel<<<cuda_blocks(output->numel), THREADS_PER_BLOCK>>>(output->data, linear->b.data, output->numel, output->strides[output->ndim - 2]);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

int linear_backward(Linear *linear, const DeviceTensor *x, const DeviceTensor *grad, DeviceTensor *output) {
    DeviceTensor x_T = {0};
    DeviceTensor W_T = {0};
    const size_t axes[] = {1, 0};
    int status = -1;

    if (tensor_alloc_2d(&x_T, x->shape[1], x->shape[0]) != 0 || tensor_alloc_2d(&W_T, linear->W.shape[1], linear->W.shape[0]) != 0) goto cleanup;
    if (tensor_transpose(x, axes, &x_T) != 0 || tensor_transpose(&linear->W, axes, &W_T) != 0) goto cleanup;

    if (tensor_matmul(&x_T, grad, &linear->dW) != 0 || tensor_matmul(grad, &W_T, output) != 0) goto cleanup;
    tensor_fill(&linear->db, 0.0f);
    linear_b_cal_kernel<<<cuda_blocks(grad->numel), THREADS_PER_BLOCK>>>(grad->data, linear->db.data, grad->numel, grad->strides[grad->ndim - 2]);
    if (cudaGetLastError() != cudaSuccess) goto cleanup;

    status = 0;

cleanup:
    tensor_free(&x_T);
    tensor_free(&W_T);
    return status;
}
