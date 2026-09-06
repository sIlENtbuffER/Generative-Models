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

__global__ static void linear_bias_forward_kernel(float *out, const float *b, size_t numel, size_t stride) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) return;

    out[i] += b[i % stride];
}

__global__ static void linear_bias_backward_kernel(float *db, const float *grad, size_t numel, size_t stride) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) return;

    atomicAdd(&db[i % stride], grad[i]);
}

int linear_alloc(Linear *linear, size_t in_dim, size_t out_dim) {
    *linear = (Linear){};
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
    tensor_free(&linear->x_T);
    tensor_free(&linear->W_T);

    *linear = (Linear){};
}

int linear_he_init(Linear *linear, uint64_t *seed) {
    float scale = sqrtf(2.0f / (float)linear->in_dim);
    linear_he_init_kernel<<<cuda_blocks(linear->W.numel), THREADS_PER_BLOCK>>>(linear->W.data, linear->W.numel, scale, *seed);
    if (tensor_fill(&linear->b, 0.0f) != 0) return -1;

    (*seed)++;
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

int linear_forward(const Linear *linear, const DeviceTensor *x, DeviceTensor *output) {
    if (tensor_matmul(x, &linear->W, output) != 0) return -1;
    linear_bias_forward_kernel<<<cuda_blocks(output->numel), THREADS_PER_BLOCK>>>(output->data, linear->b.data, output->numel, output->strides[output->ndim - 2]);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

int linear_backward(Linear *linear, const DeviceTensor *x, const DeviceTensor *grad, DeviceTensor *output) {
    const size_t axes[] = {1, 0};
    const size_t x_t_numel = x->shape[1] * x->shape[0];

    if (linear->x_T.data == NULL || x_t_numel > linear->x_T.numel) {
        tensor_free(&linear->x_T);
        if (tensor_alloc_2d(&linear->x_T, x->shape[1], x->shape[0]) != 0) return -1;
    }
    if (linear->W_T.data == NULL && tensor_alloc_2d(&linear->W_T, linear->W.shape[1], linear->W.shape[0]) != 0) return -1;

    DeviceTensor x_T_view = {};
    x_T_view.ndim = 2;
    x_T_view.shape[0] = x->shape[1];
    x_T_view.shape[1] = x->shape[0];
    x_T_view.strides[0] = x->shape[0];
    x_T_view.strides[1] = 1;
    x_T_view.numel = x_t_numel;
    x_T_view.data = linear->x_T.data;

    if (tensor_transpose(x, axes, &x_T_view) != 0 || tensor_transpose(&linear->W, axes, &linear->W_T) != 0) return -1;

    if (tensor_matmul(&x_T_view, grad, &linear->dW) != 0 || tensor_matmul(grad, &linear->W_T, output) != 0) return -1;
    if (tensor_fill(&linear->db, 0.0f) != 0) return -1;
    linear_bias_backward_kernel<<<cuda_blocks(grad->numel), THREADS_PER_BLOCK>>>(linear->db.data, grad->data, grad->numel, grad->strides[grad->ndim - 2]);
    if (cudaGetLastError() != cudaSuccess) return -1;

    return 0;
}
