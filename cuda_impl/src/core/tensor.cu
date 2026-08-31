#include "core/tensor.cuh"
#include "core/cuda.cuh"

#include <stdint.h>
#include <stdlib.h>

__global__ static void tensor_fill_kernel(float *data, size_t numel, float value) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numel) data[i] = value;
}

__global__ static void tensor_matmul_kernel(DeviceTensor a, DeviceTensor b, DeviceTensor out, size_t a_padding, size_t b_padding, size_t k_size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= out.numel) return;

    size_t batch_ndim = out.ndim - 2;
    size_t i = (idx / out.strides[out.ndim -2]) % out.shape[out.ndim - 2];
    size_t j = (idx / out.strides[out.ndim -1]) % out.shape[out.ndim - 1];

    size_t a_batch_addr = 0, b_batch_addr = 0;
    for (size_t axis=0; axis<batch_ndim; axis++) {
        size_t coordinate = (idx / out.strides[axis]) % out.shape[axis];
        if (axis >= a_padding) {
            size_t a_axis = axis - a_padding;
            a_batch_addr += (a.shape[a_axis] == 1? 0 : coordinate) * a.strides[a_axis];
        }
        if (axis >= b_padding) {
            size_t b_axis = axis - b_padding;
            b_batch_addr += (b.shape[b_axis] == 1? 0 : coordinate) * b.strides[b_axis];
        }
    }

    float sum = 0.0f;
    for (size_t k=0; k<k_size; k++) {
        sum += a.data[a_batch_addr + i * a.strides[a.ndim-2] + k * a.strides[a.ndim-1]] * b.data[b_batch_addr + k * b.strides[b.ndim-2] + j * b.strides[b.ndim-1]];
    }
    out.data[idx] = sum;
}

typedef struct { size_t v[TENSOR_MAX_DIMS]; } AxesArray;

__global__ static void tensor_transpose_kernel(DeviceTensor input, DeviceTensor out, AxesArray axes) {
    size_t output_addr = blockIdx.x * blockDim.x + threadIdx.x;
    if (output_addr >= out.numel) return;

    size_t remaining_idx = output_addr;
    size_t input_addr = 0;

    for (size_t output_axis = 0; output_axis < out.ndim; output_axis++) {
        size_t coordinate = remaining_idx / out.strides[output_axis];
        remaining_idx %= out.strides[output_axis];

        size_t input_axis = axes.v[output_axis];
        input_addr += coordinate * input.strides[input_axis];
    }

    out.data[output_addr] = input.data[input_addr];
}

int tensor_alloc(DeviceTensor *tensor, size_t ndim, const size_t *shape) {
    if (ndim > TENSOR_MAX_DIMS) return -1;

    *tensor = (DeviceTensor){};
    size_t stride = 1;

    for (size_t i=ndim; i-->0;) {
        tensor->shape[i] = shape[i];
        tensor->strides[i] = stride;

        if (shape[i] != 0 && stride > SIZE_MAX / shape[i]) goto fail;

        stride *= shape[i];
    }
    tensor->ndim = ndim;
    tensor->numel = stride;

    if (cudaMalloc(&tensor->data, tensor->numel * sizeof *tensor->data) != cudaSuccess) goto fail;
    if (cudaMemset(tensor->data, 0, tensor->numel * sizeof *tensor->data) != cudaSuccess) goto fail;

    return 0;

fail:
    *tensor = (DeviceTensor){};
    return -1;
}

int tensor_alloc_1d(DeviceTensor *tensor, size_t dim0) {
    const size_t shape[] = {dim0};
    return tensor_alloc(tensor, 1, shape);
}

int tensor_alloc_2d(DeviceTensor *tensor, size_t dim0, size_t dim1) {
    const size_t shape[] = {dim0, dim1};
    return tensor_alloc(tensor, 2, shape);
}

void tensor_free(DeviceTensor *tensor) {
    cudaFree(tensor->data);
    *tensor = (DeviceTensor){};
}


int tensor_fill(DeviceTensor *tensor, float value) {
    // whoosh!!
    tensor_fill_kernel<<<cuda_blocks(tensor->numel), THREADS_PER_BLOCK>>>(tensor->data, tensor->numel, value);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

int tensor_is_same_shape(const DeviceTensor *a, const DeviceTensor *b) {
    if (a->ndim != b->ndim) return 0;
    for (size_t i=0; i<a->ndim; i++) {
        if (a->shape[i] != b->shape[i]) return 0;
    }
    return 1;
}

int tensor_matmul(const DeviceTensor *a, const DeviceTensor *b, DeviceTensor *output) {
    size_t batch_ndim = output->ndim - 2;
    size_t a_padding = batch_ndim - (a->ndim - 2);
    size_t b_padding = batch_ndim - (b->ndim - 2);
    size_t k_size = a->shape[a->ndim - 1];

    tensor_matmul_kernel<<<cuda_blocks(output->numel), THREADS_PER_BLOCK>>>(*a, *b, *output, a_padding, b_padding, k_size);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

int tensor_transpose(const DeviceTensor *input, const size_t *axes, DeviceTensor *output) {
    if (input->numel != output->numel || input->ndim != output->ndim) return -1;
    int seen[TENSOR_MAX_DIMS] = {0};
    for (size_t i=0; i<input->ndim; i++) {
        if (axes[i] >= input->ndim) return -1;
        if (seen[axes[i]]) return -1;
        seen[axes[i]] = 1;
        if (output->shape[i] != input->shape[axes[i]]) return -1;
    }

    AxesArray axes_arr = {0};
    for (size_t i = 0; i < input->ndim; i++) axes_arr.v[i] = axes[i];

    tensor_transpose_kernel<<<cuda_blocks(output->numel), THREADS_PER_BLOCK>>>(*input, *output, axes_arr);

    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

int tensor_device_to_host(const DeviceTensor *src, float *dst) {
    return cudaMemcpy(dst, src->data, src->numel * sizeof *src->data, cudaMemcpyDeviceToHost) == cudaSuccess ? 0 : -1;
}

int tensor_host_to_device(const float *src, DeviceTensor *dst) {
    return cudaMemcpy(dst->data, src, dst->numel * sizeof *dst->data, cudaMemcpyHostToDevice) == cudaSuccess ? 0 : -1;
}
